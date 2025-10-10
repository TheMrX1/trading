import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import uuid
import yfinance as yf
import numpy as np
import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)

load_dotenv()
BOT_TOKEN = os.getenv("TEST_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("TEST_BOT_TOKEN not found in .env file")

TRUSTED_USERS = [1085064193, 7424028554]
#TRUSTED_USERS = [1085064193, 1563262750, 829213580, 1221434895, 1229198783, 1647115336, 5405897708]

USER_NAMES = {
    1085064193: "Дима О",
    1563262750: "Маша",
    1221434895: "Кира",
    1229198783: "Катя",
    829213580: "Лиза",
    1647115336: "Ульяна",
    7424028554: "MrX",
    5405897708: "Дима З"
}

user_assets = {}
user_states = {}
user_comments = {}
user_settings = {}

user_asset_names = {}

ticker_name_cache = {}

user_names_cache = {}

blacklist = {}

# Портфель и ордера
user_portfolio = {}  # {user_id: {ticker: {"qty": int, "avg_price": float}}}
user_orders = {}     # {user_id: {order_id: order_dict}}

# Временный контекст для сделок
user_trade_context = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_company_name(ticker):
    ticker = ticker.upper()
    if ticker in ticker_name_cache:
        return ticker_name_cache[ticker]

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get("shortName") or info.get("longName")
        if not name:
            fast_info = getattr(stock, "fast_info", {})
            name = fast_info.get("shortName") if fast_info else None
    except Exception as exc:
        logger.debug(f"Не удалось получить название компании для {ticker}: {exc}")
        name = None

    if not name:
        name = ticker

    ticker_name_cache[ticker] = name
    return name


def get_display_name(ticker, user_id=None):
    ticker = ticker.upper()
    company_name = None
    if user_id:
        company_name = user_asset_names.get(user_id, {}).get(ticker)
    if not company_name:
        company_name = ticker_name_cache.get(ticker)
    if not company_name:
        company_name = get_company_name(ticker)
    if user_id:
        user_asset_names.setdefault(user_id, {})[ticker] = company_name
    ticker_name_cache[ticker] = company_name
    if company_name and company_name != ticker:
        return f"{company_name} ({ticker})"
    return ticker

def main_menu():
    keyboard = [
        [InlineKeyboardButton("➕ Добавить актив", callback_data="add_asset"),
         InlineKeyboardButton("📊 Мои активы", callback_data="my_assets")],
        [InlineKeyboardButton("💼 Мой портфель", callback_data="my_portfolio")],
        [InlineKeyboardButton("👥 Активы группы", callback_data="group_assets"),
         InlineKeyboardButton("🚫 Черный список", callback_data="blacklist")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        await update.message.reply_text("⛔ У вас нет доступа к этому боту.")
        return
    await update.message.reply_text("Привет! Выбери действие:", reply_markup=main_menu())

async def show_assets_menu(query, user_id, page=0):
    assets = user_assets.get(user_id, [])
    per_page = 5
    start = page * per_page
    end = start + per_page
    page_assets = assets[start:end]

    keyboard = []
    for asset in page_assets:
        display_text = get_display_name(asset, user_id)
        keyboard.append([InlineKeyboardButton(display_text, callback_data=f"asset_{asset}")])

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("⬅️", callback_data=f"page_{page-1}"))
    if end < len(assets):
        nav_buttons.append(InlineKeyboardButton("➡️", callback_data=f"page_{page+1}"))
    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    await query.edit_message_text("Ваши активы:", reply_markup=InlineKeyboardMarkup(keyboard))

async def show_portfolio_menu(query, user_id):
    positions = user_portfolio.get(user_id, {})
    # Удаляем нулевые позиции из хранения
    tickers_to_delete = [t for t, p in positions.items() if p.get("qty", 0) <= 0]
    for t in tickers_to_delete:
        try:
            del positions[t]
        except Exception:
            pass
    orders = user_orders.get(user_id, {})
    lines = ["💼 Мой портфель:\n"]
    if not positions:
        lines.append("Пока нет позиций.")
    else:
        total_change = 0.0
        for ticker, pos in positions.items():
            qty = pos.get("qty", 0)
            avg_price = pos.get("avg_price", 0.0)
            name = get_display_name(ticker, user_id)
            # текующая цена
            current = None
            try:
                fi = getattr(yf.Ticker(ticker), "fast_info", {}) or {}
                current = fi.get("last_price")
            except Exception:
                current = None
            if current is None:
                try:
                    hist = yf.Ticker(ticker).history(period="5d")
                    if not hist.empty:
                        pc = "Adj Close" if "Adj Close" in hist.columns else "Close"
                        current = float(hist[pc].iloc[-1])
                except Exception:
                    current = None
            change_value = (current - avg_price) * qty if (current is not None) else 0.0
            total_change += change_value
            lines.append(f"• {name}, {qty} шт, {avg_price:.2f} -> { (current or 0.0):.2f} ({change_value:+.2f} USD)")
            lines.append("")
        lines.append(f"total: {total_change:+.2f} USD")
        lines.append("")

    # Открытые ордера
    # Разделитель, если нет и позиций, и ордеров
    orders = user_orders.get(user_id, {})
    if not positions and not orders:
        lines.append("\n-----------\n")

    lines.append("🧾 Opened orders:\n")
    if not orders:
        lines.append("Нет открытых ордеров.")
    else:
        for oid, od in orders.items():
            lines.append(f"#{oid[:8]} {od['side']} {od['ticker']} {od['qty']} @ {od['price']:.2f} ({od['time_in_force']})")

    keyboard = [
        [InlineKeyboardButton("📜 opened orders", callback_data="orders_open")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
    ]
    await query.edit_message_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(keyboard))

def classify_cycle(df):
    df = df.copy()
    
    price_column = "Adj Close" if "Adj Close" in df.columns else "Close"
    
    df["EMA20"] = df[price_column].ewm(span=20, adjust=False).mean()
    ema_slope = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-2] if len(df) >= 2 else 0.0
    close = df[price_column].iloc[-1]
    ema = df["EMA20"].iloc[-1]
    above = close > ema
    price_std = df[price_column].std()
    flat_threshold = price_std * 0.02 if price_std and not np.isnan(price_std) else 0.0
    if np.isnan(flat_threshold):
        flat_threshold = 0.0
    ema_abs = abs(ema) if not np.isnan(ema) else 0.0
    flat_threshold = max(flat_threshold, max(ema_abs * 1e-4, 1e-6))
    near_flat_ema = abs(ema_slope) <= flat_threshold

    delta = np.sign(df[price_column].diff().fillna(0))
    obv = (delta * df["Volume"]).fillna(0).cumsum()
    obv_slope = obv.iloc[-1] - (obv.iloc[-10] if len(obv) >= 10 else 0)

    window = df[price_column].tail(50)
    rng = (window.max() - window.min()) if len(window) > 0 else 0
    if rng == 0:
        in_range = True
    else:
        in_range = (window.min() + 0.2 * rng) < close < (window.max() - 0.2 * rng)

    if ema_slope > 0 and above and obv_slope > 0:
        return "Markup (рост)"
    if ema_slope < 0 and not above and obv_slope < 0:
        return "Markdown (падение)"
    if near_flat_ema and in_range and obv_slope >= 0:
        return "Accumulation (накопление)"
    if near_flat_ema and in_range and obv_slope < 0:
        return "Distribution (распределение)"
    return "Transition (переход)"

def estimate_liquidity(df, eps_bp=5):
    if df.empty or "Volume" not in df:
        return None
    
    price_column = "Adj Close" if "Adj Close" in df.columns else "Close"
    
    if price_column not in df:
        return None
        
    df["ret_abs"] = (df[price_column].pct_change().abs()).fillna(0)
    valid = df[(df["Volume"] > 0) & (df["ret_abs"] < 0.1)]
    if len(valid) < 20:
        return None
    lam = np.median((valid["ret_abs"] / valid["Volume"]).replace([np.inf, -np.inf], np.nan).dropna())
    if lam is None or lam <= 0:
        return None
    epsilon = eps_bp / 10000.0
    Q = epsilon / lam
    avg_vol = valid["Volume"].mean()
    Q = max(Q, 100.0)
    Q = min(Q, 10.0 * avg_vol)
    return int(Q)

def detect_last_large_buy(df, mult=2):
    if df.empty:
        return None
    
    price_column = "Adj Close" if "Adj Close" in df.columns else "Close"
    
    look = df.tail(100) if len(df) >= 100 else df
    avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()
    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        rng = (row["High"] - row["Low"]) if (row["High"] >= row["Low"]) else 0
        near_high = (rng == 0) or ((row["High"] - row[price_column]) <= 0.1 * (rng + 1e-9))
        if (row["Volume"] > mult * (avg_vol if avg_vol else 1)) and near_high:
            ts = df.index[idx].to_pydatetime()
            return ts, int(row["Volume"])
    return None

def calculate_beta(ticker, benchmark="^GSPC", period="3y"):
    stock = yf.Ticker(ticker)
    benchmark_stock = yf.Ticker(benchmark)
    
    stock_hist = stock.history(period=period)
    benchmark_hist = benchmark_stock.history(period=period)
    
    if len(stock_hist) < 30 or len(benchmark_hist) < 30:
        raise Exception("Недостаточно данных для расчета бета-коэффициента")
    
    stock_price_col = "Adj Close" if "Adj Close" in stock_hist.columns else "Close"
    benchmark_price_col = "Adj Close" if "Adj Close" in benchmark_hist.columns else "Close"
    
    stock_returns = stock_hist[stock_price_col].pct_change().dropna()
    benchmark_returns = benchmark_hist[benchmark_price_col].pct_change().dropna()
    
    aligned_data = stock_returns.align(benchmark_returns, join='inner')
    stock_returns_aligned = aligned_data[0]
    benchmark_returns_aligned = aligned_data[1]
    
    covariance = np.cov(stock_returns_aligned, benchmark_returns_aligned, ddof=1)[0][1]
    benchmark_variance = np.var(benchmark_returns_aligned, ddof=1)
    
    if benchmark_variance == 0:
        raise Exception("Дисперсия эталонного индекса равна нулю")
    
    beta = covariance / benchmark_variance
    return beta, f"https://finance.yahoo.com/quote/{ticker}/key-statistics"

def calculate_beta_5y_monthly(ticker, benchmark="^GSPC"):
    stock = yf.Ticker(ticker)
    info = stock.info
    if info.get("beta") is not None:
        return info["beta"], f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    hist = stock.history(period="5y", interval="1mo")
    benchmark_hist = yf.Ticker(benchmark).history(period="5y", interval="1mo")

    price_col_stock = "Adj Close" if "Adj Close" in hist.columns else "Close"
    price_col_bench = "Adj Close" if "Adj Close" in benchmark_hist.columns else "Close"

    stock_returns = hist[price_col_stock].pct_change().dropna()
    bench_returns = benchmark_hist[price_col_bench].pct_change().dropna()

    aligned_stock, aligned_bench = stock_returns.align(bench_returns, join="inner")

    if len(aligned_stock) < 12:
        raise Exception("Недостаточно данных для расчета 5-летнего бета коэффициента")

    covariance = np.cov(aligned_stock, aligned_bench, ddof=1)[0][1]
    variance = np.var(aligned_bench, ddof=1)
    if variance == 0:
        raise Exception("Дисперсия эталонного индекса равна нулю")

    beta = covariance / variance
    return beta, f"https://finance.yahoo.com/quote/{ticker}/key-statistics"

def calculate_cagr(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if len(hist) < 2:
        raise Exception("Недостаточно данных для расчета CAGR")
    
    if "Adj Close" in hist.columns:
        price_column = "Adj Close"
    elif "Close" in hist.columns:
        price_column = "Close"
    else:
        raise Exception("Нет доступных данных о ценах")
    
    start_price = hist[price_column].iloc[0]
    end_price = hist[price_column].iloc[-1]
    
    days = (hist.index[-1] - hist.index[0]).days
    years = days / 365.25
    
    cagr = ((end_price / start_price) ** (1.0/years)) - 1
    return cagr * 100, f"https://finance.yahoo.com/quote/{ticker}/history"

def calculate_eps(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    if "trailingEps" in info and info["trailingEps"] is not None:
        return info["trailingEps"], f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    elif "epsTrailingTwelveMonths" in info and info["epsTrailingTwelveMonths"] is not None:
        return info["epsTrailingTwelveMonths"], f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    else:
        raise Exception("EPS данные недоступны для этого актива")

def build_info_text(ticker, user_id=None):
    settings = {
        "eps_bp": 5,
        "big_buy_mult": 2,
        "analysis_days": 5,
        "cycle_tf": "5m"
    }

    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{settings['analysis_days']}d", interval=settings['cycle_tf'])
    if df.empty:
        return "Данные недостаточны для этого тикера."

    price_column = "Adj Close" if "Adj Close" in df.columns else "Close"

    # Пытаемся взять максимально актуальную цену и время с fast_info
    fi = getattr(stock, "fast_info", {})
    fast_price = fi.get("last_price")
    market_ts = fi.get("last_market_time") or fi.get("last_trading_time")

    ts = None
    price = None
    if market_ts is not None:
        try:
            ts = datetime.fromtimestamp(int(market_ts), tz=timezone.utc)
        except Exception:
            ts = None
    if fast_price is not None and ts is not None:
        try:
            price = round(float(fast_price), 4)
        except Exception:
            price = None

    if price is None or ts is None:
        last = df.iloc[-1]
        price = round(float(last[price_column]), 4)
        idx_ts = last.name
        ts = idx_ts.to_pydatetime() if hasattr(idx_ts, "to_pydatetime") else datetime.fromtimestamp(idx_ts.timestamp(), tz=timezone.utc)

    look = df.tail(100) if len(df) >= 100 else df
    avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()

    approx_book_vol = estimate_liquidity(df.tail(200), eps_bp=settings["eps_bp"])
    stage = classify_cycle(df)
    big = detect_last_large_buy(df, mult=settings["big_buy_mult"])

    info = []
    company_name = None
    if user_id:
        company_name = user_asset_names.get(user_id, {}).get(ticker)
    if not company_name:
        company_name = ticker_name_cache.get(ticker)
    if not company_name:
        company_name = get_company_name(ticker)
    info.append(f"ℹ️ {company_name} ({ticker})" if company_name != ticker else f"ℹ️ {ticker}")
    ts_msk = (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(ZoneInfo("Europe/Moscow"))
    info.append(f"🕒 Последнее обновление (MSK): {ts_msk.strftime('%Y-%m-%d %H:%M')}")
    info.append(f"💵 Цена: {price} USD")
    recommendation_key, recommendation_mean, num_analysts, distribution, rec_source = fetch_analyst_recommendation(ticker)
    recommendation_parts = []
    if recommendation_key:
        recommendation_parts.append(f"рейтинг: {recommendation_key}")
    if recommendation_mean:
        try:
            recommendation_parts.append(f"mean: {float(recommendation_mean):.2f}")
        except Exception:
            pass
    if num_analysts:
        recommendation_parts.append(f"аналитиков: {num_analysts}")
    if recommendation_parts:
        recommendation_line = "; ".join(recommendation_parts)
        info.append(f"📈 Оценка аналитиков: {recommendation_line}\nИсточник: {rec_source}")
    elif rec_source:
        info.append(f"📈 Оценка аналитиков: данные недоступны\nИсточник: {rec_source}")
    info.append(f"📊 Объём (последняя свеча {settings['analysis_days']}d/{settings['cycle_tf']}): {int(last['Volume'])}")
    
    cycle_periods = [
        (5, "5 дней", "5m"),
        (30, "1 месяц", "1d"),
        (90, "3 месяца", "1d"),
        (180, "6 месяцев", "1d"),
        (365, "1 год", "1d")
    ]
    
    cycle_lines = ["🧭 Стадия цикла:"]
    for days, label, interval in cycle_periods:
        if days <= 30:
            period_df = stock.history(period=f"{days}d", interval=interval)
        else:
            if days == 90:
                period_df = stock.history(period="3mo", interval=interval)
            elif days == 180:
                period_df = stock.history(period="6mo", interval=interval)
            elif days == 365:
                period_df = stock.history(period="1y", interval=interval)
            else:
                period_df = stock.history(period=f"{days}d", interval=interval)
        
        if not period_df.empty:
            period_stage = classify_cycle(period_df)
            cycle_lines.append(f"{label} ({days}d/{interval}): {period_stage}")
        else:
            cycle_lines.append(f"{label} ({days}d/{interval}): данные недоступны")
    
    cycle_info = "\n".join(cycle_lines)
    chart_link = f"https://finance.yahoo.com/quote/{ticker}/chart?p={ticker}"
    info.append(f"{cycle_info}\n{chart_link}")
    
    if approx_book_vol is not None:
        info.append(f"📥 Приближённая дневная ликвидность: ~{approx_book_vol} ед.")
    else:
        info.append("📥 Объем стакана (приближенный): оценка недоступна")
        
    if big:
        ts_big, vol_big = big
        ts_big_msk = (ts_big if ts_big.tzinfo else ts_big.replace(tzinfo=timezone.utc)).astimezone(ZoneInfo("Europe/Moscow"))
        info.append(f"🚀 Последняя крупная покупка: {ts_big_msk.strftime('%Y-%m-%d %H:%M')}, объём {vol_big}")
    else:
        info.append("🚀 Последняя крупная покупка: не обнаружена")

    user_comment = user_comments.get(user_id, {}).get(ticker) if user_id else None
    if user_comment:
        info.append(f"💬 Комментарий: {user_comment}")

    return "\n\n".join(info)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

    user_name = query.from_user.username
    if user_name:
        user_names_cache[user_id] = user_name

    if user_id not in TRUSTED_USERS:
        await query.edit_message_text("⛔ Нет доступа.")
        return

    if query.data == "add_asset":
        user_states[user_id] = "waiting_for_asset"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
        await query.edit_message_text("Введите тикер актива (например, AAPL):",
                                      reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "my_assets":
        assets = user_assets.get(user_id, [])
        if not assets:
            await query.edit_message_text("У вас пока нет активов.", reply_markup=main_menu())
            return
        await show_assets_menu(query, user_id, page=0)

    elif query.data == "my_portfolio":
        await show_portfolio_menu(query, user_id)

    elif query.data == "group_assets":
        keyboard = []
        has_assets = False
        for uid in TRUSTED_USERS:
            assets = user_assets.get(uid, [])
            if assets:
                has_assets = True
                user_display_name = get_user_name(uid)
                keyboard.append([InlineKeyboardButton(user_display_name, callback_data=f"group_user_{uid}")])
        if not has_assets:
            await query.edit_message_text("👥 Активы группы:\n\nПока нет активов у пользователей.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]))
            return

        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
        await query.edit_message_text("Выберите пользователя:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("group_user_"):
        target_user_id = int(query.data.split("_", 2)[2])
        assets = user_assets.get(target_user_id, [])
        comments = user_comments.get(target_user_id, {})
        names = user_asset_names.get(target_user_id, {})

        if not assets:
            text = "У выбранного пользователя пока нет активов."
        else:
            lines = [f"👤 {get_user_name(target_user_id)}"]
            for asset in assets:
                company_name = names.get(asset)
                if not company_name:
                    company_name = get_company_name(asset)
                    user_asset_names.setdefault(target_user_id, {})[asset] = company_name
                ticker_name_cache[asset] = company_name
                comment = comments.get(asset, "")
                comment_part = f": {comment}" if comment else ""
                lines.append(f"{company_name} ({asset}){comment_part}\n")
            text = "\n".join(lines)

        keyboard = [
            [InlineKeyboardButton("⬅️ Назад", callback_data="group_assets")]
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "blacklist":
        blacklist_lines = ["🚫 Черный список:\n"]
        if blacklist:
            for ticker, data in blacklist.items():
                user_name = get_user_name(data["user_id"])
                blacklist_lines.append(f"• {ticker} (добавил: {user_name}) - {data['comment']}")
        else:
            blacklist_lines.append("Черный список пуст.")
        
        keyboard = [
            [InlineKeyboardButton("➕ Добавить в ЧС", callback_data="add_to_blacklist")],
            [InlineKeyboardButton("🗑 Удалить из ЧС", callback_data="remove_from_blacklist")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
        ]
        await query.edit_message_text("\n".join(blacklist_lines), reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "add_to_blacklist":
        user_states[user_id] = "waiting_for_blacklist_ticker"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="blacklist")]]
        await query.edit_message_text("Введите тикер актива для добавления в черный список:",
                                      reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "remove_from_blacklist":
        user_states[user_id] = "waiting_for_remove_blacklist_ticker"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="blacklist")]]
        await query.edit_message_text("Введите тикер актива для удаления из черного списка:",
                                      reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("asset_"):
        ticker = query.data.split("_", 1)[1]
        comment = user_comments.get(user_id, {}).get(ticker, ticker)
        display_name = get_display_name(ticker, user_id)
        keyboard = [
            [InlineKeyboardButton("ℹ️ Информация", callback_data=f"info_{ticker}"),
             InlineKeyboardButton("🗑 Удалить актив", callback_data=f"delete_{ticker}")],
            [InlineKeyboardButton("🧮 Калькулятор", callback_data=f"calc_{ticker}")],
            [InlineKeyboardButton("➕ Купить", callback_data=f"buy_{ticker}"), InlineKeyboardButton("➖ Продать", callback_data=f"sell_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="my_assets")]
        ]
        await query.edit_message_text(f"Актив {display_name}\nКомментарий: {comment}", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("info_"):
        ticker = query.data.split("_", 1)[1]
        try:
            text = build_info_text(ticker, user_id)
            await query.edit_message_text(text)
            await query.message.reply_text("Главное меню:", reply_markup=main_menu())
        except Exception as e:
            await query.edit_message_text(f"Ошибка: {e}")
            await query.message.reply_text("Главное меню:", reply_markup=main_menu())

    elif query.data.startswith("delete_"):
        ticker = query.data.split("_", 1)[1]
        if user_id in user_assets and ticker in user_assets[user_id]:
            user_assets[user_id].remove(ticker)
            if user_id in user_comments and ticker in user_comments[user_id]:
                del user_comments[user_id][ticker]
                if not user_comments[user_id]:
                    del user_comments[user_id]
            if user_id in user_asset_names and ticker in user_asset_names[user_id]:
                del user_asset_names[user_id][ticker]
                if not user_asset_names[user_id]:
                    del user_asset_names[user_id]
            if not user_assets[user_id]:
                del user_assets[user_id]
            save_user_data()
            await query.edit_message_text(f"✅ Актив {ticker} успешно удален!", reply_markup=main_menu())
        else:
            await query.edit_message_text(f"❌ Актив {ticker} не найден в вашем списке.", reply_markup=main_menu())

    elif query.data.startswith("calc_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        keyboard = [
            [InlineKeyboardButton("CAGR", callback_data=f"cagr_{ticker}"),
             InlineKeyboardButton("EPS", callback_data=f"eps_{ticker}")],
            [InlineKeyboardButton("β", callback_data=f"beta_{ticker}"),
             InlineKeyboardButton("P/E Ratio", callback_data=f"pe_{ticker}")],
            [InlineKeyboardButton("RVOL", callback_data=f"rvol_{ticker}"),
             InlineKeyboardButton("🎯 12M Target", callback_data=f"target_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data=f"asset_{ticker}")]
        ]
        await query.edit_message_text(f"🧮 Калькулятор для {display_name}\nВыберите метрику:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("cagr_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        try:
            cagr_5y_value, source_url = calculate_cagr(ticker, period="5y")
            cagr_3y_value, _ = calculate_cagr(ticker, period="3y")
            message_text = f"📈 CAGR для {display_name}:\n\n"
            message_text += f"5-летний: {cagr_5y_value:.2f}%\n"
            message_text += f"3-летний: {cagr_3y_value:.2f}%\n\n"
            message_text += f"Источник данных: {source_url}\n"
            message_text += f"Формула: CAGR = (Конечная стоимость / Начальная стоимость)^(1/n) - 1"
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете CAGR для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("eps_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        try:
            eps_value, source_url = calculate_eps(ticker)
            message_text = f"📊 EPS для {display_name}: ${eps_value:.2f}\n\nИсточник данных: {source_url}"
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете EPS для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("beta_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        try:
            beta_3y_value, source_url = calculate_beta(ticker)
            beta_5y_value, _ = calculate_beta_5y_monthly(ticker)
            message_text = f"📊 Бета-коэффициент для {display_name}:\n\n"
            message_text += f"5-летний (месячные данные): {beta_5y_value:.2f}\n"
            message_text += f"3-летний (дневные данные): {beta_3y_value:.2f}\n\n"
            message_text += f"Источник данных: {source_url}\n"
            message_text += f"Формула: β = Cov(Ri, Rm) / Var(Rm)"
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете бета-коэффициента для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("pe_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        try:
            pe_value, source_url = calculate_pe_ratio(ticker)
            message_text = f"📊 P/E Ratio для {display_name}: {pe_value:.2f}\n\nИсточник данных: {source_url}"
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при получении P/E Ratio для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("rvol_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="30d", interval="1d")
            if df.empty:
                raise Exception("Недостаточно данных для расчета RVOL")
            last = df.iloc[-1]
            look = df.tail(100) if len(df) >= 100 else df
            avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()
            rvol = float(last["Volume"]) / avg_vol if avg_vol else 0.0
            message_text = f"📊 RVOL для {display_name}: {rvol:.2f}\n\n"
            message_text += f"Объём (последняя свеча 30d/1d): {int(last['Volume'])}\n"
            message_text += f"Средний объём: {int(avg_vol)}\n\n"
            message_text += f"Источник данных: https://finance.yahoo.com/quote/{ticker}/key-statistics"
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете RVOL для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("target_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        try:
            target_value, source_url = fetch_consensus_target(ticker)
            if target_value is None:
                raise Exception("Средняя целевая цена недоступна")
            stock = yf.Ticker(ticker)
            current_price = stock.fast_info.get("last_price") if hasattr(stock, "fast_info") else None
            if current_price is None:
                hist = stock.history(period="5d")
                if not hist.empty:
                    price_column = "Adj Close" if "Adj Close" in hist.columns else "Close"
                    current_price = float(hist[price_column].iloc[-1])
            diff_text = ""
            if current_price:
                diff = ((target_value - current_price) / current_price) * 100
                diff_text = f"\nТекущая цена Yahoo: {current_price:.2f} USD ({diff:+.2f}% к таргету)"
            message_text = f"🎯 Консенсусная 12-месячная цель для {display_name}: {target_value:.2f} USD\nИсточник: {source_url}{diff_text}"
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при получении цели для {ticker}: {e}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data in ("trade_market", "trade_day", "trade_gtc"):
        ctx = user_trade_context.get(user_id)
        if not ctx or "qty" not in ctx:
            await query.edit_message_text("Сессия сделки не найдена.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))
            return
        action = ctx["action"]
        qty = ctx["qty"]
        ticker = ctx.get("ticker")
        tif = {"trade_market": "MARKET", "trade_day": "DAY", "trade_gtc": "GTC"}[query.data]
        ctx["tif"] = tif
        if tif == "MARKET":
            # Исполнение по рынку: обновляем портфель сразу
            price_exec = None
            try:
                fi = getattr(yf.Ticker(ticker), "fast_info", {}) or {}
                price_exec = fi.get("last_price")
            except Exception:
                price_exec = None
            if price_exec is None:
                hist = yf.Ticker(ticker).history(period="5d")
                if not hist.empty:
                    pc = "Adj Close" if "Adj Close" in hist.columns else "Close"
                    price_exec = float(hist[pc].iloc[-1])
            price_exec = float(price_exec or 0.0)
            if action == "buy":
                pos = user_portfolio.setdefault(user_id, {}).setdefault(ticker, {"qty": 0, "avg_price": 0.0})
                total_cost = pos["avg_price"] * pos["qty"] + price_exec * qty
                pos["qty"] += qty
                pos["avg_price"] = total_cost / max(pos["qty"], 1)
            else:
                pos = user_portfolio.setdefault(user_id, {}).get(ticker)
                if not pos or pos.get("qty", 0) <= 0:
                    await query.edit_message_text("❌ Нечего продавать.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))
                    return
                sell_qty = min(qty, pos["qty"])
                pos["qty"] -= sell_qty
                if pos["qty"] == 0:
                    pos["avg_price"] = 0.0
                    # Удаляем пустую позицию
                    try:
                        del user_portfolio[user_id][ticker]
                    except Exception:
                        pass
            save_user_data()
            await query.edit_message_text(f"✅ Исполнено по рынку: {action} {ticker} {qty} @ {price_exec:.2f}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))
            user_trade_context.pop(user_id, None)
            await query.message.reply_text("Главное меню:", reply_markup=main_menu())
        else:
            # Запрос цены для лимитки
            ctx["step"] = "price"
            await query.edit_message_text("Введите лимитную цену:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))

    elif query.data == "trade_manual":
        ctx = user_trade_context.get(user_id)
        if not ctx or "qty" not in ctx:
            await query.edit_message_text("Сессия сделки не найдена.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))
            return
        ctx["action"] = "manual_buy"
        ctx["step"] = "price_manual"
        await query.edit_message_text("Введите цену, по которой вы ранее купили актив:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))

    elif query.data == "back":
        await query.edit_message_text("Главное меню:", reply_markup=main_menu())

    elif query.data.startswith("page_"):
        page = int(query.data.split("_")[1])
        await show_assets_menu(query, user_id, page)

    elif query.data.startswith("force_add_"):
        ticker = query.data.split("_", 2)[2]
        user_states[user_id] = f"force_add_{ticker}"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
        await query.edit_message_text(f"Введите комментарий для добавления {ticker} (актив в черном списке!):",
                                      reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("buy_") or query.data == "buy_start":
        ticker = query.data.split("_", 1)[1] if "_" in query.data else None
        user_trade_context[user_id] = {"action": "buy", "ticker": ticker, "step": "qty"}
        await query.edit_message_text("Введите количество акций для покупки:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))

    elif query.data.startswith("sell_") or query.data == "sell_start":
        ticker = query.data.split("_", 1)[1] if "_" in query.data else None
        user_trade_context[user_id] = {"action": "sell", "ticker": ticker, "step": "qty"}
        await query.edit_message_text("Введите количество акций для продажи:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))

    elif query.data == "orders_open":
        orders = user_orders.get(user_id, {})
        if not orders:
            await query.edit_message_text("Нет открытых ордеров.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))
            return
        keyboard = []
        for oid, od in orders.items():
            keyboard.append([InlineKeyboardButton(f"#{oid[:8]} {od['side']} {od['ticker']} {od['qty']} @ {od['price']:.2f}", callback_data=f"order_{oid}")])
        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")])
        await query.edit_message_text("Открытые ордера:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("order_"):
        oid = query.data.split("_", 1)[1]
        od = user_orders.get(user_id, {}).get(oid)
        if not od:
            await query.edit_message_text("Ордер не найден.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="orders_open")]]))
            return
        keyboard = [
            [InlineKeyboardButton("✏️ Изменить цену", callback_data=f"order_edit_{oid}")],
            [InlineKeyboardButton("❌ Отменить", callback_data=f"order_cancel_{oid}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="orders_open")]
        ]
        await query.edit_message_text(f"Ордер #{oid[:8]}\n{od['side']} {od['ticker']} {od['qty']} @ {od['price']:.2f} ({od['time_in_force']})", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("order_edit_"):
        oid = query.data.split("_", 2)[2]
        ctx = user_trade_context.setdefault(user_id, {})
        ctx.update({"action": "edit_order", "oid": oid, "step": "price"})
        await query.edit_message_text("Введите новую цену для ордера:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"order_{oid}")]]))

    elif query.data.startswith("order_cancel_"):
        oid = query.data.split("_", 2)[2]
        if user_id in user_orders and oid in user_orders[user_id]:
            del user_orders[user_id][oid]
            await query.edit_message_text("✅ Ордер отменён.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="orders_open")]]))
        else:
            await query.edit_message_text("Ордер не найден.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="orders_open")]]))

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return

    user_name = update.effective_user.username
    if user_name:
        user_names_cache[user_id] = user_name

    if user_states.get(user_id) == "waiting_for_asset":
        ticker = update.message.text.strip().upper()
        
        if ticker in blacklist:
            blacklist_data = blacklist[ticker]
            user_name = get_user_name(blacklist_data["user_id"])
            comment = blacklist_data["comment"]
            
            message = f"⚠️ Актив {ticker} находится в черном списке!\n"
            message += f"Добавил: {user_name}\n"
            message += f"Комментарий: {comment}"
            
            keyboard = [
                [InlineKeyboardButton("➕ Добавить", callback_data=f"force_add_{ticker}")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
            ]
            await update.message.reply_text(message, reply_markup=InlineKeyboardMarkup(keyboard))
            user_states[user_id] = None
            return
            
        user_states[user_id] = f"waiting_for_comment_{ticker}"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
        company_name = get_company_name(ticker)
        prompt_name = company_name if company_name and company_name != ticker else ticker
        await update.message.reply_text(f"Введите комментарий для актива {prompt_name} (например, один из ведущих тех-гигантов):",
                                      reply_markup=InlineKeyboardMarkup(keyboard))
                                      
    elif user_states.get(user_id, "").startswith("waiting_for_comment_"):
        parts = user_states[user_id].split("_", 3)
        if len(parts) >= 4:
            ticker = parts[3]
            comment = update.message.text.strip()
            
            user_assets.setdefault(user_id, [])
            if ticker not in user_assets[user_id]:
                user_assets[user_id].append(ticker)
            
            user_comments.setdefault(user_id, {})
            user_comments[user_id][ticker] = comment

            user_asset_names.setdefault(user_id, {})
            company_name = get_company_name(ticker)
            user_asset_names[user_id][ticker] = company_name
            ticker_name_cache[ticker] = company_name
            
            save_user_data()
            
            user_states[user_id] = None
            await update.message.reply_text(f"✅ Актив {ticker} добавлен с комментарием '{comment}'!", reply_markup=main_menu())
            
    elif user_states.get(user_id) == "waiting_for_blacklist_ticker":
        ticker = update.message.text.strip().upper()
        user_states[user_id] = f"waiting_for_blacklist_comment_{ticker}"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="blacklist")]]
        await update.message.reply_text(f"Введите комментарий для добавления {ticker} в черный список:",
                                      reply_markup=InlineKeyboardMarkup(keyboard))
                                      
    elif user_states.get(user_id, "").startswith("waiting_for_blacklist_comment_"):
        parts = user_states[user_id].split("_", 4)
        if len(parts) >= 5:
            ticker = parts[4]
            comment = update.message.text.strip()
            
            blacklist[ticker] = {"user_id": user_id, "comment": comment}
            
            save_blacklist()
            await notify_users_about_blacklist(context, ticker, user_id, comment)
            remove_asset_from_all_users(ticker)
            save_user_data()
            
            user_states[user_id] = None
            await update.message.reply_text(f"✅ Актив {ticker} добавлен в черный список с комментарием '{comment}'!", reply_markup=main_menu())
            
    elif user_states.get(user_id) == "waiting_for_remove_blacklist_ticker":
        ticker = update.message.text.strip().upper()
        
        if ticker in blacklist:
            del blacklist[ticker]
            
            save_blacklist()
            
            await update.message.reply_text(f"✅ Актив {ticker} удален из черного списка!", reply_markup=main_menu())
        else:
            await update.message.reply_text(f"❌ Актив {ticker} не найден в черном списке.", reply_markup=main_menu())
            
        user_states[user_id] = None
        
    elif user_states.get(user_id, "").startswith("force_add_"):
        parts = user_states[user_id].split("_", 2)
        if len(parts) >= 3:
            ticker = parts[2]
            comment = update.message.text.strip()
            
            user_assets.setdefault(user_id, [])
            if ticker not in user_assets[user_id]:
                user_assets[user_id].append(ticker)
            
            user_comments.setdefault(user_id, {})
            user_comments[user_id][ticker] = comment

            user_asset_names.setdefault(user_id, {})
            company_name = get_company_name(ticker)
            user_asset_names[user_id][ticker] = company_name
            ticker_name_cache[ticker] = company_name
            
            save_user_data()
            
            user_states[user_id] = None
            await update.message.reply_text(f"✅ Актив {ticker} принудительно добавлен с комментарием '{comment}'!", reply_markup=main_menu())

    # Ввод количества для сделки
    elif user_id in user_trade_context and user_trade_context.get(user_id, {}).get("step") == "qty":
        ctx = user_trade_context[user_id]
        try:
            qty = int(update.message.text.strip())
            if qty <= 0:
                raise ValueError
        except Exception:
            await update.message.reply_text("❌ Некорректное количество. Введите положительное целое число:")
            return
        ctx["qty"] = qty
        ctx["step"] = "price_mode"
        keyboard = [
            [InlineKeyboardButton("Market price", callback_data="trade_market")],
            [InlineKeyboardButton("LP till today", callback_data="trade_day")],
            [InlineKeyboardButton("LP till canceled", callback_data="trade_gtc")],
            [InlineKeyboardButton("Already bought", callback_data="trade_manual")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]
        ]
        await update.message.reply_text("Выберите режим исполнения:", reply_markup=InlineKeyboardMarkup(keyboard))

    # Ввод цены для лимитного ордера (после выбора режима) или редактирования ордера
    elif user_id in user_trade_context and user_trade_context.get(user_id, {}).get("step") == "price":
        ctx = user_trade_context[user_id]
        try:
            price = float(update.message.text.strip().replace(",", "."))
            if price <= 0:
                raise ValueError
        except Exception:
            await update.message.reply_text("❌ Некорректная цена. Введите положительное число:")
            return
        ctx["price"] = price

        # Редактирование существующего ордера
        if ctx.get("action") == "edit_order" and ctx.get("oid"):
            oid = ctx["oid"]
            od = user_orders.get(user_id, {}).get(oid)
            if not od:
                await update.message.reply_text("❌ Ордер не найден.")
            else:
                od["price"] = price
                save_user_data()
                await update.message.reply_text(f"✏️ Цена ордера #{oid[:8]} обновлена на {price:.2f}")
            user_trade_context.pop(user_id, None)
            await update.message.reply_text("Главное меню:", reply_markup=main_menu())
        else:
            # Регистрируем новый лимитный ордер
            oid = str(uuid.uuid4())
            user_orders.setdefault(user_id, {})[oid] = {
                "ticker": ctx.get("ticker") or "UNKNOWN",
                "side": ctx["action"],
                "qty": ctx["qty"],
                "price": price,
                "time_in_force": ctx.get("tif", "DAY"),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            save_user_data()
            await update.message.reply_text(f"✅ Лимитный ордер создан: #{oid[:8]} {ctx['action']} {ctx.get('ticker') or 'UNKNOWN'} {ctx['qty']} @ {price:.2f} ({ctx.get('tif','DAY')})")
            user_trade_context.pop(user_id, None)
            await update.message.reply_text("Главное меню:", reply_markup=main_menu())

    elif user_id in user_trade_context and user_trade_context.get(user_id, {}).get("step") == "price_manual":
        ctx = user_trade_context[user_id]
        try:
            price = float(update.message.text.strip().replace(",", "."))
            if price <= 0:
                raise ValueError
        except Exception:
            await update.message.reply_text("❌ Некорректная цена. Введите положительное число:")
            return
        qty = ctx.get("qty", 0)
        if qty <= 0:
            await update.message.reply_text("❌ Некорректное количество.")
            user_trade_context.pop(user_id, None)
            return
        ticker = ctx.get("ticker") or "UNKNOWN"
        pos = user_portfolio.setdefault(user_id, {}).setdefault(ticker, {"qty": 0, "avg_price": 0.0})
        total_cost = pos["avg_price"] * pos["qty"] + price * qty
        pos["qty"] += qty
        pos["avg_price"] = total_cost / max(pos["qty"], 1)
        save_user_data()
        user_trade_context.pop(user_id, None)
        await update.message.reply_text(f"✅ Добавлено в портфель: {ticker} {qty} @ {price:.2f}")
        await update.message.reply_text("Главное меню:", reply_markup=main_menu())


def load_user_data():
    """Загружает данные пользователей из файла users.txt"""
    global user_assets, user_comments, user_settings, user_asset_names, ticker_name_cache, user_portfolio, user_orders
    try:
        users_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.txt")
        
        if not os.path.exists(users_file_path):
            save_user_data()
            return
        
        with open(users_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        current_user_id = None
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
                
            if line.startswith("USER_ID:"):
                current_user_id = int(line.split(":")[1])
                user_assets[current_user_id] = []
                user_comments[current_user_id] = {}
                user_asset_names[current_user_id] = {}
                user_portfolio[current_user_id] = {}
                user_orders[current_user_id] = {}
                user_settings[current_user_id] = {
                    "eps_bp": 5,
                    "big_buy_mult": 2,
                    "analysis_days": 5,
                    "cycle_tf": "5m"
                }
            elif line.startswith("ASSETS:") and current_user_id:
                current_section = "assets"
            elif line.startswith("COMMENTS:") and current_user_id:
                current_section = "comments"
            elif line.startswith("NAMES:") and current_user_id:
                current_section = "names"
            elif line.startswith("PORTFOLIO:") and current_user_id:
                current_section = "portfolio"
            elif line.startswith("ORDERS:") and current_user_id:
                current_section = "orders"
            elif line.startswith("SETTINGS:") and current_user_id:
                current_section = "settings"
            elif current_section == "assets" and current_user_id:
                if line == "END_ASSETS":
                    current_section = None
                else:
                    user_assets[current_user_id].append(line)
            elif current_section == "comments" and current_user_id:
                if line == "END_COMMENTS":
                    current_section = None
                else:
                    if "=" in line:
                        ticker, comment = line.split("=", 1)
                        user_comments[current_user_id][ticker] = comment
            elif current_section == "settings" and current_user_id:
                if line == "END_SETTINGS":
                    current_section = None
                else:
                    if "=" in line:
                        key, value = line.split("=", 1)
                        user_settings[current_user_id][key] = value
            elif current_section == "names" and current_user_id:
                if line == "END_NAMES":
                    current_section = None
                else:
                    if "=" in line:
                        ticker, name = line.split("=", 1)
                        user_asset_names[current_user_id][ticker] = name
                        ticker_name_cache[ticker] = name
            elif current_section == "portfolio" and current_user_id:
                if line == "END_PORTFOLIO":
                    current_section = None
                else:
                    if "=" in line and "," in line:
                        t, rest = line.split("=", 1)
                        q_str, ap_str = rest.split(",", 1)
                        try:
                            user_portfolio[current_user_id][t] = {"qty": int(q_str), "avg_price": float(ap_str)}
                        except Exception:
                            pass
            elif current_section == "orders" and current_user_id:
                if line == "END_ORDERS":
                    current_section = None
                else:
                    parts = line.split("|")
                    if len(parts) >= 7:
                        oid, t, side, q, p, tif, created = parts[:7]
                        try:
                            user_orders[current_user_id][oid] = {
                                "ticker": t,
                                "side": side,
                                "qty": int(q),
                                "price": float(p),
                                "time_in_force": tif,
                                "created_at": created
                            }
                        except Exception:
                            pass
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных пользователей: {e}")
        user_assets = {}
        user_comments = {}
        user_asset_names = {}
        user_settings = {}
        ticker_name_cache = {}
        user_portfolio = {}
        user_orders = {}

def save_user_data():
    """Сохраняет данные пользователей в файл users.txt"""
    try:
        users_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.txt")
        
        with open(users_file_path, "w", encoding="utf-8") as f:
            for user_id in user_assets.keys():
                f.write(f"USER_ID:{user_id}\n")
                
                f.write("ASSETS:\n")
                for asset in user_assets.get(user_id, []):
                    f.write(f"{asset}\n")
                f.write("END_ASSETS\n")
                
                f.write("COMMENTS:\n")
                comments = user_comments.get(user_id, {})
                for ticker, comment in comments.items():
                    f.write(f"{ticker}={comment}\n")
                f.write("END_COMMENTS\n")
                
                f.write("NAMES:\n")
                names = user_asset_names.get(user_id, {})
                for ticker, name in names.items():
                    f.write(f"{ticker}={name}\n")
                f.write("END_NAMES\n")

                f.write("PORTFOLIO:\n")
                portfolio = user_portfolio.get(user_id, {})
                for t, pos in portfolio.items():
                    f.write(f"{t}={pos.get('qty', 0)},{pos.get('avg_price', 0.0)}\n")
                f.write("END_PORTFOLIO\n")

                f.write("ORDERS:\n")
                orders = user_orders.get(user_id, {})
                for oid, od in orders.items():
                    f.write(f"{oid}|{od['ticker']}|{od['side']}|{od['qty']}|{od['price']}|{od['time_in_force']}|{od['created_at']}\n")
                f.write("END_ORDERS\n")

                f.write("SETTINGS:\n")
                settings = {
                    "eps_bp": 5,
                    "big_buy_mult": 2,
                    "analysis_days": 5,
                    "cycle_tf": "5m"
                }
                for key, value in settings.items():
                    f.write(f"{key}={value}\n")
                f.write("END_SETTINGS\n")
                f.write("\n")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных пользователей: {e}")

load_user_data()

def load_blacklist():
    """Загружает черный список из файла blacklist.txt"""
    global blacklist
    try:
        blacklist_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "blacklist.txt")
        
        if not os.path.exists(blacklist_file_path):
            save_blacklist()
            return
        
        with open(blacklist_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
                
            if "=" in line:
                parts = line.split("=", 2)
                if len(parts) >= 3:
                    ticker = parts[0]
                    user_id = int(parts[1])
                    comment = parts[2]
                    blacklist[ticker] = {"user_id": user_id, "comment": comment}
    except Exception as e:
        logging.error(f"Ошибка при загрузке черного списка: {e}")
        blacklist = {}

def save_blacklist():
    """Сохраняет черный список в файл blacklist.txt"""
    try:
        blacklist_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "blacklist.txt")
        
        with open(blacklist_file_path, "w", encoding="utf-8") as f:
            for ticker, data in blacklist.items():
                f.write(f"{ticker}={data['user_id']}={data['comment']}\n")
    except Exception as e:
        logging.error(f"Ошибка при сохранении черного списка: {e}")

def get_user_name(user_id):
    """Получает имя пользователя по ID"""
    return USER_NAMES.get(user_id, f"User_{user_id}")

def remove_asset_from_all_users(ticker):
    """Удаляет актив из списков всех пользователей"""
    for user_id in user_assets:
        if ticker in user_assets[user_id]:
            user_assets[user_id].remove(ticker)
            if user_id in user_comments and ticker in user_comments[user_id]:
                del user_comments[user_id][ticker]
                if not user_comments[user_id]:
                    del user_comments[user_id]
            if user_id in user_asset_names and ticker in user_asset_names[user_id]:
                del user_asset_names[user_id][ticker]
                if not user_asset_names[user_id]:
                    del user_asset_names[user_id]

async def notify_users_about_blacklist(context, ticker, added_by_user_id, comment):
    """Отправляет уведомления пользователям об добавлении актива в черный список"""
    added_by_name = get_user_name(added_by_user_id)
    
    for user_id in user_assets:
        if ticker in user_assets[user_id]:
            try:
                message = f"⚠️ Актив {ticker} был добавлен в черный список пользователем {added_by_name}.\n"
                message += f"Комментарий: {comment}"
                await context.bot.send_message(chat_id=user_id, text=message)
            except Exception as e:
                logging.error(f"Ошибка при отправке уведомления пользователю {user_id}: {e}")

load_blacklist()

def calculate_pe_ratio(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    if "trailingPE" in info and info["trailingPE"] is not None:
        return info["trailingPE"], f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    elif "forwardPE" in info and info["forwardPE"] is not None:
        return info["forwardPE"], f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    else:
        raise Exception("P/E данные недоступны для этого актива")


def fetch_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="10d")
        if not hist.empty:
            latest = hist["Close"].dropna()
            if not latest.empty:
                return float(latest.iloc[-1]) / 1000.0, "https://finance.yahoo.com/quote/%5ETNX"
    except Exception:
        pass
    return 0.04, "https://finance.yahoo.com/quote/%5ETNX"


def estimate_market_return():
    for ticker in ["^SP500TR", "^SPXTR", "^GSPC"]:
        try:
            tr = yf.Ticker(ticker)
            hist = tr.history(period="5y")
            if len(hist) >= 2:
                price_column = "Adj Close" if "Adj Close" in hist.columns else "Close"
                start_price = hist[price_column].iloc[0]
                end_price = hist[price_column].iloc[-1]
                years = (hist.index[-1] - hist.index[0]).days / 365.25
                if start_price > 0 and years > 0:
                    market_return = (end_price / start_price) ** (1.0 / years) - 1
                    return float(market_return)
        except Exception:
            continue
    return 0.08


def calculate_dcf_valuation(ticker):
    stock = yf.Ticker(ticker)

    risk_free, risk_free_source = fetch_risk_free_rate()
    market_return = estimate_market_return()
    beta_value, beta_source = calculate_beta_5y_monthly(ticker)

    equity_cost = risk_free + beta_value * max(market_return - risk_free, 0.0)
    equity_cost = max(equity_cost, risk_free + 0.01)

    cashflow_df = stock.cashflow
    if cashflow_df is None or cashflow_df.empty or "Free Cash Flow" not in cashflow_df.index:
        raise Exception("Нет данных о свободном денежном потоке")

    fcf_series = cashflow_df.loc["Free Cash Flow"].dropna()
    if fcf_series.empty:
        raise Exception("Нет доступных значений свободного денежного потока")

    fcf_values = list(reversed(fcf_series.tolist()))
    fcf_values = [float(v) for v in fcf_values if not np.isnan(v)]
    if len(fcf_values) < 3:
        raise Exception("Недостаточно исторических значений FCF для прогноза")

    fcf_values = fcf_values[-5:]

    growth_rates = []
    for i in range(1, len(fcf_values)):
        prev = fcf_values[i - 1]
        current = fcf_values[i]
        if prev != 0:
            growth_rates.append((current / prev) - 1)

    if growth_rates:
        median_growth = float(np.median(growth_rates))
        growth_rate = max(min(median_growth, 0.25), -0.2)
    else:
        growth_rate = 0.02

    last_fcf = fcf_values[-1]
    forecast_flows = []
    projected_fcf = last_fcf
    for _ in range(5):
        projected_fcf *= (1 + growth_rate)
        forecast_flows.append(projected_fcf)

    terminal_growth = 0.02 if growth_rate > 0 else 0.01
    if equity_cost <= terminal_growth:
        terminal_growth = min(terminal_growth, equity_cost - 0.01)
        if terminal_growth < 0:
            terminal_growth = 0.0
            equity_cost = max(equity_cost, 0.05)

    discount_factor = 1 + equity_cost
    pv_flows = 0.0
    for year, flow in enumerate(forecast_flows, start=1):
        pv_flows += flow / (discount_factor ** year)

    terminal_value = forecast_flows[-1] * (1 + terminal_growth)
    denominator = equity_cost - terminal_growth if equity_cost > terminal_growth else 0.01
    terminal_value = terminal_value / denominator
    pv_terminal = terminal_value / (discount_factor ** 5)

    equity_value = pv_flows + pv_terminal

    shares_outstanding = None
    try:
        fast_info = stock.fast_info
        shares_outstanding = fast_info.get("shares_outstanding")
    except Exception:
        shares_outstanding = None

    if not shares_outstanding:
        shares_outstanding = stock.info.get("sharesOutstanding")

    if not shares_outstanding or shares_outstanding <= 0:
        raise Exception("Нет данных о количестве акций в обращении")

    intrinsic_value = equity_value / shares_outstanding

    current_price = None
    try:
        fast_info = stock.fast_info
        current_price = fast_info.get("last_price")
    except Exception:
        pass

    if current_price is None:
        hist = stock.history(period="5d")
        if not hist.empty:
            price_column = "Adj Close" if "Adj Close" in hist.columns else "Close"
            current_price = float(hist[price_column].iloc[-1])

    sources = {
        "risk_free": risk_free_source,
        "beta": beta_source,
        "cashflow": f"https://finance.yahoo.com/quote/{ticker}/cash-flow",
        "shares": f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    }

    return {
        "risk_free": risk_free,
        "market_return": market_return,
        "beta": beta_value,
        "discount_rate": equity_cost,
        "historical_fcf": fcf_values,
        "growth_rate": growth_rate,
        "forecast_flows": forecast_flows,
        "terminal_growth": terminal_growth,
        "pv_flows": pv_flows,
        "pv_terminal": pv_terminal,
        "equity_value": equity_value,
        "intrinsic_value": intrinsic_value,
        "current_price": current_price,
        "shares": shares_outstanding,
        "sources": sources
    }

def fetch_consensus_target(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    target = info.get("targetMeanPrice") or info.get("targetMedianPrice") or info.get("targetHighPrice")
    source = f"https://finance.yahoo.com/quote/{ticker}"
    if target is None:
        return None, source
    return float(target), source


def fetch_analyst_recommendation(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    recommendation_key = info.get("recommendationKey")
    recommendation_mean = info.get("recommendationMean")
    num_analysts = info.get("numberOfAnalystOpinions")
    source = f"https://finance.yahoo.com/quote/{ticker}/analysis"

    summary_lines = []
    try:
        summary = stock.recommendations_summary
        if summary is not None and not summary.empty:
            latest = summary.iloc[-1]
            categories = ["strongBuy", "buy", "hold", "sell", "strongSell"]
            for label in categories:
                value = latest.get(label)
                if value and value > 0:
                    summary_lines.append(f"{label}: {int(value)}")
            if "mean" in latest and latest["mean"]:
                recommendation_mean = latest["mean"]
            total = float(sum(latest.get(label, 0) or 0 for label in categories))
            if total > 0 and not recommendation_key:
                if (latest.get("strongBuy", 0) or 0) + (latest.get("buy", 0) or 0) > total * 0.6:
                    recommendation_key = "buy"
                elif (latest.get("sell", 0) or 0) + (latest.get("strongSell", 0) or 0) > total * 0.6:
                    recommendation_key = "sell"
                else:
                    recommendation_key = "hold"
    except Exception:
        pass

    distribution = ", ".join(summary_lines) if summary_lines else None
    return recommendation_key, recommendation_mean, num_analysts, distribution, source

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling()

if __name__ == "__main__":
    main()