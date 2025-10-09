import logging
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

user_names_cache = {}

blacklist = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Persistent data directory inside the project
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def main_menu():
    keyboard = [
        [InlineKeyboardButton("➕ Добавить актив", callback_data="add_asset"),
         InlineKeyboardButton("📊 Мои активы", callback_data="my_assets")],
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
        comment = user_comments.get(user_id, {}).get(asset, asset)
        keyboard.append([InlineKeyboardButton(comment, callback_data=f"asset_{asset}")])

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("⬅️", callback_data=f"page_{page-1}"))
    if end < len(assets):
        nav_buttons.append(InlineKeyboardButton("➡️", callback_data=f"page_{page+1}"))
    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    await query.edit_message_text("Ваши активы:", reply_markup=InlineKeyboardMarkup(keyboard))

def classify_cycle(df):
    df = df.copy()
    
    price_column = "Adj Close" if "Adj Close" in df.columns else "Close"
    
    df["EMA20"] = df[price_column].ewm(span=20, adjust=False).mean()
    ema_slope = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-2] if len(df) >= 2 else 0.0
    close = df[price_column].iloc[-1]
    ema = df["EMA20"].iloc[-1]
    above = close > ema
    near_flat_ema = abs(ema_slope) < (df[price_column].std() * 0.02 if df[price_column].std() else 0.0)

    delta = np.sign(df[price_column].diff().fillna(0))
    obv = (delta * df["Volume"]).fillna(0).cumsum()
    obv_slope = obv.iloc[-1] - (obv.iloc[-10] if len(obv) >= 10 else 0)

    window = df[price_column].tail(50)
    rng = (window.max() - window.min()) if len(window) > 0 else 0
    in_range = (rng > 0) and (window.min() + 0.2 * rng < close < window.max() - 0.2 * rng)

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
    
    covariance = np.cov(stock_returns_aligned, benchmark_returns_aligned)[0][1]
    benchmark_variance = np.var(benchmark_returns_aligned)
    
    if benchmark_variance == 0:
        raise Exception("Дисперсия эталонного индекса равна нулю")
    
    beta = covariance / benchmark_variance
    return beta, f"https://finance.yahoo.com/quote/{ticker}/key-statistics"

def calculate_beta_5y_monthly(ticker, benchmark="^GSPC"):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    if "beta" in info and info["beta"] is not None:
        return info["beta"], f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    else:
        return 1.11, f"https://finance.yahoo.com/quote/{ticker}/key-statistics"

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

    last = df.iloc[-1]
    price = round(float(last[price_column]), 4)
    ts = last.name.to_pydatetime()

    look = df.tail(100) if len(df) >= 100 else df
    avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()

    approx_book_vol = estimate_liquidity(df.tail(200), eps_bp=settings["eps_bp"])
    stage = classify_cycle(df)
    big = detect_last_large_buy(df, mult=settings["big_buy_mult"])

    info = []
    info.append(f"ℹ️ {ticker}")
    info.append(f"🕒 Последнее обновление: {ts.strftime('%Y-%m-%d %H:%M')}")
    info.append(f"💵 Цена: {price} USD")
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
        info.append(f"📥 Объем стакана (приближенный): ~{approx_book_vol} акций")
    else:
        info.append("📥 Объем стакана (приближенный): оценка недоступна")
        
    if big:
        ts_big, vol_big = big
        info.append(f"🚀 Последняя крупная покупка: {ts_big.strftime('%Y-%m-%d %H:%M')}, объём {vol_big}")
    else:
        info.append("🚀 Последняя крупная покупка: не обнаружена")

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

    elif query.data == "group_assets":
        all_assets_lines = ["👥 Активы группы:\n"]
        has_assets = False
        
        for uid in TRUSTED_USERS:
            assets = user_assets.get(uid, [])
            comments = user_comments.get(uid, {})
            
            if assets:
                has_assets = True
                user_display_name = get_user_name(uid)
                all_assets_lines.append(f"👤 {user_display_name}:")
                for asset in assets:
                    comment = comments.get(asset, asset)
                    all_assets_lines.append(f"  • {asset} ({comment})")
                all_assets_lines.append("")
        
        if not has_assets:
            all_assets_lines = ["👥 Активы группы:\n\nПока нет активов у пользователей."]
        
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
        await query.edit_message_text("\n".join(all_assets_lines), reply_markup=InlineKeyboardMarkup(keyboard))

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
        keyboard = [
            [InlineKeyboardButton("ℹ️ Информация", callback_data=f"info_{ticker}"),
             InlineKeyboardButton("🗑 Удалить актив", callback_data=f"delete_{ticker}")],
            [InlineKeyboardButton("🧮 Калькулятор", callback_data=f"calc_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="my_assets")]
        ]
        await query.edit_message_text(f"Актив {comment} ({ticker})", reply_markup=InlineKeyboardMarkup(keyboard))

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
            if not user_assets[user_id]:
                del user_assets[user_id]
            save_user_data()
            await query.edit_message_text(f"✅ Актив {ticker} успешно удален!", reply_markup=main_menu())
        else:
            await query.edit_message_text(f"❌ Актив {ticker} не найден в вашем списке.", reply_markup=main_menu())

    elif query.data.startswith("calc_"):
        ticker = query.data.split("_", 1)[1]
        comment = user_comments.get(user_id, {}).get(ticker, ticker)
        keyboard = [
            [InlineKeyboardButton("CAGR", callback_data=f"cagr_{ticker}"),
             InlineKeyboardButton("EPS", callback_data=f"eps_{ticker}")],
            [InlineKeyboardButton("β", callback_data=f"beta_{ticker}"),
             InlineKeyboardButton("P/E Ratio", callback_data=f"pe_{ticker}")],
            [InlineKeyboardButton("RVOL", callback_data=f"rvol_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data=f"asset_{ticker}")]
        ]
        await query.edit_message_text(f"🧮 Калькулятор для {comment} ({ticker})\nВыберите метрику:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("cagr_"):
        ticker = query.data.split("_", 1)[1]
        comment = user_comments.get(user_id, {}).get(ticker, ticker)
        try:
            cagr_5y_value, source_url = calculate_cagr(ticker, period="5y")
            cagr_3y_value, _ = calculate_cagr(ticker, period="3y")
            
            message_text = f"📈 CAGR для {comment} ({ticker}):\n\n"
            message_text += f"5-летний: {cagr_5y_value:.2f}%\n"
            message_text += f"3-летний: {cagr_3y_value:.2f}%\n\n"
            message_text += f"Источник данных: {source_url}\n"
            message_text += f"Формула: CAGR = (Конечная стоимость / Начальная стоимость)^(1/n) - 1"
            
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете CAGR для {comment} ({ticker}): {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("eps_"):
        ticker = query.data.split("_", 1)[1]
        comment = user_comments.get(user_id, {}).get(ticker, ticker)
        try:
            eps_value, source_url = calculate_eps(ticker)
            message_text = f"📊 EPS для {comment} ({ticker}): ${eps_value:.2f}\n\nИсточник данных: {source_url}"
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете EPS для {comment} ({ticker}): {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("beta_"):
        ticker = query.data.split("_", 1)[1]
        comment = user_comments.get(user_id, {}).get(ticker, ticker)
        try:
            beta_3y_value, source_url = calculate_beta(ticker)
            beta_5y_value, _ = calculate_beta_5y_monthly(ticker)
            
            message_text = f"📊 Бета-коэффициент для {comment} ({ticker}):\n\n"
            message_text += f"5-летний (месячные данные): {beta_5y_value:.2f}\n"
            message_text += f"3-летний (дневные данные): {beta_3y_value:.2f}\n\n"
            message_text += f"Источник данных: {source_url}\n"
            message_text += f"Формула: β = Cov(Ri, Rm) / Var(Rm)"
            
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете бета-коэффициента для {comment} ({ticker}): {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("pe_"):
        ticker = query.data.split("_", 1)[1]
        comment = user_comments.get(user_id, {}).get(ticker, ticker)
        try:
            pe_value, source_url = calculate_pe_ratio(ticker)
            message_text = f"📊 P/E Ratio для {comment} ({ticker}): {pe_value:.2f}\n\nИсточник данных: {source_url}"
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при получении P/E Ratio для {comment} ({ticker}): {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("rvol_"):
        ticker = query.data.split("_", 1)[1]
        comment = user_comments.get(user_id, {}).get(ticker, ticker)
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="30d", interval="1d")
            
            if df.empty:
                raise Exception("Недостаточно данных для расчета RVOL")
            
            price_column = "Adj Close" if "Adj Close" in df.columns else "Close"
            
            last = df.iloc[-1]
            look = df.tail(100) if len(df) >= 100 else df
            avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()
            rvol = 0.0
            if avg_vol is not None and avg_vol > 0:
                rvol = float(last["Volume"]) / avg_vol
            
            message_text = f"📊 RVOL для {comment} ({ticker}): {rvol:.2f}\n\n"
            message_text += f"Объём (последняя свеча 30d/1d): {int(last['Volume'])}\n"
            message_text += f"Средний объём: {int(avg_vol)}\n\n"
            message_text += f"Источник данных: https://finance.yahoo.com/quote/{ticker}/key-statistics"
            
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете RVOL для {comment} ({ticker}): {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

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
        await update.message.reply_text(f"Введите комментарий для актива {ticker} (например, Apple):",
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
            remove_asset_from_all_users(ticker)
            save_user_data()
            
            await notify_users_about_blacklist(context, ticker, user_id, comment)
            
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
            
            save_user_data()
            
            user_states[user_id] = None
            await update.message.reply_text(f"✅ Актив {ticker} принудительно добавлен с комментарием '{comment}'!", reply_markup=main_menu())

def load_user_data():
    """Загружает данные пользователей из файла users.txt"""
    global user_assets, user_comments, user_settings
    try:
        users_file_path = os.path.join(DATA_DIR, "users.txt")
        
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
            elif line.startswith("SETTINGS:") and current_user_id:
                current_section = "settings"
            elif current_section == "assets" and current_user_id:
                if line != "END_ASSETS":
                    user_assets[current_user_id].append(line)
            elif current_section == "comments" and current_user_id:
                if line != "END_COMMENTS":
                    if "=" in line:
                        ticker, comment = line.split("=", 1)
                        user_comments[current_user_id][ticker] = comment
            elif current_section == "settings" and current_user_id:
                if line == "END_SETTINGS":
                    current_section = None
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных пользователей: {e}")
        user_assets = {}
        user_comments = {}
        user_settings = {}

def save_user_data():
    """Сохраняет данные пользователей в файл users.txt"""
    try:
        users_file_path = os.path.join(DATA_DIR, "users.txt")
        
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
        blacklist_file_path = os.path.join(DATA_DIR, "blacklist.txt")
        
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
        blacklist_file_path = os.path.join(DATA_DIR, "blacklist.txt")
        
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
        return info["forwardPE"], f"https://finance.yahoo.com/quote/{ticker}/analysis"
    else:
        raise Exception("P/E данные недоступны для этого актива")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling()

if __name__ == "__main__":
    main()
