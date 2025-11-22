import logging
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import yfinance as yf
import numpy as np
import os
import time
from urllib.parse import quote_plus
from html import escape
from telegram.constants import ParseMode
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InlineQueryResultArticle, InputTextMessageContent, WebAppInfo
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes, InlineQueryHandler
)
from uuid import uuid4

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in .env file")

#TRUSTED_USERS = [1085064193, 7424028554]
TRUSTED_USERS = [1085064193, 1563262750, 829213580, 1221434895, 1229198783, 1647115336, 5405897708]

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

# Портфель
user_portfolio = {}  # {user_id: {ticker: {"qty": int, "avg_price": float}}}

# Временный контекст для сделок
user_trade_context = {}

# Дополнительные выведенные средства
user_extra_funds = {}  # {user_id: float}

# Кэш статистики группы
group_stats_cache = {}  # {"last_update": ts, "data": {...}}

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


def get_finviz_chart_url(ticker: str, period: str = "i") -> str:
    encoded_ticker = quote_plus(ticker.upper())
    cache_bust = int(time.time())
    # period: i = intraday, d = daily. Add cache-busting param to avoid Telegram CDN caching
    return f"https://finviz.com/chart.ashx?t={encoded_ticker}&ty=c&ta=1&p={period}&s=l&_={cache_bust}"


def format_source(url: str) -> str:
    safe_url = escape(url, quote=True)
    return f"<b><i><a href=\"{safe_url}\">источник</a></i></b>"


def map_to_tradingview(ticker: str) -> str:
    """Maps Yahoo Finance tickers to TradingView tickers."""
    ticker = ticker.upper()
    
    # Crypto (e.g., BTC-USD -> BINANCE:BTCUSD)
    if ticker.endswith("-USD"):
        # Remove -USD and add BINANCE: prefix (common for major cryptos)
        # TradingView usually resolves BTCUSD to a major exchange, but BINANCE is a safe default for crypto
        clean_ticker = ticker.replace("-USD", "USD")
        return f"BINANCE:{clean_ticker}"
    
    # Forex (e.g., EURUSD=X -> FX:EURUSD)
    if ticker.endswith("=X"):
        clean_ticker = ticker.replace("=X", "")
        return f"FX:{clean_ticker}"
    
    # MOEX (e.g., SBER.ME -> MOEX:SBER)
    if ticker.endswith(".ME"):
        clean_ticker = ticker.replace(".ME", "")
        return f"MOEX:{clean_ticker}"
        
    # Default (US Stocks usually match, e.g., AAPL -> AAPL)
    return ticker


def map_to_tradingview(ticker: str) -> str:
    """Maps Yahoo Finance tickers to TradingView tickers."""
    ticker = ticker.upper()
    
    # Crypto (e.g., BTC-USD -> BINANCE:BTCUSD)
    if ticker.endswith("-USD"):
        # Remove -USD and add BINANCE: prefix (common for major cryptos)
        # TradingView usually resolves BTCUSD to a major exchange, but BINANCE is a safe default for crypto
        clean_ticker = ticker.replace("-USD", "USD")
        return f"BINANCE:{clean_ticker}"
    
    # Forex (e.g., EURUSD=X -> FX:EURUSD)
    if ticker.endswith("=X"):
        clean_ticker = ticker.replace("=X", "")
        return f"FX:{clean_ticker}"
    
    # MOEX (e.g., SBER.ME -> MOEX:SBER)
    if ticker.endswith(".ME"):
        clean_ticker = ticker.replace(".ME", "")
        return f"MOEX:{clean_ticker}"
        
    # Default (US Stocks usually match, e.g., AAPL -> AAPL)
    # Some indices might need mapping (e.g., ^GSPC -> SP:SPX), but let's stick to basics first
    return ticker


def fetch_finviz_insights(ticker: str) -> list:
    """Attempt to extract AI-summarized insight from finviz quote page.
    Returns at most one main summary plus optional secondary sentences.
    """
    url = f"https://finviz.com/quote.ashx?t={quote_plus(ticker.upper())}&p=d"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8"
    }
    insights: list[str] = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return insights
        soup = BeautifulSoup(resp.text, "html.parser")

        # Primary: FINVIZ embeds an init JSON with id 'why-stock-moving-init-data-<n>'
        try:
            init_scripts = soup.find_all("script", id=lambda x: isinstance(x, str) and x.startswith("why-stock-moving-init-data"))
            for sc in init_scripts:
                raw = (sc.string or sc.get_text("", strip=True) or "").strip()
                if not raw:
                    continue
                data = json.loads(raw)
                wm = data.get("whyMoving") if isinstance(data, dict) else None
                if wm:
                    headline = wm.get("headline") or wm.get("summary")
                    if headline:
                        insights.append(" ".join(str(headline).split()))
                        return insights[:1]
        except Exception:
            pass
        # If no init JSON found, считаем что инсайтов нет — возвращаем пусто
        return insights
    except Exception:
        return insights

def get_msk_time_str(ts=None):
    """Возвращает строку времени в MSK (UTC+3)"""
    if ts is None:
        ts = time.time()
    elif isinstance(ts, datetime):
        ts = ts.timestamp()
    
    # Direct conversion from timestamp to MSK datetime
    dt = datetime.fromtimestamp(ts, tz=ZoneInfo("Europe/Moscow"))
    return dt.strftime('%d.%m.%Y %H:%M')
def main_menu():
    keyboard = [
        [InlineKeyboardButton("➕ Добавить актив", callback_data="add_asset"),
         InlineKeyboardButton("📊 Мои активы", callback_data="my_assets")],
        [InlineKeyboardButton("💡 Инсайты", callback_data="insights")],
        [InlineKeyboardButton("💼 Мой портфель", callback_data="my_portfolio"),
         InlineKeyboardButton("👥 Инвестиции группы", callback_data="group_investments")],
        [InlineKeyboardButton("👥 Активы группы", callback_data="group_assets"),
         InlineKeyboardButton("🚫 Черный список", callback_data="blacklist")]
    ]
    keyboard.append([InlineKeyboardButton("🏷 Сектор актива", callback_data="sectors"),
                     InlineKeyboardButton("ℹ️ Информация по тикеру", callback_data="ticker_info")])
    keyboard.append([InlineKeyboardButton("⚙️ Настройки", callback_data="settings")])
    return InlineKeyboardMarkup(keyboard)

def settings_menu():
    keyboard = [
        [InlineKeyboardButton("📊 Тип графика", callback_data="chart_settings")],
        [InlineKeyboardButton("🧠 Советы (Таймфрейм)", callback_data="advises_settings")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="show_main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def chart_settings_menu(user_id):
    current_setting = user_settings.get(user_id, {}).get("chart_type", "static")
    static_text = "✅ Статический (Картинка)" if current_setting == "static" else "Статический (Картинка)"
    dynamic_text = "✅ Динамический (Widget)" if current_setting == "dynamic" else "Динамический (Widget)"
    
    keyboard = [
        [InlineKeyboardButton(static_text, callback_data="set_chart_static")],
        [InlineKeyboardButton(dynamic_text, callback_data="set_chart_dynamic")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="settings")]
    ]
    return InlineKeyboardMarkup(keyboard)

def advises_settings_menu(user_id):
    current_setting = user_settings.get(user_id, {}).get("advises_interval", "1M")
    w1_text = "✅ 1 Неделя" if current_setting == "1W" else "1 Неделя"
    m1_text = "✅ 1 Месяц" if current_setting == "1M" else "1 Месяц"
    
    keyboard = [
        [InlineKeyboardButton(w1_text, callback_data="set_advises_1W")],
        [InlineKeyboardButton(m1_text, callback_data="set_advises_1M")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="settings")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        await update.message.reply_text("⛔ У вас нет доступа к этому боту.")
        return
    
    name = get_user_name(user_id)
    text = (f"👋 Привет, {name}!\n\n"
            "Я разработан... Да впринципе Вам пофигу, Кем. А в остальном, желаю удачи и приятного использования")
    
    keyboard = [[InlineKeyboardButton("главное меню", callback_data="show_main_menu")]]
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

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
    await query.edit_message_text("📋 <b>Ваши активы:</b>", parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))

def get_portfolio_text_and_keyboard(user_id):
    positions = user_portfolio.get(user_id, {})
    # Удаляем нулевые позиции из хранения
    tickers_to_delete = [t for t, p in positions.items() if p.get("qty", 0) <= 0]
    for t in tickers_to_delete:
        try:
            del positions[t]
        except Exception:
            pass
    lines = ["💼 <b>Мой портфель:</b>", ""]
    if not positions:
        lines.append("<i>Пока нет позиций.</i>")
    else:
        total_change = 0.0
        total_invested = 0.0
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
            
            current_val = current or 0.0
            change_value = (current_val - avg_price) * qty if (current is not None) else 0.0
            total_change += change_value
            total_invested += (avg_price * qty)
            
            lines.append(f"• <b>{name}</b>, {qty} шт, {avg_price:.2f} -> {current_val:.2f} ({change_value:+.2f} USD)")
            lines.append("")
            
    lines.append(f"💰 <b>Инвестировано:</b> {total_invested:.2f} USD")
    
    extra = user_extra_funds.get(user_id, 0.0)
    if extra != 0:
        lines.append(f"💵 <b>Выведенные из активов:</b> {extra:.2f} USD")
    
    total_profit = total_change + extra
    pct_change = (total_profit / total_invested * 100.0) if total_invested > 0 else 0.0
    
    lines.append(f"<b>Заработок:</b> {total_profit:+.2f} USD ({pct_change:+.2f}%)")
    lines.append("")
    
    keyboard = [
        [InlineKeyboardButton("➕ extra", callback_data="add_extra_funds")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
    ]
    return "\n".join(lines), InlineKeyboardMarkup(keyboard)

async def show_portfolio_menu(query, user_id):
    text, reply_markup = get_portfolio_text_and_keyboard(user_id)
    await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)

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
    # Избегаем предупреждения SettingWithCopyWarning при работе с срезами
    df = df.copy()
        
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

    # Получаем актуальную цену точно так же, как в портфеле
    current_price = None
    try:
        fi = getattr(yf.Ticker(ticker), "fast_info", {}) or {}
        current_price = fi.get("last_price")
    except Exception:
        current_price = None
    if current_price is None:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                pc = "Adj Close" if "Adj Close" in hist.columns else "Close"
                current_price = float(hist[pc].iloc[-1])
        except Exception:
            current_price = None
    
    # Получаем время из fast_info
    ts = None
    try:
        fi = getattr(yf.Ticker(ticker), "fast_info", {}) or {}
        market_ts = fi.get("last_market_time") or fi.get("last_trading_time")
        if market_ts is not None:
            ts = datetime.fromtimestamp(int(market_ts), tz=timezone.utc)
    except Exception:
        ts = None
    
    # Fallback к историческим данным если время не получено
    if ts is None:
        last = df.iloc[-1]
        idx_ts = last.name
        ts = idx_ts.to_pydatetime() if hasattr(idx_ts, "to_pydatetime") else datetime.fromtimestamp(idx_ts.timestamp(), tz=timezone.utc)
    
    # Получаем объем из fast_info
    volume = None
    try:
        fi = getattr(yf.Ticker(ticker), "fast_info", {}) or {}
        fast_volume = fi.get("last_volume")
        if fast_volume is not None:
            volume = int(fast_volume)
    except Exception:
        volume = None
    
    # Fallback к историческим данным если объем не получен
    if volume is None:
        last = df.iloc[-1]
        volume = int(last['Volume'])
    
    # Используем полученную цену
    price = round(current_price, 4) if current_price is not None else round(float(df[price_column].iloc[-1]), 4)

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
    ts_msk_str = get_msk_time_str(ts)
    info.append(f"🕒 Последнее обновление (MSK): {ts_msk_str}")
    info.append(f"💵 Цена: {price} USD")
    # Советы удалены из текста, теперь кнопка
    info.append(f"📊 Объём (последняя свеча {settings['analysis_days']}d/{settings['cycle_tf']}): {volume}")
    
    cycle_periods = [
        (5, "5d", "5m"),
        (30, "1mo", "1d"),
        (90, "3mo", "1d"),
        (180, "6mo", "1d"),
        (365, "1y", "1d")
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
            cycle_lines.append(f"{label}: {period_stage}")
        else:
            cycle_lines.append(f"{label}: данные недоступны")
    
    cycle_info = "\n".join(cycle_lines)
    chart_link = f"https://finance.yahoo.com/quote/{ticker}/chart?p={ticker}"
    info.append(f"{cycle_info}\n{format_source(chart_link)}")
    
    if approx_book_vol is not None:
        info.append(f"📥 Приближённая дневная ликвидность: ~{approx_book_vol} ед.")
    else:
        info.append("📥 Объем стакана (приближенный): оценка недоступна")
        
    if big:
        ts_big, vol_big = big
        ts_big_msk_str = get_msk_time_str(ts_big)
    
        info.append(f"🚀 Последняя крупная покупка: {ts_big_msk_str}, объём {vol_big}")
    else:
        info.append("🚀 Последняя крупная покупка: не обнаружена")

    user_comment = user_comments.get(user_id, {}).get(ticker) if user_id else None
    if user_id and ticker not in user_assets.get(user_id, []):
        info.append("💬 Комментарий: Вы не добавили этот актив в Ваши активы")
    elif user_comment:
        info.append(f"💬 Комментарий: {user_comment}")

    return "\n\n".join(info)

def build_ticker_info_text(ticker, user_id=None):
    """Упрощенная версия информации для команды /ticker"""
    stock = yf.Ticker(ticker)
    
    # Получаем актуальную цену
    current_price = None
    try:
        fi = getattr(yf.Ticker(ticker), "fast_info", {}) or {}
        current_price = fi.get("last_price")
    except Exception:
        current_price = None
    if current_price is None:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                pc = "Adj Close" if "Adj Close" in hist.columns else "Close"
                current_price = float(hist[pc].iloc[-1])
        except Exception:
            current_price = None
    
    price = round(current_price, 4) if current_price is not None else None
    
    # Получаем название компании
    company_name = None
    if user_id:
        company_name = user_asset_names.get(user_id, {}).get(ticker)
    if not company_name:
        company_name = ticker_name_cache.get(ticker)
    if not company_name:
        company_name = get_company_name(ticker)
    
    info = []
    info.append(f"ℹ️ {company_name} ({ticker})" if company_name != ticker else f"ℹ️ {ticker}")
    
    # Цена
    if price is not None:
        info.append(f"💵 Цена: {price} USD")
    else:
        info.append("💵 Цена: данные недоступны")
    
    # Советы удалены из текста, теперь кнопка
    
    # Стадии цикла
    cycle_periods = [
        (5, "5d", "5m"),
        (30, "1mo", "1d"),
        (90, "3mo", "1d"),
        (180, "6mo", "1d"),
        (365, "1y", "1d")
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
            cycle_lines.append(f"{label}: {period_stage}")
        else:
            cycle_lines.append(f"{label}: данные недоступны")
    
    cycle_info = "\n".join(cycle_lines)
    chart_link = f"https://finance.yahoo.com/quote/{ticker}/chart?p={ticker}"
    info.append(f"{cycle_info}\n{format_source(chart_link)}")
    
    return "\n\n".join(info)

def build_sector_text(ticker, user_id=None):
    ticker = ticker.upper()
    stock = yf.Ticker(ticker)
    lines = []
    name = get_display_name(ticker, user_id)
    lines.append(f"🏷 Сектор для {name}")
    sector = None
    industry = None
    try:
        info = stock.info or {}
        sector = info.get("sector")
        industry = info.get("industry")
    except Exception:
        pass

    weights = None
    try:
        weights = getattr(stock, "fund_sector_weightings", None)
    except Exception:
        weights = None

    if weights and isinstance(weights, list) and len(weights) > 0:
        lines.append("Распределение по секторам (ETF/фонд):")
        for item in weights:
            for k, v in item.items():
                try:
                    pct = float(v) * 100.0
                    lines.append(f"- {k}: {pct:.1f}%")
                except Exception:
                    lines.append(f"- {k}: {v}")
        lines.append(format_source(f"https://finance.yahoo.com/quote/{ticker}/holdings"))
    else:
        if sector:
            lines.append(f"Сектор: {sector}")
        if industry:
            lines.append(f"Отрасль: {industry}")
        lines.append(format_source(f"https://finance.yahoo.com/quote/{ticker}/profile"))

    return "\n".join(lines)

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

    elif query.data == "show_main_menu":
        await query.edit_message_text("Главное меню:", reply_markup=main_menu())

    elif query.data == "settings":
        await query.edit_message_text("⚙️ Настройки:", reply_markup=settings_menu())

    elif query.data == "chart_settings":
        await query.edit_message_text("Выберите тип графика:", reply_markup=chart_settings_menu(user_id))

    elif query.data == "set_chart_static":
        user_settings.setdefault(user_id, {})["chart_type"] = "static"
        await query.edit_message_text("Выберите тип графика:", reply_markup=chart_settings_menu(user_id))

    elif query.data == "set_chart_dynamic":
        user_settings.setdefault(user_id, {})["chart_type"] = "dynamic"
        await query.edit_message_text("Выберите тип графика:", reply_markup=chart_settings_menu(user_id))

    elif query.data == "advises_settings":
        await query.edit_message_text("Выберите таймфрейм для советов:", reply_markup=advises_settings_menu(user_id))

    elif query.data == "set_advises_1W":
        user_settings.setdefault(user_id, {})["advises_interval"] = "1W"
        await query.edit_message_text("Выберите таймфрейм для советов:", reply_markup=advises_settings_menu(user_id))

    elif query.data == "set_advises_1M":
        user_settings.setdefault(user_id, {})["advises_interval"] = "1M"
        await query.edit_message_text("Выберите таймфрейм для советов:", reply_markup=advises_settings_menu(user_id))

    elif query.data == "add_asset":
        user_states[user_id] = "waiting_for_asset"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
        await query.edit_message_text("Введите тикер актива (например, AAPL):",
                                      reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "my_assets":
        assets = user_assets.get(user_id, [])
        if not assets:
            await query.edit_message_text("У вас пока нет активов.", reply_markup=main_menu())
            return
        user_trade_context.pop(user_id, None)
        await show_assets_menu(query, user_id, page=0)

    elif query.data == "my_portfolio":
        user_trade_context.pop(user_id, None)
        await show_portfolio_menu(query, user_id)

    elif query.data == "group_investments":
        # Используем кэш
        if not group_stats_cache:
            await query.edit_message_text("⏳ Обновляю данные группы, подождите...", reply_markup=None)
            await update_group_stats()
            
        # Формируем отчет из кэша
        data = group_stats_cache.get("data", {})
        if not data:
             await query.edit_message_text("Инвестиции группы пока пусты.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]))
             return

        lines = ["👥 <b>Инвестиции группы</b>", ""]
        
        sector_data = data.get("sectors", {})
        sorted_sectors = sorted(sector_data.keys())
        
        total_invested_all = data.get("total_invested", 0.0)
        total_current_all = data.get("total_current", 0.0)
        total_extra_all = data.get("total_extra", 0.0)
        
        for sec in sorted_sectors:
            lines.append(f"🏷 <b>{sec}</b>")
            sec_items = sector_data[sec]
            sec_invested = 0.0
            sec_current = 0.0
            
            for item in sec_items:
                # item: {ticker, user, invested, current, profit_abs, profit_pct}
                lines.append(f"• {item['ticker']} - {item['user']} - влож: {item['invested']:.0f}$ - сейчас: {item['current']:.0f}$ ({item['profit_abs']:+.0f}$ / {item['profit_pct']:+.1f}%)")
                sec_invested += item['invested']
                sec_current += item['current']
            
            sec_profit = sec_current - sec_invested
            sec_pct = (sec_profit / sec_invested * 100.0) if sec_invested > 0 else 0.0
            lines.append(f"<i>Total {sec}: {sec_invested:.0f}$ -> {sec_current:.0f}$ ({sec_profit:+.0f}$ / {sec_pct:+.1f}%)</i>")
            lines.append("")

        if total_extra_all != 0:
             lines.append(f"💵 <b>Выведенные из активов (все): {total_extra_all:.0f}$</b>")

        total_profit_all = (total_current_all - total_invested_all) + total_extra_all
        total_pct_all = (total_profit_all / total_invested_all * 100.0) if total_invested_all > 0 else 0.0
        
        lines.append(f"<b>TOTAL ALL: {total_invested_all:.0f}$ -> {total_current_all:.0f}$ ({total_profit_all:+.0f}$ / {total_pct_all:+.1f}%)</b>")
        
        # Добавляем время обновления
        upd_ts = group_stats_cache.get("last_update")
        if upd_ts:
             dt = get_msk_time_str(upd_ts)
             lines.append(f"\n🕒 Обновлено: {dt} (MSK)")

        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
        # Если сообщение отправлено как "Обновляю...", то edit, иначе тоже edit
        try:
            await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception:
             # Если сообщение не изменилось или устарело
             await context.bot.send_message(chat_id=query.message.chat_id, text="\n".join(lines), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "group_assets":
        keyboard = []
        has_assets = False
        for uid in TRUSTED_USERS:
            assets = user_assets.get(uid, [])
            # Также проверяем портфель, вдруг там есть позиции, но нет в списке наблюдения
            portfolio = user_portfolio.get(uid, {})
            has_portfolio = any(p.get("qty", 0) > 0 for p in portfolio.values())
            
            if assets or has_portfolio:
                has_assets = True
                user_display_name = get_user_name(uid)
                keyboard.append([InlineKeyboardButton(user_display_name, callback_data=f"group_user_{uid}")])
        if not has_assets:
            await query.edit_message_text("👥 Активы группы:\n\nПока нет активов у пользователей.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]))
            return

        keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
        await query.edit_message_text("Выберите пользователя:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "insights":
        assets = user_assets.get(user_id, [])
        if not assets:
            await query.edit_message_text("У вас нет активов для сбора инсайтов.", reply_markup=main_menu())
            return
        lines = ["💡 Инсайты по вашим активам:", ""]
        for t in assets:
            insights = fetch_finviz_insights(t)
            if insights:
                lines.append(f"{get_display_name(t, user_id)}:")
                for s in insights:
                    lines.append(f"• {s}")
                lines.append("")
        if len(lines) <= 2:
            lines.append("Инсайты не найдены.")
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
        await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("group_user_") and not query.data.startswith("group_user_assets_") and not query.data.startswith("group_user_portfolio_"):
        target_user_id = int(query.data.split("_", 2)[2])
        u_name = get_user_name(target_user_id)
        
        keyboard = [
            [InlineKeyboardButton("Активы пользователя", callback_data=f"group_user_assets_{target_user_id}")],
            [InlineKeyboardButton("Портфель пользователя", callback_data=f"group_user_portfolio_{target_user_id}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="group_assets")]
        ]
        await query.edit_message_text(f"Пользователь: {u_name}\nВыберите действие:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("group_user_assets_"):
        target_user_id = int(query.data.split("_", 3)[3])
        assets = user_assets.get(target_user_id, [])
        comments = user_comments.get(target_user_id, {})
        names = user_asset_names.get(target_user_id, {})
        u_name = get_user_name(target_user_id)

        if not assets:
            text = f"У пользователя {u_name} нет отслеживаемых активов."
        else:
            lines = [f"👤 Активы пользователя {u_name}:", ""]
            for asset in assets:
                company_name = names.get(asset)
                if not company_name:
                    company_name = get_company_name(asset)
                    user_asset_names.setdefault(target_user_id, {})[asset] = company_name
                ticker_name_cache[asset] = company_name
                comment = comments.get(asset, "")
                comment_part = f"\n💬 {comment}" if comment else ""
                lines.append(f"• <b>{company_name} ({asset})</b>{comment_part}")
                lines.append("")
            text = "\n".join(lines)

        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data=f"group_user_{target_user_id}")]]
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("group_user_portfolio_"):
        target_user_id = int(query.data.split("_", 3)[3])
        positions = user_portfolio.get(target_user_id, {})
        u_name = get_user_name(target_user_id)
        
        # Удаляем нулевые позиции
        tickers_to_delete = [t for t, p in positions.items() if p.get("qty", 0) <= 0]
        for t in tickers_to_delete:
            try:
                del positions[t]
            except Exception:
                pass
                
        lines = [f"💼 Портфель пользователя {u_name}:\n"]
        if not positions:
            lines.append("Пока нет позиций.")
        else:
            total_change = 0.0
            total_invested = 0.0
            for ticker, pos in positions.items():
                qty = pos.get("qty", 0)
                avg_price = pos.get("avg_price", 0.0)
                
                # Получаем имя
                name = user_asset_names.get(target_user_id, {}).get(ticker)
                if not name:
                    name = get_company_name(ticker)
                
                # Текущая цена
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
                total_invested += (avg_price * qty)
                
                lines.append(f"• {name} ({ticker}), {qty} шт, {avg_price:.2f} -> { (current or 0.0):.2f} ({change_value:+.2f} USD)")
                lines.append("")
            
            lines.append(f"инвестировано: {total_invested:.2f} USD")
            pct_change = (total_change / total_invested * 100.0) if total_invested > 0 else 0.0
            lines.append(f"заработок: {total_change:+.2f} USD ({pct_change:+.2f}%)")
            lines.append("")

        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data=f"group_user_{target_user_id}")]]
        await query.edit_message_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(keyboard))

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
        user_trade_context.pop(user_id, None)
        keyboard = [
            [InlineKeyboardButton("ℹ️ Информация", callback_data=f"info_{ticker}"),
             InlineKeyboardButton("🗑 Удалить актив", callback_data=f"delete_{ticker}")],
            [InlineKeyboardButton("🧮 Калькулятор", callback_data=f"calc_{ticker}")],
            [InlineKeyboardButton("➕ Купить", callback_data=f"buy_{ticker}"), InlineKeyboardButton("➖ Продать", callback_data=f"sell_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="my_assets")]
        ]
        text = f"Актив {display_name}\nКомментарий: {comment}"
        markup = InlineKeyboardMarkup(keyboard)
        message = query.message
        chat_id = message.chat_id if message else query.from_user.id
        if message and message.photo:
            try:
                await message.delete()
            except Exception:
                pass
            await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=markup)
        elif message:
            await query.edit_message_text(text, reply_markup=markup)
        else:
            await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=markup)

    elif query.data.startswith("info_"):
        ticker = query.data.split("_", 1)[1]
        try:
            text = build_info_text(ticker, user_id)
            keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data=f"asset_{ticker}")]]
            
            chart_type = user_settings.get(user_id, {}).get("chart_type", "static")
            advises_interval = user_settings.get(user_id, {}).get("advises_interval", "1M")
            web_app_url = os.getenv("WEB_APP_BASE_URL")
            
            # Кнопка советов
            advises_btn = None
            if web_app_url:
                tv_ticker = map_to_tradingview(ticker)
                advises_url = f"{web_app_url}/advises.html?symbol={tv_ticker}&interval={advises_interval}"
                advises_btn = InlineKeyboardButton("🧠 Советы", web_app=WebAppInfo(url=advises_url))

            if chart_type == "dynamic":
                if web_app_url:
                    tv_ticker = map_to_tradingview(ticker)
                    full_url = f"{web_app_url}/chart.html?symbol={tv_ticker}"
                    row = [InlineKeyboardButton("📈 Открыть график", web_app=WebAppInfo(url=full_url))]
                    if advises_btn:
                        row.append(advises_btn)
                    keyboard.insert(0, row)
                    
                    message = query.message
                    chat_id = message.chat_id if message else query.from_user.id
                    
                    # Если было фото, удаляем его и шлем текст
                    if message and message.photo:
                        try:
                            await message.delete()
                        except Exception:
                            pass
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=text,
                            parse_mode=ParseMode.HTML,
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                    # Если был текст, редактируем
                    elif message:
                         await query.edit_message_text(
                            text=text,
                            parse_mode=ParseMode.HTML,
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                    else:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=text,
                            parse_mode=ParseMode.HTML,
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                else:
                     await query.answer("URL для Web App не настроен (WEB_APP_BASE_URL)", show_alert=True)
                     # Fallback to static
                     photo_url = get_finviz_chart_url(ticker)
                     message = query.message
                     chat_id = message.chat_id if message else query.from_user.id
                     await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo_url,
                        caption=text,
                        parse_mode=ParseMode.HTML,
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                     try:
                        if message:
                            await message.delete()
                     except Exception:
                        pass
            else:
                # Static (default)
                photo_url = get_finviz_chart_url(ticker)
                message = query.message
                chat_id = message.chat_id if message else query.from_user.id
                
                # Для статики кнопку советов добавляем в клавиатуру
                if advises_btn:
                    keyboard.insert(0, [advises_btn])
                
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_url,
                    caption=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
                try:
                    if message:
                        await message.delete()
                except Exception:
                    pass
            return
        except Exception as e:
            keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data=f"asset_{ticker}")]]
            await query.edit_message_text(f"Ошибка: {e}", reply_markup=InlineKeyboardMarkup(keyboard))

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
            # Обновляем кэш группы в фоне
            context.application.create_task(update_group_stats())
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
            message_text += f"{format_source(source_url)}\n"
            message_text += f"Формула: CAGR = (Конечная стоимость / Начальная стоимость)^(1/n) - 1"
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(message_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))
        except Exception as e:
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(f"❌ Ошибка при расчете CAGR для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))

    elif query.data.startswith("eps_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        try:
            eps_value, source_url = calculate_eps(ticker)
            message_text = f"📊 EPS для {display_name}: ${eps_value:.2f}\n\n{format_source(source_url)}"
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(message_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))
        except Exception as e:
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(f"❌ Ошибка при расчете EPS для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))

    elif query.data.startswith("beta_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        try:
            beta_3y_value, source_url = calculate_beta(ticker)
            beta_5y_value, _ = calculate_beta_5y_monthly(ticker)
            message_text = f"📊 Бета-коэффициент для {display_name}:\n\n"
            message_text += f"5-летний (месячные данные): {beta_5y_value:.2f}\n"
            message_text += f"3-летний (дневные данные): {beta_3y_value:.2f}\n\n"
            message_text += f"{format_source(source_url)}\n"
            message_text += f"Формула: β = Cov(Ri, Rm) / Var(Rm)"
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(message_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))
        except Exception as e:
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(f"❌ Ошибка при расчете бета-коэффициента для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))

    elif query.data.startswith("pe_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        try:
            pe_value, source_url = calculate_pe_ratio(ticker)
            message_text = f"📊 P/E Ratio для {display_name}: {pe_value:.2f}\n\n{format_source(source_url)}"
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(message_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))
        except Exception as e:
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(f"❌ Ошибка при получении P/E Ratio для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))

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
            message_text += f"{format_source(f'https://finance.yahoo.com/quote/{ticker}/key-statistics')}"
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(message_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))
        except Exception as e:
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(f"❌ Ошибка при расчете RVOL для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))

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
            message_text = f"🎯 Консенсусная 12-месячная цель для {display_name}: {target_value:.2f} USD\n{format_source(source_url)}{diff_text}"
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(message_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))
        except Exception as e:
            back_cb = f"calc_{ticker}" if ticker in user_assets.get(user_id, []) else f"calcany_{ticker}"
            await query.edit_message_text(f"❌ Ошибка при получении цели для {ticker}: {e}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_cb)]]))

    elif query.data == "trade_market":
        ctx = user_trade_context.get(user_id)
        if not ctx or "qty" not in ctx:
            await query.edit_message_text("Сессия сделки не найдена.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))
            return
        action = ctx["action"]
        qty = ctx["qty"]
        ticker = ctx.get("ticker")
        # Исполнение по рынку: обновляем портфель сразу (только покупка)
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
            save_user_data()
            context.application.create_task(update_group_stats())
            await query.edit_message_text(f"✅ Покупка по рынку: {ticker} {qty} @ {price_exec:.2f}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))
            user_trade_context.pop(user_id, None)
            await query.message.reply_text("Главное меню:", reply_markup=main_menu())
        else:
            await query.edit_message_text("❌ Неверный режим для продажи.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))

    elif query.data == "trade_manual":
        ctx = user_trade_context.get(user_id)
        if not ctx or "qty" not in ctx:
            await query.edit_message_text("Сессия сделки не найдена.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]))
            return
        ctx["action"] = "manual_buy"
        ctx["step"] = "price_manual"
        back_to = ctx.get("back_to") or (f"asset_{ctx.get('ticker')}" if ctx.get('ticker') else "my_portfolio")
        await query.edit_message_text("Введите цену, по которой вы ранее купили актив:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_to)]]))

    elif query.data == "sectors":
        user_states[user_id] = "waiting_for_sector_ticker"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
        await query.edit_message_text("Введите тикер актива для получения сектора/распределения:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "ticker_info":
        user_states[user_id] = "waiting_for_ticker_info"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="back")]]
        user_trade_context.pop(user_id, None)
        await query.edit_message_text("Введите тикер для получения информации/калькулятора:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("ticker_info_menu_"):
        ticker = query.data.split("_", 3)[3]
        display_name = get_display_name(ticker, user_id)
        keyboard = [
            [InlineKeyboardButton("ℹ️ Информация", callback_data=f"infoany_{ticker}"),
             InlineKeyboardButton("🧮 Калькулятор", callback_data=f"calcany_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
        ]
        text = f"Тикер {display_name}. Выберите действие:"
        markup = InlineKeyboardMarkup(keyboard)
        message = query.message
        chat_id = message.chat_id if message else query.from_user.id
        if message and message.photo:
            try:
                await message.delete()
            except Exception:
                pass
            await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=markup)
        elif message:
            await query.edit_message_text(text, reply_markup=markup)
        else:
            await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=markup)

    elif query.data.startswith("calcany_"):
        ticker = query.data.split("_", 1)[1]
        display_name = get_display_name(ticker, user_id)
        keyboard = [
            [InlineKeyboardButton("CAGR", callback_data=f"cagr_{ticker}"),
             InlineKeyboardButton("EPS", callback_data=f"eps_{ticker}")],
            [InlineKeyboardButton("β", callback_data=f"beta_{ticker}"),
             InlineKeyboardButton("P/E Ratio", callback_data=f"pe_{ticker}")],
            [InlineKeyboardButton("RVOL", callback_data=f"rvol_{ticker}"),
             InlineKeyboardButton("🎯 12M Target", callback_data=f"target_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data=f"ticker_info_menu_{ticker}")]
        ]
        await query.edit_message_text(f"🧮 Калькулятор для {display_name}\nВыберите метрику:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("infoany_"):
        ticker = query.data.split("_", 1)[1]
        try:
            text = build_info_text(ticker, user_id)
            keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data=f"ticker_info_menu_{ticker}")]]
            
            chart_type = user_settings.get(user_id, {}).get("chart_type", "static")
            advises_interval = user_settings.get(user_id, {}).get("advises_interval", "1M")
            web_app_url = os.getenv("WEB_APP_BASE_URL")
            
            # Кнопка советов
            advises_btn = None
            if web_app_url:
                tv_ticker = map_to_tradingview(ticker)
                advises_url = f"{web_app_url}/advises.html?symbol={tv_ticker}&interval={advises_interval}"
                advises_btn = InlineKeyboardButton("🧠 Советы", web_app=WebAppInfo(url=advises_url))

            if chart_type == "dynamic":
                if web_app_url:
                    tv_ticker = map_to_tradingview(ticker)
                    full_url = f"{web_app_url}/chart.html?symbol={tv_ticker}"
                    row = [InlineKeyboardButton("📈 Открыть график", web_app=WebAppInfo(url=full_url))]
                    if advises_btn:
                        row.append(advises_btn)
                    keyboard.insert(0, row)
                    
                    message = query.message
                    chat_id = message.chat_id if message else query.from_user.id
                    
                    if message and message.photo:
                        try:
                            await message.delete()
                        except Exception:
                            pass
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=text,
                            parse_mode=ParseMode.HTML,
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                    elif message:
                         await query.edit_message_text(
                            text=text,
                            parse_mode=ParseMode.HTML,
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                    else:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=text,
                            parse_mode=ParseMode.HTML,
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                else:
                     await query.answer("URL для Web App не настроен (WEB_APP_BASE_URL)", show_alert=True)
                     # Fallback
                     photo_url = get_finviz_chart_url(ticker)
                     message = query.message
                     chat_id = message.chat_id if message else query.from_user.id
                     await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo_url,
                        caption=text,
                        parse_mode=ParseMode.HTML,
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                     try:
                        if message:
                            await message.delete()
                     except Exception:
                        pass
            else:
                photo_url = get_finviz_chart_url(ticker)
                message = query.message
                chat_id = message.chat_id if message else query.from_user.id
                
                if advises_btn:
                    keyboard.insert(0, [advises_btn])
                    
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_url,
                    caption=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
                try:
                    if message:
                        await message.delete()
                except Exception:
                    pass
            return
        except Exception as e:
            keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data=f"ticker_info_menu_{ticker}")]]
            await query.edit_message_text(f"Ошибка: {e}", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data == "back":
        user_states[user_id] = None
        user_trade_context.pop(user_id, None)
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
        back_to = f"asset_{ticker}" if ticker else "my_portfolio"
        user_trade_context[user_id] = {"action": "buy", "ticker": ticker, "step": "qty", "back_to": back_to}
        await query.edit_message_text("Введите количество акций для покупки:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_to)]]))

    elif query.data == "add_extra_funds":
        user_states[user_id] = "waiting_for_extra_funds"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="my_portfolio")]]
        await query.edit_message_text("Введите сумму выведенных средств (в USD):", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("sell_") or query.data == "sell_start":
        ticker = query.data.split("_", 1)[1] if "_" in query.data else None
        back_to = f"asset_{ticker}" if ticker else "my_portfolio"
        user_trade_context[user_id] = {"action": "sell", "ticker": ticker, "step": "qty", "back_to": back_to}
        await query.edit_message_text("Введите количество акций для продажи:", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=back_to)]]))

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return

    user_name = update.effective_user.username
    if user_name:
        user_names_cache[user_id] = user_name

    # Сначала проверяем торговый контекст (приоритет над состояниями пользователя)
    if user_id in user_trade_context and user_trade_context.get(user_id, {}).get("step") == "qty":
        ctx = user_trade_context[user_id]
        try:
            qty = int(update.message.text.strip())
            if qty <= 0:
                raise ValueError
        except Exception:
            await update.message.reply_text("❌ Некорректное количество. Введите положительное целое число:")
            return
        ctx["qty"] = qty
        back_to = ctx.get("back_to") or (f"asset_{ctx.get('ticker')}" if ctx.get('ticker') else "my_portfolio")
        if ctx.get("action") == "sell":
            # Immediate sell by quantity only
            ticker = ctx.get("ticker")
            pos = user_portfolio.setdefault(user_id, {}).get(ticker)
            if not pos or pos.get("qty", 0) <= 0:
                await update.message.reply_text("❌ Нечего продавать.")
            else:
                sell_qty = min(qty, pos["qty"])
                pos["qty"] -= sell_qty
                if pos["qty"] == 0:
                    pos["avg_price"] = 0.0
                    try:
                        del user_portfolio[user_id][ticker]
                    except Exception:
                        pass
                save_user_data()
                context.application.create_task(update_group_stats())
                await update.message.reply_text(f"✅ Продажа выполнена: {ticker} {sell_qty}")
            user_trade_context.pop(user_id, None)
            await update.message.reply_text("Главное меню:", reply_markup=main_menu())
        else:
            ctx["step"] = "price_mode"
            keyboard = [
                [InlineKeyboardButton("Market price", callback_data="trade_market")],
                [InlineKeyboardButton("Already bought", callback_data="trade_manual")],
                [InlineKeyboardButton("⬅️ Назад", callback_data=back_to)]
            ]
            await update.message.reply_text("Выберите режим исполнения:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif user_id in user_trade_context and user_trade_context.get(user_id, {}).get("step") == "price":
        # Limit orders removed
        user_trade_context.pop(user_id, None)
        await update.message.reply_text("Режим лимитных ордеров отключен.")
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
        context.application.create_task(update_group_stats())
        user_trade_context.pop(user_id, None)
        await update.message.reply_text(f"✅ Добавлено в портфель: {ticker} {qty} @ {price:.2f}")
        await update.message.reply_text("Главное меню:", reply_markup=main_menu())

    elif user_states.get(user_id) == "waiting_for_sector_ticker":
        ticker = update.message.text.strip().upper()
        try:
            text = build_sector_text(ticker, user_id)
        except Exception as e:
            text = f"❌ Не удалось получить сектор для {ticker}: {e}"
        user_states[user_id] = None
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        await update.message.reply_text("Главное меню:", reply_markup=main_menu())

    elif user_states.get(user_id) == "waiting_for_ticker_info":
        ticker = update.message.text.strip().upper()
        user_states[user_id] = None
        display_name = get_display_name(ticker, user_id)
        keyboard = [
            [InlineKeyboardButton("ℹ️ Информация", callback_data=f"infoany_{ticker}"),
             InlineKeyboardButton("🧮 Калькулятор", callback_data=f"calcany_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
        ]
        await update.message.reply_text(f"Тикер {display_name}. Выберите действие:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif user_states.get(user_id) == "waiting_for_asset":
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
                                      
    elif user_states.get(user_id) and user_states[user_id].startswith("waiting_for_comment_"):
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
            context.application.create_task(update_group_stats())
            
            user_states[user_id] = None
            await update.message.reply_text(f"✅ Актив {ticker} добавлен с комментарием '{comment}'!", reply_markup=main_menu())
            
    elif user_states.get(user_id) == "waiting_for_blacklist_ticker":
        ticker = update.message.text.strip().upper()
        user_states[user_id] = f"waiting_for_blacklist_comment_{ticker}"
        keyboard = [[InlineKeyboardButton("⬅️ Назад", callback_data="blacklist")]]
        await update.message.reply_text(f"Введите комментарий для добавления {ticker} в черный список:",
                                      reply_markup=InlineKeyboardMarkup(keyboard))
                                      
    elif (user_states.get(user_id) or "").startswith("waiting_for_blacklist_comment_"):
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
        
    elif (user_states.get(user_id) or "").startswith("force_add_"):
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
            context.application.create_task(update_group_stats())
            
            user_states[user_id] = None
            await update.message.reply_text(f"✅ Актив {ticker} принудительно добавлен с комментарием '{comment}'!", reply_markup=main_menu())

    elif user_states.get(user_id) == "waiting_for_extra_funds":
        try:
            amount = float(update.message.text.strip().replace(",", "."))
            # Суммируем с текущим значением
            current_extra = user_extra_funds.get(user_id, 0.0)
            new_extra = current_extra + amount
            user_extra_funds[user_id] = new_extra
            save_user_data()
            # Обновляем кэш, так как extra влияет на total
            context.application.create_task(update_group_stats())
            
            user_states[user_id] = None
            
            await update.message.reply_text(f"✅ Добавлено: {amount:.2f} USD. Всего выведено: {new_extra:.2f} USD")
            
            # Автоматически показываем портфель новым сообщением
            text, reply_markup = get_portfolio_text_and_keyboard(user_id)
            await update.message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
            
        except Exception:
            await update.message.reply_text("❌ Некорректная сумма. Введите число (можно отрицательное):")


def load_user_data():
    """Загружает данные пользователей из файла users.txt"""
    global user_assets, user_comments, user_settings, user_asset_names, ticker_name_cache, user_portfolio, user_extra_funds
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
                user_extra_funds[current_user_id] = 0.0
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
            elif line.startswith("SETTINGS:") and current_user_id:
                current_section = "settings"
            elif line.startswith("EXTRA_FUNDS:") and current_user_id:
                current_section = "extra_funds"
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
            elif current_section == "extra_funds" and current_user_id:
                if line == "END_EXTRA_FUNDS":
                    current_section = None
                else:
                    try:
                        user_extra_funds[current_user_id] = float(line)
                    except Exception:
                        pass
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных пользователей: {e}")
        user_assets = {}
        user_comments = {}
        user_asset_names = {}
        user_settings = {}
        ticker_name_cache = {}
        ticker_name_cache = {}
        user_portfolio = {}
        user_extra_funds = {}

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

                f.write("EXTRA_FUNDS:\n")
                f.write(f"{user_extra_funds.get(user_id, 0.0)}\n")
                f.write("END_EXTRA_FUNDS\n")

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

def load_group_cache():
    global group_stats_cache
    try:
        cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "group_stats.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                group_stats_cache = json.load(f)
    except Exception as e:
        logging.error(f"Ошибка загрузки кэша группы: {e}")
        group_stats_cache = {}

def save_group_cache():
    try:
        cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "group_stats.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(group_stats_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Ошибка сохранения кэша группы: {e}")

load_group_cache()

async def update_group_stats():
    """Фоновая задача для обновления статистики группы"""
    global group_stats_cache
    
    sector_data = {}
    total_invested_all = 0.0
    total_current_all = 0.0
    
    # Считаем extra funds
    total_extra_all = sum(user_extra_funds.values())

    for uid in TRUSTED_USERS:
        u_name = get_user_name(uid)
        portfolio = user_portfolio.get(uid, {})
        
        for ticker, pos in portfolio.items():
            qty = pos.get("qty", 0)
            if qty <= 0:
                continue
            
            avg_price = pos.get("avg_price", 0.0)
            invested = qty * avg_price
            
            # Получаем текущую цену
            current_price = 0.0
            try:
                fi = getattr(yf.Ticker(ticker), "fast_info", {}) or {}
                current_price = fi.get("last_price")
            except Exception:
                pass
            
            if not current_price:
                try:
                    hist = yf.Ticker(ticker).history(period="5d")
                    if not hist.empty:
                        pc = "Adj Close" if "Adj Close" in hist.columns else "Close"
                        current_price = float(hist[pc].iloc[-1])
                except Exception:
                    pass
            
            current_price = float(current_price or 0.0)
            current_val = qty * current_price
            
            profit_abs = current_val - invested
            profit_pct = (profit_abs / invested * 100.0) if invested > 0 else 0.0
            
            total_invested_all += invested
            total_current_all += current_val
            
            # Получаем сектор
            sector = "Неизвестный сектор"
            try:
                info = yf.Ticker(ticker).info or {}
                sector = info.get("sector") or "Неизвестный сектор"
            except Exception:
                pass
            
            if sector not in sector_data:
                sector_data[sector] = []
            
            sector_data[sector].append({
                "ticker": ticker,
                "user": u_name,
                "invested": invested,
                "current": current_val,
                "profit_abs": profit_abs,
                "profit_pct": profit_pct
            })

    group_stats_cache = {
        "last_update": time.time(),
        "data": {
            "sectors": sector_data,
            "total_invested": total_invested_all,
            "total_current": total_current_all,
            "total_extra": total_extra_all
        }
    }
    save_group_cache()

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

def fetch_finviz_screener(signal="ta_topgainers"):
    """Fetches top gainers or losers from Finviz screener."""
    url = f"https://finviz.com/screener.ashx?v=110&s={signal}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        # Finviz screener table usually has class "table-light" or similar. 
        # We look for rows in the main table.
        # A robust way is to find the table with id "screener-views-table" or similar, 
        # but Finviz changes often. Let's try to find rows with ticker links.
        
        rows = soup.select("tr[valign='top']") # Common for finviz data rows
        results = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 10:
                continue
            
            # Usually: 1=Ticker, 8=Price, 9=Change
            # This is fragile and depends on the specific view (v=110 is Overview)
            # v=110 Overview columns: No, Ticker, Company, Sector, Industry, Country, Market Cap, P/E, Price, Change, Volume
            
            try:
                ticker_col = cols[1].text.strip()
                price_col = cols[8].text.strip()
                change_col = cols[9].text.strip()
                
                # Verify it looks like a ticker
                if not ticker_col.isalpha() or len(ticker_col) > 6:
                    continue
                    
                results.append({
                    "ticker": ticker_col,
                    "price": price_col,
                    "change": change_col
                })
                if len(results) >= 10:
                    break
            except Exception:
                continue
                
        return results
    except Exception as e:
        logging.error(f"Error fetching screener {signal}: {e}")
        return []

def fetch_finviz_news(ticker):
    """Fetches news from Finviz quote page."""
    url = f"https://finviz.com/quote.ashx?t={quote_plus(ticker.upper())}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        # News is usually in a table with id "news-table"
        news_table = soup.find("table", id="news-table")
        if not news_table:
            return []
            
        rows = news_table.find_all("tr")
        results = []
        for row in rows:
            try:
                # Format: Date/Time | Headline (Link)
                cols = row.find_all("td")
                if len(cols) < 2:
                    continue
                
                date_str = cols[0].text.strip()
                link_tag = cols[1].find("a")
                headline = link_tag.text.strip()
                link = link_tag["href"]
                
                results.append({
                    "date": date_str,
                    "headline": headline,
                    "link": link
                })
                if len(results) >= 5:
                    break
            except Exception:
                continue
        return results
    except Exception as e:
        logging.error(f"Error fetching news for {ticker}: {e}")
        return []

def fetch_finviz_insider(ticker):
    """Fetches insider trading from Finviz quote page."""
    url = f"https://finviz.com/quote.ashx?t={quote_plus(ticker.upper())}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        # Insider trading is usually in a table class "body-table" (there are many), 
        # but it's typically the one after the news or financial statements.
        # A better way is to look for the header "Insider Trading"
        
        # Finviz structure for insider is complex. Let's look for a table that contains "Relationship" or "Transaction"
        tables = soup.find_all("table", class_="body-table")
        insider_table = None
        for table in tables:
            headers_row = table.find("tr")
            if headers_row and "Relationship" in headers_row.text and "Transaction" in headers_row.text:
                insider_table = table
                break
        
        if not insider_table:
            return []
            
        rows = insider_table.find_all("tr")[1:] # Skip header
        results = []
        for row in rows:
            try:
                cols = row.find_all("td")
                if len(cols) < 5:
                    continue
                
                # Columns: Owner, Relationship, Date, Transaction, Cost, #Shares, Value, #Shares Total, SEC Form 4
                owner = cols[0].text.strip()
                relationship = cols[1].text.strip()
                date = cols[2].text.strip()
                transaction = cols[3].text.strip()
                # cost = cols[4].text.strip()
                shares = cols[5].text.strip()
                
                results.append({
                    "owner": owner,
                    "relationship": relationship,
                    "date": date,
                    "transaction": transaction,
                    "shares": shares
                })
                if len(results) >= 5:
                    break
            except Exception:
                continue
        return results
    except Exception as e:
        logging.error(f"Error fetching insider for {ticker}: {e}")
        return []

async def post_init(application: Application):
    """Выполняется после инициализации приложения, но до начала polling"""
    logging.info("Запуск обновления статистики группы при старте...")
    # Запускаем обновление в фоне (или ждем завершения, если критично)
    # Лучше подождать, чтобы данные были готовы сразу
    await update_group_stats()
    logging.info("Статистика группы обновлена.")

async def inline_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик inline-запросов"""
    query = update.inline_query.query.strip().upper()
    user_id = update.inline_query.from_user.id

    if user_id not in TRUSTED_USERS:
        return

    results = []

    # 1. Если запрос пустой - показываем общие опции
    if not query:
        # Портфель
        port_text, _ = get_portfolio_text_and_keyboard(user_id)
        results.append(
            InlineQueryResultArticle(
                id=str(uuid4()),
                title="💼 Мой портфель",
                description="Отправить сводку портфеля",
                input_message_content=InputTextMessageContent(
                    message_text=port_text,
                    parse_mode=ParseMode.HTML
                )
            )
        )
        
        # Группа
        data = group_stats_cache.get("data", {})
        if data:
            total_invested = data.get("total_invested", 0.0)
            total_current = data.get("total_current", 0.0)
            total_profit = total_current - total_invested
            pct = (total_profit / total_invested * 100) if total_invested else 0
            
            group_text = (f"👥 <b>Инвестиции группы</b>\n"
                          f"Вложено: {total_invested:.0f} USD\n"
                          f"Сейчас: {total_current:.0f} USD\n"
                          f"Прибыль: {total_profit:+.0f} USD ({pct:+.1f}%)")
            
            results.append(
                InlineQueryResultArticle(
                    id=str(uuid4()),
                    title="👥 Инвестиции группы",
                    description=f"Total: {total_current:.0f}$ ({pct:+.1f}%)",
                    input_message_content=InputTextMessageContent(
                        message_text=group_text,
                        parse_mode=ParseMode.HTML
                    )
                )
            )
            
        results.append(
             InlineQueryResultArticle(
                id=str(uuid4()),
                title="🔍 Поиск котировок",
                description="Введите тикер (например, AAPL)",
                input_message_content=InputTextMessageContent(
                    message_text="Чтобы найти котировку, введите @botname TICKER"
                )
            )
        )
    
    # 2. Обработка специальных команд
    elif query == "PORTFOLIO" or query == "ПОРТФЕЛЬ":
        text, _ = get_portfolio_text_and_keyboard(user_id)
        results.append(
            InlineQueryResultArticle(
                id=str(uuid4()),
                title="💼 Мой портфель",
                description="Текущее состояние портфеля",
                input_message_content=InputTextMessageContent(
                    message_text=text,
                    parse_mode=ParseMode.HTML
                )
            )
        )

    elif query == "GROUP" or query == "ГРУППА":
        # Генерируем текст группы (упрощенно, без кэша update если он стар, но мы берем из кэша)
        data = group_stats_cache.get("data", {})
        if data:
            total_invested = data.get("total_invested", 0.0)
            total_current = data.get("total_current", 0.0)
            total_profit = total_current - total_invested
            pct = (total_profit / total_invested * 100) if total_invested else 0
            
            text = (f"👥 <b>Инвестиции группы</b>\n"
                    f"Вложено: {total_invested:.0f} USD\n"
                    f"Сейчас: {total_current:.0f} USD\n"
                    f"Прибыль: {total_profit:+.0f} USD ({pct:+.1f}%)")
            
            results.append(
                InlineQueryResultArticle(
                    id=str(uuid4()),
                    title="👥 Инвестиции группы",
                    description=f"Total: {total_current:.0f}$ ({pct:+.1f}%)",
                    input_message_content=InputTextMessageContent(
                        message_text=text,
                        parse_mode=ParseMode.HTML
                    )
                )
            )

    # 3. Поиск тикера (если длина < 6 и это буквы)
    elif len(query) < 6 and query.isalpha():
        ticker = query
        try:
            # Пытаемся получить данные
            stock = yf.Ticker(ticker)
            fi = getattr(stock, "fast_info", {})
            price = fi.get("last_price")
            
            if price:
                # Получаем имя
                name = get_company_name(ticker)
                
                # Формируем текст
                text = f"ℹ️ <b>{name} ({ticker})</b>\n"
                text += f"💵 Цена: {price:.2f} USD\n"
                
                # Можно добавить изменение за день если есть
                try:
                    prev_close = fi.get("previous_close")
                    if prev_close:
                        change = price - prev_close
                        pct = (change / prev_close) * 100
                        emoji = "🟢" if change >= 0 else "🔴"
                        text += f"Изменение: {emoji} {change:+.2f} ({pct:+.2f}%)"
                except:
                    pass

                results.append(
                    InlineQueryResultArticle(
                        id=str(uuid4()),
                        title=f"{ticker} - {price:.2f} USD",
                        description=f"{name}",
                        input_message_content=InputTextMessageContent(
                            message_text=text,
                            parse_mode=ParseMode.HTML
                        )
                    )
                )
        except Exception:
            pass

    await update.inline_query.answer(results, cache_time=10) # cache_time=0 for debug

async def cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return
    text, _ = get_portfolio_text_and_keyboard(user_id)
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

async def cmd_investisions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return
    
    # Используем кэш
    if not group_stats_cache:
        await update.message.reply_text("⏳ Обновляю данные группы, подождите...")
        await update_group_stats()
        
    data = group_stats_cache.get("data", {})
    if not data:
         await update.message.reply_text("Инвестиции группы пока пусты.")
         return

    lines = ["👥 <b>Инвестиции группы</b>", ""]
    
    sector_data = data.get("sectors", {})
    sorted_sectors = sorted(sector_data.keys())
    
    total_invested_all = data.get("total_invested", 0.0)
    total_current_all = data.get("total_current", 0.0)
    total_extra_all = data.get("total_extra", 0.0)
    
    for sec in sorted_sectors:
        lines.append(f"🏷 <b>{sec}</b>")
        sec_items = sector_data[sec]
        sec_invested = 0.0
        sec_current = 0.0
        
        for item in sec_items:
            lines.append(f"• {item['ticker']} - {item['user']} - влож: {item['invested']:.0f}$ - сейчас: {item['current']:.0f}$ ({item['profit_abs']:+.0f}$ / {item['profit_pct']:+.1f}%)")
            sec_invested += item['invested']
            sec_current += item['current']
        
        sec_profit = sec_current - sec_invested
        sec_pct = (sec_profit / sec_invested * 100.0) if sec_invested > 0 else 0.0
        lines.append(f"<i>Total {sec}: {sec_invested:.0f}$ -> {sec_current:.0f}$ ({sec_profit:+.0f}$ / {sec_pct:+.1f}%)</i>")
        lines.append("")

    if total_extra_all != 0:
         lines.append(f"💵 <b>Выведенные из активов (все): {total_extra_all:.0f}$</b>")

    total_profit_all = (total_current_all - total_invested_all) + total_extra_all
    total_pct_all = (total_profit_all / total_invested_all * 100.0) if total_invested_all > 0 else 0.0
    
    lines.append(f"<b>TOTAL ALL: {total_invested_all:.0f}$ -> {total_current_all:.0f}$ ({total_profit_all:+.0f}$ / {total_pct_all:+.1f}%)</b>")
    
    upd_ts = group_stats_cache.get("last_update")
    if upd_ts:
         dt = get_msk_time_str(upd_ts)
         lines.append(f"\n🕒 Обновлено: {dt} (MSK)")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

async def cmd_ticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return
    
    if not context.args:
        await update.message.reply_text("Использование: /ticker <TICKER>")
        return
    
    ticker = context.args[0].upper()
    try:
        text = build_ticker_info_text(ticker, user_id)
    
        # Клавиатура для тикера
        keyboard = []
        chart_type = user_settings.get(user_id, {}).get("chart_type", "static")
        advises_interval = user_settings.get(user_id, {}).get("advises_interval", "1M")
        web_app_url = os.getenv("WEB_APP_BASE_URL")
        
        # Кнопка советов
        advises_btn = None
        if web_app_url:
            tv_ticker = map_to_tradingview(ticker)
            advises_url = f"{web_app_url}/advises.html?symbol={tv_ticker}&interval={advises_interval}"
            advises_btn = InlineKeyboardButton("🧠 Советы", web_app=WebAppInfo(url=advises_url))

        if chart_type == "dynamic":
            if web_app_url:
                tv_ticker = map_to_tradingview(ticker)
                full_url = f"{web_app_url}/chart.html?symbol={tv_ticker}"
                row = [InlineKeyboardButton("📈 Открыть график", web_app=WebAppInfo(url=full_url))]
                if advises_btn:
                    row.append(advises_btn)
                keyboard.append(row)
                
                await update.message.reply_text(
                    text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
            else:
                 # Fallback to static behavior (text with image preview)
                 photo_url = get_finviz_chart_url(ticker)
                 # Add invisible link for preview
                 text_with_preview = f"<a href='{photo_url}'>&#8205;</a>{text}"
                 
                 if advises_btn:
                    keyboard.append([advises_btn])
                 
                 await update.message.reply_text(
                    text_with_preview,
                    parse_mode=ParseMode.HTML,
                    reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None
                )
        else:
            # Static
            photo_url = get_finviz_chart_url(ticker)
            # Add invisible link for preview so it looks like a photo message
            text_with_preview = f"<a href='{photo_url}'>&#8205;</a>{text}"
            
            if advises_btn:
                keyboard.append([advises_btn])
                
            await update.message.reply_text(
                text_with_preview,
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None
            )
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

async def cmd_hotmap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return
    
    # Finviz map is dynamic, so we send the Sector Performance image as a proxy
    # and a link to the map.
    
    image_url = "https://finviz.com/grp_image.ashx?bar_sector_t.png"
    map_url = "https://finviz.com/map.ashx"
    
    temp_file = None
    try:
        # Download image
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        }
        resp = requests.get(image_url, headers=headers, timeout=10)
        if resp.status_code == 200 and resp.content and len(resp.content) > 0:
            # Use absolute path for temp file
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"temp_map_{uuid4()}.png")
            
            with open(temp_file, "wb") as f:
                f.write(resp.content)
            
            # Verify file was written
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                caption = f"📊 <b>Карта рынка (Sectors)</b>\n\n<a href='{map_url}'>🔗 Открыть интерактивную карту Finviz</a>"
                
                with open(temp_file, "rb") as f:
                    await update.message.reply_photo(photo=f, caption=caption, parse_mode=ParseMode.HTML)
            else:
                await update.message.reply_text(f"❌ Не удалось сохранить изображение. <a href='{map_url}'>Ссылка</a>", parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(f"❌ Не удалось загрузить карту (код: {resp.status_code}). <a href='{map_url}'>Ссылка</a>", parse_mode=ParseMode.HTML)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")
    finally:
        # Delete file in finally block to ensure cleanup
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

async def cmd_top_gainers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return
    
    await update.message.reply_text("⏳ Загружаю Top Gainers...")
    data = fetch_finviz_screener("ta_topgainers")
    if not data:
        await update.message.reply_text("❌ Не удалось получить данные.")
        return
        
    lines = ["🚀 <b>Top Gainers (Today)</b>", ""]
    for item in data:
        lines.append(f"• <b>{item['ticker']}</b>: {item['price']} ({item['change']})")
    
    lines.append(f"\n<a href='https://finviz.com/screener.ashx?v=110&s=ta_topgainers'>🔗 Finviz Screener</a>")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

async def cmd_top_losers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return
    
    await update.message.reply_text("⏳ Загружаю Top Losers...")
    data = fetch_finviz_screener("ta_toplosers")
    if not data:
        await update.message.reply_text("❌ Не удалось получить данные.")
        return
        
    lines = ["🔻 <b>Top Losers (Today)</b>", ""]
    for item in data:
        lines.append(f"• <b>{item['ticker']}</b>: {item['price']} ({item['change']})")
    
    lines.append(f"\n<a href='https://finviz.com/screener.ashx?v=110&s=ta_toplosers'>🔗 Finviz Screener</a>")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return
    
    if not context.args:
        await update.message.reply_text("Использование: /news <TICKER>")
        return
    
    ticker = context.args[0].upper()
    await update.message.reply_text(f"⏳ Ищу новости для {ticker}...")
    
    news = fetch_finviz_news(ticker)
    if not news:
        await update.message.reply_text(f"❌ Новости для {ticker} не найдены.")
        return
        
    lines = [f"📰 <b>Новости {ticker}</b>", ""]
    for item in news:
        lines.append(f"• {item['date']} - <a href='{item['link']}'>{item['headline']}</a>")
        
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML, disable_web_page_preview=True)

async def cmd_insider(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return
    
    if not context.args:
        await update.message.reply_text("Использование: /insider <TICKER>")
        return
    
    ticker = context.args[0].upper()
    await update.message.reply_text(f"⏳ Проверяю инсайдеров для {ticker}...")
    
    data = fetch_finviz_insider(ticker)
    if not data:
        await update.message.reply_text(f"❌ Инсайдерские сделки для {ticker} не найдены.")
        return
        
    lines = [f"🕵️ <b>Инсайдеры {ticker}</b>", ""]
    for item in data:
        # Emoji based on transaction type
        emoji = "🛒" if "Buy" in item['transaction'] else "💸" if "Sale" in item['transaction'] else "📝"
        lines.append(f"{emoji} <b>{item['date']}</b>: {item['owner']} ({item['relationship']})")
        lines.append(f"   {item['transaction']} {item['shares']} shares")
        lines.append("")
        
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

def main():
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("portfolio", cmd_portfolio))
    app.add_handler(CommandHandler("investisions", cmd_investisions))
    app.add_handler(CommandHandler("ticker", cmd_ticker))
    #app.add_handler(CommandHandler("hotmap", cmd_hotmap))
    app.add_handler(CommandHandler("top_gainers", cmd_top_gainers))
    app.add_handler(CommandHandler("top_losers", cmd_top_losers))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("insider", cmd_insider))
    
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(InlineQueryHandler(inline_query_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
