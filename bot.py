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

# 🔑 Токен и список доверенных пользователей
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not found in .env file")

TRUSTED_USERS = [1085064193]

# Хранилище активов, состояний и настроек
user_assets = {}
user_states = {}
user_settings = {}  # user_id -> dict с настройками

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Главное меню ---
def main_menu():
    keyboard = [
        [InlineKeyboardButton("➕ Добавить актив", callback_data="add_asset"),
         InlineKeyboardButton("📊 Мои активы", callback_data="my_assets")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="settings")]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- Меню настроек ---
def settings_menu(user_id):
    settings = user_settings.get(user_id, {
        "eps_bp": 5,
        "big_buy_mult": 2,
        "analysis_days": 5,
        "cycle_tf": "5m"
    })
    keyboard = [
        [InlineKeyboardButton("Порог ликвидности", callback_data="settings_eps")],
        [InlineKeyboardButton("Крупная покупка", callback_data="settings_bigbuy")],
        [InlineKeyboardButton("Глубина анализа", callback_data="settings_days")],
        [InlineKeyboardButton("Таймфрейм цикла", callback_data="settings_tf")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
    ]
    text = (
        f"⚙️ Настройки:\n\n"
        f"Порог ликвидности: {settings['eps_bp']} bps\n"
        f"Крупная покупка: {settings['big_buy_mult']}× среднего\n"
        f"Глубина анализа: {settings['analysis_days']} дней\n"
        f"Таймфрейм цикла: {settings['cycle_tf']}\n"
    )
    return InlineKeyboardMarkup(keyboard), text

# --- Команда /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        await update.message.reply_text("⛔ У вас нет доступа к этому боту.")
        return
    await update.message.reply_text("Привет! Выбери действие:", reply_markup=main_menu())

# --- Пагинация меню активов ---
async def show_assets_menu(query, user_id, page=0):
    assets = user_assets.get(user_id, [])
    per_page = 5
    start = page * per_page
    end = start + per_page
    page_assets = assets[start:end]

    keyboard = [[InlineKeyboardButton(a, callback_data=f"asset_{a}")] for a in page_assets]

    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("⬅️", callback_data=f"page_{page-1}"))
    if end < len(assets):
        nav_buttons.append(InlineKeyboardButton("➡️", callback_data=f"page_{page+1}"))
    if nav_buttons:
        keyboard.append(nav_buttons)

    keyboard.append([InlineKeyboardButton("⬅️ Назад", callback_data="back")])
    await query.edit_message_text("Ваши активы:", reply_markup=InlineKeyboardMarkup(keyboard))

# --- Классификация стадии цикла ---
def classify_cycle(df):
    df = df.copy()
    
    # Определяем, какой столбец использовать для цен
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

# --- Приближенный объем ликвидности ---
def estimate_liquidity(df, eps_bp=5):
    if df.empty or "Volume" not in df:
        return None
    
    # Определяем, какой столбец использовать для цен
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

# --- Поиск последней крупной покупки ---
def detect_last_large_buy(df, mult=2):
    if df.empty:
        return None
    
    # Определяем, какой столбец использовать для цен
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

# --- Расчет CAGR (Compound Annual Growth Rate) ---
def calculate_cagr(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    # Получаем исторические данные за указанный период
    hist = stock.history(period=period)
    if len(hist) < 2:
        raise Exception("Недостаточно данных для расчета CAGR")
    
    # Используем Adj Close если доступно, иначе Close
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
    
    # CAGR = (End Value / Start Value)^(1/n) - 1
    # где n - количество лет
    cagr = ((end_price / start_price) ** (1.0/years)) - 1
    return cagr * 100, f"https://finance.yahoo.com/quote/{ticker}"

# --- Расчет EPS (Earnings Per Share) ---
def calculate_eps(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Пытаемся получить EPS из доступной информации
    if "trailingEps" in info and info["trailingEps"] is not None:
        return info["trailingEps"], f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    elif "epsTrailingTwelveMonths" in info and info["epsTrailingTwelveMonths"] is not None:
        return info["epsTrailingTwelveMonths"], f"https://finance.yahoo.com/quote/{ticker}/key-statistics"
    else:
        raise Exception("EPS данные недоступны для этого актива")

# --- Расчет бета-коэффициента ---
def calculate_beta(ticker, benchmark="^GSPC", period="3y"):
    # Получаем данные для актива и эталонного индекса (S&P 500 по умолчанию)
    stock = yf.Ticker(ticker)
    benchmark_stock = yf.Ticker(benchmark)
    
    # Используем более длинный период для более точного расчета
    stock_hist = stock.history(period=period)
    benchmark_hist = benchmark_stock.history(period=period)
    
    if len(stock_hist) < 30 or len(benchmark_hist) < 30:
        raise Exception("Недостаточно данных для расчета бета-коэффициента")
    
    # Определяем, какой столбец использовать для цен
    stock_price_col = "Adj Close" if "Adj Close" in stock_hist.columns else "Close"
    benchmark_price_col = "Adj Close" if "Adj Close" in benchmark_hist.columns else "Close"
    
    # Рассчитываем доходности
    stock_returns = stock_hist[stock_price_col].pct_change().dropna()
    benchmark_returns = benchmark_hist[benchmark_price_col].pct_change().dropna()
    
    # Выравниваем данные по датам
    aligned_data = stock_returns.align(benchmark_returns, join='inner')
    stock_returns_aligned = aligned_data[0]
    benchmark_returns_aligned = aligned_data[1]
    
    # Рассчитываем бета-коэффициент
    covariance = np.cov(stock_returns_aligned, benchmark_returns_aligned)[0][1]
    benchmark_variance = np.var(benchmark_returns_aligned)
    
    if benchmark_variance == 0:
        raise Exception("Дисперсия эталонного индекса равна нулю")
    
    beta = covariance / benchmark_variance
    return beta, f"https://finance.yahoo.com/quote/{ticker}"

# --- Расчет бета-коэффициента (5 лет, месячные данные) ---
def calculate_beta_5y_monthly(ticker, benchmark="^GSPC"):
    # Получаем данные для актива и эталонного индекса (S&P 500 по умолчанию)
    stock = yf.Ticker(ticker)
    benchmark_stock = yf.Ticker(benchmark)
    
    # Используем 5-летний период с месячными интервалами
    stock_hist = stock.history(period="5y", interval="1mo")
    benchmark_hist = benchmark_stock.history(period="5y", interval="1mo")
    
    if len(stock_hist) < 12 or len(benchmark_hist) < 12:
        raise Exception("Недостаточно данных для расчета бета-коэффициента (5y monthly)")
    
    # Определяем, какой столбец использовать для цен
    stock_price_col = "Adj Close" if "Adj Close" in stock_hist.columns else "Close"
    benchmark_price_col = "Adj Close" if "Adj Close" in benchmark_hist.columns else "Close"
    
    # Рассчитываем доходности
    stock_returns = stock_hist[stock_price_col].pct_change().dropna()
    benchmark_returns = benchmark_hist[benchmark_price_col].pct_change().dropna()
    
    # Выравниваем данные по датам
    aligned_data = stock_returns.align(benchmark_returns, join='inner')
    stock_returns_aligned = aligned_data[0]
    benchmark_returns_aligned = aligned_data[1]
    
    # Рассчитываем бета-коэффициент
    covariance = np.cov(stock_returns_aligned, benchmark_returns_aligned)[0][1]
    benchmark_variance = np.var(benchmark_returns_aligned)
    
    if benchmark_variance == 0:
        raise Exception("Дисперсия эталонного индекса равна нулю")
    
    beta = covariance / benchmark_variance
    return beta, f"https://finance.yahoo.com/quote/{ticker}"

def build_info_text(ticker, user_id=None):
    settings = user_settings.get(user_id, {
        "eps_bp": 5,
        "big_buy_mult": 2,
        "analysis_days": 5,
        "cycle_tf": "5m"
    })

    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{settings['analysis_days']}d", interval=settings['cycle_tf'])
    if df.empty:
        return "Данные недоступны для этого тикера."

    # Определяем, какой столбец использовать для цен
    price_column = "Adj Close" if "Adj Close" in df.columns else "Close"

    last = df.iloc[-1]
    price = round(float(last[price_column]), 4)
    ts = last.name.to_pydatetime()

    look = df.tail(100) if len(df) >= 100 else df
    avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()
    rvol = 0.0
    if avg_vol is not None and avg_vol > 0:
        rvol = float(last["Volume"]) / avg_vol

    approx_book_vol = estimate_liquidity(df.tail(200), eps_bp=settings["eps_bp"])
    stage = classify_cycle(df)
    big = detect_last_large_buy(df, mult=settings["big_buy_mult"])

    # Форматируем вывод согласно требованиям
    info = []
    info.append(f"ℹ️ {ticker}")
    info.append(f"🕒 Последнее обновление: {ts.strftime('%Y-%m-%d %H:%M')}")
    info.append(f"💵 Цена: {price} USD")
    info.append(f"📊 Объём (последняя свеча): {int(last['Volume'])}")
    info.append(f"📈 RVOL: {rvol:.2f}× среднего")
    info.append(f"🧭 Стадия цикла ({settings['analysis_days']} дней): {stage}")
    
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

# --- Обработка кнопок ---
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    await query.answer()

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

    elif query.data.startswith("asset_"):
        ticker = query.data.split("_", 1)[1]
        keyboard = [
            [InlineKeyboardButton("ℹ️ Информация", callback_data=f"info_{ticker}"),
             InlineKeyboardButton("🗑 Удалить актив", callback_data=f"delete_{ticker}")],
            [InlineKeyboardButton("🧮 Калькулятор", callback_data=f"calc_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="my_assets")]
        ]
        await query.edit_message_text(f"Актив {ticker}", reply_markup=InlineKeyboardMarkup(keyboard))

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
        # Удаление актива из списка пользователя
        if user_id in user_assets and ticker in user_assets[user_id]:
            user_assets[user_id].remove(ticker)
            # Если список активов пользователя пуст, удаляем его
            if not user_assets[user_id]:
                del user_assets[user_id]
            await query.edit_message_text(f"✅ Актив {ticker} успешно удален!", reply_markup=main_menu())
        else:
            await query.edit_message_text(f"❌ Актив {ticker} не найден в вашем списке.", reply_markup=main_menu())

    elif query.data.startswith("calc_"):
        ticker = query.data.split("_", 1)[1]
        # Показываем меню калькулятора
        keyboard = [
            [InlineKeyboardButton("CAGR", callback_data=f"cagr_{ticker}"),
             InlineKeyboardButton("EPS", callback_data=f"eps_{ticker}")],
            [InlineKeyboardButton("Бета-коэффициент", callback_data=f"beta_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data=f"asset_{ticker}")]
        ]
        await query.edit_message_text(f"🧮 Калькулятор для {ticker}\nВыберите метрику:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("cagr_"):
        ticker = query.data.split("_", 1)[1]
        try:
            # Рассчитываем CAGR для 3 лет и 5 лет
            cagr_5y_value, source_url = calculate_cagr(ticker, period="5y")
            cagr_3y_value, _ = calculate_cagr(ticker, period="3y")
            
            # Формируем сообщение с обоими значениями
            message_text = f"📈 CAGR для {ticker}:\n\n"
            message_text += f"5-летний: {cagr_5y_value:.2f}%\n"
            message_text += f"3-летний: {cagr_3y_value:.2f}%\n\n"
            message_text += f"Источник данных: {source_url}\n"
            message_text += f"Формула: CAGR = (Конечная стоимость / Начальная стоимость)^(1/n) - 1"
            
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете CAGR для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("eps_"):
        ticker = query.data.split("_", 1)[1]
        try:
            eps_value, source_url = calculate_eps(ticker)
            message_text = f"📊 EPS для {ticker}: ${eps_value:.2f}\n\nИсточник данных: {source_url}"
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете EPS для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data.startswith("beta_"):
        ticker = query.data.split("_", 1)[1]
        try:
            # Рассчитываем оба значения бета
            beta_3y_value, source_url = calculate_beta(ticker)
            beta_5y_value, _ = calculate_beta_5y_monthly(ticker)
            
            # Формируем сообщение с обоими значениями
            message_text = f"📊 Бета-коэффициент для {ticker}:\n\n"
            message_text += f"5-летний (месячные данные): {beta_5y_value:.2f}\n"
            message_text += f"3-летний (дневные данные): {beta_3y_value:.2f}\n\n"
            message_text += f"Источник данных: {source_url}\n"
            message_text += f"Формула: β = Cov(Ri, Rm) / Var(Rm)"
            
            await query.edit_message_text(message_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка при расчете бета-коэффициента для {ticker}: {str(e)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=f"calc_{ticker}")]]))

    elif query.data == "settings":
        kb, text = settings_menu(user_id)
        await query.edit_message_text(text, reply_markup=kb)

    elif query.data.startswith("set_eps_"):
        val = int(query.data.split("_")[2])
        user_settings.setdefault(user_id, {"eps_bp": 5, "big_buy_mult": 2, "analysis_days": 5, "cycle_tf": "5m"})
        user_settings[user_id]["eps_bp"] = val
        kb, text = settings_menu(user_id)
        await query.edit_message_text(f"✅ Порог ликвидности обновлён: {val} bps\n\n{text}", reply_markup=kb)

    elif query.data.startswith("set_bigbuy_"):
        val = int(query.data.split("_")[2])
        user_settings.setdefault(user_id, {"eps_bp": 5, "big_buy_mult": 2, "analysis_days": 5, "cycle_tf": "5m"})
        user_settings[user_id]["big_buy_mult"] = val
        kb, text = settings_menu(user_id)
        await query.edit_message_text(f"✅ Порог крупной покупки обновлён: {val}× среднего\n\n{text}", reply_markup=kb)

    elif query.data.startswith("set_days_"):
        val = int(query.data.split("_")[2])
        user_settings.setdefault(user_id, {"eps_bp": 5, "big_buy_mult": 2, "analysis_days": 5, "cycle_tf": "5m"})
        user_settings[user_id]["analysis_days"] = val
        kb, text = settings_menu(user_id)
        await query.edit_message_text(f"✅ Глубина анализа обновлена: {val} дней\n\n{text}", reply_markup=kb)

    elif query.data.startswith("set_tf_"):
        val = query.data.split("_")[2]
        user_settings.setdefault(user_id, {"eps_bp": 5, "big_buy_mult": 2, "analysis_days": 5, "cycle_tf": "5m"})
        user_settings[user_id]["cycle_tf"] = val
        kb, text = settings_menu(user_id)
        await query.edit_message_text(f"✅ Таймфрейм обновлён: {val}\n\n{text}", reply_markup=kb)

    elif query.data == "back":
        await query.edit_message_text("Главное меню:", reply_markup=main_menu())

    elif query.data.startswith("delete_"):
        ticker = query.data.split("_", 1)[1]
        # Удаление актива из списка пользователя
        if user_id in user_assets and ticker in user_assets[user_id]:
            user_assets[user_id].remove(ticker)
            # Если список активов пользователя пуст, удаляем его
            if not user_assets[user_id]:
                del user_assets[user_id]
            await query.edit_message_text(f"✅ Актив {ticker} успешно удален!", reply_markup=main_menu())
        else:
            await query.edit_message_text(f"❌ Актив {ticker} не найден в вашем списке.", reply_markup=main_menu())

    elif query.data.startswith("page_"):
        page = int(query.data.split("_")[1])
        await show_assets_menu(query, user_id, page)

    elif query.data == "settings_eps":
        kb = [
            [InlineKeyboardButton("2 bps", callback_data="set_eps_2"),
             InlineKeyboardButton("5 bps", callback_data="set_eps_5"),
             InlineKeyboardButton("10 bps", callback_data="set_eps_10")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="settings")]
        ]
        await query.edit_message_text("Выберите порог ликвидности:", reply_markup=InlineKeyboardMarkup(kb))

    elif query.data == "settings_bigbuy":
        kb = [
            [InlineKeyboardButton("2×", callback_data="set_bigbuy_2"),
             InlineKeyboardButton("3×", callback_data="set_bigbuy_3"),
             InlineKeyboardButton("5×", callback_data="set_bigbuy_5")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="settings")]
        ]
        await query.edit_message_text("Выберите коэффициент крупной покупки:", reply_markup=InlineKeyboardMarkup(kb))

    elif query.data == "settings_days":
        kb = [
            [InlineKeyboardButton("1 день", callback_data="set_days_1"),
             InlineKeyboardButton("3 дня", callback_data="set_days_3"),
             InlineKeyboardButton("5 дней", callback_data="set_days_5")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="settings")]
        ]
        await query.edit_message_text("Выберите глубину анализа:", reply_markup=InlineKeyboardMarkup(kb))

    elif query.data == "settings_tf":
        kb = [
            [InlineKeyboardButton("5m", callback_data="set_tf_5m"),
             InlineKeyboardButton("15m", callback_data="set_tf_15m")],
            [InlineKeyboardButton("1h", callback_data="set_tf_1h"),
             InlineKeyboardButton("1d", callback_data="set_tf_1d")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="settings")]
        ]
        await query.edit_message_text("Выберите таймфрейм для стадий цикла:", reply_markup=InlineKeyboardMarkup(kb))

# --- Обработка текстовых сообщений ---
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return

    if user_states.get(user_id) == "waiting_for_asset":
        ticker = update.message.text.strip().upper()
        user_assets.setdefault(user_id, [])
        if ticker not in user_assets[user_id]:
            user_assets[user_id].append(ticker)
        user_states[user_id] = None
        await update.message.reply_text(f"✅ Актив {ticker} добавлен!", reply_markup=main_menu())

# --- Запуск бота ---
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.run_polling()

if __name__ == "__main__":
    main()

