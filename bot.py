import logging
import yfinance as yf
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)

# 🔑 Токен и список доверенных пользователей
BOT_TOKEN = "8465557074:AAHTm9eINh3XSJBJW-tXTSO4g2v1I95vOoM"
TRUSTED_USERS = [1085064193]

# Хранилище активов и состояний
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

# --- Классификация стадии цикла ---
def classify_cycle(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    ema_slope = df["EMA20"].iloc[-1] - df["EMA20"].iloc[-2] if len(df) >= 2 else 0.0
    close = df["Close"].iloc[-1]
    ema = df["EMA20"].iloc[-1]
    above = close > ema
    near_flat_ema = abs(ema_slope) < (df["Close"].std() * 0.02 if df["Close"].std() else 0.0)

    delta = np.sign(df["Close"].diff().fillna(0))
    obv = (delta * df["Volume"]).fillna(0).cumsum()
    obv_slope = obv.iloc[-1] - (obv.iloc[-10] if len(obv) >= 10 else 0)

    window = df["Close"].tail(50)
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
    if df.empty or "Close" not in df or "Volume" not in df:
        return None
    df["ret_abs"] = (df["Close"].pct_change().abs()).fillna(0)
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
    look = df.tail(100) if len(df) >= 100 else df
    avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()
    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        rng = (row["High"] - row["Low"]) if (row["High"] >= row["Low"]) else 0
        near_high = (rng == 0) or ((row["High"] - row["Close"]) <= 0.1 * (rng + 1e-9))
        if (row["Volume"] > mult * (avg_vol if avg_vol else 1)) and near_high:
            ts = df.index[idx].to_pydatetime()
            return ts, int(row["Volume"])
    return None

# --- Сбор информации по тикеру ---
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

    last = df.iloc[-1]
    price = round(float(last["Close"]), 4)
    ts = last.name.to_pydatetime()

    look = df.tail(100) if len(df) >= 100 else df
    avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()
    rvol = (float(last["Volume"]) / avg_vol) if (avg_vol and avg_vol > 0) else 0.0

    approx_book_vol = estimate_liquidity(df.tail(200), eps_bp=settings["eps_bp"])
    stage = classify_cycle(df)
    big = detect_last_large_buy(df, mult=settings["big_buy_mult"])

    info = []
    info.append(f"ℹ️ {ticker}")
    info.append(f"💵 Цена: {price} USD")
    info.append(f"🕒 Последнее обновление: {ts.strftime('%Y-%m-%d %H:%M')}")
    info.append(f"📊 Объём (последняя свеча): {int(last['Volume'])}")
    info.append(f"📈 RVOL: {rvol:.2f}× среднего")
    info.append(f"🧭 Стадия цикла: {stage}")

    if approx_book_vol is not None:
        info.append(f"📥 Приближенный объем ликвидности (±{settings['eps_bp']/100:.2f}%): ~{approx_book_vol} акций")
    else:
        info.append("📥 Приближенный объем ликвидности: оценка недоступна")

    if big:
        ts_big, vol_big = big
        info.append(f"🚀 Последняя крупная покупка: {ts_big.strftime('%Y-%m-%d %H:%M')}, объём {vol_big}")
    else:
        info.append("🚀 Последняя крупная покупка: не обнаружена")

    return "\n".join(info)

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
            [InlineKeyboardButton("ℹ️ Информация", callback_data=f"info_{ticker}")],
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

    elif query.data.startswith("page_"):
        page = int(query.data.split("_")[1])
        await show_assets_menu(query, user_id, page)

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

