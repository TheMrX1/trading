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
user_settings = {}  # user_id -> {"eps_bp": int}

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
    current = user_settings.get(user_id, {}).get("eps_bp", 5)
    keyboard = [
        [InlineKeyboardButton("2 bps", callback_data="set_eps_2"),
         InlineKeyboardButton("5 bps", callback_data="set_eps_5"),
         InlineKeyboardButton("10 bps", callback_data="set_eps_10")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back")]
    ]
    return InlineKeyboardMarkup(keyboard), current

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
def detect_last_large_buy(df):
    if df.empty:
        return None
    look = df.tail(100) if len(df) >= 100 else df
    avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()
    for idx in range(len(df) - 1, -1, -1):
        row = df.iloc[idx]
        rng = (row["High"] - row["Low"]) if (row["High"] >= row["Low"]) else 0
        near_high = (rng == 0) or ((row["High"] - row["Close"]) <= 0.1 * (rng + 1e-9))
        if (row["Volume"] > 2 * (avg_vol if avg_vol else 1)) and near_high:
            ts = df.index[idx].to_pydatetime()
            return ts, int(row["Volume"])
    return None

# --- Сбор информации по тикеру ---
def build_info_text(ticker, user_id=None):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5d", interval="5m")
    if df.empty:
        return "Данные недоступны для этого тикера."

    last = df.iloc[-1]
    price = round(float(last["Close"]), 4)
    ts = last.name.to_pydatetime()

    look = df.tail(100) if len(df) >= 100 else df
    avg_vol = look["Volume"].mean() if len(look) > 0 else df["Volume"].mean()
    rvol = (float(last["Volume"]) / avg_vol) if (avg_vol and avg_vol > 0) else 0.0

    eps_bp = 5
    if user_id and user_id in user_settings:
        eps_bp = user_settings[user_id].get("eps_bp", 5)

    approx_book_vol = estimate_liquidity(df.tail(200), eps_bp=eps_bp)
    stage = classify_cycle(df)
    big = detect_last_large_buy(df)

    info = []
    info.append(f"ℹ️ {ticker}")
    info.append(f"💵 Цена: {price} USD")
    info.append(f"🕒 Последнее обновление: {ts.strftime('%Y-%m-%d %H:%M')}")
    info.append(f"📊 Объём (последняя свеча): {int(last['Volume'])}")
    info.append(f"📈 RVOL: {rvol:.2f}× среднего")
    info.append(f"🧭 Стадия цикла: {stage}")

    if approx_book_vol is not None:
        info.append(f"📥 Приближенный объем ликвидности (±{eps_bp/100:.2f}%): ~{approx_book_vol} акций")
    else:
                info.append("📥 Приближенный объем ликвидности: оценка недоступна")

    if big:
        ts_big, vol_big = big
        info.append(f"🚀 Последняя крупная покупка: {ts_big.strftime('%Y-%m-%d %H:%M')}, объём {vol_big}")
    else:
        info.append("🚀 Последняя крупная покупка: не обнаружена")

    return "\n".join(info)

# --- Обработка нажатий кнопок ---
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
        kb, current = settings_menu(user_id)
        await query.edit_message_text(
            f"⚙️ Настройки\n\nТекущий порог ликвидности: {current} bps",
            reply_markup=kb
        )

    elif query.data.startswith("set_eps_"):
        val = int(query.data.split("_")[2])
        user_settings.setdefault(user_id, {})["eps_bp"] = val
        kb, current = settings_menu(user_id)
        await query.edit_message_text(
            f"✅ Порог ликвидности обновлён: {val} bps",
            reply_markup=kb
        )

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
