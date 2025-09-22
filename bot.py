import logging
import yfinance as yf
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)

# 🔑 Токен и список доверенных пользователей
BOT_TOKEN = "8465557074:AAHTm9eINh3XSJBJW-tXTSO4g2v1I95vOoM"
TRUSTED_USERS = [1085064193]

# Хранилище активов по пользователям
user_assets = {}
user_states = {}

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Главное меню ---
def main_menu():
    keyboard = [
        [InlineKeyboardButton("➕ Добавить актив", callback_data="add_asset"),
         InlineKeyboardButton("📊 Мои активы", callback_data="my_assets")]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- Команда /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        await update.message.reply_text("⛔ У вас нет доступа к этому боту.")
        return
    await update.message.reply_text("Привет! Выбери действие:", reply_markup=main_menu())

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
            [InlineKeyboardButton("💵 Цена", callback_data=f"price_{ticker}")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="my_assets")]
        ]
        await query.edit_message_text(f"Актив {ticker}", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("price_"):
        ticker = query.data.split("_", 1)[1]
        price = get_price(ticker)
        await query.edit_message_text(f"💵 {ticker}: {price} USD", reply_markup=main_menu())

    elif query.data == "back":
        await query.edit_message_text("Главное меню:", reply_markup=main_menu())

    elif query.data.startswith("page_"):
        page = int(query.data.split("_")[1])
        await show_assets_menu(query, user_id, page)

# --- Показ меню активов с пагинацией ---
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

# --- Получение цены через yfinance ---
def get_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        return round(data["Close"].iloc[-1], 2)
    except Exception as e:
        logger.error(f"Ошибка получения цены {ticker}: {e}")
        return "Ошибка"

# --- Обработка текстовых сообщений ---
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in TRUSTED_USERS:
        return

    if user_states.get(user_id) == "waiting_for_asset":
        ticker = update.message.text.strip().upper()
        user_assets.setdefault(user_id, []).append(ticker)
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
