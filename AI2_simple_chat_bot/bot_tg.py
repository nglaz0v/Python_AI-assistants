import sys
import os
import logging
from telegram.ext import Application, MessageHandler, filters
from dotenv import load_dotenv

# Добавляем путь к родительской директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.message_handler_tg import handle_message

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
prompt_logger = logging.getLogger("prompt_history")

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        prompt_logger.error("❌ Токен не установлен. Установите: export TELEGRAM_BOT_TOKEN='ваш_токен'")
        return

    # Создаем приложение
    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Инициализируем словари для хранения истории диалогов
    application.bot_data['chat_histories'] = {}
    application.bot_data['llm_messages'] = {}
    
    application.run_polling(allowed_updates=None, drop_pending_updates=True)
    prompt_logger.info("🤖 Простой диалоговый Telegram чат-бот запущен...")

if __name__ == "__main__":
    main()
