import sys
import os
import logging
import time
from telegram.ext import Application, MessageHandler, filters, CommandHandler

# Добавляем путь к родительской директории для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_stores.base import books_service_vector_store, organization_vector_store

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    force=True
)
# Отключаем подробные логи от HTTP-запросов
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
prompt_logger = logging.getLogger("prompt_history")


def initialize_vector_stores():
    """
    Инициализирует все векторные хранилища при запуске бота
    """
    logger.info("🔄 Начинается инициализация векторных хранилищ...")
    
    start_time = time.time()
    
    try:
        # Инициализируем хранилище книг и правил обслуживания
        books_service_vector_store.initialize_vector_store()
        logger.info("✅ Хранилище книг и правил обслуживания инициализировано")
        
        # Инициализируем хранилище документов организации
        organization_vector_store.initialize_vector_store()
        logger.info("✅ Хранилище документов организации инициализировано")
        
        elapsed_time = time.time() - start_time
        logger.info(f"🎉 Все векторные хранилища успешно инициализированы за {elapsed_time:.2f} секунд")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при инициализации векторных хранилищ: {e}")
        return False


def main():

    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        logger.error("❌ Токен не установлен. Установите: export TELEGRAM_BOT_TOKEN='ваш_токен'")
        return

    # Инициализируем векторные хранилища перед запуском бота
    logger.info("🚀 Запуск инициализации системы...")
    
    if not initialize_vector_stores():
        logger.error("❌ Не удалось инициализировать векторные хранилища. Бот не может быть запущен.")
        return

    # Импортируем обработчики только ПОСЛЕ инициализации векторных хранилищ,
    # так как они зависят от цепочек, которые в свою очередь требуют инициализированных хранилищ
    from handlers.handlers_tg import (
        handle_message,
        handle_start_command,
        handle_help_command,
        handle_reload_command,
    )

    # Создаем приложение
    application = Application.builder().token(token).build()
    
    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", handle_start_command))
    application.add_handler(CommandHandler("help", handle_help_command))
    application.add_handler(CommandHandler("reload", handle_reload_command))
    
    # Добавляем обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    logger.info("🤖 ИИ агент книжного магазина запущен и готов к работе!")
    application.run_polling(allowed_updates=None, drop_pending_updates=True)

if __name__ == "__main__":
    main()