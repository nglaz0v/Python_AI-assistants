import sys
import os
import logging
import time
from dotenv import load_dotenv

from vkbottle import Bot, GroupEventType
from vkbottle.bot import BotLabeler, Message

# Добавляем путь к родительской директории для импорта модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import validate_config
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


# Создаем лейблер (маршрутизатор) для обработки сообщений
# BotLabeler - диспетчер, который принимает сообщения от ВКонтакте
# и связывает входящее сообщение с функцией-обработчиком
labeler = BotLabeler()


def main():
    token = os.getenv("VK_GROUP_TOKEN")

    if not token:
        logger.error("❌ Токен не установлен. Установите: export VK_GROUP_TOKEN='ваш_токен'")
        return

    # Инициализируем векторные хранилища перед запуском бота
    logger.info("🚀 Запуск инициализации системы...")
    
    if not initialize_vector_stores():
        logger.error("❌ Не удалось инициализировать векторные хранилища. Бот не может быть запущен.")
        return

    # Импортируем обработчики только ПОСЛЕ инициализации векторных хранилищ,
    # так как они зависят от цепочек, которые в свою очередь требуют инициализированных хранилищ
    from handlers.handlers_vk import (
        handle_message,
        handle_start_command,
        handle_help_command,
        handle_reload_command,
    )

    # Регистрируем обработчики команд
    @labeler.message(text="/start")
    async def start_handler(message: Message) -> None:
        """Обрабатывает команду /start"""
        await handle_start_command(message)

    @labeler.message(text="/help")
    async def help_handler(message: Message) -> None:
        """Обрабатывает команду /help"""
        await handle_help_command(message)

    @labeler.message(text="/reload")
    async def reload_handler(message: Message) -> None:
        """Обрабатывает команду /reload"""
        await handle_reload_command(message)

    # Регистрируем обработчик текстовых сообщений
    @labeler.message()
    async def message_handler(message: Message) -> None:
        """Обрабатывает все текстовые сообщения"""
        await handle_message(message)

    # Создаем бота
    bot = Bot(token=token, labeler=labeler)
    
    # Запускаем бота
    logger.info("🤖 ИИ агент книжного магазина запущен и готов к работе!")
    bot.run_forever()

if __name__ == "__main__":
    main()
