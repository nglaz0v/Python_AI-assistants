import sys
import os
import logging
from dotenv import load_dotenv

from vkbottle import Bot, GroupEventType
from vkbottle.bot import BotLabeler, Message

# Добавляем путь к родительской директории
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.message_handler_vk import handle_message

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
prompt_logger = logging.getLogger("prompt_history")

# Создаем лейблер (маршрутизатор) для обработки сообщений
labeler = BotLabeler()

# Глобальные словари для хранения истории диалогов
chat_histories = {}
llm_messages = {}

# message_handler - функция-обёртка для передачи данных в handle_message
@labeler.message()
async def message_handler(message: Message) -> None:
    """Обрабатывает все текстовые сообщения"""
    await handle_message(message, chat_histories, llm_messages)

def main():
    token = os.getenv("VK_GROUP_TOKEN")

    if not token:
        prompt_logger.error("❌ Токен не установлен. Установите: export VK_GROUP_TOKEN='ваш_токен'")
        return

    # Создаем бота
    bot = Bot(token=token, labeler=labeler)
    
    # Инициализируем словари для хранения истории диалогов
    chat_histories.clear()
    llm_messages.clear()
    
    prompt_logger.info("🤖 Простой диалоговый VK чат-бот запущен...")
    
    # Запускаем бота в режиме polling
    bot.run_forever()

if __name__ == "__main__":
    main()