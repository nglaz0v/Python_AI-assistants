import sys
import os
import logging
from dotenv import load_dotenv

from vkbottle import Bot, GroupEventType
from vkbottle.bot import BotLabeler, Message, MessageEvent

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.handlers_vk import (
    start, handle_task_selection, handle_prompt_input,
    handle_callback
)
from modules.config import user_states


# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Создаем лейблер (маршрутизатор) для обработки сообщений
# BotLabeler - диспетчер, который принимает сообщения от ВКонтакте
# и связывает входящее сообщение с функцией-обработчиком
labeler = BotLabeler()

# Декоратор @labeler.message(text="/start") связывает сообщение
# "/start", с функцией обработчиком
@labeler.message(text="/start")
async def start_handler(message: Message) -> None:
    """Обрабатывает команду /start"""
    await start(message)

# Декоратор @labeler.message() связывает сообщение с любым текстом
# с обработчиком (кроме /start, его обработчик зарегистрирован ранее)
@labeler.message()
async def message_handler(message: Message) -> None:
    """Обрабатывает все текстовые сообщения"""
    # Логируем входящее сообщение
    logger.info(f"message_handler: message.text={message.text}, message.from_id={message.from_id}")
    
    user_id = message.from_id
    text = message.text
    
    # Проверяем состояние пользователя: если есть запись в user_states, значит WAITING_FOR_PROMPT
    # В VK нет встроенного ConversationHandler как в Telegram, поэтому мы сами проверяем состояние
    if user_id in user_states:
        await handle_prompt_input(message)
        return
    
    # Иначе обрабатываем как выбор задачи
    await handle_task_selection(message)

# Декоратор @labeler.raw_event() связывает событие
# MESSAGE_EVENT (аналог CallbackQuery в Telegram) с соотвествующим обработчиком
@labeler.raw_event(GroupEventType.MESSAGE_EVENT, MessageEvent)
async def callback_handler(event: MessageEvent) -> None:
    """Обрабатывает callback-события от кнопок"""
    # Логируем входящее событие
    logger.info(f"callback_handler: event.user_id={event.user_id}, event.payload={event.payload}")
    await handle_callback(event)

def main() -> None:
    """Запуск бота."""
    # Получаем токен из переменных окружения
    token = os.getenv('VK_GROUP_TOKEN')
    if not token:
        print("❌ Токен не установлен. Установите: export VK_GROUP_TOKEN='ваш_токен'")
        return

    # Создаем экземпляр бота VK
    bot = Bot(token=token, labeler=labeler)
    
    # Выводим информацию о запуске бота
    print("🤖 VK бот запущен...")
    print("📱 Используйте /start для начала работы")
    print("⏹️  Для остановки нажмите Ctrl+C")

    # Запускаем бота в режиме polling
    bot.run_forever()

if __name__ == "__main__":
    main()