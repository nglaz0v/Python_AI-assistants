import sys
import os
import logging
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, filters, CallbackQueryHandler
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.handlers_tg import (
    start, handle_task_selection, handle_prompt_input,
    handle_callback, CHOOSING_TASK, WAITING_FOR_PROMPT
)


# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def main() -> None:
    """Запуск бота."""
    # Получаем токен из переменных окружения
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        print("❌ Токен не установлен. Установите: export TELEGRAM_BOT_TOKEN='ваш_токен'")
        return

    try:
        # Создаем экземпляр приложения Telegram бота
        application = Application.builder().token(token).build()

        # Определяем обработчик разговоров
        conv_handler = ConversationHandler(
            # Точка входа в разговор - команда /start
            entry_points=[CommandHandler("start", start)],
            states={
                # Состояние выбора задачи пользователем
                CHOOSING_TASK: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, handle_task_selection),
                    CallbackQueryHandler(handle_callback),
                ],
                # Состояние ожидания ввода промта от пользователя
                WAITING_FOR_PROMPT: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, handle_prompt_input),
                ],
            },
            # Обработчик возврата к началу разговора
            fallbacks=[CommandHandler("start", start)],
            # Настройки обработчика: отдельно для каждого пользователя и чата
            per_user=True,
            per_chat=True,
        )

        # Добавляем обработчик разговоров в приложение
        application.add_handler(conv_handler)
        
        # Выводим информацию о запуске бота
        print("🤖 Telegram бот запущен...")
        print("📱 Используйте /start для начала работы")
        print("⏹️  Для остановки нажмите Ctrl+C")

        # Запускаем бота в режиме polling
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

    except Exception as e:
        # В случае ошибки выводим сообщение об ошибке
        print(f"❌ Ошибка запуска бота: {e}")

if __name__ == "__main__":
    main()
