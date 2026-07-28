import logging
from datetime import datetime
from telegram import Update, ReplyKeyboardRemove
from telegram.ext import ContextTypes
import asyncio
from .config import TaskType, user_states, UserState, welcome_text_template
from .keyboard_tg import get_main_keyboards, get_cancel_keyboard
from llm_providers.openAI_API import OpenAIAPIProvider
from llm_providers.gigachat_API import GigaChatAPIProvider
from llm_providers.yandexgpt_API import YandexGPTAPIProvider

# Состояния диалога бота
CHOOSING_TASK = 0      # Состояние выбора задачи
WAITING_FOR_PROMPT = 1 # Состояние ожидания ввода текста для обработки

logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает команду /start"""
    # Получаем данные пользователя
    user = update.effective_user
    
    # Создаем инлайн клавиатуру
    _, inline_kb = get_main_keyboards()
    
    # Выводим привественное сообщение в чате
    await update.message.reply_text(welcome_text_template.format(first_name=user.first_name), reply_markup=inline_kb)
    
    return CHOOSING_TASK


async def handle_task_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает выбор задачи пользователем."""
    if not update.message or not update.message.text or not update.effective_user:
        return CHOOSING_TASK
    text = update.message.text
    user_id = update.effective_user.id

    if text == "❌ Отмена":
        # Сначала скрываем клавиатуру с кнопкой отмены
        await update.message.reply_text(
            "❌ Операция отменена.",
            reply_markup=ReplyKeyboardRemove()
        )
        # Затем показываем инлайн меню с выбором задач
        _, inline_kb = get_main_keyboards()
        await update.message.reply_text(
            "Выберите задачу:",
            reply_markup=inline_kb
        )
        if user_id in user_states:
            del user_states[user_id]
        return CHOOSING_TASK

    reply_kb, inline_kb = get_main_keyboards()
    await update.message.reply_text("❌ Неизвестная команда. Используйте кнопки меню.", reply_markup=inline_kb)
    return CHOOSING_TASK

async def handle_prompt_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает ввод текста пользователем для выбранной задачи."""
    # Получаем текст сообщения
    text = update.message.text

    # Проверяем, что не нажата кнопка "Отмена"
    if text == "❌ Отмена":
        # Сначала скрываем клавиатуру с кнопкой отмены
        await update.message.reply_text(
            "❌ Операция отменена.",
            reply_markup=ReplyKeyboardRemove()
        )
        # Затем показываем инлайн меню с выбором задач
        _, inline_kb = get_main_keyboards()
        await update.message.reply_text(
            "Выберите задачу:",
            reply_markup=inline_kb
        )

        return CHOOSING_TASK
    
    # Получаем данные пользователя и его состояние
    user_id = update.effective_user.id
    user_state = user_states.get(user_id)
    if user_state:
        # Извлекаем задачу, выбранную пользователем на предыдущем шаге
        task = user_state.task
        
        # Показываем сообщение о начале обработки запроса
        processing_msg = await update.message.reply_text(
            f"🧠 Обрабатываю запрос...\nЗадача: {task}"
        )
        
        # Создаем клавиатуру для последующего использования в ответах
        _, inline_kb = get_main_keyboards()
        
        try:
            # Отправляем запрос к LLM для обработки текста пользователя
            response = await handle_request(task, text)
            
            # Отправляем результат пользователю
            await update.message.reply_text(
                f"✅ **Результат:**\n\n{response}",
                reply_markup=inline_kb
            )
        except Exception as e:
            # Логируем ошибку и сообщаем о ней пользователю
            logger.error(f"Ошибка при обработке запроса: {e}")
            await update.message.reply_text(
                f"❌ Ошибка: {str(e)}",
                reply_markup=inline_kb
            )
        finally:
            # Удаляем сообщение о процессе обработки
            await processing_msg.delete()
            
            # Очищаем состояние пользователя, завершая сессию
            if user_id in user_states:
                del user_states[user_id]
    
    # Возвращаемся к состоянию выбора задачи
    return CHOOSING_TASK

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обрабатывает callback-запросы от кнопок."""
    query = update.callback_query
    
    await query.answer()
    data = query.data
        
    # Извлекаем имя задачи из callback_data
    task_name = data.split("_")[1]
    
    # Динамически генерируем task_map на основе TaskType
    task_map = {
        # Ключ: название задачи в нижнем регистре
        # Значение: оригинальное название задачи
        getattr(TaskType, attr).lower(): getattr(TaskType, attr)
        for attr in dir(TaskType)
        if not attr.startswith('_') and isinstance(getattr(TaskType, attr), str)
    }
        
    # Получаем оригинальное название задачи
    task = task_map[task_name]
    
    # Сохраняем состояние пользователя
    user_id = query.from_user.id
    user_states[user_id] = UserState(task=task, created_at=datetime.now())
    
    # Отправляем описание задачи
    description = TaskType.task_description(task)
    await query.edit_message_text(description, reply_markup=None)
    
    # Отправляем кнопку отмены
    cancel_kb = get_cancel_keyboard()
    await context.bot.send_message(
        chat_id=query.message.chat.id,
        text="Для отмены используйте кнопку ниже:",
        reply_markup=cancel_kb
    )
        
    return WAITING_FOR_PROMPT


async def handle_request(task: str, prompt: str) -> str:
    """Отправляет запрос к LLM и возвращает ответ."""
    # Формируем messages
    messages = [
        {
            "role": "system",
            "content": TaskType.system_prompt(task)
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Для изменения провайдера необходимо и изменить эту строку
    provider = OpenAIAPIProvider()
    if provider:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, provider._make_request, messages, TaskType.temperature(task), TaskType.max_tokens(task))
    else:
        return "Ошибка: не удалось подключиться к модели."
