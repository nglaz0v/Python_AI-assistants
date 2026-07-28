import logging
from datetime import datetime
from vkbottle.bot import Message, MessageEvent
import asyncio
from .config import TaskType, user_states, UserState, welcome_text_template
from .keyboard_vk import get_main_keyboards, get_cancel_keyboard
from llm_providers.openAI_API import OpenAIAPIProvider
from llm_providers.gigachat_API import GigaChatAPIProvider
from llm_providers.yandexgpt_API import YandexGPTAPIProvider

# Состояния диалога бота
CHOOSING_TASK = 0      # Состояние выбора задачи
WAITING_FOR_PROMPT = 1 # Состояние ожидания ввода текста для обработки

# Список названий задач для распознавания в текстовых сообщениях (нужно для VK)
TASK_NAMES = [
    getattr(TaskType, attr)
    for attr in dir(TaskType)
    if not attr.startswith('_') and isinstance(getattr(TaskType, attr), str)
]

logger = logging.getLogger(__name__)

async def start(message: Message) -> int:
    """Обрабатывает команду /start"""
    # Получаем данные пользователя
    user_id = message.from_id
    
    # Создаем инлайн клавиатуру
    _, inline_kb = get_main_keyboards()
    
    # Выводим привественное сообщение в чате
    await message.answer(welcome_text_template.format(first_name="Пользователь"), keyboard=inline_kb.get_json() if inline_kb else None)
    
    return CHOOSING_TASK


async def handle_task_selection(message: Message) -> int:
    """Обрабатывает выбор задачи пользователем."""
    if not message.text:
        return CHOOSING_TASK
    text = message.text
    user_id = message.from_id

    if text == "❌ Отмена":
        # Сначала скрываем клавиатуру с кнопкой отмены
        await message.answer(
            "❌ Операция отменена.",
            keyboard=None  # Скрываем клавиатуру
        )
        # Затем показываем инлайн меню с выбором задач
        _, inline_kb = get_main_keyboards()
        await message.answer(
            "Выберите задачу:",
            keyboard=inline_kb.get_json() if inline_kb else None
        )
        if user_id in user_states:
            del user_states[user_id]
        return CHOOSING_TASK

    # Проверяем, является ли текст названием задачи
    # В VK при нажатии на инлайн кнопку текст кнопки отправляется как обычное сообщение,
    # а не как callback-событие. Поэтому нам нужно распознавать названия задач в текстовых сообщениях.
    if text in TASK_NAMES:
        # Пользователь выбрал задачу через текст (нажал на кнопку в VK)
        task = text
        
        # Сохраняем состояние пользователя
        user_states[user_id] = UserState(task=task, created_at=datetime.now())
        
        # Отправляем описание задачи
        description = TaskType.task_description(task)
        await message.answer(description, keyboard=None)
        
        # Отправляем кнопку отмены
        cancel_kb = get_cancel_keyboard()
        await message.answer(
            "Для отмены используйте кнопку ниже:",
            keyboard=cancel_kb.get_json() if cancel_kb else None
        )
        
        return WAITING_FOR_PROMPT

    reply_kb, inline_kb = get_main_keyboards()
    await message.answer("❌ Неизвестная команда. Используйте кнопки меню.", keyboard=inline_kb.get_json() if inline_kb else None)
    return CHOOSING_TASK

async def handle_prompt_input(message: Message) -> int:
    """Обрабатывает ввод текста пользователем для выбранной задачи."""
    # Получаем текст сообщения
    text = message.text

    # Проверяем, что не нажата кнопка "Отмена"
    if text == "❌ Отмена":
        # Сначала скрываем клавиатуру с кнопкой отмены
        await message.answer(
            "❌ Операция отменена.",
            keyboard=None
        )
        # Затем показываем инлайн меню с выбором задач
        _, inline_kb = get_main_keyboards()
        await message.answer(
            "Выберите задачу:",
            keyboard=inline_kb.get_json() if inline_kb else None
        )

        return CHOOSING_TASK
    
    # Получаем данные пользователя и его состояние
    user_id = message.from_id
    user_state = user_states.get(user_id)
    if user_state:
        # Извлекаем задачу, выбранную пользователем на предыдущем шаге
        task = user_state.task
        
        # Показываем сообщение о начале обработки запроса
        processing_msg = await message.answer(
            f"🧠 Обрабатываю запрос...\nЗадача: {task}"
        )
        
        # Создаем клавиатуру для последующего использования в ответах
        _, inline_kb = get_main_keyboards()
        
        try:
            # Отправляем запрос к LLM для обработки текста пользователя
            response = await handle_request(task, text)
            
            # Отправляем результат пользователю
            await message.answer(
                f"✅ **Результат:**\n\n{response}",
                keyboard=inline_kb.get_json() if inline_kb else None
            )
        except Exception as e:
            # Логируем ошибку и сообщаем о ней пользователю
            logger.error(f"Ошибка при обработке запроса: {e}")
            await message.answer(
                f"❌ Ошибка: {str(e)}",
                keyboard=inline_kb.get_json() if inline_kb else None
            )
        finally:
            
            # Очищаем состояние пользователя, завершая сессию
            if user_id in user_states:
                del user_states[user_id]
    
    # Возвращаемся к состоянию выбора задачи
    return CHOOSING_TASK

async def handle_callback(event: MessageEvent) -> int:
    """Обрабатывает callback-запросы от кнопок."""
    await event.show_snackbar("")  
    payload = event.payload
    
    # Извлекаем имя задачи из payload
    if payload and "task" in payload:
        task_name = payload["task"]
    else:
        return CHOOSING_TASK
    
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
    user_id = event.user_id
    user_states[user_id] = UserState(task=task, created_at=datetime.now())
    
    # Отправляем описание задачи
    description = TaskType.task_description(task)
    await event.edit_message(description, keyboard=None)
    
    # Отправляем кнопку отмены
    cancel_kb = get_cancel_keyboard()
    await event.send_message(
        "Для отмены используйте кнопку ниже:",
        keyboard=cancel_kb.get_json() if cancel_kb else None
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