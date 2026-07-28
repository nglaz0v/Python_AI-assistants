import asyncio
import logging
from vkbottle.bot import Message

# Импортируем конфигурацию
from .config import PROVIDER, DEFAULT_TEMPERATURE, MAX_TOKENS, MAX_HISTORY, RECENT_MESSAGES

# Импортируем сервисы
from .llm_service import get_system_prompt, generate_summary


# Настройка логирования
logger = logging.getLogger(__name__)
prompt_logger = logging.getLogger("prompt_history")

async def handle_message(message: Message, chat_histories, llm_messages) -> None:
    # Получаем текущий событийный цикл
    loop = asyncio.get_event_loop()

    # Получаем текст сообщения
    if not message.text:
        return
    user_prompt = message.text
    
    # Получаем ID пользователя для хранения индивидуальной истории
    user_id = message.from_id
    user_name = None  # В VK можно получить имя через API, но пока оставим None

    # Загружаем системный промпт
    system_prompt = get_system_prompt()  
    
    # Инициализируем историю для пользователя, если её нет
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        llm_messages[user_id] = []
        llm_messages[user_id].append({"role": "system", "content": system_prompt})   
        
    # Основная логика обработки сообщения
    # Добавляем новое сообщение пользователя в историю диалога
    chat_histories[user_id].append({"role": "user", "content": user_prompt})
    llm_messages[user_id].append({"role": "user", "content": user_prompt})
    
    # Делаем саммари каждые MAX_HISTORY сообщений
    if len(llm_messages[user_id]) >= MAX_HISTORY:
        # Делаем саммари и обновляем системный промпт
        summary_text = await generate_summary(
            history=llm_messages[user_id],
            loop=loop
        )
        
        # Формируем новую историю диалога
        original_system_prompt = get_system_prompt()
        llm_messages[user_id] = []
        llm_messages[user_id].append({
            "role": "system", 
            "content": f"{original_system_prompt}\n\n---\n\nСводка предыдущего диалога:\n{summary_text}"
        })
        llm_messages[user_id].extend(chat_histories[user_id][-RECENT_MESSAGES:])
            
    try:
        # Вызов LLM
        response = await loop.run_in_executor(
            None,
            PROVIDER.send_request,
            llm_messages[user_id], 
            DEFAULT_TEMPERATURE,
            MAX_TOKENS
        )
        
        # Добавляем ответ ассистента в историю диалога
        chat_histories[user_id].append({"role": "assistant", "content": response})
        llm_messages[user_id].append({"role": "assistant", "content": response})

        # Логируем историю диалога с привязкой к user_id
        log_lines = [f"User ID: {user_id}"]
        for msg in llm_messages[user_id]:
            if msg["role"] in ("user", "assistant"):
                content_preview = msg['content'][:60]  
                log_lines.append(f"{msg['role']}: {content_preview}")
        prompt_logger.info(f"ТТТUser: {user_name}\n" + "\n".join(log_lines))

        # Отправляем ответ пользователю
        await message.answer(response)
        
    except Exception as e:
        logger.error(f"Ошибка при обращении к LLM: {e}")
        await message.answer(f"❌ Ошибка: {e}")