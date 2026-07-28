import asyncio
import logging
from telegram import Update
from telegram.ext import ContextTypes

# Импортируем конфигурацию
from .config import PROVIDER, DEFAULT_TEMPERATURE, MAX_TOKENS, MAX_HISTORY, RECENT_MESSAGES

# Импортируем сервисы
from .llm_service import get_system_prompt, generate_summary


# Настройка логирования
logger = logging.getLogger(__name__)
prompt_logger = logging.getLogger("prompt_history")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Получаем текущий событийный цикл
    loop = asyncio.get_event_loop()

    # Получаем текст сообщения
    if not update.message or not update.message.text:
        return
    user_prompt = update.message.text
    
    # Получаем ID пользователя для хранения индивидуальной истории
    user_id = update.effective_user.id if update.effective_user else 0
    user_name = update.effective_user.full_name if update.effective_user else None

    # Загружаем системный промпт
    system_prompt = get_system_prompt()  
    
    # Инициализируем историю для пользователя, если её нет
    if user_id not in context.bot_data['chat_histories']:
        context.bot_data['chat_histories'][user_id] = []
        context.bot_data['llm_messages'][user_id] = []
        context.bot_data['llm_messages'][user_id].append({"role": "system", "content": system_prompt})   
        
    # Основная логика обработки сообщения
    # Добавляем новое сообщение пользователя в историю диалога
    context.bot_data['chat_histories'][user_id].append({"role": "user", "content": user_prompt})
    context.bot_data['llm_messages'][user_id].append({"role": "user", "content": user_prompt})
    
    # Делаем саммари каждые MAX_HISTORY сообщений
    if len(context.bot_data['llm_messages'][user_id]) >= MAX_HISTORY:
        # Делаем саммари и обновляем системный промпт
        summary_text = await generate_summary(
            history=context.bot_data['llm_messages'][user_id],
            loop=loop
        )
        
        # Формируем новую историю диалога
        original_system_prompt = get_system_prompt()
        context.bot_data['llm_messages'][user_id] = []
        context.bot_data['llm_messages'][user_id].append({
            "role": "system", 
            "content": f"{original_system_prompt}\n\n---\n\nСводка предыдущего диалога:\n{summary_text}"
        })
        context.bot_data['llm_messages'][user_id].extend(context.bot_data['chat_histories'][user_id][-RECENT_MESSAGES:])
            
    try:
        # Вызов LLM
        response = await loop.run_in_executor(
            None,
            PROVIDER.send_request,
            context.bot_data['llm_messages'][user_id], 
            DEFAULT_TEMPERATURE,
            MAX_TOKENS
        )
        
        # Добавляем ответ ассистента в историю диалога
        context.bot_data['chat_histories'][user_id].append({"role": "assistant", "content": response})
        context.bot_data['llm_messages'][user_id].append({"role": "assistant", "content": response})

        # Логируем историю диалога с привязкой к user_id
        log_lines = [f"User ID: {user_id}"]
        for message in context.bot_data['llm_messages'][user_id]:
            if message["role"] in ("user", "assistant"):
                content_preview = message['content'][:60]  
                log_lines.append(f"{message['role']}: {content_preview}")
        prompt_logger.info(f"ТТТUser: {user_name}\n" + "\n".join(log_lines))

        # Отправляем ответ пользователю
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Ошибка при обращении к LLM: {e}")
        await update.message.reply_text(f"❌ Ошибка: {e}")