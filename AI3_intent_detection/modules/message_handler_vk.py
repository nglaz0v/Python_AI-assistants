import asyncio
import logging
import json
from vkbottle.bot import Message

# Импортируем конфигурацию
from .config import PROVIDER, DEFAULT_TEMPERATURE, MAX_TOKENS, MAX_HISTORY, RECENT_MESSAGES

# Импортируем сервисы
from .llm_service import get_system_prompt, generate_summary

# Импортируем обработчики намерений
from .intent_handlers_vk import (handle_order_intent, handle_confirmed_intent, handle_delivery_intent,
    handle_show_cart_intent, handle_unknown_intent)

# Настройка логирования
logger = logging.getLogger(__name__)
prompt_logger = logging.getLogger("prompt_history")

# Очистка ответов LLM
def strip_code_fences(raw_text: str) -> str:
    """Удаляет обрамление в виде кавычек и тройных бэктиков ```/```json."""
    # Снимаем внешние одинарные/двойные кавычки, если вся строка в них
    if (raw_text.startswith("'") and raw_text.endswith("'")) or \
       (raw_text.startswith('"') and raw_text.endswith('"')):
        raw_text = raw_text[1:-1]
    if raw_text.startswith("```"):
        # удалить первую строку (``` или ```json)
        nl = raw_text.find("\n")
        if nl != -1:
            raw_text = raw_text[nl + 1 :]
        # удалить завершающие бэктики
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
    return raw_text

async def handle_message(message: Message, chat_histories: dict, llm_messages: dict) -> None:
    """Обрабатывает входящее сообщение от пользователя в VK."""
    # Получаем текущий событийный цикл
    loop = asyncio.get_event_loop()

    # Получаем текст сообщения
    if not message.text:
        return
    user_prompt = message.text
    
    # Получаем ID пользователя для хранения индивидуальной истории
    user_id = message.from_id
    user_name = message.from_id  # В VK нет прямого доступа к имени пользователя в сообщении

    # Загружаем системный промпт
    system_prompt = get_system_prompt(user_id)
    
    # Инициализируем историю для пользователя, если её нет
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        llm_messages[user_id] = []
        llm_messages[user_id].append({"role": "system", "content": system_prompt})
      
    ## Основная логика обработки сообщения
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
        original_system_prompt = get_system_prompt(user_id)
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

        # Получаем JSON из ответа LLM и очищаем его
        try:
            cleaned = strip_code_fences(response)
            parsed_response = json.loads(cleaned)
        except Exception as parse_error:
            logger.error(f"Ошибка парсинга JSON от LLM: {parse_error}")
            final_text = "❌ Ошибка обработки ответа модели (некорректный JSON)"
        else:
            # Добавляем ответ ассистента в историю диалога
            chat_histories[user_id].append({"role": "assistant", "content": parsed_response["answer_llm"]})
            llm_messages[user_id].append({"role": "assistant", "content": parsed_response.get("answer_llm")})

            # Логируем историю диалога с привязкой к user_id
            log_lines = [f"User ID: {user_id}"]
            for msg in llm_messages[user_id]:
                if msg["role"] in ("user", "assistant"):
                    content_preview = msg['content'][:60]  
                    log_lines.append(f"{msg['role']}: {content_preview}")
            prompt_logger.info(f"ТТТUser: {user_name}\n" + "\n".join(log_lines))
            
            # Диспетчеризация в зависимости от намерения покупателя
            intent = parsed_response.get("intent")
            
            intent_handlers = {
                "order": lambda: handle_order_intent(user_id, parsed_response),
                "confirmed": lambda: handle_confirmed_intent(user_id, message.peer_id, parsed_response, message, llm_messages),
                "delivery": lambda: handle_delivery_intent(user_id, message.peer_id, parsed_response, message),
                "show_cart": lambda: handle_show_cart_intent(user_id, parsed_response),
                "unknown": lambda: handle_unknown_intent(user_id, parsed_response),
            }

            # Получаем обработчик для намерения
            handler = intent_handlers.get(intent)
            if handler is None:
                handler = lambda: handle_unknown_intent(user_id, parsed_response)
            logger.info("="*20)
            logger.info(f"Обрабатываем намерение: {intent}")
            
            # Выполняем обработчик
            result = handler()
            if asyncio.iscoroutine(result):
                final_text, followup_coro = await result
            else:
                final_text, followup_coro = result
                
            if followup_coro:
                asyncio.create_task(followup_coro)

    except Exception as e:
        logger.error(f"Ошибка при обращении к LLM: {e}")
        final_text = f"❌ Ошибка: {e}"

    # Отправка ответа
    await message.answer(final_text)
