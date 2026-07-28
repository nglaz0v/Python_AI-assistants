import os
import datetime
import logging
from typing import Dict
from .config import PROVIDER, SUMMARY_TEMPERATURE, MAX_TOKENS

from AI3_intent_detection.mock.mock import mock_cart_get


# Настройка логирования
logger = logging.getLogger(__name__)

def get_system_prompt(user_id: int):
    """Читает и объединяет системные промпты из нескольких файлов с корзиной покупателя."""
    base_dir = os.path.dirname(__file__)
    prompts_dir = os.path.join(base_dir, '..', 'prompts')
    
    # Список файлов для объединения
    prompt_files = [
        'system_prompt.txt',
        'product.txt',
        'promo.txt',
        'delyvery.txt'
    ]
    
    # Читаем и объединяем содержимое файлов
    system_prompt = ""
    for file_name in prompt_files:
        file_path = os.path.join(prompts_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as f:
                system_prompt += f.read() + "\n\n"
    
    # Добавляем текущую дату в промпт
    system_prompt += f"Сегодняшнее число: {datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    # Добавляем корзину покупателя
    user_cart_text = mock_cart_get(user_id)
    system_prompt = f"{system_prompt}\n\n{user_cart_text}"

    # Убираем лишние переносы в конце
    system_prompt = system_prompt.strip()
        
    return system_prompt

def get_summary_prompt():
    """Читает промпт для саммаризации диалога."""
    base_dir = os.path.dirname(__file__)
    prompts_dir = os.path.join(base_dir, '..', 'prompts')
    
    # Читаем саммари-промпт
    summary_prompt_path = os.path.join(prompts_dir, 'summary.txt')
    with open(summary_prompt_path, encoding='utf-8') as f:
        summary_prompt = f.read().strip()
        
    return summary_prompt


async def generate_summary(history, loop):
    """Генерирует саммари диалога и обновляет контекст."""
    try:
        # Получаем промпт для саммаризации
        summary_prompt = get_summary_prompt()

        # Фильтруем только пользовательские и ассистентские сообщения для саммари
        user_assistant_messages = [msg for msg in history if msg["role"] in ("user", "assistant")]
        user_summary_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in user_assistant_messages])

        # Создаем сообщения для запроса саммари
        summary_messages = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": user_summary_prompt},
        ]

        # Запрашиваем саммари
        summary = await loop.run_in_executor(
            None,
            PROVIDER.send_request,
            summary_messages, 
            SUMMARY_TEMPERATURE,
            MAX_TOKENS
        )
        
        summary = f"{summary}\n\nТы уже здоровался с пользователем"
        logger.info(f"Сгенерировано саммари: {summary}")
        
        return summary

    except Exception as e:
        logger.error(f"Ошибка генерации саммари: {e}")
        # В случае ошибки возвращаем пустую строку
        return "Извините, непредвиденная ошибка"