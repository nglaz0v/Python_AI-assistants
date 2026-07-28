import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from AI3_intent_detection.mock.mock import (
    mock_payment_process,
    mock_delivery_process,
    mock_cart_get,
    mock_cart_update,
    mock_cart_clear,
)
from typing import Dict, Any, Tuple
import asyncio

from vkbottle.bot import Message
from .llm_service import get_system_prompt
        
def format_cart_text(cart: Dict[str, int]) -> str:
    """Возвращает форматированный текст корзины."""
    lines = ["Ваша корзина:"]
    if not cart:
        lines.append("пуста")
    else:
        for name, qty in cart.items():
            lines.append(f"- {name}: {qty} шт.")
    return "\n".join(lines)

async def handle_order_intent(user_id: int, parsed_response: Dict[str, Any]) -> Tuple[str, None]:
    """Обработка намерения order — обновление корзины."""
    # Получаем новое состояние корзины от LLM
    new_cart = parsed_response.get("cart") or {}
        
    # Обновляем корзину новым состоянием
    user_cart = mock_cart_update(user_id, new_cart)
    
    # Форматируем ответ с корзиной
    cart_text = format_cart_text(user_cart)
    return (f"{parsed_response.get('answer_llm')}\n\n{cart_text}", None)


async def handle_confirmed_intent(user_id: int, peer_id: int, parsed_response: Dict[str, Any], message: Message, llm_messages: dict):
    """Обработка намерения confirmed — создание заказа и оплата (MOK)."""
    user_cart = mock_cart_get(user_id)

    # Создаем заказ и запускаем имитацию оплаты
    # В VK используем message.answer() для отправки сообщений в тот же чат
    async def _send_message(target_peer_id: int, text: str) -> None:
        # В VK отправляем ответ в тот же чат, откуда пришло сообщение
        await message.answer(text)

    async def _after_paid() -> None:
        # Очищаем корзину и сбрасываем контекст после подтверждения оплаты
        mock_cart_clear(user_id)
        
        # Сбрасывает контекст LLM для пользователя
        system_prompt = get_system_prompt(user_id)
        
        # Сбрасываем llm_messages для пользователя
        if user_id in llm_messages:
            llm_messages[user_id] = []
            llm_messages[user_id].append({"role": "system", "content": system_prompt})

    order_id, payment_url = await mock_payment_process(
        user_id=user_id,
        chat_id=peer_id,
        items=user_cart.copy(),
        send_message=_send_message,
        on_confirm=_after_paid,
    )

    # Готовим ответ пользователю
    answer_llm = parsed_response.get("answer_llm")
    details_text = f"Номер заказа: {order_id}\nСсылка на оплату: {payment_url}"
    initial_text = f"{answer_llm}\n\n{details_text}"

    return (initial_text, None)


async def handle_delivery_intent(user_id: int, peer_id: int, parsed_response: Dict[str, Any], message: Message):
    """Обработка намерения delivery — создание заявки на доставку (MOK)."""
    user_cart = mock_cart_get(user_id)

    # Проверяем, что корзина не пуста
    if not user_cart:
        return ("Корзина пуста. Сначала добавьте товары в корзину.", None)

    delivery_info = {
        "address": parsed_response.get("delivery_address"),
        "phone_number": parsed_response.get("phone_number"),
        "delivery_date_time": parsed_response.get("delivery_date_time"),
    }
  
    # Имитация доставки (создание + подтверждение)
    # В VK используем message.answer() для отправки сообщений в тот же чат
    async def _send_delivery_message(target_peer_id: int, text: str) -> None:
        # В VK отправляем ответ в тот же чат, откуда пришло сообщение
        await message.answer(text)

    await mock_delivery_process(
        user_id=user_id,
        chat_id=peer_id,
        items=user_cart.copy(),
        delivery_info=delivery_info,
        send_message=_send_delivery_message,
    )
  
    # Формируем текст ответа после получения номера доставки
    answer_llm = parsed_response.get("answer_llm")
    answer_llm = f"{answer_llm}\n\nЗаявка на доставку зарегистрирована."
  
    return (answer_llm, None)


async def handle_show_cart_intent(user_id: int, parsed_response: Dict[str, Any]) -> Tuple[str, None]:
    """Возвращает ответ модели и текст корзины."""
    # Получаем текущую корзину
    user_cart = mock_cart_get(user_id)
    
    # Форматируем ответ с корзиной
    cart_text = format_cart_text(user_cart)
    return (f"{parsed_response.get('answer_llm')}\n\n{cart_text}", None)


async def handle_unknown_intent(user_id: int, parsed_response: Dict[str, Any]) -> Tuple[str, None]:
    """Обработка неопределенного намерения."""
    
    return (parsed_response["answer_llm"], None)
