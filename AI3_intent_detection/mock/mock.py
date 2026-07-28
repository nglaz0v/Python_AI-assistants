import asyncio
import datetime
import random
from typing import Dict, Any, Awaitable, Callable, Tuple, List, Optional


# =====================
# Хранилища состояния MOK
# =====================

# Оплаченные заказы по пользователю
_PAID_STORE: Dict[int, Dict[str, Any]] = {}


# =====================
# MOK: оплата
# =====================

async def mock_payment_process(
    user_id: int,
    chat_id: int,
    items: Dict[str, int],
    send_message: Callable[[int, str], Awaitable[None]],
    on_confirm: Optional[Callable[[], Awaitable[None]]] = None,
) -> Tuple[str, str]:
    """MOK: создание платежа и последующее подтверждение через пару секунд.
    Возвращает (order_id, payment_url) сразу, а подтверждение отправляет асинхронно.
    Если передан on_confirm, вызывает его после отправки подтверждения.
    """
    # Генерирует простой псевдослучайный номер заказа.
    order_id = f"ORD-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}-{random.randint(100, 999)}"
    payment_url = f"https://pay.example.com/{order_id}"

    async def _confirm_later() -> None:
        await asyncio.sleep(2)
        status = "PAID"
        # сохраняем оплаченные товары и номер заказа
        _PAID_STORE[user_id] = {
            "order_id": order_id,
            "items": dict(items),
            "paid_at": datetime.datetime.now().isoformat(),
        }
        text = f"Оплата подтверждена. Номер заказа: {order_id}. Статус: {status}."
        await send_message(chat_id, text)
        if on_confirm is not None:
            try:
                await on_confirm()
            except Exception:
                pass

    asyncio.create_task(_confirm_later())
    return order_id, payment_url


# =====================
# MOK: доставка
# =====================

async def mock_delivery_process(
    user_id: int,
    chat_id: int,
    items: Dict[str, int],
    delivery_info: Dict[str, Any],
    send_message: Callable[[int, str], Awaitable[None]],
) -> str:
    """MOK: создание доставки и подтверждение приёма несколько пару секунд.
    Автоматически создает заказ и оплату, если их нет.
    Возвращает delivery_id сразу, подтверждение уходит асинхронно.
    """
    # Проверяем, есть ли оплаченный заказ
    paid = _PAID_STORE.get(user_id)
    
    if not paid or not paid.get("items"):
        # Если заказ не оплачен, создаем его автоматически
        order_id = f"ORD-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}-{random.randint(100, 999)}"
        payment_url = f"https://pay.example.com/{order_id}"
        
        # Сохраняем как оплаченный заказ
        _PAID_STORE[user_id] = {
            "order_id": order_id,
            "items": dict(items),
            "paid_at": datetime.datetime.now().isoformat(),
        }
        
        # Отправляем сообщение о создании заказа
        await send_message(chat_id, f"💳 Заказ автоматически создан и оплачен!\nНомер заказа: {order_id}\nСсылка на оплату: {payment_url}")
        
        # Имитируем подтверждение оплаты через 2 секунды
        async def _confirm_payment_later() -> None:
            await asyncio.sleep(2)
            await send_message(chat_id, f"Оплата подтверждена. Номер заказа: {order_id}. Статус: PAID.")
        
        asyncio.create_task(_confirm_payment_later())

    delivery_id = f"DLV-{random.randint(100000, 999999)}"

    async def _confirm_later() -> None:
        await asyncio.sleep(2)
        text = f"Доставка зарегистрирована. Номер доставки: {delivery_id}. Статус: REGISTRED."
        await send_message(chat_id, text)

    asyncio.create_task(_confirm_later())
    return delivery_id


# =====================
# MOK: корзина пользователя
# =====================

_CART_STORE: Dict[int, Dict[str, int]] = {}


def mock_cart_get(user_id: int) -> Dict[str, int]:
    """Возвращает копию корзины пользователя."""
    return _CART_STORE.get(user_id, {}).copy()


def mock_cart_update(user_id: int, new_cart: Dict[str, int]) -> Dict[str, int]:
    """Обновляет корзину пользователя новым состоянием."""
    _CART_STORE[user_id] = new_cart.copy()
    return new_cart.copy()


def mock_cart_clear(user_id: int) -> None:
    """Очищает корзину пользователя."""
    _CART_STORE[user_id] = {}


