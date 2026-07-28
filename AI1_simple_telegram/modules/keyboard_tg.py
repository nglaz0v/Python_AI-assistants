from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from .config import TaskType

def get_main_keyboards():
    """
    Возвращает кортеж из двух клавиатур:
    (ReplyKeyboardMarkup, InlineKeyboardMarkup)
    """
    # ReplyKeyboardMarkup (обычная клавиатура)
    keyboard = ReplyKeyboardMarkup([[KeyboardButton("❌ Отмена")]], resize_keyboard=True, one_time_keyboard=False)

# InlineKeyboardMarkup (инлайн клавиатура)
    # Получаем все названия задач из TaskType
    task_names = [
        getattr(TaskType, attr)
        for attr in dir(TaskType)
        if not attr.startswith('_') and isinstance(getattr(TaskType, attr), str)
    ]
    
    # Получаем имена атрибутов для генерации callback_data
    task_attrs = [
        attr for attr in dir(TaskType)
        if not attr.startswith('_') and isinstance(getattr(TaskType, attr), str)
    ]
    
    # Создаём кнопки: текст = название задачи, callback_data = task_ + значение_атрибута_в_lower
    inline_buttons = [
        InlineKeyboardButton(label, callback_data=f"task_{getattr(TaskType, attr).lower()}")
        for attr, label in zip(task_attrs, task_names)
    ]
    
    # Разбиваем кнопки на строки по 2 в каждой
    inline_keyboard = []
    for i in range(0, len(inline_buttons), 2):
        inline_keyboard.append(inline_buttons[i:i+2])
    inline = InlineKeyboardMarkup(inline_keyboard)

    return keyboard, inline

def get_cancel_keyboard():
    """
    Возвращает клавиатуру только с кнопкой отмены
    """
    return ReplyKeyboardMarkup([[KeyboardButton("❌ Отмена")]], resize_keyboard=True, one_time_keyboard=True)