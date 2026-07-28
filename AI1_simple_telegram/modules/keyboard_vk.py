from vkbottle.tools import Keyboard, KeyboardButtonColor, Text
from .config import TaskType

def get_main_keyboards():
    """
    Возвращает кортеж из двух клавиатур:
    (Keyboard с кнопкой отмены, Keyboard с кнопками задач)
    """
    # Обычная клавиатура (inline=False)
    keyboard = Keyboard(one_time=False, inline=False)
    keyboard.add(Text("❌ Отмена", payload={"type": "cancel"}))

# Инлайн клавиатура (inline=True)
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
    
    # Создаём кнопки: текст = название задачи, payload = task_ + значение_атрибута_в_lower
    inline_buttons = [
        Text(label, payload={"type": "task", "task": getattr(TaskType, attr).lower()})
        for attr, label in zip(task_attrs, task_names)
    ]
    
    # Разбиваем кнопки на строки по 2 в каждой
    inline_keyboard = Keyboard(inline=True)
    for i in range(0, len(inline_buttons), 2):
        inline_keyboard.add(inline_buttons[i], color=KeyboardButtonColor.PRIMARY)
        if i + 1 < len(inline_buttons):
            inline_keyboard.add(inline_buttons[i + 1], color=KeyboardButtonColor.PRIMARY)
        inline_keyboard.row()

    return keyboard, inline_keyboard

def get_cancel_keyboard():
    """
    Возвращает клавиатуру только с кнопкой отмены
    """
    keyboard = Keyboard(one_time=True, inline=False)
    keyboard.add(Text("❌ Отмена", payload={"type": "cancel"}))
    return keyboard