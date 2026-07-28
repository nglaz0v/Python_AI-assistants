from dataclasses import dataclass
from datetime import datetime
from typing import Dict

# Типы задач для генерации контента
class TaskType:
    MARKETING = "Маркетинговые рассылки"
    BUSINESS_LETTER = "Деловое письмо"
    TRANSLATION = "Перевод"

# Возвращает системный промпт в зависимости от типа задачи
    @staticmethod
    def system_prompt(task_type):
        prompts = {
            TaskType.MARKETING: "Ты — эксперт по маркетинговым рассылкам. Помоги создать эффективный текст рассылки.",
            TaskType.BUSINESS_LETTER: "Ты — помощник по деловой переписке. Составь грамотное деловое письмо.",
            TaskType.TRANSLATION: "Ты — профессиональный переводчик. Переведи текст максимально точно и естественно."
        }
        return prompts.get(task_type)
    
# Возвращает описание задачи для отображения пользователю
    @staticmethod
    def task_description(task_type):
        descriptions = {
            TaskType.MARKETING: "📧 Маркетинговые рассылки\n\nВведите описание продукта и аудиторию:",
            TaskType.BUSINESS_LETTER: "📄 Деловое письмо\n\nОпишите ситуацию и цель письма:",
            TaskType.TRANSLATION: "🌐 Перевод\n\nВведите текст для перевода:"
        }
        return descriptions.get(task_type, "")

# Возвращает температуру генерации в зависимости от типа задачи
    @staticmethod
    def temperature(task_type):
        temperatures = {
            TaskType.MARKETING: 0.7,
            TaskType.BUSINESS_LETTER: 0.3,
            TaskType.TRANSLATION: 0.1
        }
        return temperatures.get(task_type, 0.5)

# Возвращает максимальное количество токенов в зависимости от типа задачи
    @staticmethod
    def max_tokens(task_type):
        max_tokens = {
            TaskType.MARKETING: 800,
            TaskType.BUSINESS_LETTER: 1200,
            TaskType.TRANSLATION: 2000
        }
        return max_tokens.get(task_type, 1000)

# Хранит состояние пользователя: текущую задачу и временные метки
@dataclass
class UserState:
    task: str
    created_at: datetime

user_states: Dict[int, UserState] = {}


# Шаблон приветственного сообщения для пользователя
welcome_text_template = (
    "👋 Привет, {first_name}!\n\n"
    "🤖 Я AI-помощник для создания контента.\n\n"
    "📋 Доступные задачи:\n"
    "• 📧 Маркетинговые рассылки\n"
    "• 📄 Деловые письма\n"
    "• 🌐 Переводы\n\n"
    "Выберите задачу:"
)
