# Импортируем функции отправки запросов к LLM
from llm_providers.openAI_API import OpenAIAPIProvider
from llm_providers.gigachat_API import GigaChatAPIProvider
from llm_providers.yandexgpt_API import YandexGPTAPIProvider

# Глобальные параметры
DEFAULT_TEMPERATURE = 0.5  # Температура для обычных запросов
SUMMARY_TEMPERATURE = 0.1  # Температура для саммаризации
MAX_TOKENS = 512  # Максимальное количество токенов
MAX_HISTORY = 6  # Максимальная длина истории диалога
RECENT_MESSAGES = 2  # Количество последних сообщений для отправки в LLM после саммаризации

# Для изменения провайдера необходимо изменить эту строку
PROVIDER = OpenAIAPIProvider()