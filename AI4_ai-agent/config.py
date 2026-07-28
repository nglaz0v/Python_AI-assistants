"""
Конфигурационный модуль для ИИ агента книжного магазина
"""

import os
import logging
from dotenv import load_dotenv

# Настройка логирования для конфига
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из корневого .env файла
load_dotenv()

# Настройки LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
MAX_TOKENS = 512  # Ограничение на количество генерируемых токенов
EMBEDDING_CHUNK_SIZE = 5  # Размер чанка для генерации эмбеддингов
EMBEDDING_MAX_RETRIES = 3  # Максимальное количество попыток при ошибках API

# Telegram Bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# VK Bot
VK_GROUP_TOKEN = os.getenv("VK_GROUP_TOKEN")

# ChromaDB настройки
CHROMA_PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Пути к данным
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATHS = {
    "organization": os.path.join(PROJECT_ROOT, "test_data/Организация/ЛНА"),
    "service_rules": os.path.join(PROJECT_ROOT, "test_data/Организация/Правила обслуживания"),
    "sql_database": os.path.join(PROJECT_ROOT, "test_data/Книги/library.db")
}

# ID сотрудников для доступа к внутренней информации
EMPLOYEE_IDS = [81900000, 210000, 9870098]  # Примерные ID сотрудников, нужно заменить на реальные

# Настройки RAG
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "similarity_top_k": 3
}

# Настройки памяти
MEMORY_CONFIG = {
    "k": 5,  # Количество пар сообщений для хранения в контексте
    "history_dir": os.path.join(CHROMA_PERSIST_DIRECTORY, "history")
}

# Проверка обязательных настроек
def validate_config():
    """Проверяет наличие обязательных настроек"""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY не установлен")
    
    if not API_BASE_URL:
        errors.append("API_BASE_URL не установлен")
    
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN не установлен")
    
    if errors:
        raise ValueError(f"Ошибки конфигурации: {', '.join(errors)}")
    
    logger.info("✅ Конфигурация загружена успешно")
    logger.info(f"📚 Модель LLM: {LLM_MODEL_NAME}")
    logger.info(f"🔗 API Base: {API_BASE_URL}")
    logger.info(f"👥 Сотрудников: {len(EMPLOYEE_IDS)}")
    
    return True

# Проверка доступа сотрудника
def is_employee(user_id: int) -> bool:
    """Проверяет, является ли пользователь сотрудником"""
    return user_id in EMPLOYEE_IDS

if __name__ == "__main__":
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"❌ Ошибка конфигурации: {e}")