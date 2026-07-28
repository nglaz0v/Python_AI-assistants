import os
import requests
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# Получаем API ключ из .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Ошибка: OPENAI_API_KEY не найден в .env")

# Получаем URL API из .env
API_BASE_URL = os.getenv("API_BASE_URL")
if not API_BASE_URL:
    raise ValueError("Ошибка: API_BASE_URL не найден в .env")

# Получаем имя модели из .env
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
if not LLM_MODEL_NAME:
    raise ValueError("Ошибка: LLM_MODEL_NAME не найден в .env")

class OpenAIAPIProvider:
    def _make_request(self, messages, temperature: float, max_tokens: int) -> str:
        url = f"{API_BASE_URL}/chat/completions"   # Задаем url
        
        # Формируем заголовок запроса
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # Формируем тело запроса
        payload = {
            "model": LLM_MODEL_NAME,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Направляем запрос в LLM
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() 
        
        result = response.json() 
        return result.get("choices", [{}])[0].get("message", {}).get("content", "Нет ответа от API.")
    
    def send_request(self, messages, temperature, max_tokens) -> str:
        try:
            return self._make_request(messages, temperature, max_tokens)
        except Exception as e:
            raise Exception(f"Ошибка при запросе к ChatGPT: {str(e)}")
