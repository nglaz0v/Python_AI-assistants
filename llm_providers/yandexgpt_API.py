import os
import requests
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# Получаем API ключ и folder_id из .env
OPENAI_API_KEY = os.getenv("YANDEXGPT_API_KEY")
FOLDER_ID = os.getenv("FOLDER_ID_YANDEX")

if not OPENAI_API_KEY:
    raise ValueError("Ошибка: YANDEXGPT_API_KEY не найден в .env")
if not FOLDER_ID:
    raise ValueError("Ошибка: FOLDER_ID_YANDEX не найден в .env")

class YandexGPTAPIProvider:
    def _make_request(self, messages, temperature: float, max_tokens: int) -> str:
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"  # Задаем url

        # Формируем заголовок запроса
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "x-folder-id": FOLDER_ID
        }

        # Преобразуем сообщения в формат YandexGPT (text вместо content)
        yandex_messages = []
        for message in messages:
            yandex_messages.append({
                "role": message["role"],
                "text": message["content"]
            })

        # Формируем тело запроса
        payload = {
            "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": max_tokens,
                "reasoningOptions": {
                    "mode": "DISABLED"
                }
            },
            "messages": yandex_messages
        }
        
        # Направляем запрос в LLM
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "Нет ответа от API.")
    
    def send_request(self, messages, temperature, max_tokens) -> str:
        try:
            return self._make_request(messages, temperature, max_tokens)
        except Exception as e:
            raise Exception(f"Ошибка при запросе к YandexGPT: {str(e)}")
