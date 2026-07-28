import os
import time
import requests
import urllib3
import uuid
from dotenv import load_dotenv

# Подавление предупреждения о небезопасном HTTPS-соединении
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Загружаем переменные окружения из .env
load_dotenv()

# Получаем ключ авторизации из .env
AUTH_KEY = os.getenv("AUTHORIZATION_KEY_SBER")
if not AUTH_KEY:
    raise ValueError("Ошибка: AUTHORIZATION_KEY_SBER не найден в .env")

class GigaChatAPIProvider:
    # Инициализируем состояние токена доступа
    def __init__(self):
        self.access_token = None
        self.token_expires_at = 0

    def _get_access_token(self):
        # Получаем токен доступа от SberBank
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        
        # Генерируем UUID для запроса
        rquid = str(uuid.uuid4())
        
        # Заголовки для получения токена
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "RqUID": rquid,
            "Authorization": f"Basic {AUTH_KEY}"
        }
        
        # Тело запроса для получения токена
        data = 'scope=GIGACHAT_API_PERS'

        try:
            response = requests.post(url, headers=headers, data=data, verify=False)
            response.raise_for_status()
            token_data = response.json()
            
            # Возвращаем токен и время истечения
            expires_at = token_data["expires_at"] / 1000  # конвертируем в секунды
            self.token_expires_at = expires_at
            return token_data["access_token"]
        except requests.exceptions.RequestException as e:
            return f"Ошибка получения токена: {str(e)}"

    def _make_request(self, messages, temperature: float, max_tokens: int) -> str:
        # Обновляем токен при необходимости
        if not self.access_token or time.time() >= self.token_expires_at:
            token = self._get_access_token()
            if "Ошибка" in token:
                return token
            self.access_token = token

        # URL для отправки запроса к GigaChat
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        
        # Заголовки аутентификации
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        # Параметры запроса к GigaChat
        data = {
            "model": "GigaChat-2",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(url, headers=headers, json=data, verify=False)
            response.raise_for_status()
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "Нет ответа от API.")
        except requests.exceptions.RequestException as e:
            return f"Ошибка API: {str(e)}"
    
    def send_request(self, messages, temperature, max_tokens) -> str:
        try:
            return self._make_request(messages, temperature, max_tokens)
        except Exception as e:
            raise Exception(f"Ошибка при запросе к GigaChat: {str(e)}")
