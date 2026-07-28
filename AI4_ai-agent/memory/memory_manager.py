"""
Менеджер памяти для управления историей диалогов пользователей
"""

import os
import json
import re
from datetime import datetime
from typing import Dict
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import BaseMessage, messages_to_dict

from pathlib import Path
from config import MEMORY_CONFIG

class ReadableFileChatMessageHistory(FileChatMessageHistory):
    """
    Для сохранения кирилицы в читаемом виде делаем расширенную версию
    FileChatMessageHistory, которая сохраняет текст 
    в читаемом виде (без экранирования Unicode).
    """
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        # Сначала убеждаемся, что директория существует, 
        f_path = Path(file_path).absolute()
        f_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем базовый класс
        super().__init__(str(f_path), encoding=encoding)
        
        # Переопределяем self.file_path как объект Path
        self.file_path = f_path
        
        # Создаем пустой JSON-массив, если файл пуст или не существует
        if not self.file_path.exists() or self.file_path.stat().st_size == 0:
            with open(self.file_path, "w", encoding=self.encoding) as f:
                f.write("[]")
            print(f"📝 Создан новый файл истории: {self.file_path}")

    def add_message(self, message: BaseMessage) -> None:
        # Сначала добавляем в память (базовый класс сам сохранит в ASCII)
        super().add_message(message)
        # Затем перезаписываем в читаемом виде
        self._save_readable()

    def _save_readable(self) -> None:
        """Сохраняет сообщения в файл и добавляет метку времени"""
        try:
            messages = messages_to_dict(self.messages)
            
            # Добавляем метку времени в каждое сообщение только для записи в файл
            # Проверяем, нет ли уже метки времени в начале сообщения
            timestamp_pattern = r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]"
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for msg in messages:
                if 'data' in msg and 'content' in msg['data']:
                    content = msg['data']['content']
                    # Если метки еще нет, добавляем её
                    if not re.match(timestamp_pattern, content):
                        msg['data']['content'] = f"[{current_timestamp}] {content}"

            with open(self.file_path, "w", encoding=self.encoding) as f:
                json.dump(messages, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"⚠️ Ошибка при сохранении истории: {e}")

class MemoryManager:
    """Класс для управления памятью пользователей с сохранением в JSON файлы"""
    
    def __init__(self, k: int = None, history_dir: str = None):
        """
        Инициализация менеджера памяти
        """
        self.memories: Dict[int, ConversationBufferWindowMemory] = {}
        self.k = k if k is not None else MEMORY_CONFIG["k"]
        self.history_dir = history_dir if history_dir is not None else MEMORY_CONFIG["history_dir"]
        
    def _get_history_path(self, user_id: int) -> str:
        """Возвращает путь к файлу истории пользователя"""
        return os.path.join(self.history_dir, f"user_{user_id}.json")

    def get_memory(self, user_id: int) -> ConversationBufferWindowMemory:
        """
        Получает или создает память для пользователя с привязкой к файлу
        """
        if user_id not in self.memories:
            # Путь к файлу истории конкретного пользователя
            history_file = self._get_history_path(user_id)
            
            # Используем ReadableFileChatMessageHistory для читаемости
            chat_history = ReadableFileChatMessageHistory(history_file)
            
            self.memories[user_id] = ConversationBufferWindowMemory(
                k=self.k,
                memory_key="chat_history",
                chat_memory=chat_history,
                return_messages=False  # Возвращаем строку для промпта
            )
        return self.memories[user_id]
    

    def get_history_str(self, user_id: int) -> str:
        """
        Получает историю диалога в виде строки
        """
        memory = self.get_memory(user_id)
        return memory.load_memory_variables({})['chat_history']

    def add_interaction(self, user_id: int, input_text: str, output_text: str):
        """
        Добавляет сохранение в память
        """
        memory = self.get_memory(user_id)
        memory.save_context({"input": input_text}, {"output": output_text})

# Глобальный экземпляр
memory_manager = MemoryManager()
