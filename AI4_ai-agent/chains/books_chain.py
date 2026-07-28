"""
Цепочка для информации по книгам через SQL
"""

import sqlite3
import logging
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from config import DATA_PATHS
from .base_chain import BaseRAGChain

logger = logging.getLogger(__name__)


class BooksChain(BaseRAGChain):
    """Цепочка для работы с информацией о книгах через SQL"""
    
    def __init__(self):
        super().__init__(prompt_file="books_chain.txt", temperature=0.1)
        
        # 1. Загружаем схему БД (чтобы LLM не галлюцинировала названия таблиц)
        self.db_schema = self._get_schema()
        
        # 2. Загружаем шаблон для генерации SQL
        self.sql_prompt_template = self._load_prompt("sql_generation_prompt.txt")
        
        # 3. Создаем основной промпт для финального ответа
        self.prompt = self._create_prompt(
            input_variables=["sql_context", "question", "chat_history", "current_time"]
        )
        
        # 4. Создаем цепочку
        self.chain = self._create_generation_chain(self._prepare_inputs)
    
    def _get_schema(self) -> str:
        """Получает DDL схемы базы данных для контекста LLM."""
        try:
            db_path = DATA_PATHS["sql_database"]
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Получаем CREATE TABLE стейтменты
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
            rows = cursor.fetchall()
            conn.close()
            
            schema = "\n".join([row[0] for row in rows if row[0]])
            logger.info("Схема БД успешно загружена")
            return schema
        except Exception as e:
            logger.error(f"Ошибка получения схемы БД: {e}")
            return "Схема недоступна."

    def _prepare_inputs(self, inputs: dict) -> dict:
        """
        Оркестратор: 
        1. Генерирует SQL на основе вопроса и схемы.
        2. Выполняет SQL.
        3. Собирает всё в словарь для финального промпта.
        """
        question = inputs["question"]
        chat_history = inputs.get("chat_history", "")
        
        logger.info("📊 [BooksChain] Генерация и выполнение SQL...")
        
        # 1. Генерируем SQL
        sql_query = self._generate_sql_query(question, chat_history)
        
        # 2. Выполняем SQL и получаем текст результатов
        sql_results = self._get_sql_context(sql_query)
        
        # 3. Возвращаем данные для self.prompt
        return {
            "sql_context": sql_results,
            "question": question,
            "chat_history": chat_history,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def _generate_sql_query(self, question: str, chat_history: str) -> str:
        """Генерирует SQL-запрос с помощью LLM."""
        
        # Создаем локальный промпт для SQL
        # ВАЖНО: В файле sql_generation_prompt.txt должна быть переменная {schema}
        sql_prompt = PromptTemplate(
            input_variables=["question", "chat_history", "schema"], 
            template=self.sql_prompt_template
        )
        
        # Цепочка для генерации SQL
        sql_chain = (
            sql_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        try:
            raw_query = sql_chain.invoke({
                "question": question, 
                "chat_history": chat_history,
                "schema": self.db_schema
            }).strip()
            
            # Очистка от markdown (```sql ... ```)
            cleaned_query = raw_query.replace("```sql", "").replace("```", "").strip()
            
            logger.info(f"Сгенерированный SQL: {cleaned_query}")
            return cleaned_query
        except Exception as e:
            logger.error(f"Ошибка генерации SQL: {e}")
            return ""

    def _get_sql_context(self, sql_query: str) -> str:
        """Выполняет SQL и форматирует результат."""
        if not sql_query:
            return "Не удалось сформировать запрос к базе данных."
            
        try:
            # Простая защита
            if not sql_query.upper().startswith("SELECT"):
                return "Ошибка безопасности: разрешены только SELECT-запросы."

            db_path = DATA_PATHS["sql_database"]
            # Используем контекстный менеджер для надежного закрытия
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sql_query)
                rows = cursor.fetchall()

            if not rows:
                return "По вашему запросу в базе данных ничего не найдено."

            # Форматируем результаты
            results = []
            for row in rows[:15]: # Ограничиваем выдачу (защита от переполнения контекста)
                items = [f"{key}: {value}" for key, value in dict(row).items()]
                results.append("- " + ", ".join(items))
            
            return "\n".join(results)
            
        except sqlite3.Error as e:
            logger.error(f"Ошибка выполнения SQL: {e}")
            return "Ошибка базы данных при выполнении запроса."
        except Exception as e:
            logger.error(f"Общая ошибка SQL: {e}")
            return "Произошла ошибка при получении данных."

    def get_chain_info(self) -> dict:
        """Возвращает метаданные цепочки."""
        return {
            "name": "Цепочка книг (SQL)",
            "description": "Поиск информации о наличии, ценах и авторах книг в базе данных",
            "restricted": False,
            "vector_store": None # У этой цепочки нет векторного хранилища
        }


# Глобальный экземпляр
books_chain = BooksChain()