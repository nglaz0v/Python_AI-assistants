"""
Инструменты для ИИ агента, оборачивающие существующие цепочки.
"""

import logging
from typing import Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from chains.books_chain import books_chain
from chains.service_chain import service_chain
from chains.organization_chain import organization_chain
from config import is_employee
from memory.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class SearchInput(BaseModel):
    query: str = Field(description="Поисковый запрос или вопрос пользователя")

class BooksSearchTool(BaseTool):
    name: str = "search_books"
    description: str = "Поиск информации о книгах: наличие, цены, авторы, описания. Используй для любых вопросов о книгах."
    args_schema: Type[BaseModel] = SearchInput
    user_id: int = 0

    def _run(self, query: str) -> str:
        logger.info(f"🛠️ Вызов инструмента search_books с запросом: {query} (User ID: {self.user_id})")
        chat_history = memory_manager.get_history_str(self.user_id)
        return books_chain.process_query(query, chat_history=chat_history)

class ServiceRulesSearchTool(BaseTool):
    name: str = "search_service_rules"
    description: str = "Поиск информации о правилах обслуживания: доставка, оплата, возврат, условия покупки. Используй для вопросов о работе магазина."
    args_schema: Type[BaseModel] = SearchInput
    user_id: int = 0

    def _run(self, query: str) -> str:
        logger.info(f"🛠️ Вызов инструмента search_service_rules с запросом: {query} (User ID: {self.user_id})")
        chat_history = memory_manager.get_history_str(self.user_id)
        return service_chain.process_query(query, chat_history=chat_history)

class OrganizationDocsSearchTool(BaseTool):
    name: str = "search_organization_docs"
    description: str = "Поиск во внутренних документах организации: должностные инструкции, положения, регламенты. ТОЛЬКО ДЛЯ СОТРУДНИКОВ."
    args_schema: Type[BaseModel] = SearchInput
    
    # Мы будем передавать user_id при инициализации или через контекст
    user_id: int = 0

    def _run(self, query: str) -> str:
        logger.info(f"🛠️ Вызов инструмента search_organization_docs с запросом: {query} (User ID: {self.user_id})")
        
        if not is_employee(self.user_id):
            logger.warning(f"🚫 Доступ к инструменту search_organization_docs запрещен для пользователя {self.user_id}")
            return "ОШИБКА: У вас нет прав доступа к внутренним документам организации. Эта информация доступна только сотрудникам."
            
        chat_history = memory_manager.get_history_str(self.user_id)
        return organization_chain.process_query(query, chat_history=chat_history)

def get_tools(user_id: int):
    """Возвращает список инструментов, настроенных для конкретного пользователя"""
    return [
        BooksSearchTool(user_id=user_id),
        ServiceRulesSearchTool(user_id=user_id),
        OrganizationDocsSearchTool(user_id=user_id)
    ]
