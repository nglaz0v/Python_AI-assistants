"""
Цепочка для информации по правилам обслуживания
"""

import logging
from vector_stores.base import books_service_vector_store
from typing import Optional
from .base_chain import BaseRAGChain

logger = logging.getLogger(__name__)


class ServiceChain(BaseRAGChain):
    """Цепочка для работы с правилами обслуживания"""
    
    def __init__(self):
        super().__init__(prompt_file="service_chain.txt", temperature=0.1)
        
        # Создаем промт
        self.prompt = self._create_prompt(input_variables=["context", "question", "chat_history", "current_time"])
        
        # Получаем ретривер
        self.retriever = books_service_vector_store.get_retriever()
        
        # Создаем стандартную RAG-цепочку
        self.chain = self._create_rag_chain(self.retriever, prefix="Service ")
    
    def reload(self, vector_store=None):
        """Перезагружает ретривер и цепочку"""
        super().reload()
        if vector_store:
            self.retriever = vector_store.get_retriever()
            self.chain = self._create_rag_chain(self.retriever, prefix="Service ")
            logger.info("✅ Ретривер и цепочка обслуживания обновлены")

    def get_chain_info(self) -> dict:
        """Возвращает информацию о цепочке"""
        return {
            "name": "Цепочка обслуживания",
            "description": "Предоставляет информацию о правилах покупки, доставки и возврата товаров",
            "restricted": False,
            "vector_store": "books_service"
        }


# Создаем глобальный экземпляр для использования
service_chain = ServiceChain()
