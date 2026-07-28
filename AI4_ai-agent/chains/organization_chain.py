"""
Цепочка для информации сотрудников об организации
"""

import logging
from vector_stores.base import organization_vector_store
from typing import Optional
from .base_chain import BaseRAGChain

logger = logging.getLogger(__name__)


class OrganizationChain(BaseRAGChain):
    """Цепочка для работы с внутренними документами организации"""
    
    def __init__(self):
        super().__init__(prompt_file="organization_chain.txt", temperature=0.1)
        
        # Создаем промт
        self.prompt = self._create_prompt(input_variables=["context", "question", "chat_history", "current_time"])
        
        # Получаем ретривер
        self.retriever = organization_vector_store.get_retriever()
        
        # Создаем стандартную RAG-цепочку
        self.chain = self._create_rag_chain(self.retriever, prefix="Org ")
    
    def reload(self, vector_store=None):
        """Перезагружает ретривер и цепочку"""
        super().reload()
        if vector_store:
            self.retriever = vector_store.get_retriever()
            self.chain = self._create_rag_chain(self.retriever, prefix="Org ")
            logger.info("✅ Ретривер и цепочка организации обновлены")

    def get_chain_info(self) -> dict:
        """Возвращает информацию о цепочке"""
        return {
            "name": "Цепочка организации",
            "description": "Предоставляет информацию из внутренних документов компании для сотрудников",
            "restricted": True,
            "vector_store": "organization_docs"
        }


# Глобальный экземпляр
organization_chain = OrganizationChain()
