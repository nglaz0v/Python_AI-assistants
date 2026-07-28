"""
Модуль для работы с векторными базами данных Chroma
"""

from .base import VectorStoreManager, books_service_vector_store, organization_vector_store

__all__ = [
    'VectorStoreManager',
    'books_service_vector_store',
    'organization_vector_store'
]