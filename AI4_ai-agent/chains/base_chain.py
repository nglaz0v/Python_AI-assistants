"""
Базовый класс для всех цепочек RAG с общими компонентами.
"""
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Union, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.utils import convert_to_secret_str

from config import LLM_MODEL_NAME, OPENAI_API_KEY, API_BASE_URL, MAX_TOKENS

logger = logging.getLogger(__name__)


class BaseRAGChain:
    """
    Базовый класс для цепочек RAG.
    Обеспечивает единый интерфейс инициализации LLM, загрузки промптов и логирования.
    """
    
    def __init__(self, prompt_file: str, temperature: float = 0.1):
        """
        Инициализирует базовые компоненты цепочки.
        
        Args:
            prompt_file: Имя файла с промптом (относительно директории prompts)
            temperature: Температура для LLM
        """
        self.prompt_file = prompt_file
        self.temperature = temperature
        
        # 1. Загружаем текст шаблона из файла
        self.prompt_template = self._load_prompt()
        
        # 2. Создаем LLM
        self.llm = self._create_llm()
        
        # 3. Эти поля должны быть переопределены/созданы в дочерних классах
        self.prompt = None  # Объект PromptTemplate
        self.chain = None   # Итоговая Runnable цепочка
        
    def _load_prompt(self, prompt_file: Optional[str] = None) -> str:
        """Загружает текст промпта из файла."""
        file_to_load = prompt_file or self.prompt_file
        prompt_path = os.path.join(
            os.path.dirname(__file__), 
            '../prompts', 
            file_to_load
        )
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Файл промпта не найден: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        logger.info(f"Загружен промт из {prompt_path}")
        return prompt_template
    
    def _create_llm(self) -> ChatOpenAI:
        """Создает экземпляр LLM с настройками из конфига."""
        
        logger.info(f"Создание ChatOpenAI: URL={API_BASE_URL}, Model={LLM_MODEL_NAME}")
        
        return ChatOpenAI(
            base_url=API_BASE_URL,
            model=LLM_MODEL_NAME,
            api_key=convert_to_secret_str(OPENAI_API_KEY or "empty"),
            temperature=self.temperature,
            max_tokens=MAX_TOKENS
        )
    
    def _create_prompt(self, input_variables: List[str]) -> PromptTemplate:
        """Создает объект PromptTemplate с указанными переменными."""
        return PromptTemplate(
            input_variables=input_variables,
            template=self.prompt_template
        )


    def _create_generation_chain(self, input_runnable: Union[dict, Any]):
        """
        Создает финальную цепочку: [Подготовка данных] -> Промпт -> Лог -> LLM -> Парсер.
        """
        if self.prompt is None:
            raise ValueError("self.prompt не инициализирован. Вызовите self._create_prompt перед созданием цепочки.")

        return (
            input_runnable
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _create_rag_chain(self, retriever, prefix: str = ""):
        """
        Специализированный метод для ВЕКТОРНОГО поиска.
        Автоматически настраивает input_runnable для работы с ретривером.
        """
        # Определяем логику сбора данных для векторного RAG
        rag_inputs = {
            # Важно: ищем документы только по вопросу, игнорируя историю
            "context": (lambda x: x["question"]) 
            | retriever 
            | (lambda docs: self._format_documents(docs, prefix=prefix)),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", ""),
            "current_time": lambda x: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Передаем готовую логику сбора данных в универсальный генератор
        return self._create_generation_chain(rag_inputs)

    def _format_documents(self, docs: List[Document], prefix: str = "") -> str:
        """Форматирует найденные документы в строку для контекста."""
        logger.info(f"Получено {len(docs)} документов от {prefix}ретривера".strip())
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            # Логируем только начало документа, чтобы не засорять логи
            content_preview = doc.page_content[:200].replace('\n', ' ')
            logger.info(f"Docs[{i}] ({source}): {content_preview}...")
            
        formatted = "\n\n".join([doc.page_content for doc in docs])
        return formatted
    

    def process_query(self, question: str, chat_history: str = "") -> str:
        """
        Основной метод запуска цепочки.
        """
        if not question.strip():
            return "Пожалуйста, задайте ваш вопрос."
            
        if self.chain is None:
            logger.error(f"CRITICAL: Цепочка RAG не инициализирована в классе {self.__class__.__name__}")
            return "Внутренняя ошибка конфигурации бота (Chain not initialized)."

        try:
            logger.info(f"Обработка запроса: '{question}'")
            # Запускаем цепочку
            return self.chain.invoke({
                "question": question, 
                "chat_history": chat_history or "История отсутствует"
            })
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            return "Извините, произошла ошибка при обработке вашего запроса."
    
    def reload(self, vector_store=None):
        """
        Перезагружает компоненты цепочки.
        Должен быть реализован в дочерних классах, если требуется специфическая логика.
        """
        logger.info(f"🔄 Перезагрузка цепочки {self.__class__.__name__}")
        # По умолчанию просто перезагружаем промпт
        self.prompt_template = self._load_prompt()
        if self.prompt and hasattr(self.prompt, 'input_variables'):
            self.prompt = self._create_prompt(self.prompt.input_variables)

    def get_chain_info(self) -> Dict[str, Any]:
        """Возвращает метаданные цепочки (имя, описание, права доступа)."""
        raise NotImplementedError("Метод get_chain_info должен быть реализован в дочернем классе")