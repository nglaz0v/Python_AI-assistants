import logging
from llama_index.core.callbacks import CallbackManager, CBEventType
from typing import Any, Dict, List, Optional
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

# Настройка логирования
def setup_logging():
    """Настройка базового логирования"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Отключение логов от сторонних библиотек
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Пользовательский обработчик для логирования этапов и шагов работы LlamaIndex
class DetailedLogger(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[]
        )
        # Флаг для отслеживания, что мы в процессе индексации
        self.indexing_started = False
        # Словари для хранения информации о документах
        self.doc_chunks_count = {}
        self.doc_embeddings_count = {}
        # Список для отслеживания уже выведенных документов
        self.processed_docs = set()
        # Флаги для отслеживания вывода шагов
        self.chunking_step_shown = False
        self.embedding_step_shown = False
        self.llm_step_shown = False
        # Счетчик документов
        self.doc_counter = 0
        # Флаг для отслеживания первого запроса к LLM в текущем этапе
        self.first_llm_request = True
        # Счетчик для уточняющих запросов
        self.refine_query_count = 0
        # Флаг для отслеживания, что мы в этапе индексации
        self.in_indexing_phase = True
        # Количество загруженных документов
        self.loaded_docs_count = None
    
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        logger = logging.getLogger(__name__)
        
        # Отладочная информация: выводим все события
        logger.debug(f"Event start: {event_type}")
        
        if event_type == CBEventType.NODE_PARSING and not self.indexing_started:
            logger.info("=" * 60)
            logger.info("Этап подготовки. Индексация данных.")
            logger.info("=" * 60)
            logger.info("  Шаг 1. Загрузка документов")
            if self.loaded_docs_count:
                logger.info(f"    Загружено {self.loaded_docs_count} документов")
            self.indexing_started = True
            # Сброс флагов для следующей индексации
            self.chunking_step_shown = False
            self.embedding_step_shown = False
            self.llm_step_shown = False
            self.doc_chunks_count = {}
            self.doc_embeddings_count = {}
            self.processed_docs = set()
            self.doc_counter = 0
            self.first_llm_request = True
            self.refine_query_count = 0
            self.in_indexing_phase = True
        
        if event_type == CBEventType.CHUNKING and not self.chunking_step_shown:
            logger.info("  Шаг 2. Разбивка документов - на чанки")
            self.chunking_step_shown = True
        elif event_type == CBEventType.EMBEDDING and not self.embedding_step_shown:
            logger.info("  Шаг 3. Генерация эмбеддингов")
            self.embedding_step_shown = True
            self.in_indexing_phase = False  # Завершаем этап индексации
        elif event_type == CBEventType.RETRIEVE:
            logger.info("=" * 60)
            logger.info("Этап 1. Поиск релевантных данных (Retrieval)")
            logger.info("=" * 60)
            if payload:
                query_str = payload.get("query_str", "")
                if query_str:
                    logger.info("  Шаг 1. Поиск данных")
                    logger.info(f"    Поиск фрагментов по запросу: {query_str}")
        elif event_type == CBEventType.LLM:
            if not self.llm_step_shown:
                logger.info("=" * 60)
                logger.info("Этап 2. Генерация ответа")
                logger.info("=" * 60)
                self.llm_step_shown = True
                # Сбрасываем флаг первого запроса к LLM для нового этапа
                self.first_llm_request = True
                # Сбрасываем счетчик уточняющих запросов
                self.refine_query_count = 0
            
            if payload:
                messages = payload.get("messages", [])
                
                # Определяем тип запроса и формируем соответствующее сообщение
                if self.first_llm_request:
                    logger.info(f"    Запрос к LLM")
                    self.first_llm_request = False
                else:
                    # Для уточняющих запросов показываем номер
                    self.refine_query_count += 1
                    logger.info("-" * 60)
                    logger.info(f"    Уточняющий запрос к LLM № {self.refine_query_count}:")
                
                # Для всех сообщений выводим их в формате "Роль ...: контекст..."
                logger.info(f"    Всего сообщений в запросе: {len(messages)}")
                for i, msg in enumerate(messages):
                    # Проверяем, есть ли у сообщения атрибут role и content
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        role = msg.role
                        content = msg.content
                        
                        # Выводим сообщение в формате "Роль ...: контекст..."
                        # Ограничиваем длину контекста для читаемости
                        if len(content) > 200:
                            content_preview = content[:200] + "..."
                        else:
                            content_preview = content
                            
                        logger.info(f"      Роль {role}: {content_preview}")
                    elif hasattr(msg, 'content'):
                        # Если role нет, но есть content, выводим как есть
                        content = msg.content
                        
                        # Выводим сообщение с пометкой "Сообщение"
                        if len(content) > 200:
                            content_preview = content[:200] + "..."
                        else:
                            content_preview = content
                            
                        logger.info(f"      Сообщение: {content_preview}")
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        logger = logging.getLogger(__name__)
        
        if event_type == CBEventType.CHUNKING:
            if payload and self.in_indexing_phase:
                chunks = payload.get("chunks", [])
                # Выводим информацию о количестве чанков с номером документа
                if chunks and len(chunks) > 0:
                    count = len(chunks)
                    self.doc_counter += 1
                    logger.info(f"    Документ {self.doc_counter} разбит на {count} чанка")
        elif event_type == CBEventType.EMBEDDING:
            if payload:
                chunks = payload.get("chunks", [])
                # Получаем информацию о документе
                if chunks and len(chunks) > 0:
                    first_chunk = chunks[0]
                    if hasattr(first_chunk, 'node') and hasattr(first_chunk.node, 'metadata'):
                        file_name = first_chunk.node.metadata.get('file_name', 'Unknown')
                        self.doc_embeddings_count[file_name] = self.doc_embeddings_count.get(file_name, 0) + len(chunks)
        elif event_type == CBEventType.RETRIEVE and payload:
            nodes = payload.get("nodes", [])
            logger.info(f"    Найдено фрагментов: {len(nodes)}")
            # Показываем все фрагменты, которые будут отправлены в LLM
            for i, node in enumerate(nodes):  # Показываем все фрагменты
                score = getattr(node, 'score', None)
                if score is not None:
                    logger.info(f"    Фрагмент {i+1} (релевантность: {score:.4f}): {node.node.get_content()[:100]}...")
                else:
                    logger.info(f"    Фрагмент {i+1}: {node.node.get_content()[:100]}...")
        elif event_type == CBEventType.LLM and payload:
            response = payload.get("response", "")
            response_str = str(response)
            
            # Извлекаем основную часть ответа (без "assistant: " и лишних пробелов)
            if response_str.startswith("assistant: "):
                response_str = response_str[10:].strip()
            
            # Логируем ответ в формате "Роль ASSISTANT: контекст..."
            # Показываем полное содержимое ответа
            logger.info(f"      Роль ASSISTANT: {response_str}")
    
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass
    
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass
    
    def set_loaded_docs_count(self, count: int) -> None:
        """Установка количества загруженных документов"""
        self.loaded_docs_count = count

def setup_callback_manager():
    """Настройка менеджера колбэков с кастомным логгером"""
    # Убираем LlamaDebugHandler чтобы избежать избыточного логирования
    custom_logger = DetailedLogger()
    callback_manager = CallbackManager([custom_logger])
    return callback_manager, custom_logger