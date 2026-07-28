import os
from dotenv import load_dotenv
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.prompts import PromptTemplate

# Импортируем модули из папки utils
from utils.logger import setup_logging, setup_callback_manager

# Настройка логирования
logger = setup_logging()

# Настройка менеджера колбэков для логирования
callback_manager, custom_logger = setup_callback_manager()
Settings.callback_manager = callback_manager

# Загружаем переменные окружения из .env файла
load_dotenv()

def get_api_key():
    return os.getenv("OPENAI_API_KEY")

def get_api_base_url():
    return os.getenv("API_BASE_URL")

def get_llm_model_name():
    return os.getenv("LLM_MODEL_NAME")

# ============================================================
# Этап подготовки. Индексация данных.
# ============================================================

# Шаг 1. Загрузка документов
reader = SimpleDirectoryReader(
    input_dir="test_data/Организация/ЛНА",  # Путь к папке с документами
    recursive=True,         # Рекурсивное чтение вложенных папок
    required_exts=[".txt", ".pdf", ".docx", ".md"],  # Ограничение типов файлов
)
documents = reader.load_data()

# Устанавливаем количество документов для логера
custom_logger.set_loaded_docs_count(len(documents))

# Шаг 2. Разбивка документов - на чанки
text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)
nodes = text_splitter.get_nodes_from_documents(documents)

# ============================================================
# Настройка LLM для извлечения сущностей и связей!
# ============================================================
llm = OpenAILike(
    api_base=get_api_base_url(),
    model=get_llm_model_name(),
    is_chat_model=True,
    api_key=get_api_key(),
    temperature=0.2,
)
Settings.llm = llm  # Задаем LLM по умолчанию
Settings.chunk_size = 512

# Шаг 3. Создание Knowledge Graph Index
# LLM используется для извлечения триплетов (субъект, предикат, объект)
index = KnowledgeGraphIndex(
    nodes=nodes,
    max_triplets_per_chunk=2,
    show_progress=True,
    include_embeddings=False,
)

# ============================================================
# Визуализация связей графа в консоли
# ============================================================
logger.info("=" * 60)
logger.info("СВЯЗИ В ГРАФЕ ЗНАНИЙ (Knowledge Graph):")
logger.info("=" * 60)

# Получаем граф через NetworkX представление
graph = index.get_networkx_graph()

if graph.number_of_edges() > 0:
    for edge in graph.edges(data=True):
        subject = edge[0]
        obj = edge[1]
        # В LlamaIndex предикат обычно хранится в атрибуте 'title' или 'label'
        relation = edge[2].get('title', 'связан с')
        logger.info(f"[{subject}] --({relation})--> [{obj}]")
else:
    logger.warning("Граф пуст. Возможно, LLM не смогла извлечь триплеты.")

logger.info(f"Всего узлов: {graph.number_of_nodes()}")
logger.info(f"Всего связей: {graph.number_of_edges()}")
logger.info("=" * 60)

# ============================================================
# Этап 1. Поиск релевантных данных (Retrieval)
# ============================================================

# Шаблон для начального запроса
qa_prompt = PromptTemplate("""
Ты - профессиональный ассистент по анализу документов компании «Книжный Мир». Твоя задача - строго и точно извлекать информацию из предоставленного контекста.
Используй знания из графа связей.

Ниже приведена контекстная информация.
---------------------
{context_str}
---------------------

Правила ответа:
1. Используй ТОЛЬКО информацию из контекста выше.
2. НЕ ДОБАВЛЯЙ никакой информации, которой нет в контексте.
3. НЕ ИСПОЛЬЗУЙ свои предварительные знания по теме вопроса.
4. Извлекай только ту информацию, которая прямо отвечает на вопрос.
5. Если в контексте нет информации для ответа на вопрос, честно скажи об этом.
6. Отвечай кратко и по существу, в виде структурированного списка.

Вопрос: {query_str}
""")

# Шаг 1. Поиск данных
query_engine = index.as_query_engine(
    include_text=True, # Включаем текст чанков для полноты ответа
    response_mode='simple_summarize',
    text_qa_template=qa_prompt,
)

# ============================================================
# Этап 2. Генерация ответа
# ============================================================

# Запрос к LLM
while True:
    query_text = input("\nЗадайте свой вопрос по графу (0 - выход): ").strip()

    # Условие выхода
    if query_text == "0":
        logger.info("Выход из программы.")
        break

    response = query_engine.query(query_text)

    # Вывод ответа
    logger.info("=" * 60)
    logger.info("ОТВЕТ НА ЗАПРОС:")
    logger.info("=" * 60)
    logger.info(response)
    logger.info("=" * 60)
