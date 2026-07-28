import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
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

# Получаем переменные из окружения
def get_api_key():
    return os.getenv("OPENAI_API_KEY")

def get_api_base_url():
    return os.getenv("API_BASE_URL")

def get_embedding_model_name():
    return os.getenv("EMBEDDING_MODEL_NAME")

def get_llm_model_name():
    return os.getenv("LLM_MODEL_NAME")

# ============================================================
# Этап настройки llamaindex.
# ============================================================

# Настройка LLM
llm = OpenAILike(
    api_base=get_api_base_url(),
    model=get_llm_model_name(),
    is_chat_model=True,  # Режим работы с чатом
    api_key=get_api_key(), # Получаем API-ключ из .env
    temperature=0.2,  # Температура генерации
)

# Настройка модели эмбеддинга
embed_model = OpenAIEmbedding(
    api_base=get_api_base_url(),
    model_name=get_embedding_model_name(),
    api_key=get_api_key(),
)

# Создание шаблонов запросов
# Шаблон для начального запроса
qa_prompt = PromptTemplate("""
Ты - профессиональный ассистент по анализу документов компании «Книжный Мир». Твоя задача - строго и точно извлекать информацию из предоставленного контекста.

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

# Шаблон для уточняющих запросов
refine_prompt = PromptTemplate("""
Ты - профессиональный ассистент по анализу документов компании «Книжный Мир». Твоя задача - улучшить ответ, используя только новый контекст.

Исходный вопрос: {query_str}
Предыдущий ответ: {existing_answer}

Ниже приведен дополнительный контекст:
------------
{context_msg}
------------

Правила уточнения:
1. Используй ТОЛЬКО информацию из нового контекста выше.
2. Добавь в ответ только ту информацию, которая есть в новом контексте и относится к вопросу.
3. НЕ ДОБАВЛЯЙ никакой другой информации, даже если она кажется логичной.
4. НЕ ИСПОЛЬЗУЙ свои предварительные знания по теме вопроса.
5. НЕ ПОВТОРЯЙ информацию, которая уже есть в предыдущем ответе.
6. Если новый контекст не содержит полезной информации по вопросу, просто повтори предыдущий ответ без изменений.
7. Предоставь только окончательный ответ без комментариев о процессе уточнения.

Уточненный ответ:
""")

# # ============================================================
# Этап подготовки. Индексация данных.
# ============================================================

# Шаг 1. Загрузка документов
reader = SimpleDirectoryReader(
    input_dir="test_data", # Путь к папке с документами
    recursive=True,  # Рекурсивное чтение вложенных папок
    required_exts=[".txt", ".pdf", ".docx", ".md"],
)
documents = reader.load_data()

# Устанавливаем количество документов для логера
custom_logger.set_loaded_docs_count(len(documents))

# Шаг 2. Разбивка документов - на чанки
# Настройка параметров разбиения текста на чанки
text_splitter = SentenceSplitter(
    chunk_size=512,  # Количество токенов в чанке
    chunk_overlap=50  # Перекрытие в токенах
)

# Шаг 3. Генерация эмбеддингов
Settings.embed_model = embed_model

# Создаем Chroma клиент
chroma_client = chromadb.PersistentClient(path="llamaindex/chroma_db")

# Создаем коллекцию Chroma
chroma_collection = chroma_client.get_or_create_collection("bookstore_collection")

# Создаем хранилище векторов Chroma
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Создаем контекст хранения
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Проверяем, есть ли данные в коллекции Chroma
collection_count = chroma_collection.count()
if collection_count > 0:
    # Загружаем индекс из векторного хранилища
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    logger.info("Индекс успешно загружен из векторной БД.")
else:
    # Создаем индекс заново
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        text_splitter=text_splitter
    )
    logger.info("Индекс успешно создан и сохранен в векторную БД.")


# ============================================================
# Этап 1. Поиск релевантных данных (Retrieval)
# ============================================================

# Шаг 1. Поиск данных
# Настройка RAG
query_engine = index.as_query_engine(
    llm=llm,
    text_qa_template=qa_prompt,
    refine_template=refine_prompt,
    similarity_top_k=3,  # Количество фрагментов для поиска
    response_mode='simple_summarize'  # Режим работы llamaindex
)

# ============================================================
# Этап 2. Генерация ответа
# ============================================================

# Запрос к LLM
while True:
    query_text = input("\nЗадайте свой вопрос (0 - выход): ").strip()

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