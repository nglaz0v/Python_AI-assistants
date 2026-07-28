import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings


# Загружаем переменные окружения
load_dotenv()

# Загрузка документов с рекурсивным поиском
print("Поиск документов в папке test_data...")
loader = DirectoryLoader(
    "test_data",
    glob="**/*.md",  # Ищем только .md файлы
    recursive=True,
    show_progress=True,
    loader_cls=TextLoader 
)
documents = loader.load()

    
# Вывод информации о загруженных документах
print(f"Загружено документов: {len(documents)}")
for i, doc in enumerate(documents[:3]):  # Показываем первые 3 документа
    print(f"Документ {i+1}: {doc.metadata.get('source', 'Неизвестный источник')}")
if len(documents) > 3:
    print(f"... и ещё {len(documents) - 3} документов")


# Разбивка документов на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# Раскоментируйте необходимый вам вариант создания эмбеддингов
# Создание эмбеддингов через HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL_NAME"),
    model_kwargs={
        "device": "cpu",  # или "cuda" если есть GPU
        "trust_remote_code": True  
    },
    encode_kwargs={
        "batch_size": 8,
        "normalize_embeddings": True 
    }
)

# Создание эмбеддингов через OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#     base_url=os.getenv("API_BASE_URL"),
#     model=os.getenv("EMBEDDING_MODEL_NAME"),
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# Вывод информации о чанках
print(f"Создано чанков: {len(splits)}")

# Создание векторного хранилища
print("\n=== СОЗДАНИЕ ВЕКТОРНОГО ХРАНИЛИЩА ===")
vectorstore = FAISS.from_documents(splits, embeddings)
print("Векторное хранилище создано успешно")

# Создание retriever
print("\n=== СОЗДАНИЕ RETRIEVER ===")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Retriever создан успешно")

# Настройка LLM
print("\n=== НАСТРОЙКА LLM ===")
llm = ChatOpenAI(
    base_url=os.getenv("API_BASE_URL"),
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2
)
print("LLM настроена успешно")

# Создание шаблона запроса
qa_prompt = PromptTemplate.from_template("""Ты - профессиональный ассистент по анализу документов компании «Книжный Мир». Твоя задача - строго и точно извлекать информацию из предоставленного контекста.

Контекст:
{context}

Вопрос: {question}

Правила ответа:
1. Используй ТОЛЬКО информацию из контекста выше.
2. НЕ ДОБАВЛЯЙ никакой информации, которой нет в контексте.
3. НЕ ИСПОЛЬЗУЙ свои предварительные знания по теме вопроса.
4. Извлекай только ту информацию, которая прямо отвечает на вопрос.
5. Если в контексте нет информации для ответа на вопрос, честно скажи об этом.
6. Отвечай кратко и по существу, в виде структурированного списка.

Ответ:""")

# Создание цепочки RAG (Chain)
# Цепочка объединяет несколько компонентов в последовательность операций:
# 1. Retriever - ищет релевантные документы в векторной базе
# 2. PromptTemplate - форматирует контекст и вопрос в шаблон
# 3. LLM - генерирует ответ на основе контекста и вопроса
# 4. StrOutputParser - преобразует ответ в строку
print("\n=== СОЗДАНИЕ ЦЕПОЧКИ (CHAIN) ===")
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | qa_prompt
    | llm
    | StrOutputParser()
)

# Интерактивный запрос к LLM
print("\n=== ИНТЕРАКТИВНЫЙ РЕЖИМ ЗАПРОСОВ ===")
print("Введите 0 для выхода")
while True:
    query_text = input("\nЗадайте свой вопрос (0 - выход): ").strip()

    # Условие выхода
    if query_text == "0":
        print("Выход из программы.")
        break

    print("=" * 60)
    print("ОТВЕТ НА ЗАПРОС:")
    print("=" * 60)
    
    # Отладочный вывод: что возвращает retriever 
    print("\n Поиск релевантных чанков для запроса:")
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(query_text, k=3)
    for i, (doc, score) in enumerate(retrieved_docs_with_scores):
        print(f" Чанк {i+1} (сходство: {score:.3f}): {doc.page_content[:200]}...")
        print(f" Источник: {doc.metadata.get('source', 'неизвестно')}")
        print()
    
    response = rag_chain.invoke(query_text)
    print(response)
    print("=" * 60)