import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel


# Загружаем переменные окружения из .env файла
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)
 
# Цепочка 1
summary_chain = (
    ChatPromptTemplate.from_template("Кратко резюмируй: {input}") 
    | llm
)

# Цепочка 2
facts_chain = (
    ChatPromptTemplate.from_template("Перечисли факты списком: {input}") 
    | llm
)

# Цепочка 3
keywords_chain = (
    ChatPromptTemplate.from_template("Извлеки 5 ключевых слов: {input}") |
    llm
)
 
# Цепочка из 3-х паралельных
parallel_chain = RunnableParallel(
    summary=summary_chain,
    facts=facts_chain,
    keywords=keywords_chain
)
 
while True:
    
    query_text = input("\nЗадайте свой вопрос (0 - выход): ").strip()

    # Условие выхода
    if query_text == "0":
        print("Выход из программы.")
        break

    print("Направлен запрос, ожидайте ответа")
    response = parallel_chain.invoke(query_text)

    print("=" * 60)
    print("ОТВЕТ НА ЗАПРОС:")
    print("=" * 60)
        
    print(f"Цепочка 1. РЕЗЮМЕ:\n{response['summary'].content}")
    print("-" * 30)
    print(f"Цепочка 2. ФАКТЫ:\n{response['facts'].content}")
    print("-" * 30)
    print(f"Цепочка 3. КЛЮЧЕВЫЕ СЛОВА:\n{response['keywords'].content}")
    print("=" * 60)