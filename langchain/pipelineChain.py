import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)

# Создаем шаблон промпта
prompt = PromptTemplate.from_template("{query}")

# Создаем цепочку
chain = (
    {"query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query_text = input("\nЗадайте свой вопрос (0 - выход): ").strip()

    # Условие выхода
    if query_text == "0":
        print("Выход из программы.")
        break

    print("Направлен запрос, ожидайте ответа")
    response = chain.invoke(query_text)
    
    print("=" * 60)
    print("ОТВЕТ НА ЗАПРОС:")
    print("=" * 60)

    print(response)
    print("=" * 60)
