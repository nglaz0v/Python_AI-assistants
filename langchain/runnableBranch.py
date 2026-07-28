import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda


# Загружаем переменные окружения из .env файла
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)
 
# Определяем предикаты — простые функции, возвращающие True/False
def is_short_query(x: dict) -> bool:
    # Извлекаем текст запроса из словаря
    query = x.get("query", "")
    return len(query) < 20

def is_greeting(x: dict) -> bool:
    # Извлекаем текст запроса из словаря
    query = x.get("query", "")
    return any(word in query.lower() for word in ["привет", "здравствуй", "hello", "hi"])

# Определяем разные цепочки для разных случаев
short_answer_chain = (
    ChatPromptTemplate.from_template("Кратко ответь на вопрос: {query}. Обязательно начни ответ со слова is_short_query")
    | llm
)

greeting_chain = (
    ChatPromptTemplate.from_template("Вежливо поздоровайся в ответ на: {query}. Обязательно начни ответ со слова is_greeting")
    | llm
)

default_chain = (
    ChatPromptTemplate.from_template("Подробно ответь на вопрос: {query}. Обязательно начни ответ со слова default")
    | llm
)

# Собираем условную маршрутизацию
branch_chain = RunnableBranch(
    (is_greeting, greeting_chain),      # если запрос — приветствие
    (is_short_query, short_answer_chain),  # если запрос короткий
    default_chain                       # во всех остальных случаях
)
 
while True:
    
    query_text = input("\nЗадайте свой вопрос (0 - выход): ").strip()

    # Условие выхода
    if query_text == "0":
        print("Выход из программы.")
        break

    print("Направлен запрос, ожидайте ответа")
    response = branch_chain.invoke({"query": query_text})

    print("=" * 60)
    print("ОТВЕТ НА ЗАПРОС:")
    print("=" * 60)
    print(response.content)
    print("=" * 60)