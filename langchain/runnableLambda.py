import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate


# Загружаем переменные окружения
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)

# Цепочка без временной метки
prompt_without_time = ChatPromptTemplate.from_template(
    "Какое сейчас время и дата?"
)
chain_without = (prompt_without_time |
                 llm
)

# Пользовательская функция
def add_timestamp(input_data):
    # Создаём копию входных данных
    output_data = input_data.copy()
    # Добавляем текущее время
    output_data["current_time"] = datetime.now()
    return output_data

prompt_with_time = ChatPromptTemplate.from_template(
    "Текущее время: {current_time}\n\nКакое сейчас время и дата?"
)

# Цепочка с временной меткой
chain_with = (RunnableLambda(add_timestamp) 
              | prompt_with_time 
              | llm
)

response1 = chain_without.invoke({})
response2 = chain_with.invoke({})

print("Без временной метки:")
print(response1.content)
print("\nС временной меткой:")
print(response2.content)