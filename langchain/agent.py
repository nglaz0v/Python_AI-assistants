import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor


# Загружаем переменные окружения
load_dotenv()
 
 
# Создаём LLM
LLM = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)


@tool
def get_current_date() -> str:
    """Возвращает текущую дату в формате ДД.ММ.ГГГГ."""
    from datetime import datetime
    return datetime.now().strftime("%d.%m.%Y")
 
 
@tool
def calculate_years_difference(year1: int, year2: int) -> int:
    """Вычисляет разницу между двумя годами."""
    return abs(year2 - year1)
 
tools = [get_current_date, calculate_years_difference]
 

# Промпт для агента
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — полезный помощник. Ты можешь использовать инструменты для получения информации или выполнения вычислений. "
               "Если вопрос можно ответить без инструментов — отвечай сразу. "
               "Если нужны инструменты — вызывай их. Не придумывай данные."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
 
 
agent = create_tool_calling_agent(LLM, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
 
 
# Вызов агента
response = agent_executor.invoke({"input": "Сколько лет прошло с момента бородинского сражения?"})
print(f"Ответ агента: {response['output']}")
