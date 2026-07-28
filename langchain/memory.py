import os
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# Загружаем переменные окружения из .env файла
load_dotenv()

def get_api_key():
    return os.getenv("OPENAI_API_KEY")

def get_api_base_url():
    return os.getenv("API_BASE_URL")

def get_llm_model_name():
    return os.getenv("LLM_MODEL_NAME")

llm = ChatOpenAI(
    base_url=get_api_base_url(),
    model=get_llm_model_name(),
    api_key=get_api_key(),
    temperature=0.3
)

# Промпт для суммаризации на русском языке
summary_prompt = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template="""Текущее резюме диалога:
{summary}

Новые сообщения:
{new_lines}

Создай новое резюме диалога на том же языке, что и новые сообщения.
Резюме должно быть кратким, но содержать важную информацию из диалога.
Не меняй язык оригинала - если сообщения на русском, пиши на русском, если на английском - на английском.

Новое резюме:"""
)

# 1. ConversationSummaryMemory - хранит сжатое содержание диалога
# Используем кастомный промпт на русском языке
memory_summary = ConversationSummaryMemory(llm=llm, prompt=summary_prompt)

# 2. ConversationBufferWindowMemory - хранит последние K сообщений (в данном случае 5)
memory_buffer = ConversationBufferWindowMemory(k=5)

# Заполняем обе памяти расширенным начальным контекстом для сравнения
initial_history = [
    ({"input": "Здравствуйте, меня зовут Алексей. Я врач терапевт. Что вас беспокоит?"}, {"output": "Приятно познакомиться, Алексей!. У меня болит горло"}),
    ({"input": "Как давно это вас беспокоит"}, {"output": "Уже 3 дня"}),
    ({"input": "Какие лекарства вы уже принимали?"}, {"output": "Принимал парацетамол вчера вечером"}),
    ({"input": "Какая у вас сейчас температура?"}, {"output": "Температура 37.8"})
]

for inp, out in initial_history:
    memory_summary.save_context(inp, out)
    memory_buffer.save_context(inp, out)

print("=== СРАВНЕНИЕ ТИПОВ ПАМЯТИ ===")

print("\n1. SUMMARY MEMORY (Сжатие):")
print(memory_summary.load_memory_variables({})['history'])

print("\n2. BUFFER WINDOW MEMORY (Окно):")
print(memory_buffer.load_memory_variables({})['history'])
