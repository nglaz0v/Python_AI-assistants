"""
Модуль ИИ агента на базе инструментов.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils import convert_to_secret_str

from config import LLM_MODEL_NAME, OPENAI_API_KEY, API_BASE_URL, MAX_TOKENS
from .tools import get_tools
from memory.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class BookStoreAgent:
    """Агент для управления запросами через инструменты"""
    
    def __init__(self):
        # 1. Создаем LLM
        api_key = OPENAI_API_KEY or "empty"
        self.llm = ChatOpenAI(
            base_url=API_BASE_URL,
            model=LLM_MODEL_NAME,
            api_key=convert_to_secret_str(api_key),
            temperature=0.1,
            max_tokens=MAX_TOKENS
        )
        
        # 2. Загружаем системный промпт
        prompt_path = os.path.join(os.path.dirname(__file__), '../prompts/agent_system_prompt.txt')
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt_text = f.read()
            
        # 3. Создаем шаблон промпта
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        logger.info("ИИ Агент инициализирован")

    def process_message(self, user_message: str, user_id: int = 0) -> Dict[str, Any]:
        """
        Обрабатывает сообщение пользователя через агента с инструментами
        """
        try:
            # Получаем историю диалога
            chat_history = memory_manager.get_history_str(user_id)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Получаем инструменты для конкретного пользователя
            tools = get_tools(user_id)
            
            # Создаем агента
            agent = create_openai_tools_agent(self.llm, tools, self.prompt)
            
            # Создаем исполнителя агента
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=False,
                handle_parsing_errors=True
            )
            
            logger.info(f"👤 Вопрос пользователя: {user_message} (ID: {user_id})")
            
            # Запускаем агента
            result = agent_executor.invoke({
                "input": user_message,
                "chat_history": chat_history or "История отсутствует",
                "current_time": current_time
            })
            
            response = result["output"]
            
            # Сохраняем в память
            memory_manager.add_interaction(user_id, user_message, response)
            
            return {
                "success": True,
                "response": response,
                "intent": "agent_decision" # Решение принимает агент
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка в агенте: {e}", exc_info=True)
            return {
                "success": False,
                "response": "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
                "error": str(e)
            }

# Глобальный экземпляр
book_store_agent = BookStoreAgent()
