"""
Обработчик сообщений для Telegram бота
"""

import os
import time
import logging
from telegram import Update
from telegram.ext import ContextTypes

from modules.agent import book_store_agent
from config import is_employee
from vector_stores.base import books_service_vector_store, organization_vector_store
from chains.service_chain import service_chain
from chains.organization_chain import organization_chain


logger = logging.getLogger(__name__)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает входящие сообщения от пользователей Telegram
    
    Args:
        update: Объект обновления от Telegram
        context: Контекст бота
    """
    try:
        # Проверяем, что сообщение существует
        if not update.message or not update.message.text:
            await update.message.reply_text("❌ Получено пустое сообщение. Пожалуйста, напишите ваш вопрос.")
            return
        
        # Получаем сообщение и информацию о пользователе
        user_message = update.message.text
        user_id = update.effective_user.id if update.effective_user else 0
        username = update.effective_user.username if update.effective_user else "Unknown"
        
        logger.info(f"📨 Получено сообщение от {username} (ID: {user_id}): {user_message}")
        
        # Отправляем статус "печатает"
        await update.message.chat.send_action(action="typing")
        
        # Обрабатываем сообщение через агента
        logger.info(f"🔄 Обработка сообщения через ИИ агента...")
        result = book_store_agent.process_message(user_message, user_id)
        logger.info(f"📋 Получен результат от агента: успех={result['success']}, намерение={result.get('intent')}")
        
        # Отправляем ответ пользователю
        if result["success"]:
            response = result["response"]
            
            # Отправка основного ответа
            logger.info(f"📨 Отправка основного ответа пользователю {username}")
            await update.message.reply_text(response)
            logger.info(f"✅ Основной ответ отправлен пользователю {username}")
                
        else:
            # Обработка ошибок
            logger.info(f"⚠️ Подготовка ответа об ошибке для пользователя {username}")
            error_response = result["response"]
            if result.get("access_denied"):
                error_response += "\n\n🔒 *Эта информация доступна только сотрудникам компании.*"
            
            logger.info(f"📨 Отправка ответа об ошибке пользователю {username}")
            await update.message.reply_text(error_response)
            logger.info(f"✅ Ответ об ошибке отправлен пользователю {username}")
            
        logger.info(f"🎉 Обработка сообщения завершена для пользователя {username}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке сообщения: {e}")
        
        # Отправляем сообщение об ошибке пользователю
        error_message = (
            "❌ Произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте позже или обратитесь к администратору."
        )
        
        try:
            await update.message.reply_text(error_message)
        except Exception as send_error:
            logger.error(f"❌ Не удалось отправить сообщение об ошибке: {send_error}")


async def handle_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает команду /start
    
    Args:
        update: Объект обновления от Telegram
        context: Контекст бота
    """
    welcome_message = """
🤖 *Добро пожаловать в книжный магазин "Книжный мир"!*

Я - ваш ИИ помощник, готовый помочь с:

📚 *Информацией о книгах*
• Наличие и стоимость книг
• Поиск по авторам и названиям  
• Информация о содержании

🛒 *Правилами обслуживания*
• Доставка и сроки
• Оплата и возврат
• Условия покупки

🏢 *Внутренней информацией* (для сотрудников)
• Документы организации
• Должностные инструкции
• Положения отделов

*Просто задайте ваш вопрос, и я постараюсь помочь!*

*Примеры запросов:*
• "Какие книги Толстого есть в наличии?"
• "Сколько стоит доставка?"
• "Какие правила возврата товара?"
    """
    
    await update.message.reply_text(welcome_message, parse_mode='Markdown')


async def handle_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает команду /help
    
    Args:
        update: Объект обновления от Telegram
        context: Контекст бота
    """
    help_message = """
🆘 *Помощь по использованию бота*

*Доступные команды:*
/start - Начать работу с ботом
/help - Показать эту справку

*Категории запросов:*

📚 *Книги*
• "Какие книги есть в наличии?"
• "Сколько стоит 'Война и мир'?"
• "Найти книги про приключения"

🛒 *Обслуживание*  
• "Как оформить доставку?"
• "Какие способы оплаты?"
• "Можно ли вернуть книгу?"

🏢 *Организация* (только для сотрудников)
• "Какие должностные обязанности у помощника?"
• "Положение о продажах"

*Просто напишите ваш вопрос, и я определю наиболее подходящую категорию!*
    """
    
    await update.message.reply_text(help_message, parse_mode='Markdown')


async def handle_reload_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает команду /reload для перезагрузки данных векторных хранилищ
    
    Args:
        update: Объект обновления от Telegram
        context: Контекст бота
    """
    try:
        # Проверяем, является ли пользователь сотрудником
        user_id = update.effective_user.id if update.effective_user else 0
        
        if not is_employee(user_id):
            await update.message.reply_text(
                "❌ *Эта команда доступна только сотрудникам компании.*",
                parse_mode='Markdown'
            )
            return
        
        # Отправляем статус "печатает"
        await update.message.chat.send_action(action="typing")
        
        logger.info(f"🔄 Пользователь {user_id} запросил перезагрузку данных")
        
        # Обновляем векторные хранилища
        await update.message.reply_text("🔄 Начинается обновление данных...")
        
        # Сбрасываем кэш хранилищ для принудительного сканирования файлов
        for store in [books_service_vector_store, organization_vector_store]:
            try:
                # Сбрасываем кэш хранилища (закрываем соединение)
                store.reset()
                logger.info(f"🔄 Сброшен кэш для {store.collection_name}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось сбросить хранилище {store.collection_name}: {e}")

        
        # Даем время для освобождения ресурсов
        time.sleep(0.5)
        
        # Обновляем хранилище книг и правил обслуживания
        await update.message.reply_text("📚 Обновление хранилища книг и правил обслуживания...")
        books_service_vector_store.update_vector_store()
        
        # Обновляем цепочку обслуживания
        service_chain.reload(vector_store=books_service_vector_store)
        
        await update.message.reply_text("✅ Хранилище книг и правил обслуживания обновлено")
        
        # Обновляем хранилище документов организации
        await update.message.reply_text("🏢 Обновление хранилища документов организации...")
        organization_vector_store.update_vector_store()
        
        # Обновляем цепочку организации
        organization_chain.reload(vector_store=organization_vector_store)
        
        await update.message.reply_text("✅ Хранилище документов организации обновлено")
        
        await update.message.reply_text(
            "🎉 Все данные успешно обновлены!\n\n"
            "✅ Проверены и обновлены:\n"
            "• Правила обслуживания\n"
            "• Документы организации\n\n"
            "Система готова к работе с актуальными данными."
        )
        
        logger.info(f"✅ Данные успешно обновлены пользователем {user_id}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при перезагрузке данных: {e}")
        await update.message.reply_text(
            "❌ Произошла ошибка при перезагрузке данных. "
            "Пожалуйста, попробуйте позже или обратитесь к администратору."
        )