import os
from dotenv import load_dotenv
from llama_index.core import TreeIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import MarkdownNodeParser
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

# Для TreeIndex важно установить LLM в Settings, так как он используется для построения дерева
Settings.llm = llm

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

# # ============================================================
# Этап подготовки. Индексация данных.
# ============================================================

# Шаг 1. Загрузка документов
# Используем конкретный файл, как указано в задании
reader = SimpleDirectoryReader(
    input_files=[r"test_data/НПА/Закон о защите прав потребителей.md"]
)
documents = reader.load_data()

# Устанавливаем количество документов для логера
custom_logger.set_loaded_docs_count(len(documents))

# Шаг 2. Разбивка документов на основе структуры Markdown
# MarkdownNodeParser разбивает документ по заголовкам (#, ##, ###),
# что соответствует структуре Закон -> Глава -> Статья.
parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# Анализ узлов: подсчёт по уровням заголовков
heading_counts = {}
for node in nodes:
    # MarkdownNodeParser в текущей версии может сохранять путь заголовков в header_path
    header_path = node.metadata.get("header_path", "")
    # Считаем количество уровней в пути (разделенных /)
    if header_path and header_path != "/":
        # Путь вида '/**О ЗАЩИТЕ ПРАВ ПОТРЕБИТЕЛЕЙ**\r/**Глава I.../'
        # Считаем количество вхождений заголовков (обычно они начинаются с # в контенте,
        # но в метаданных мы можем посчитать количество сегментов пути)
        levels = header_path.strip("/").split("\r/")
        level = len(levels)
        heading_counts[level] = heading_counts.get(level, 0) + 1
    else:
        # Узлы без заголовка (преамбула)
        heading_counts[0] = heading_counts.get(0, 0) + 1

logger.info("=" * 60)
logger.info("АНАЛИЗ УЗЛОВ MarkdownNodeParser:")
logger.info("=" * 60)
for level in sorted(heading_counts.keys()):
    label = f"Уровень {level}"
    logger.info(f"{label}: {heading_counts[level]} узлов")
logger.info(f"Всего узлов: {len(nodes)}")
logger.info("=" * 60)

# Шаг 3. Создание TreeIndex
index = TreeIndex(
    nodes=nodes,
    show_progress=True,
    num_children=10 # Количество листьев в ветке
)

# ============================================================
# Визуализация структуры индекса
# ============================================================
logger.info("=" * 60)
logger.info("ВИЗУАЛИЗАЦИЯ ДЕРЕВА (TreeIndex):")
logger.info("=" * 60)

index_struct = index.index_struct
docstore = index.docstore

def get_node_title(node_id):
    node = docstore.get_node(node_id, raise_error=False)
    if not node:
        return "[Узел не найден]"
    
    # Для визуализации лучше всегда использовать начало контента, 
    # так как метаданные заголовков (header_path) могут дублироваться для разных чанков одной секции.
    content = node.get_content()
    
    # Очистка текста от переносов строк и лишних пробелов
    title = content.replace("\r", "").replace("\n", " ").strip()
    
    # Ограничиваем длину
    return (title[:300] + "...") if len(title) > 100 else title

# Определяем корневые узлы
root_nodes = index_struct.root_nodes
if isinstance(root_nodes, dict):
    root_ids = list(root_nodes.values())
else:
    root_ids = root_nodes

# Создаем множество ID листовых узлов для быстрой проверки
leaf_ids = set(n.node_id for n in nodes)

# Рекурсивная функция для отрисовки дерева
def draw_tree(node_id, level=0, visited=None):
    if visited is None:
        visited = set()
    
    if node_id in visited:
        return
    visited.add(node_id)

    title = get_node_title(node_id)
    indent = "  " * level
    prefix = "└── " if level > 0 else ""
    
    # Помечаем, является ли узел суммаризацией или листовым узлом
    is_leaf = node_id in leaf_ids
    node_type = "[Leaf]" if is_leaf else "[Summary]"
    
    # Если это не первый уровень (не корень), добавляем отступ и префикс
    # Если это корень, выводим без отступа
    display_prefix = f"{indent}{prefix}"
    
    logger.info(f"{display_prefix}{node_type} {title} (ID: {node_id[:8]})")
    
    # В TreeIndex (IndexGraph) связи хранятся в node_id_to_children_ids
    child_ids = []
    if hasattr(index_struct, 'node_id_to_children_ids') and node_id in index_struct.node_id_to_children_ids:
        child_ids = index_struct.node_id_to_children_ids[node_id]

    if child_ids:
        for child_id in child_ids:
            draw_tree(child_id, level + 1, visited)

# Запуск отрисовки от корней
for r_id in root_ids:
    draw_tree(r_id)

logger.info("=" * 60)

# ============================================================
# Этап 1. Поиск релевантных данных (Retrieval)
# ============================================================

# Шаг 1. Поиск данных
# Настройка RAG для TreeIndex
query_engine = index.as_query_engine(
    llm=llm,
    text_qa_template=qa_prompt,
    # Для TreeIndex используется режим 'tree_summarize', который рекурсивно
    # объединяет ответы от разных узлов дерева.
    response_mode='tree_summarize'
)

# ============================================================
# Этап 2. Генерация ответа
# ============================================================

# Запрос к LLM
while True:
    query_text = input("\nЗадайте свой вопрос по Закону о защите прав потребителей (0 - выход): ").strip()

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
