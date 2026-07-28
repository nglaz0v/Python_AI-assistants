import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Инициализация LLM и Embeddings
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
    temperature=0.0
)

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
#     model=os.getenv("EMBEDDING_MODEL_NAME"),
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url=os.getenv("API_BASE_URL")
# )

# Тестовые данные
test_data = [
    {
        "question": "Какие документы нужны для командировки?",
        "contexts": [
            "Отпуск оформляется за 14 дней через HR-систему. Больничный лист предоставляется в течение 3 дней после выздоровления.",  # Нерелевантный контекст
            "Для оформления командировки требуется: заявление по форме К-1, копия паспорта и согласование руководителя отдела.", # Релевантный
            "Командировочные расходы оплачиваются по предъявлении авансового отчета в течение 3 дней.", # Релевантный
        ],
        "answer": "Нужно подать заявление по форме К-1 и копию паспорта.",  # Неполный ответ - нет информации о согласовании
        "reference": "Для оформления командировки нужны: заявление по форме К-1, копия паспорта и согласование руководителя отдела."
    },
    {
        "question": "Можно ли работать удалённо без согласования?",
        "contexts": [
            "Удалённый режим работы возможен только при наличии письменного согласования с непосредственным руководителем и HR-менеджером.", # Релевантный
            "Командировки оформляются через заявление К-1. Отпускные рассчитываются исходя из среднего заработка.",  # Нерелевантный
            "Гибкий график работы обсуждается индивидуально с руководителем. Дресс-код в офисе - бизнес-кэжуал.",  # Частично релевантный
        ],
        "answer": "Да, можно работать удалённо в любое время по своему усмотрению.",  # Неверный ответ
        "reference": "Удалённая работа возможна только при наличии письменного согласования с непосредственным руководителем и HR-менеджером."
    },
    {
        "question": "Как оформить отпуск и какие сроки?",
        "contexts": [
            "Для оформления ежегодного отпуска сотрудник подаёт заявление в HR-систему не позднее чем за 14 календарных дней до начала отпуска.", # Релевантный
            "Больничный лист предоставляется в бухгалтерию. Командировочные расходы оплачиваются по возвращении.",  # Нерелевантный
            "Отпуск может быть перенесен по согласованию с руководителем. Максимальная продолжительность отпуска - 28 дней."  # Релевантный
        ],
        "answer": "Подайте заявление в HR-систему за 2 недели до отпуска. Также нужно согласовать с руководителем.",  # Частично верный + лишняя информация
        "reference": "Для оформления ежегодного отпуска сотрудник подаёт заявление в HR-систему не позднее чем за 14 календарных дней до начала отпуска."
    }
]

dataset = Dataset.from_list(test_data)

# Оценка
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings
)

# Вывод результатов
print("\n=== Результаты оценки RAGAS ===")
df = results.to_pandas()


print("\n=== Результаты по примерам ===")
print(f"{'№':<3} {'Вопрос':<35} {'Faithfulness':<12} {'Answer Rel':<12} {'Context Prec':<12} {'Context Rec':<12}")
print("-" * 90)


for i, row in df.iterrows():
    question_short = row['user_input'][:32] + "..." if len(row['user_input']) > 35 else row['user_input']
    print(f"{i+1:<3} {question_short:<35} {row['faithfulness']:<12.3f} {row['answer_relevancy']:<12.3f} {row['context_precision']:<12.3f} {row['context_recall']:<12.3f}")

# Средние значения
print(f"\nСредние оценки:")
print(f"Faithfulness: {df['faithfulness'].mean():.3f}")
print(f"Answer Relevancy: {df['answer_relevancy'].mean():.3f}")
print(f"Context Precision: {df['context_precision'].mean():.3f}")
print(f"Context Recall: {df['context_recall'].mean():.3f}")
