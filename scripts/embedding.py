import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

## Загружаем модель с поддержкой русского языка
# Попробуйте заменить на другую модель, например: "sberbank-ai/ruRoberta-large", "paraphrase-multilingual-MiniLM-L12-v2",
# "intfloat/multilingual-e5-large", "DeepPavlov/rubert-base-cased" или иные
model_name = "DeepPavlov/rubert-base-cased"  # или используйте os.getenv("MODEL_NAME") если хотите через переменные окружения
model = SentenceTransformer(model_name)

# Слова для составления эмбеддингов
words = ["Кошка", "Собака", "Тигр", "Мышь", "Автомобиль"]

# Вычисляем эмбеддинги
embeddings = model.encode(words)
embeddings = np.array(embeddings)

# Выводим векторы эмбеддингов (только первые 10 чисел для краткости)
print("Эмбеддинги слов (первые 10 компонент вектора):")
print("-" * 60)
for word, embedding in zip(words, embeddings):
    # Округляем и выводим первые 10 элементов
    emb_short = np.round(embedding[:10], 3)
    print(f"{word:8} → [{', '.join(map(str, emb_short))}, ...]")

print("\n" + "="*60)

# Выводим косинусное сходство
print("Косинусное сходство (семантическая близость):")
print("-" * 50)
pairs = [
    ("Кошка", "Собака"),
    ("Кошка", "Тигр"),
    ("Кошка", "Мышь"),
    ("Кошка", "Автомобиль")
]

for w1, w2 in pairs:
    sim = cosine_similarity([embeddings[words.index(w1)]], [embeddings[words.index(w2)]])[0][0]
    print(f"{w1:8} ↔ {w2:8} : {sim:.3f}")
