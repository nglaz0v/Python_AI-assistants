from transformers import AutoTokenizer
import tiktoken

# Применяем разные токенизаторы
tokenizers = {
    "BERT Tiny": {"type": "huggingface", "model": "cointegrated/rubert-tiny2"},
    "BERT Base": {"type": "huggingface", "model": "DeepPavlov/rubert-base-cased"},
    "DeepSeek": {"type": "huggingface", "model": "deepseek-ai/deepseek-llm-7b-base"},
    "LLaMA": {"type": "huggingface", "model": "huggyllama/llama-7b"},
    "Qwen": {"type": "huggingface", "model": "Qwen/Qwen-7B"},
    "TikToken cl100k_base": {"type": "tiktoken", "encoding": "cl100k_base"},
    "TikToken o200k_base": {"type": "tiktoken", "encoding": "o200k_base"},
    "TikToken r50k_base": {"type": "tiktoken", "encoding": "r50k_base"}
}

# Тестовый текст
text = "настольная игра"

print(f"Текст: {text}")
print(f"Длина текста в символах: {len(text)}")
print("-" * 50)

# Создаем токенизаторы tiktoken для разных кодировок
tiktoken_tokenizers = {
    "cl100k_base": tiktoken.get_encoding("cl100k_base"),
    "o200k_base": tiktoken.get_encoding("o200k_base"),
    "r50k_base": tiktoken.get_encoding("r50k_base")
}

# Тестируем каждый токенизатор
for name, tokenizer_info in tokenizers.items():
    if tokenizer_info["type"] == "huggingface":
        # Загружаем HuggingFace токенизатор
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_info["model"], trust_remote_code=True)
        
        # Подсчет токенов
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Декодирование для HuggingFace
        decode_func = lambda t: tokenizer.decode([t])
        
    elif tokenizer_info["type"] == "tiktoken":
        # Используем соответствующий tiktoken токенизатор
        encoding_name = tokenizer_info["encoding"]
        tokenizer = tiktoken_tokenizers[encoding_name]
        tokens = tokenizer.encode(text)
        
        # Декодирование для TikToken
        decode_func = lambda t: tokenizer.decode([t])
    
    # Выводим результаты
    print(f"{name}:")
    print(f"  Количество токенов: {len(tokens)}")
    print(f"  Отношение символов/токенов: {len(text) / len(tokens):.2f}")
    print("\nПервые 10 токенов:")
    for i, token in enumerate(tokens[:10]):
        print(f"  Токен {i}: {token} -> '{decode_func(token)}'")
    print("-" * 50)