"""
Базовый класс для управления векторными базами данных с механизмом обновления по хэшу.
"""

import hashlib
import logging
import os
from pathlib import Path

# Отключаем телеметрию ChromaDB для предотвращения ошибок совместимости с posthog
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import Dict, List, Tuple, Union

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    EMBEDDING_MODEL_NAME,
    OPENAI_API_KEY,
    API_BASE_URL,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_MAX_RETRIES,
    RAG_CONFIG,
    DATA_PATHS,
    CHROMA_PERSIST_DIRECTORY,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SUPPORTED_EXTENSIONS = (".md", ".txt")

class HashManager:
    """Управление хэшами файлов для отслеживания изменений."""

    def __init__(self, hash_file_path: Union[str, Path], data_paths: List[Path]):
        self.hash_file_path = Path(hash_file_path)
        self.data_paths = data_paths

    def load(self) -> Dict[str, str]:
        """Загружает сохраненные хэши из файла."""
        if not self.hash_file_path.exists():
            return {}
        hashes = {}
        try:
            with open(self.hash_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "|" in line:
                        file_path, file_hash = line.split("|", 1)
                        hashes[file_path] = file_hash
            logger.info(f"Загружено {len(hashes)} хэшей из {self.hash_file_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки хэш-файла {self.hash_file_path}: {e}")
        return hashes

    def save(self, hashes: Dict[str, str]) -> None:
        """Сохраняет хэши в файл."""
        self.hash_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_file_path, "w", encoding="utf-8") as f:
            for file_path, file_hash in hashes.items():
                f.write(f"{file_path}|{file_hash}\n")
        logger.info(f"Сохранено {len(hashes)} хэшей в {self.hash_file_path}")

    @staticmethod
    def compute_file_hash(file_path: Union[str, Path]) -> str:
        """Вычисляет MD5 хэш файла."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_current_hashes(self) -> Dict[str, str]:
        """Сканирует директории и вычисляет хэши всех поддерживаемых файлов."""
        current_hashes = {}
        for data_path in self.data_paths:
            logger.info(f"Сканирование директории: {data_path}")
            for file in data_path.rglob("*"):
                if file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    current_hashes[str(file)] = self.compute_file_hash(file)
        logger.info(f"Найдено и обработано {len(current_hashes)} файлов")
        return current_hashes

    @staticmethod
    def classify_files(current: Dict[str, str], saved: Dict[str, str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Сравнивает хэши и возвращает списки: неизмененные, измененные, новые, удаленные."""
        current_set, saved_set = set(current.keys()), set(saved.keys())
        new, deleted = list(current_set - saved_set), list(saved_set - current_set)
        common = current_set & saved_set
        modified = [p for p in common if current[p] != saved[p]]
        unchanged = [p for p in common if current[p] == saved[p]]
        return unchanged, modified, new, deleted


class VectorStoreManager:
    """Базовый класс для управления векторными БД с обновлением по хэшу."""

    def __init__(self, collection_name: str, data_paths: Union[str, List[str]]):
        self.collection_name = collection_name
        self.data_paths = [Path(p) for p in ([data_paths] if isinstance(data_paths, str) else data_paths)]
        self._vector_store = None
        
        # Создание эмбеддингов через HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
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
        # self.embeddings = OpenAIEmbeddings(
        #     model=EMBEDDING_MODEL_NAME,
        #     api_key=convert_to_secret_str(OPENAI_API_KEY),
        #     base_url=API_BASE_URL,
        #     chunk_size=EMBEDDING_CHUNK_SIZE,
        #     max_retries=EMBEDDING_MAX_RETRIES,
        # )
        
        
        self.default_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"],
            separators=["\n# ", "\n## "],
            is_separator_regex=False,
        )

        # Используем путь из конфигурации
        self.persist_directory = Path(CHROMA_PERSIST_DIRECTORY) / collection_name
        self.hash_file = self.persist_directory.parent / f".{collection_name}_hashes.txt"

        self.hash_manager = HashManager(self.hash_file, self.data_paths)
        logger.info(f"✅ VectorStoreManager инициализирован: {collection_name}")

    def _load_documents(self, file_paths: List[str]) -> List[Document]:
        """Загружает документы из указанных файлов."""
        documents = []
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": os.path.basename(file_path),
                        "type": self.collection_name,
                        "file_path": file_path,
                    },
                ))
                logger.info(f"Загружен: {file_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки {file_path}: {e}")
        return documents

    def _create_chroma_instance(self) -> Chroma:
        """Создает или загружает экземпляр Chroma."""
        return Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _add_documents_to_store(self, vector_store: Chroma, file_paths: List[str]) -> None:
        """Добавляет документы из файлов в хранилище."""
        documents = self._load_documents(file_paths)
        if not documents:
            return
        chunks = self.default_text_splitter.split_documents(documents)
        if chunks:
            vector_store.add_documents(chunks)
            logger.info(f"Добавлено {len(chunks)} чанков в {self.collection_name}")

    def _delete_documents_from_store(self, vector_store: Chroma, file_paths: List[str]) -> None:
        """Удаляет чанки, принадлежащие указанным файлам."""
        for file_path in file_paths:
            try:
                vector_store.delete(where={"file_path": file_path})
                logger.info(f"Удален файл из БД: {file_path}")
            except Exception as e:
                logger.warning(f"Ошибка удаления {file_path}: {e}")

    def reset(self) -> None:
        """Сбрасывает текущее соединение с векторным хранилищем."""
        self._vector_store = None
        logger.info(f"Соединение с хранилищем '{self.collection_name}' сброшено.")

    def initialize_vector_store(self) -> Chroma:
        """Инициализирует векторное хранилище."""
        if self._vector_store is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._vector_store = self._create_chroma_instance()
            
            # Если хэш-файл отсутствует, значит данные еще не индексировались.
            # Выполняем обновление. Это безопасно, так как self._vector_store уже не None,
            # и повторный вызов initialize_vector_store из update_vector_store не приведет к рекурсии.
            if not self.hash_file.exists():
                logger.info(f"Хэш-файл не найден. Выполняем первичную индексацию для {self.collection_name}")
                self.update_vector_store()
                
        return self._vector_store

    def update_vector_store(self) -> Chroma:
        """Выполняет инкрементальное обновление данных."""
        logger.info(f"Запуск обновления хранилища: {self.collection_name}")
        
        # Проверяем существование БД ДО инициализации
        db_exists = (self.persist_directory / "chroma.sqlite3").exists()
        
        if not db_exists and self.hash_file.exists():
            logger.info(f"Файл БД не найден. Сброс хэшей для полной переиндексации.")
            self.hash_file.unlink()

        vector_store = self.initialize_vector_store()

        saved_hashes = self.hash_manager.load()
        current_hashes = self.hash_manager.get_current_hashes()
        unchanged, modified, new, deleted = self.hash_manager.classify_files(current_hashes, saved_hashes)

        logger.info(f"Статус '{self.collection_name}': новых {len(new)}, изменённых {len(modified)}, удалённых {len(deleted)}")

        # Если БД пустая или нет сохраненных хэшей - полная переиндексация
        if not db_exists or not saved_hashes:
            # Очищаем существующие данные, если они есть
            if (results := vector_store.get()) and results["ids"]:
                vector_store.delete(ids=results["ids"])
            # Добавляем все текущие файлы
            self._add_documents_to_store(vector_store, list(current_hashes.keys()))
        # Если есть изменения - инкрементальное обновление
        elif new or modified or deleted:
            # Удаляем измененные и удаленные файлы
            if to_delete := (modified + deleted):
                self._delete_documents_from_store(vector_store, to_delete)
            # Добавляем новые и измененные файлы
            if to_add := (new + modified):
                self._add_documents_to_store(vector_store, to_add)
        
        # Сохраняем хэши если были изменения или это первая индексация
        if new or modified or deleted or not saved_hashes:
            self.hash_manager.save(current_hashes)
            
        return vector_store

    def get_retriever(self):
        """Возвращает ретривер."""
        return self.initialize_vector_store().as_retriever(
            search_kwargs={"k": RAG_CONFIG["similarity_top_k"]}
        )


# Глобальные экземпляры для использования в цепочках (сохраняем обратную совместимость)

books_service_vector_store = VectorStoreManager(
    collection_name="books_service",
    data_paths=[DATA_PATHS["service_rules"]],
)

organization_vector_store = VectorStoreManager(
    collection_name="organization_docs",
    data_paths=[DATA_PATHS["organization"]],
)
