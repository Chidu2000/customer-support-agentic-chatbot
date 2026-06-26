from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key or None,
    )


def get_vector_store() -> Chroma:
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="policy_documents",
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(chunks: list[Document]) -> int:
    if not chunks:
        return 0
    store = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)


def search_policies(query: str, k: int = 4) -> list[Document]:
    store = get_vector_store()
    if store._collection.count() == 0:
        return []
    return store.similarity_search(query, k=k)


def clear_collection() -> int:
    """Delete all documents from the vector store. Returns count removed."""
    store = get_vector_store()
    count = store._collection.count()
    if count:
        all_ids = store._collection.get(include=[])["ids"]
        store._collection.delete(ids=all_ids)
    return count


def list_indexed_sources() -> list[str]:
    store = get_vector_store()
    if store._collection.count() == 0:
        return []
    result = store._collection.get(include=["metadatas"])
    sources = {
        meta.get("source_file") or meta.get("source", "unknown")
        for meta in (result.get("metadatas") or [])
        if meta
    }
    return sorted(sources)
