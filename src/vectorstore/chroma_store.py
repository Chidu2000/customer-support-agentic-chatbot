from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings

_COLLECTION_NAME = "policy_documents"


def get_embeddings() -> OpenAIEmbeddings:
    api_key = settings.openai_api_key or None
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file and restart the app."
        )
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=api_key,
    )


def _raw_collection() -> chromadb.Collection:
    """Return the raw chromadb collection without loading embeddings."""
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(_COLLECTION_NAME)


def get_vector_store() -> Chroma:
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=_COLLECTION_NAME,
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
    col = _raw_collection()
    if col.count() == 0:
        return []
    store = get_vector_store()
    return store.similarity_search(query, k=k)


def list_indexed_sources() -> list[str]:
    """Return deduplicated source filenames — does NOT need the embedding model."""
    col = _raw_collection()
    if col.count() == 0:
        return []
    result = col.get(include=["metadatas"])
    sources = {
        (meta.get("source_file") or meta.get("source", "unknown"))
        for meta in (result.get("metadatas") or [])
        if meta
    }
    return sorted(sources)


def clear_collection() -> int:
    """Delete all documents from the vector store. Returns count removed."""
    col = _raw_collection()
    count = col.count()
    if count:
        all_ids = col.get(include=[])["ids"]
        col.delete(ids=all_ids)
    return count
