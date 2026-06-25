from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def load_pdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = path.name
    return docs


def load_text_policy(path: Path) -> list[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = path.name
    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        documents = load_pdf(path)
    elif suffix in {".txt", ".md"}:
        documents = load_text_policy(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return chunk_documents(documents)


def list_policy_files() -> list[Path]:
    policies_dir = Path(settings.policies_dir)
    policies_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for pattern in ("*.pdf", "*.txt", "*.md"):
        files.extend(sorted(policies_dir.glob(pattern)))
    return files
