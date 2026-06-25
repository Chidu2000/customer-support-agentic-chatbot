"""Ingest sample or uploaded policy documents into Chroma."""

from src.data.sample_policies import write_sample_policies
from src.ingestion.pdf_processor import ingest_file, list_policy_files
from src.vectorstore.chroma_store import add_documents


def ingest_all() -> None:
    files = list_policy_files()
    if not files:
        files = write_sample_policies()

    total = 0
    for path in files:
        chunks = ingest_file(path)
        total += add_documents(chunks)
        print(f"Ingested {len(chunks)} chunks from {path.name}")

    print(f"Done. Total chunks indexed: {total}")


if __name__ == "__main__":
    ingest_all()
