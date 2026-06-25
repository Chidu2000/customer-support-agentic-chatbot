from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    llm_provider: str = "openai"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    database_url: str = f"sqlite:///{ROOT_DIR / 'data' / 'customers.db'}"
    chroma_persist_dir: str = str(ROOT_DIR / "data" / "chroma")
    policies_dir: str = str(ROOT_DIR / "data" / "policies")

    mcp_server_name: str = "customer-support-mcp"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765


settings = Settings()
