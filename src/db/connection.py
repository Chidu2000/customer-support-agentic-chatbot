from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import ROOT_DIR, settings


def get_engine() -> Engine:
    db_path = settings.database_url.replace("sqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(settings.database_url, connect_args={"check_same_thread": False})


def init_schema(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    schema_path = ROOT_DIR / "src" / "db" / "schema.sql"
    with engine.begin() as conn:
        for statement in schema_path.read_text(encoding="utf-8").split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
