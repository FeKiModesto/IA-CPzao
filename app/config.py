import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# A cota gratuita é por modelo (~20 chamadas/dia cada), então usamos uma cadeia de reservas.
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.7-flash,gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite",
    ).split(",")
    if m.strip()
]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 768

SITE_URL = "https://arnaldojr.github.io/DisruptiveArchitectures/"
DOCS_ZIP_URL = "https://github.com/arnaldojr/DisruptiveArchitectures/archive/refs/heads/master.zip"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))


def database_url() -> str:
    url = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}")
    # O Render entrega "postgres://", o SQLAlchemy precisa do driver explícito.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url
