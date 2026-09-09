"""Central configuration for the Fact Knowledge Layer."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

EXTRACTION_MODEL = os.environ.get("FKL_EXTRACTION_MODEL", "openai/gpt-oss-120b")
RECONCILIATION_MODEL = os.environ.get("FKL_RECONCILIATION_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("FKL_EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_MAX_CHARS = int(os.environ.get("FKL_CHUNK_MAX_CHARS", 6000))
CHUNK_OVERLAP_RATIO = float(os.environ.get("FKL_CHUNK_OVERLAP_RATIO", 0.12))
CHUNK_OVERLAP_CHARS = int(CHUNK_MAX_CHARS * CHUNK_OVERLAP_RATIO)

SIMILARITY_THRESHOLD = float(os.environ.get("FKL_SIMILARITY_THRESHOLD", 0.62))
CANDIDATES_PER_FACT = int(os.environ.get("FKL_CANDIDATES_PER_FACT", 6))
MAX_CLUSTER_SIZE = int(os.environ.get("FKL_MAX_CLUSTER_SIZE", 5))

CHUNK_CONCURRENCY = int(os.environ.get("FKL_CHUNK_CONCURRENCY", 3))
DOCUMENT_CONCURRENCY = int(os.environ.get("FKL_DOCUMENT_CONCURRENCY", 3))
RECONCILE_CONCURRENCY = int(os.environ.get("FKL_RECONCILE_CONCURRENCY", 5))

# --- Database and File Paths ---
CHROMA_COLLECTION = "facts"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
SQLITE_PATH = Path(__file__).parent.parent / "facts.db"
PDF_STORE_DIR = Path(__file__).parent.parent / "pdf_store"
PDF_STORE_DIR.mkdir(parents=True, exist_ok=True)

# --- AUTO-CREATE DIRECTORIES ---
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
PDF_STORE_DIR.mkdir(parents=True, exist_ok=True)