"""
Central configuration for the Fact Knowledge Layer.

All tunables live here so nothing downstream hard-codes a filename, model
name, path, or magic number in more than one place. Every value can be
overridden via environment variables (see .env.example).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("FKL_DATA_DIR", BASE_DIR / "data"))
PDF_STORE_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_PATH = DATA_DIR / "fact_layer.db"

PDF_STORE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ---- LLM / embedding config --------------------------------------------------
# Any OpenAI-compatible endpoint works (OpenAI, Azure OpenAI via base_url,
# or a local proxy). Swap this file's client construction for LiteLLM's
# `litellm.completion` if you prefer a multi-provider router - the rest of
# the pipeline only depends on the Pydantic contracts in models.py, not on
# which SDK produced them.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EXTRACTION_MODEL = os.getenv("FKL_EXTRACTION_MODEL", "gpt-4o-mini")
RECONCILIATION_MODEL = os.getenv("FKL_RECONCILIATION_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("FKL_EMBEDDING_MODEL", "text-embedding-3-small")

# ---- Chunking ----------------------------------------------------------------
# Facts rarely span more than a couple of pages of context, and keeping
# chunks small keeps quotes traceable to a narrow page range and keeps
# extraction JSON small enough to stay reliable.
CHUNK_MAX_CHARS = int(os.getenv("FKL_CHUNK_MAX_CHARS", "6000"))
# Overlap as a fraction of chunk size (10-15% keeps a fact that straddles a
# page/paragraph boundary fully inside at least one chunk without wasting
# too much token budget re-sending text). Falls back to an explicit char
# override (FKL_CHUNK_OVERLAP_CHARS) if you'd rather set it directly.
CHUNK_OVERLAP_RATIO = float(os.getenv("FKL_CHUNK_OVERLAP_RATIO", "0.12"))
CHUNK_OVERLAP_CHARS = int(os.getenv("FKL_CHUNK_OVERLAP_CHARS", str(int(CHUNK_MAX_CHARS * CHUNK_OVERLAP_RATIO))))

# ---- Concurrency ---------------------------------------------------------------
# How many chunks of a SINGLE document are sent to the extraction LLM at
# once. Bounded (rather than unbounded asyncio.gather) so a 200-page PDF
# doesn't fire hundreds of simultaneous requests and blow through the
# provider's rate limit.
CHUNK_CONCURRENCY = int(os.getenv("FKL_CHUNK_CONCURRENCY", "5"))
# How many PDFs in one batch upload are processed at once. Each document's
# own chunk-level concurrency multiplies with this, so keep the product
# reasonable relative to your API rate limit (default 3 x 5 = 15 in-flight
# extraction calls at peak).
DOCUMENT_CONCURRENCY = int(os.getenv("FKL_DOCUMENT_CONCURRENCY", "3"))
# How many reconciliation clusters are judged concurrently per reconcile run.
RECONCILE_CONCURRENCY = int(os.getenv("FKL_RECONCILE_CONCURRENCY", "5"))

# ---- Reconciliation ------------------------------------------------------------
# Cosine-similarity threshold (Chroma returns distance, we convert) above
# which two facts are considered "candidates" for a relationship and get
# sent to the reconciliation LLM together. Tuned loose on purpose: recall
# matters more here than precision, because the LLM does the real judgment.
SIMILARITY_THRESHOLD = float(os.getenv("FKL_SIMILARITY_THRESHOLD", "0.62"))
CANDIDATES_PER_FACT = int(os.getenv("FKL_CANDIDATES_PER_FACT", "6"))
MAX_CLUSTER_SIZE = int(os.getenv("FKL_MAX_CLUSTER_SIZE", "5"))

CHROMA_COLLECTION = "facts"
