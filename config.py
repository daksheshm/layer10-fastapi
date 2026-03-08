import os
from pathlib import Path

# ==============================
# Base paths
# ==============================

BASE_DIR = Path(__file__).parent

DATA_DIR       = BASE_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
ARTIFACTS_DIR  = DATA_DIR / "artifacts"
CHUNKS_DIR     = DATA_DIR / "chunks"
EXTRACTIONS_DIR = DATA_DIR / "extractions"
GRAPH_DIR      = DATA_DIR / "graph"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESOLVED_DIR   = DATA_DIR / "resolved"
SCHEMA_DIR     = BASE_DIR / "schema"

# ==============================
# GitHub ingestion
# ==============================

GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
REPO_OWNER    = os.getenv("REPO_OWNER", "tiangolo")
REPO_NAME     = os.getenv("REPO_NAME", "fastapi")
FETCH_STATE   = "closed"   # closed issues contain full discussion + resolution
FETCH_LIMIT   = 100

# ==============================
# Extraction (Ollama)
# ==============================

OLLAMA_URL        = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_TIMEOUT    = 120
MAX_OUTPUT_TOKENS = 600

# Minimum confidence score (0–1) to accept a claim into the graph
EXTRACTION_MIN_CONFIDENCE = 0.5

# ==============================
# Chunking
# ==============================

CHUNK_SIZE    = 2000   # characters (~500 tokens)
CHUNK_OVERLAP = 200

# ==============================
# Deduplication / canonicalization
# ==============================

# Jaccard similarity threshold for merging entity names
ENTITY_SIM_THRESHOLD = 0.82

# Jaccard similarity threshold for merging claims with identical (subject, relation)
CLAIM_SIM_THRESHOLD = 0.78

# ==============================
# Retrieval
# ==============================

TOP_K_CHUNKS  = 8    # number of evidence chunks to return per query
TOP_K_CLAIMS  = 5    # number of graph claims to return per query

# ==============================
# Helpers
# ==============================

def ensure_dirs():
    """Create all data directories if they don't exist."""
    for d in [
        RAW_DIR, ARTIFACTS_DIR, CHUNKS_DIR,
        EXTRACTIONS_DIR, GRAPH_DIR, EMBEDDINGS_DIR, RESOLVED_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)