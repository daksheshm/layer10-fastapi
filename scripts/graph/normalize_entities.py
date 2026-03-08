"""
build_embeddings.py

Builds TF-IDF embeddings over all non-bot chunks so that retrieval can use
cosine similarity to find evidence relevant to a natural-language question.

Stores:
  data/embeddings/tfidf_matrix.npz    — sparse TF-IDF matrix (chunks × terms)
  data/embeddings/tfidf_vocab.json    — feature names
  data/embeddings/chunk_index.json    — ordered list of chunk_ids (row mapping)

These files are loaded at query time by retrieve_context.py and the API.

Run:
    python scripts/retrieval/build_embeddings.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import CHUNKS_DIR, EMBEDDINGS_DIR, ensure_dirs

INPUT_FILE   = CHUNKS_DIR     / "chunks.json"
MATRIX_FILE  = EMBEDDINGS_DIR / "tfidf_matrix.npz"
VOCAB_FILE   = EMBEDDINGS_DIR / "tfidf_vocab.json"
INDEX_FILE   = EMBEDDINGS_DIR / "chunk_index.json"


def main():
    ensure_dirs()

    print(f"Loading chunks from {INPUT_FILE} …")
    chunks = json.load(open(INPUT_FILE))

    # Filter out bot chunks — they only add noise to the index
    chunks = [c for c in chunks if not c.get("is_bot", False)]
    print(f"  Non-bot chunks to embed : {len(chunks)}\n")

    texts     = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    print("Fitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(
        analyzer      = "word",
        min_df        = 2,          # ignore terms appearing in < 2 chunks
        max_df        = 0.95,       # ignore near-universal terms
        sublinear_tf  = True,       # log(1+tf) to dampen high-frequency terms
        ngram_range   = (1, 2),     # unigrams + bigrams
        max_features  = 50_000,
        strip_accents = "unicode",
        token_pattern = r"(?u)\b\w[\w./-]{1,}\b"   # allow dots/slashes (file paths)
    )

    matrix = vectorizer.fit_transform(texts)

    # L2-normalise rows so cosine similarity == dot product
    matrix = normalize(matrix, norm="l2", copy=False)

    print(f"  Matrix shape     : {matrix.shape}")
    print(f"  Vocabulary size  : {len(vectorizer.get_feature_names_out())}")

    # ---- Persist ----
    sp.save_npz(str(MATRIX_FILE), matrix)

    with open(VOCAB_FILE, "w") as f:
        json.dump(vectorizer.get_feature_names_out().tolist(), f)

    with open(INDEX_FILE, "w") as f:
        json.dump(chunk_ids, f, indent=2)

    print(f"\nSaved TF-IDF matrix  → {MATRIX_FILE}")
    print(f"Saved vocabulary     → {VOCAB_FILE}")
    print(f"Saved chunk index    → {INDEX_FILE}")


if __name__ == "__main__":
    main()