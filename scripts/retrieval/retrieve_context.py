"""
retrieve_context.py

Given a natural-language question, returns a grounded context pack:
  - Ranked evidence chunks (TF-IDF cosine similarity)
  - Graph claims linked to the found entities
  - Full provenance for every returned item

Usage:
    python scripts/retrieval/retrieve_context.py "Who reviews PRs about routing?"
    python scripts/retrieval/retrieve_context.py "What does routing.py depend on?"
"""

import sys
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    CHUNKS_DIR, EMBEDDINGS_DIR, GRAPH_DIR,
    TOP_K_CHUNKS, TOP_K_CLAIMS, ensure_dirs
)

CHUNKS_FILE  = CHUNKS_DIR     / "chunks.json"
MATRIX_FILE  = EMBEDDINGS_DIR / "tfidf_matrix.npz"
VOCAB_FILE   = EMBEDDINGS_DIR / "tfidf_vocab.json"
INDEX_FILE   = EMBEDDINGS_DIR / "chunk_index.json"
NODES_FILE   = GRAPH_DIR      / "nodes_normalized.json"
EDGES_FILE   = GRAPH_DIR      / "edges_normalized.json"


# ==============================
# Lazy-load globals (reused across calls)
# ==============================

_cache = {}


def load_resources():
    if _cache:
        return _cache

    print("  Loading retrieval index …", flush=True)

    chunks    = json.load(open(CHUNKS_FILE))
    chunk_map = {c["chunk_id"]: c for c in chunks}

    matrix     = sp.load_npz(str(MATRIX_FILE))
    vocab      = json.load(open(VOCAB_FILE))
    chunk_ids  = json.load(open(INDEX_FILE))

    nodes = json.load(open(NODES_FILE))
    edges = json.load(open(EDGES_FILE))

    node_map = {n["id"]: n for n in nodes}

    # Rebuild vectorizer vocab for query transform
    vectorizer = TfidfVectorizer(
        vocabulary     = vocab,
        analyzer       = "word",
        sublinear_tf   = True,
        ngram_range    = (1, 2),
        strip_accents  = "unicode",
        token_pattern  = r"(?u)\b\w[\w./-]{1,}\b"
    )
    # Fit on a dummy corpus to initialise (vocab already set)
    vectorizer.fit([" ".join(vocab[:100])])

    _cache.update({
        "chunk_map":  chunk_map,
        "matrix":     matrix,
        "vectorizer": vectorizer,
        "chunk_ids":  chunk_ids,
        "node_map":   node_map,
        "edges":      edges
    })

    return _cache


# ==============================
# Retrieval logic
# ==============================

def embed_query(query, vectorizer):
    """Transform a query string into a normalised TF-IDF vector."""
    vec = vectorizer.transform([query])
    return normalize(vec, norm="l2", copy=False)


def top_chunks(query_vec, matrix, chunk_ids, top_k):
    """Return (chunk_id, score) pairs ordered by cosine similarity."""
    scores = (matrix @ query_vec.T).toarray().flatten()
    idxs   = np.argsort(-scores)[:top_k * 3]   # oversample, filter below

    results = []
    for i in idxs:
        score = float(scores[i])
        if score < 1e-6:
            break
        results.append((chunk_ids[i], score))

    return results[:top_k]


def extract_entity_mentions(chunk_text, node_map):
    """
    Rough entity mention detection: check if any node id appears in the text.
    Case-insensitive substring match.  Adequate for demonstration.
    """
    text_lower = chunk_text.lower()
    mentioned  = []
    for nid in node_map:
        if len(nid) > 3 and nid in text_lower:
            mentioned.append(nid)
    return mentioned


def get_graph_claims(entity_ids, edges, top_k, include_superseded=False):
    """
    Return the most-supported claims involving any of the given entities.
    """
    entity_set = set(entity_ids)
    relevant   = []

    for edge in edges:
        if edge["subject"] in entity_set or edge["object"] in entity_set:
            if not include_superseded and edge.get("superseded", False):
                continue
            relevant.append(edge)

    # Sort by support_count desc, then confidence desc
    relevant.sort(key=lambda e: (-e.get("support_count", 1), -e.get("confidence", 0)))
    return relevant[:top_k]


def format_context_pack(question, chunk_hits, graph_claims, chunk_map, node_map):
    """
    Assemble a structured context pack dict.
    """
    evidence_snippets = []
    all_entities      = set()

    for chunk_id, score in chunk_hits:
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue

        entities_in_chunk = extract_entity_mentions(chunk["text"], node_map)
        all_entities.update(entities_in_chunk)

        evidence_snippets.append({
            "chunk_id":      chunk_id,
            "score":         round(score, 4),
            "text":          chunk["text"],
            "source_url":    chunk["source_url"],
            "author":        chunk["author"],
            "timestamp":     chunk["timestamp"],
            "issue_number":  chunk["issue_number"],
            "offset_start":  chunk["offset_start"],
            "offset_end":    chunk["offset_end"],
            "entities_found": entities_in_chunk
        })

    formatted_claims = []
    for edge in graph_claims:
        # Attach primary evidence (first item in evidence list)
        primary_ev = edge.get("evidence", [{}])[0] if edge.get("evidence") else {}
        formatted_claims.append({
            "claim_id":      edge["edge_id"],
            "statement":     f"{edge['subject']} --{edge['relation']}--> {edge['object']}",
            "subject":       edge["subject"],
            "relation":      edge["relation"],
            "object":        edge["object"],
            "confidence":    edge.get("confidence", 0.0),
            "support_count": edge.get("support_count", 1),
            "superseded":    edge.get("superseded", False),
            "conflict_with": edge.get("conflict_with", []),
            "primary_evidence": primary_ev,
            "all_evidence_count": len(edge.get("evidence", []))
        })

    return {
        "question":         question,
        "evidence_snippets": evidence_snippets,
        "graph_claims":     formatted_claims,
        "entities_found":   sorted(all_entities),
        "total_evidence":   len(evidence_snippets),
        "total_claims":     len(formatted_claims)
    }


# ==============================
# Public API
# ==============================

def retrieve(question, top_k_chunks=None, top_k_claims=None, include_superseded=False):
    """
    Main retrieval entry point.  Returns a context pack dict.
    """
    top_k_chunks = top_k_chunks or TOP_K_CHUNKS
    top_k_claims = top_k_claims or TOP_K_CLAIMS

    r = load_resources()

    query_vec = embed_query(question, r["vectorizer"])
    hits      = top_chunks(query_vec, r["matrix"], r["chunk_ids"], top_k_chunks)

    # Gather entities from all top chunks for graph expansion
    all_entities = set()
    for chunk_id, _ in hits:
        chunk = r["chunk_map"].get(chunk_id)
        if chunk:
            all_entities.update(extract_entity_mentions(chunk["text"], r["node_map"]))

    # Also expand: any entity whose name appears in the query itself
    question_lower = question.lower()
    for nid in r["node_map"]:
        if len(nid) > 3 and nid in question_lower:
            all_entities.add(nid)

    claims = get_graph_claims(all_entities, r["edges"], top_k_claims, include_superseded)

    return format_context_pack(question, hits, claims, r["chunk_map"], r["node_map"])


# ==============================
# CLI
# ==============================

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieve_context.py <question>")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nQuery: {question}\n")

    pack = retrieve(question)

    print(f"Evidence snippets : {pack['total_evidence']}")
    print(f"Graph claims      : {pack['total_claims']}")
    print(f"Entities found    : {pack['entities_found']}\n")

    print("─" * 60)
    print("TOP EVIDENCE CHUNKS")
    print("─" * 60)
    for i, ev in enumerate(pack["evidence_snippets"], 1):
        print(f"\n[{i}] score={ev['score']:.4f}  issue={ev['issue_number']}  author={ev['author']}")
        print(f"    {ev['source_url']}")
        snippet = ev["text"][:250].replace("\n", " ")
        print(f"    \"{snippet}…\"")

    print("\n" + "─" * 60)
    print("GRAPH CLAIMS")
    print("─" * 60)
    for i, c in enumerate(pack["graph_claims"], 1):
        sup = " [SUPERSEDED]" if c["superseded"] else ""
        print(f"\n[{i}] conf={c['confidence']:.2f} support={c['support_count']}{sup}")
        print(f"    {c['statement']}")
        ev = c.get("primary_evidence", {})
        if ev:
            print(f"    Evidence: \"{ev.get('excerpt','')[:120]}\"")
            print(f"    Source  : {ev.get('source_url','')}")


if __name__ == "__main__":
    main()