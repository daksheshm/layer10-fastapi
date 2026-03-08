"""
api/main.py

FastAPI memory graph API.

Endpoints:
  GET /                      — health check
  GET /nodes                 — list all nodes (with optional type filter)
  GET /node/{entity}         — full node record + neighbour summary
  GET /memory                — outgoing claims for an entity
  GET /neighbors             — all edges touching an entity
  GET /search                — TF-IDF evidence search
  GET /context               — full context pack (evidence + claims) for a question
  GET /evidence/{claim_id}   — all evidence for a specific claim
  GET /conflicts             — list of conflicting claim pairs
  GET /stats                 — graph-level statistics

Run:
    cd ddgop/layer10-fastapi-memory
    uvicorn api.main:app --reload --port 8000
"""

import sys
import json
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parents[1]))
from config import GRAPH_DIR

# ==============================
# Load graph data at startup
# ==============================

NODES_FILE    = GRAPH_DIR / "nodes_normalized.json"
EDGES_FILE    = GRAPH_DIR / "edges_normalized.json"
CONFLICT_FILE = GRAPH_DIR / "conflict_report.json"

def _load():
    nodes = json.load(open(NODES_FILE))  if NODES_FILE.exists()    else []
    edges = json.load(open(EDGES_FILE))  if EDGES_FILE.exists()    else []
    conflicts = json.load(open(CONFLICT_FILE)) if CONFLICT_FILE.exists() else []
    return nodes, edges, conflicts

_nodes_list, _edges_list, _conflicts = _load()

# Index structures
_nodes  = {n["id"]: n for n in _nodes_list}
_edges  = _edges_list

# ==============================
# Lazy-load retrieval engine
# ==============================

_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from scripts.retrieval.retrieve_context import retrieve
        _retriever = retrieve
    return _retriever


# ==============================
# App
# ==============================

app = FastAPI(
    title="Layer10 Memory Graph API",
    description=(
        "Grounded long-term memory extracted from FastAPI GitHub issues. "
        "Every claim is traceable to evidence with source URLs and offsets."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"]
)


# ==============================
# Endpoints
# ==============================

@app.get("/")
def root():
    return {
        "message":    "Layer10 Memory Graph API",
        "nodes":      len(_nodes),
        "edges":      len(_edges),
        "docs":       "/docs"
    }


@app.get("/stats")
def stats():
    active     = [e for e in _edges if not e.get("superseded", False)]
    superseded = [e for e in _edges if e.get("superseded", False)]
    conflicted = [e for e in _edges if e.get("conflict_with")]

    type_counts = {}
    for n in _nodes.values():
        t = n.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    relation_counts = {}
    for e in active:
        r = e.get("relation", "unknown")
        relation_counts[r] = relation_counts.get(r, 0) + 1

    return {
        "total_nodes":       len(_nodes),
        "total_edges":       len(_edges),
        "active_edges":      len(active),
        "superseded_edges":  len(superseded),
        "conflicted_edges":  len(conflicted),
        "node_types":        type_counts,
        "relation_types":    relation_counts
    }


@app.get("/nodes")
def list_nodes(
    type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(100, le=1000)
):
    result = list(_nodes.values())
    if type:
        result = [n for n in result if n.get("type") == type]
    result.sort(key=lambda n: -n.get("frequency", 1))
    return {"count": len(result), "nodes": result[:limit]}


@app.get("/node/{entity}")
def get_node(entity: str):
    entity = entity.lower()
    node   = _nodes.get(entity)
    if not node:
        raise HTTPException(status_code=404, detail=f"Entity '{entity}' not found")

    outgoing = [e for e in _edges if e["subject"] == entity and not e.get("superseded")]
    incoming = [e for e in _edges if e["object"]  == entity and not e.get("superseded")]

    return {
        "node":               node,
        "outgoing_count":     len(outgoing),
        "incoming_count":     len(incoming),
        "outgoing_relations": list({e["relation"] for e in outgoing}),
        "incoming_relations": list({e["relation"] for e in incoming})
    }


@app.get("/memory")
def query_memory(
    entity: str,
    relation: Optional[str] = None,
    include_superseded: bool = False
):
    entity  = entity.lower()
    results = []

    for e in _edges:
        if e["subject"] != entity:
            continue
        if relation and e["relation"] != relation:
            continue
        if not include_superseded and e.get("superseded", False):
            continue

        ev_primary = e["evidence"][0] if e.get("evidence") else {}
        results.append({
            "claim_id":      e["edge_id"],
            "relation":      e["relation"],
            "object":        e["object"],
            "object_type":   _nodes.get(e["object"], {}).get("type", "unknown"),
            "confidence":    e.get("confidence", 0.0),
            "support_count": e.get("support_count", 1),
            "superseded":    e.get("superseded", False),
            "primary_evidence": ev_primary
        })

    results.sort(key=lambda r: (-r["support_count"], -r["confidence"]))
    return {"entity": entity, "count": len(results), "relations": results}


@app.get("/neighbors")
def neighbors(
    entity: str,
    include_superseded: bool = False
):
    entity  = entity.lower()
    results = []

    for e in _edges:
        if e["subject"] != entity and e["object"] != entity:
            continue
        if not include_superseded and e.get("superseded", False):
            continue
        results.append(e)

    results.sort(key=lambda e: (-e.get("support_count", 1), -e.get("confidence", 0)))
    return {"entity": entity, "count": len(results), "edges": results}


@app.get("/search")
def search(
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(8, le=50),
    include_superseded: bool = False
):
    """
    TF-IDF semantic search over evidence chunks.
    Returns ranked snippets with source metadata.
    """
    try:
        retrieve = get_retriever()
        pack = retrieve(q, top_k_chunks=top_k, include_superseded=include_superseded)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

    return {
        "query":    q,
        "count":    pack["total_evidence"],
        "results":  pack["evidence_snippets"]
    }


@app.get("/context")
def context(
    q: str = Query(..., description="Natural language question"),
    top_k_chunks: int = Query(8, le=50),
    top_k_claims: int = Query(5, le=20),
    include_superseded: bool = False
):
    """
    Full grounded context pack: evidence + graph claims.
    Use this to answer questions with provenance.
    """
    try:
        retrieve = get_retriever()
        pack = retrieve(
            q,
            top_k_chunks=top_k_chunks,
            top_k_claims=top_k_claims,
            include_superseded=include_superseded
        )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

    return pack


@app.get("/evidence/{claim_id}")
def get_evidence(
    claim_id: str,
    include_superseded: bool = False
):
    """All evidence supporting a specific claim (by claim_id / edge_id)."""
    for e in _edges:
        if e["edge_id"] == claim_id:
            if not include_superseded and e.get("superseded"):
                raise HTTPException(
                    status_code=410,
                    detail="This claim has been superseded by a more-supported claim"
                )
            return {
                "claim_id":      e["edge_id"],
                "statement":     f"{e['subject']} --{e['relation']}--> {e['object']}",
                "confidence":    e.get("confidence"),
                "support_count": e.get("support_count"),
                "superseded":    e.get("superseded", False),
                "conflict_with": e.get("conflict_with", []),
                "first_seen":    e.get("first_seen"),
                "last_seen":     e.get("last_seen"),
                "evidence":      e.get("evidence", [])
            }

    raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found")


@app.get("/conflicts")
def get_conflicts(limit: int = Query(50, le=500)):
    """List claim pairs with conflicting relations for the same entity pair."""
    return {
        "count":   len(_conflicts),
        "conflicts": _conflicts[:limit]
    }
