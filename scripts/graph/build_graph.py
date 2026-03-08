"""
build_graph.py

Assembles the final memory graph from deduplicated claims and the entity
alias index.

Graph structure:
  nodes.json  — entity nodes with type, aliases, frequency, first/last seen
  edges.json  — claim edges with evidence list, confidence, temporal metadata

Produces:
  data/graph/nodes.json
  data/graph/edges.json

Run:
    python scripts/graph/build_graph.py
"""

import sys
import json
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    GRAPH_DIR, RESOLVED_DIR, ensure_dirs
)

CLAIMS_FILE   = GRAPH_DIR    / "claims_deduped.json"
ALIAS_FILE    = RESOLVED_DIR / "alias_index.json"
OUTPUT_NODES  = GRAPH_DIR    / "nodes.json"
OUTPUT_EDGES  = GRAPH_DIR    / "edges.json"


# ==============================
# Helpers
# ==============================

def safe_min(values):
    filtered = [v for v in values if v]
    return min(filtered) if filtered else None


def safe_max(values):
    filtered = [v for v in values if v]
    return max(filtered) if filtered else None


# ==============================
# Main pipeline
# ==============================

def main():
    ensure_dirs()

    print("Loading deduplicated claims …")
    claims = json.load(open(CLAIMS_FILE))
    print(f"  Loaded {len(claims)} claims\n")

    print("Loading alias index …")
    alias_index = json.load(open(ALIAS_FILE))

    # ---- Collect all node names referenced in claims ----
    node_names = set()
    for claim in claims:
        node_names.add(claim["subject"])
        node_names.add(claim["object"])

    # ---- Build node records ----
    # Compute per-node temporal range from claims
    node_first_seen = defaultdict(list)
    node_last_seen  = defaultdict(list)

    for claim in claims:
        for name in [claim["subject"], claim["object"]]:
            node_first_seen[name].append(claim["first_seen"])
            node_last_seen[name].append(claim["last_seen"])

    nodes = []

    for name in sorted(node_names):
        alias_info = alias_index.get(name, {})
        node = {
            "id":          name,
            "label":       name,
            "type":        alias_info.get("type", "unknown"),
            "aliases":     alias_info.get("aliases", []),
            "frequency":   alias_info.get("frequency", 1),
            "first_seen":  safe_min(node_first_seen[name]),
            "last_seen":   safe_max(node_last_seen[name])
        }
        nodes.append(node)

    # ---- Build edge records from claims ----
    edges = []

    for claim in claims:
        edge = {
            "edge_id":      claim["claim_id"],
            "subject":      claim["subject"],
            "relation":     claim["relation"],
            "object":       claim["object"],
            "confidence":   claim["confidence"],
            "support_count": claim["support_count"],
            "evidence":     claim["evidence"],
            "first_seen":   claim["first_seen"],
            "last_seen":    claim["last_seen"],
            "superseded":   claim["superseded"],
            "conflict_with": claim.get("conflict_with", [])
        }
        edges.append(edge)

    # ---- Write outputs ----
    with open(OUTPUT_NODES, "w") as f:
        json.dump(nodes, f, indent=2)

    with open(OUTPUT_EDGES, "w") as f:
        json.dump(edges, f, indent=2)

    active_edges = sum(1 for e in edges if not e["superseded"])

    print(f"  Nodes            : {len(nodes)}")
    print(f"  Edges (total)    : {len(edges)}")
    print(f"  Edges (active)   : {active_edges}")
    print(f"  Edges (superseded): {len(edges) - active_edges}")
    print(f"\nSaved nodes → {OUTPUT_NODES}")
    print(f"Saved edges → {OUTPUT_EDGES}")


if __name__ == "__main__":
    main()