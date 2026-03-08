"""
graph_search.py

CLI tool to explore the memory graph by entity name.

Supports:
  - Direct relation lookup (subject → *)
  - Reverse lookup (* → object)
  - Neighbourhood expansion to depth N
  - Filtering by relation type

Usage:
    python scripts/retrieval/graph_search.py <entity>
    python scripts/retrieval/graph_search.py <entity> --depth 2
    python scripts/retrieval/graph_search.py <entity> --relation fixes
    python scripts/retrieval/graph_search.py <entity> --reverse
"""

import sys
import json
import argparse
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import GRAPH_DIR

NODES_FILE = GRAPH_DIR / "nodes_normalized.json"
EDGES_FILE = GRAPH_DIR / "edges_normalized.json"


# ==============================
# Graph loading
# ==============================

def load_graph():
    nodes = {n["id"]: n for n in json.load(open(NODES_FILE))}
    edges = json.load(open(EDGES_FILE))
    return nodes, edges


# ==============================
# Search functions
# ==============================

def find_outgoing(entity, edges, relation_filter=None):
    """All edges where entity is the subject."""
    return [
        e for e in edges
        if e["subject"] == entity
        and (relation_filter is None or e["relation"] == relation_filter)
        and not e.get("superseded", False)
    ]


def find_incoming(entity, edges, relation_filter=None):
    """All edges where entity is the object."""
    return [
        e for e in edges
        if e["object"] == entity
        and (relation_filter is None or e["relation"] == relation_filter)
        and not e.get("superseded", False)
    ]


def bfs_neighbourhood(start, edges, max_depth=2, relation_filter=None):
    """
    BFS expansion from `start` up to `max_depth` hops.
    Returns a list of (depth, edge) pairs.
    """
    visited  = {start}
    queue    = deque([(start, 0)])
    results  = []

    while queue:
        node, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for edge in find_outgoing(node, edges, relation_filter):
            neighbour = edge["object"]
            results.append((depth + 1, edge))
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))

        for edge in find_incoming(node, edges, relation_filter):
            neighbour = edge["subject"]
            results.append((depth + 1, edge))
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, depth + 1))

    return results


def print_edge(edge, nodes):
    """Pretty-print a single edge with primary evidence."""
    sub  = edge["subject"]
    rel  = edge["relation"]
    obj  = edge["object"]
    conf = edge.get("confidence", 0.0)
    sup  = edge.get("support_count", 1)
    superseded = edge.get("superseded", False)
    flag = " [SUPERSEDED]" if superseded else ""

    sub_type = nodes.get(sub, {}).get("type", "?")
    obj_type = nodes.get(obj, {}).get("type", "?")

    print(f"  [{sub_type}] {sub}  --{rel}-->  [{obj_type}] {obj}")
    print(f"  conf={conf:.2f}  support={sup}{flag}")

    evidence_list = edge.get("evidence", [])
    if evidence_list:
        ev = evidence_list[0]
        excerpt = ev.get("excerpt", "").replace("\n", " ")[:120]
        print(f"  ↳ \"{excerpt}\"")
        print(f"    {ev.get('source_url', '')}")
    print()


# ==============================
# CLI
# ==============================

def main():
    parser = argparse.ArgumentParser(description="Explore the memory graph by entity")
    parser.add_argument("entity",   help="Entity name to look up (case-insensitive)")
    parser.add_argument("--depth",    type=int, default=1, help="BFS expansion depth (default: 1)")
    parser.add_argument("--relation", default=None, help="Filter by relation type")
    parser.add_argument("--reverse",  action="store_true", help="Show incoming edges too")
    args = parser.parse_args()

    entity  = args.entity.lower().strip()
    nodes, edges = load_graph()

    if entity not in nodes:
        # Fuzzy hint
        similar = [nid for nid in nodes if entity[:5] in nid][:5]
        print(f"\nEntity '{entity}' not found in graph.")
        if similar:
            print(f"Did you mean: {similar}")
        return

    node_info = nodes[entity]
    print(f"\nEntity : {entity}  (type={node_info.get('type','?')})")
    if node_info.get("aliases"):
        print(f"Aliases: {node_info['aliases']}")
    print(f"Freq   : {node_info.get('frequency', 1)}")
    print(f"Seen   : {node_info.get('first_seen','?')} → {node_info.get('last_seen','?')}")

    if args.depth > 1:
        print(f"\n{'─'*60}")
        print(f"BFS neighbourhood (depth={args.depth})")
        print("─" * 60)
        hits = bfs_neighbourhood(entity, edges, args.depth, args.relation)
        if not hits:
            print("  (no relations found)")
        for depth, edge in hits:
            print(f"\n  depth={depth}")
            print_edge(edge, nodes)
    else:
        print(f"\n{'─'*60}")
        print("Outgoing claims")
        print("─" * 60)
        out = find_outgoing(entity, edges, args.relation)
        if not out:
            print("  (none)")
        for edge in out:
            print_edge(edge, nodes)

        if args.reverse:
            print("─" * 60)
            print("Incoming claims")
            print("─" * 60)
            inc = find_incoming(entity, edges, args.relation)
            if not inc:
                print("  (none)")
            for edge in inc:
                print_edge(edge, nodes)


if __name__ == "__main__":
    main()