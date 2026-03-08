"""
graph_viewer.py

Generates a self-contained interactive HTML graph visualisation using pyvis.

Features:
  - Node colour and size encoded by type and frequency
  - Edge thickness encoded by support_count
  - Superseded edges shown as dashed grey
  - Hovering a node shows type, aliases, first/last seen
  - Hovering an edge shows the claim statement + primary evidence excerpt
  - Optional filters: --type, --relation, --min-support, --limit

Produces:
  visualization/memory_graph.html  (open in any browser)

Run:
    python visualization/graph_viewer.py
    python visualization/graph_viewer.py --limit 150 --relation fixes
    python visualization/graph_viewer.py --type Person
"""

import sys
import json
import argparse
from pathlib import Path
import pyvis.network

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GRAPH_DIR

try:
    from pyvis.network import Network
except ImportError:
    print("pyvis not installed.  Run: pip install pyvis")
    sys.exit(1)

NODES_FILE  = GRAPH_DIR / "nodes_normalized.json"
EDGES_FILE  = GRAPH_DIR / "edges_normalized.json"
OUTPUT_FILE = Path(__file__).parent / "memory_graph.html"


# ==============================
# Colour palette per entity type
# ==============================

TYPE_COLORS = {
    "Person":      "#4A90D9",   # blue
    "Component":   "#E67E22",   # orange
    "Feature":     "#2ECC71",   # green
    "Bug":         "#E74C3C",   # red
    "PullRequest": "#9B59B6",   # purple
    "Issue":       "#E91E63",   # pink
    "Library":     "#1ABC9C",   # teal
    "Concept":     "#F1C40F",   # yellow
    "Config":      "#95A5A6",   # grey
    "Test":        "#3498DB",   # light blue
    "unknown":     "#BDC3C7",   # light grey
}

RELATION_COLORS = {
    "fixes":        "#E74C3C",
    "closes":       "#E67E22",
    "depends_on":   "#3498DB",
    "uses":         "#2ECC71",
    "defines":      "#9B59B6",
    "imports":      "#1ABC9C",
    "reviews":      "#F39C12",
    "authors":      "#4A90D9",
    "supersedes":   "#95A5A6",
    "tests":        "#27AE60",
    "configures":   "#E91E63",
    "requires":     "#E74C3C",
    "supports":     "#2ECC71",
    "mentions":     "#BDC3C7",
    "relates_to":   "#BDC3C7",
}


def freq_to_size(freq, min_size=15, max_size=50):
    """Map a frequency value to a node display size."""
    try:
        freq = int(freq)
    except (TypeError, ValueError):
        freq = 1
    return min(max_size, min_size + freq * 2)


def support_to_width(support, min_w=1, max_w=10):
    try:
        support = int(support)
    except (TypeError, ValueError):
        support = 1
    return min(max_w, min_w + support)


# ==============================
# Main builder
# ==============================

def build_graph(
    type_filter=None,
    relation_filter=None,
    min_support=1,
    node_limit=300
):
    nodes = json.load(open(NODES_FILE))
    edges = json.load(open(EDGES_FILE))

    # Apply edge filters
    if relation_filter:
        edges = [e for e in edges if e.get("relation") == relation_filter]
    edges = [e for e in edges if e.get("support_count", 1) >= min_support]

    # Identify nodes referenced by filtered edges
    referenced_nodes = set()
    for e in edges:
        referenced_nodes.add(e["subject"])
        referenced_nodes.add(e["object"])

    # Apply node type filter
    if type_filter:
        nodes = [n for n in nodes if n.get("type") == type_filter]
        valid_ids = {n["id"] for n in nodes}
        # Keep edges where BOTH endpoints match (if type-filtering)
        edges = [e for e in edges if e["subject"] in valid_ids and e["object"] in valid_ids]
        referenced_nodes = {e["subject"] for e in edges} | {e["object"] for e in edges}

    # Keep only nodes that appear in edges
    nodes = [n for n in nodes if n["id"] in referenced_nodes]

    # Sort by frequency desc and apply limit
    nodes.sort(key=lambda n: -n.get("frequency", 1))
    nodes = nodes[:node_limit]
    node_ids = {n["id"] for n in nodes}

    # Only keep edges where both nodes survived the limit
    edges = [e for e in edges if e["subject"] in node_ids and e["object"] in node_ids]

    return nodes, edges


def render_html(nodes, edges):
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#eee",
        directed=True
    )

    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 140,
          "springConstant": 0.04
        },
        "stabilization": {"iterations": 150}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
        "smooth": {"type": "dynamic"}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    for node in nodes:
        nid   = node["id"]
        ntype = node.get("type", "unknown")
        color = TYPE_COLORS.get(ntype, TYPE_COLORS["unknown"])
        size  = freq_to_size(node.get("frequency", 1))

        aliases_str = ", ".join(node.get("aliases", [])) or "—"
        tooltip = (
            f"<b>{nid}</b><br>"
            f"Type: {ntype}<br>"
            f"Frequency: {node.get('frequency', 1)}<br>"
            f"Aliases: {aliases_str}<br>"
            f"First seen: {node.get('first_seen', '?')}<br>"
            f"Last seen: {node.get('last_seen', '?')}"
        )

        net.add_node(
            nid,
            label=nid[:30],
            title=tooltip,
            color=color,
            size=size
        )

    for edge in edges:
        sub       = edge["subject"]
        obj       = edge["object"]
        rel       = edge.get("relation", "")
        conf      = edge.get("confidence", 0.0)
        support   = edge.get("support_count", 1)
        superseded = edge.get("superseded", False)

        color = "#555" if superseded else RELATION_COLORS.get(rel, "#aaa")
        width = support_to_width(support)
        dashes = superseded

        primary_ev = (edge.get("evidence") or [{}])[0]
        excerpt    = primary_ev.get("excerpt", "")[:150]
        src_url    = primary_ev.get("source_url", "")

        tooltip = (
            f"<b>{sub} --{rel}→ {obj}</b><br>"
            f"Confidence: {conf:.2f}<br>"
            f"Support: {support} chunk(s)<br>"
            f"Superseded: {superseded}<br>"
            f"Evidence: {excerpt}<br>"
            f"Source: {src_url}"
        )

        net.add_edge(
            sub, obj,
            title=tooltip,
            label=rel,
            color=color,
            width=width,
            dashes=dashes
        )

    return net


def main():
    parser = argparse.ArgumentParser(description="Generate interactive memory graph HTML")
    parser.add_argument("--type",        default=None, help="Filter nodes by entity type")
    parser.add_argument("--relation",    default=None, help="Filter edges by relation type")
    parser.add_argument("--min-support", type=int, default=1, help="Min support_count for edges")
    parser.add_argument("--limit",       type=int, default=300, help="Max nodes to render")
    parser.add_argument("--output",      default=str(OUTPUT_FILE), help="Output HTML file path")
    args = parser.parse_args()

    print("Building graph …")
    nodes, edges = build_graph(
        type_filter=args.type,
        relation_filter=args.relation,
        min_support=args.min_support,
        node_limit=args.limit
    )
    print(f"  Nodes: {len(nodes)}  Edges: {len(edges)}")

    net = render_html(nodes, edges)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))

    print(f"\nVisualization saved → {output_path}")
    print("Open it in your browser: file://" + str(output_path.resolve()))


if __name__ == "__main__":
    main()