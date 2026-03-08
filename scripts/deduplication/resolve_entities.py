"""
resolve_entities.py

Canonicalises entity names extracted from the corpus by:
  1. Normalising name strings (lowercase, strip punctuation).
  2. Clustering near-duplicate names using Jaccard similarity over character
     trigrams (via rapidfuzz).
  3. Choosing the most-frequent / longest form as the canonical name.
  4. Writing a canonical_map.json that maps every raw name → canonical name,
     plus an alias_index.json listing aliases per canonical.

All merges are logged so they can be audited and reversed.

Produces:
  data/resolved/canonical_map.json
  data/resolved/alias_index.json
  data/resolved/merge_log.json

Run:
    python scripts/deduplication/resolve_entities.py
"""

import sys
import json
import re
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    EXTRACTIONS_DIR, RESOLVED_DIR,
    ENTITY_SIM_THRESHOLD, ensure_dirs
)

INPUT_FILE      = EXTRACTIONS_DIR / "validated_extractions.json"
CANONICAL_MAP   = RESOLVED_DIR / "canonical_map.json"
ALIAS_INDEX     = RESOLVED_DIR / "alias_index.json"
MERGE_LOG       = RESOLVED_DIR / "merge_log.json"


# ==============================
# Normalisation
# ==============================

def normalize_name(name):
    """
    Produce a stable comparison key from an entity name.
    Does NOT modify the display name.
    """
    # Unicode → ASCII
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")

    name = name.lower().strip()

    # Strip backticks, quotes
    name = name.strip("`'\"`")

    # Collapse whitespace and common separators to single space
    name = re.sub(r'[\s\-_/\\.]+', ' ', name)

    # Remove common noise words
    noise = {"the", "a", "an", "of", "for", "in", "on", "at", "to", "by"}
    tokens = [t for t in name.split() if t not in noise]

    return " ".join(tokens).strip()


# ==============================
# Similarity
# ==============================

def sim(a, b):
    """Normalised token-set similarity (0–1)."""
    return fuzz.token_set_ratio(a, b) / 100.0


# ==============================
# Union-find for clustering
# ==============================

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self):
        groups = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return list(groups.values())


# ==============================
# Main pipeline
# ==============================

def main():
    ensure_dirs()

    print(f"Loading validated extractions from {INPUT_FILE} …")
    records = json.load(open(INPUT_FILE))

    # ---- Collect all entity names + their types and frequency ----
    name_freq  = Counter()      # raw name → occurrence count
    name_type  = {}             # raw name → most-common type

    type_votes = defaultdict(Counter)   # raw name → {type: count}

    for rec in records:
        for entity in rec["extraction"].get("entities", []):
            name = entity["name"].strip()
            if not name:
                continue
            name_freq[name]              += 1
            type_votes[name][entity["type"]] += 1

    for name in name_freq:
        name_type[name] = type_votes[name].most_common(1)[0][0]

    all_names = list(name_freq.keys())
    print(f"  Unique raw entity names : {len(all_names)}\n")

    # ---- Build normalised → raw mapping ----
    norm_to_raws = defaultdict(list)
    for name in all_names:
        norm_to_raws[normalize_name(name)].append(name)

    # ---- Cluster using similarity on normalised keys ----
    uf         = UnionFind(all_names)
    merge_log  = []

    norm_keys  = list(norm_to_raws.keys())

    # O(n²) — acceptable for typical corpus sizes (<5 000 unique entities)
    for i, nk_a in enumerate(norm_keys):
        for nk_b in norm_keys[i+1:]:
            s = sim(nk_a, nk_b)
            if s >= ENTITY_SIM_THRESHOLD:
                for ra in norm_to_raws[nk_a]:
                    for rb in norm_to_raws[nk_b]:
                        if ra != rb and uf.find(ra) != uf.find(rb):
                            uf.union(ra, rb)
                            merge_log.append({
                                "name_a":     ra,
                                "name_b":     rb,
                                "similarity": round(s, 4),
                                "reason":     "fuzzy_token_set"
                            })

    print(f"  Merges performed : {len(merge_log)}")

    # ---- Choose canonical name per cluster ----
    # Rule: most frequent; tie-break → longest; tie-break → alphabetical
    clusters    = uf.clusters()
    canon_map   = {}     # raw name → canonical name
    alias_index = {}     # canonical name → {aliases, type}

    for cluster in clusters:
        if len(cluster) == 1:
            name = cluster[0]
            canon_map[name] = name
            alias_index[name] = {
                "aliases":    [],
                "type":       name_type.get(name, "unknown"),
                "frequency":  name_freq[name]
            }
        else:
            # Pick canonical
            canonical = max(
                cluster,
                key=lambda n: (name_freq[n], len(n), n)
            )
            for name in cluster:
                canon_map[name] = canonical

            aliases = [n for n in cluster if n != canonical]
            alias_index[canonical] = {
                "aliases":   aliases,
                "type":      name_type.get(canonical, "unknown"),
                "frequency": sum(name_freq[n] for n in cluster)
            }

    with open(CANONICAL_MAP, "w") as f:
        json.dump(canon_map, f, indent=2, sort_keys=True)

    with open(ALIAS_INDEX, "w") as f:
        json.dump(alias_index, f, indent=2, sort_keys=True)

    with open(MERGE_LOG, "w") as f:
        json.dump(merge_log, f, indent=2)

    print(f"  Canonical entities : {len(alias_index)}")
    print(f"  Raw names mapped   : {len(canon_map)}")
    print(f"\nSaved canonical_map → {CANONICAL_MAP}")
    print(f"Saved alias_index   → {ALIAS_INDEX}")
    print(f"Saved merge_log     → {MERGE_LOG}")


if __name__ == "__main__":
    main()