"""
deduplicate_claims.py

Merges duplicate and near-duplicate claims produced by extraction.

Deduplication strategy:
  1. Apply the canonical entity map so claims reference stable node IDs.
  2. Group claims by (canonical_subject, relation, canonical_object).
  3. Within each group, merge evidence from all supporting chunks into a
     single "deduplicated" claim with a multi-source evidence list.
  4. If conflicting claims exist (same subject/object, different relation),
     store them as separate claim variants with a "conflict" flag.
  5. Historical claims (where newer evidence contradicts older) are kept
     with a "superseded" flag rather than deleted.

Produces:
  data/graph/claims_deduped.json      — one record per canonical claim
  data/graph/conflict_report.json     — pairs of conflicting claims

Run:
    python scripts/deduplication/deduplicate_claims.py
"""

import sys
import json
import hashlib
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    EXTRACTIONS_DIR, GRAPH_DIR, RESOLVED_DIR,
    CLAIM_SIM_THRESHOLD, ensure_dirs
)

INPUT_EXTRACTIONS = EXTRACTIONS_DIR / "validated_extractions.json"
CANONICAL_MAP_FILE = RESOLVED_DIR   / "canonical_map.json"
OUTPUT_CLAIMS     = GRAPH_DIR       / "claims_deduped.json"
CONFLICT_REPORT   = GRAPH_DIR       / "conflict_report.json"


# ==============================
# Helpers
# ==============================

def claim_id(subject, relation, obj):
    """Deterministic ID for a canonical (subject, relation, object) triple."""
    key = f"{subject.lower()}|{relation.lower()}|{obj.lower()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def evidence_sim(ev_a, ev_b):
    """String similarity between two evidence excerpts."""
    return fuzz.token_set_ratio(ev_a, ev_b) / 100.0


# ==============================
# Main pipeline
# ==============================

def main():
    ensure_dirs()

    print(f"Loading validated extractions …")
    records = json.load(open(INPUT_EXTRACTIONS))

    print(f"Loading canonical map …")
    canon_map = json.load(open(CANONICAL_MAP_FILE))

    # ---- Collect all raw claims ----
    raw_claims = []

    for rec in records:
        for claim in rec["extraction"].get("claims", []):
            subject  = canon_map.get(claim["subject"],  claim["subject"])
            obj      = canon_map.get(claim["object"],   claim["object"])
            relation = claim["relation"]

            raw_claims.append({
                "subject":          subject,
                "relation":         relation,
                "object":           obj,
                "confidence":       claim.get("confidence", 0.7),
                "evidence_excerpt": claim.get("evidence_excerpt", ""),
                "source_url":       rec.get("source_url", ""),
                "artifact_id":      rec.get("artifact_id", ""),
                "chunk_id":         rec.get("chunk_id", ""),
                "issue_number":     rec.get("issue_number"),
                "author":           rec.get("author"),
                "timestamp":        rec.get("timestamp"),
                "offset_start":     rec.get("offset_start", 0),
                "offset_end":       rec.get("offset_end",   0)
            })

    print(f"  Raw claims loaded : {len(raw_claims)}\n")

    # ---- Group by (subject, relation, object) ----
    groups = defaultdict(list)
    for c in raw_claims:
        key = (c["subject"].lower(), c["relation"], c["object"].lower())
        groups[key].append(c)

    # ---- Merge within each group ----
    deduped_claims = []
    conflict_pairs = []

    for (subject, relation, obj), group in groups.items():
        # Sort by timestamp so we know which evidence is newest
        group.sort(key=lambda x: x.get("timestamp") or "")

        # Deduplicate evidence by similarity
        unique_evidence = []
        for c in group:
            ev = c["evidence_excerpt"]
            is_dup = any(
                evidence_sim(ev, ue["excerpt"]) >= CLAIM_SIM_THRESHOLD
                for ue in unique_evidence
            )
            if not is_dup:
                unique_evidence.append({
                    "excerpt":       ev,
                    "source_url":    c["source_url"],
                    "artifact_id":   c["artifact_id"],
                    "chunk_id":      c["chunk_id"],
                    "issue_number":  c["issue_number"],
                    "author":        c["author"],
                    "timestamp":     c["timestamp"],
                    "offset_start":  c["offset_start"],
                    "offset_end":    c["offset_end"]
                })

        # Aggregate confidence: mean of all confidences in group
        avg_conf = round(sum(c["confidence"] for c in group) / len(group), 4)

        cid = claim_id(subject, relation, obj)

        deduped_claims.append({
            "claim_id":         cid,
            "subject":          subject,
            "relation":         relation,
            "object":           obj,
            "confidence":       avg_conf,
            "support_count":    len(group),
            "evidence":         unique_evidence,
            "first_seen":       group[0].get("timestamp"),
            "last_seen":        group[-1].get("timestamp"),
            "superseded":       False,   # updated below if conflict detected
            "conflict_with":    []
        })

    # ---- Detect conflicts ----
    # A conflict is: same (subject, object) pair but different relations
    # or same (subject, relation) but contradictory objects.
    # We use a simple approach: flag pairs where same subject+object appear
    # with different relations.

    by_subject_object = defaultdict(list)
    for claim in deduped_claims:
        key = (claim["subject"], claim["object"])
        by_subject_object[key].append(claim)

    for key, claims in by_subject_object.items():
        if len(claims) > 1:
            relations = [c["relation"] for c in claims]
            for i, ca in enumerate(claims):
                for cb in claims[i+1:]:
                    if ca["relation"] != cb["relation"]:
                        conflict_pairs.append({
                            "claim_a": ca["claim_id"],
                            "claim_b": cb["claim_id"],
                            "subject": ca["subject"],
                            "object":  ca["object"],
                            "relation_a": ca["relation"],
                            "relation_b": cb["relation"],
                            "note": "same_subject_object_different_relation"
                        })
                        ca["conflict_with"].append(cb["claim_id"])
                        cb["conflict_with"].append(ca["claim_id"])

    # Mark low-support claims superseded by high-support ones
    for claims in by_subject_object.values():
        if len(claims) > 1:
            best = max(claims, key=lambda c: (c["support_count"], c["confidence"]))
            for c in claims:
                if c["claim_id"] != best["claim_id"] and c["support_count"] < best["support_count"]:
                    c["superseded"] = True

    with open(OUTPUT_CLAIMS, "w") as f:
        json.dump(deduped_claims, f, indent=2)

    with open(CONFLICT_REPORT, "w") as f:
        json.dump(conflict_pairs, f, indent=2)

    superseded_count = sum(1 for c in deduped_claims if c["superseded"])

    print(f"  Raw claims        : {len(raw_claims)}")
    print(f"  Deduped claims    : {len(deduped_claims)}")
    print(f"  Superseded        : {superseded_count}")
    print(f"  Conflicts found   : {len(conflict_pairs)}")
    print(f"\nSaved claims_deduped  → {OUTPUT_CLAIMS}")
    print(f"Saved conflict_report → {CONFLICT_REPORT}")


if __name__ == "__main__":
    main()