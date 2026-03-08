"""
validate_extractions.py

Validates raw extraction output against extraction_schema.json, repairs
common structural issues, and filters out low-confidence claims.

Writes a cleaned copy to data/extractions/validated_extractions.json and
a report to data/extractions/validation_report.json.

Run:
    python scripts/extraction/validate_extractions.py
"""

import sys
import json
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    EXTRACTIONS_DIR, SCHEMA_DIR,
    EXTRACTION_MIN_CONFIDENCE, ensure_dirs
)

INPUT_FILE       = EXTRACTIONS_DIR / "raw_extractions.json"
OUTPUT_FILE      = EXTRACTIONS_DIR / "validated_extractions.json"
REPORT_FILE      = EXTRACTIONS_DIR / "validation_report.json"
SCHEMA_FILE      = SCHEMA_DIR      / "extraction_schema.json"

VALID_RELATIONS  = {
    "fixes", "closes", "depends_on", "uses", "defines", "imports",
    "reviews", "authors", "supersedes", "tests", "configures",
    "requires", "supports", "mentions", "relates_to"
}

VALID_ENTITY_TYPES = {
    "Person", "Component", "Feature", "Bug", "PullRequest", "Issue",
    "Library", "Concept", "Config", "Test", "unknown"
}


# ==============================
# Repair helpers
# ==============================

def repair_entity(e):
    """Coerce an entity dict into a valid shape."""
    if not isinstance(e, dict):
        return None
    name = str(e.get("name", "")).strip()
    if not name:
        return None
    etype = e.get("type", "unknown")
    if etype not in VALID_ENTITY_TYPES:
        etype = "unknown"
    return {"name": name, "type": etype}


def repair_claim(c, entity_names):
    """Coerce a claim dict into a valid shape; return None to discard."""
    if not isinstance(c, dict):
        return None

    subject  = str(c.get("subject",  "")).strip()
    relation = str(c.get("relation", "")).strip().lower()
    obj      = str(c.get("object",   "")).strip()
    evidence = str(c.get("evidence_excerpt", "")).strip()

    # Hard discard conditions
    if not subject or not relation or not obj:
        return None
    if subject == obj:
        return None
    if relation not in VALID_RELATIONS:
        # Try fuzzy repair: if the relation contains a known word
        for vr in VALID_RELATIONS:
            if vr in relation:
                relation = vr
                break
        else:
            return None

    # Evidence is required — if missing we synthesise a placeholder
    if not evidence:
        evidence = f"{subject} {relation} {obj}"

    # Confidence
    conf = c.get("confidence", 0.7)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.7
    conf = max(0.0, min(1.0, conf))

    if conf < EXTRACTION_MIN_CONFIDENCE:
        return None

    return {
        "subject":          subject,
        "relation":         relation,
        "object":           obj,
        "evidence_excerpt": evidence[:500],
        "confidence":       round(conf, 4),
        "subject_type":     c.get("subject_type", "unknown"),
        "object_type":      c.get("object_type", "unknown"),
        "negated":          bool(c.get("negated", False))
    }


# ==============================
# Validate & repair a single extraction
# ==============================

def validate_and_repair(raw_extraction, schema):
    """
    Attempt to validate raw extraction dict.
    Returns (repaired_extraction, issues_list).
    """
    issues = []

    if not isinstance(raw_extraction, dict):
        return {"entities": [], "claims": []}, ["extraction_not_dict"]

    # Repair entities
    raw_entities    = raw_extraction.get("entities", [])
    repaired_entities = []
    entity_names    = set()

    for e in raw_entities:
        fixed = repair_entity(e)
        if fixed:
            repaired_entities.append(fixed)
            entity_names.add(fixed["name"])
        else:
            issues.append(f"dropped_entity:{e}")

    # Repair claims
    raw_claims      = raw_extraction.get("claims", [])
    repaired_claims = []

    for c in raw_claims:
        fixed = repair_claim(c, entity_names)
        if fixed:
            # Auto-add missing entity nodes referenced in claims
            for name, etype_key in [(fixed["subject"], "subject_type"), (fixed["object"], "object_type")]:
                if name not in entity_names:
                    repaired_entities.append({"name": name, "type": fixed.get(etype_key, "unknown")})
                    entity_names.add(name)
                    issues.append(f"auto_added_entity:{name}")
            repaired_claims.append(fixed)
        else:
            issues.append(f"dropped_claim:{c.get('subject','?')}-{c.get('relation','?')}-{c.get('object','?')}")

    repaired = {"entities": repaired_entities, "claims": repaired_claims}

    # Final schema validation
    try:
        jsonschema.validate(instance=repaired, schema=schema)
    except jsonschema.ValidationError as ve:
        issues.append(f"schema_error:{ve.message[:120]}")

    return repaired, issues


# ==============================
# Main pipeline
# ==============================

def main():
    ensure_dirs()

    print(f"Loading raw extractions from {INPUT_FILE} …")
    records = json.load(open(INPUT_FILE))
    schema  = json.load(open(SCHEMA_FILE))
    print(f"  Loaded {len(records)} extraction records\n")

    validated    = []
    report_rows  = []

    total_entities_before = 0
    total_claims_before   = 0
    total_entities_after  = 0
    total_claims_after    = 0

    for rec in records:
        raw_ext = rec.get("extraction", {"entities": [], "claims": []})

        eb = len(raw_ext.get("entities", []))
        cb = len(raw_ext.get("claims",   []))
        total_entities_before += eb
        total_claims_before   += cb

        repaired, issues = validate_and_repair(raw_ext, schema)

        ea = len(repaired["entities"])
        ca = len(repaired["claims"])
        total_entities_after += ea
        total_claims_after   += ca

        new_rec = {k: v for k, v in rec.items() if k != "extraction"}
        new_rec["extraction"]        = repaired
        new_rec["validation_issues"] = issues

        validated.append(new_rec)

        report_rows.append({
            "chunk_id":       rec.get("chunk_id"),
            "entities_before": eb, "entities_after": ea,
            "claims_before":  cb,  "claims_after":   ca,
            "issues":         issues
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(validated, f, indent=2)

    report = {
        "total_records":          len(records),
        "total_entities_before":  total_entities_before,
        "total_entities_after":   total_entities_after,
        "total_claims_before":    total_claims_before,
        "total_claims_after":     total_claims_after,
        "entity_retention_pct":   round(100 * total_entities_after / max(total_entities_before, 1), 1),
        "claim_retention_pct":    round(100 * total_claims_after   / max(total_claims_before,   1), 1),
        "rows":                   report_rows
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Entities : {total_entities_before} → {total_entities_after} ({report['entity_retention_pct']}% retained)")
    print(f"  Claims   : {total_claims_before}   → {total_claims_after}   ({report['claim_retention_pct']}% retained)")
    print(f"\nSaved validated extractions → {OUTPUT_FILE}")
    print(f"Saved validation report     → {REPORT_FILE}")


if __name__ == "__main__":
    main()