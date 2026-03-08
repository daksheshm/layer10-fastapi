# Layer10 Take-Home — Grounded Long-Term Memory via Structured Extraction

A pipeline that converts GitHub issue and pull-request discussions into a **queryable, grounded memory graph** with strong evidence provenance, deduplication, and an interactive visualization layer.

The system demonstrates how scattered technical discussions can be transformed into **structured long-term memory** while preserving traceability back to the original source text.

---

# Corpus

**Source:** `fastapi/fastapi` GitHub repository

**Artifacts:** closed issues, pull requests, comments, and review events

Approximate dataset size:

| Item            | Count     |
| --------------- | --------- |
| Issues / PRs    | ~100      |
| Comments        | ~400–900  |
| Total artifacts | ~500–1000 |

Why this corpus:

* rich technical discussions
* decisions and reversals
* cross-references between files/components
* multiple maintainers and reviewers
* identity resolution challenges (aliases, bots)

These properties make the dataset suitable for testing **long-term knowledge extraction**.

---

# Data Ingestion

Artifacts are fetched from the GitHub REST API and normalized into a flat artifact list.

```
Issue Body → artifact
Issue Comment → artifact
Pull Request Review → artifact
```

Each artifact includes:

```
artifact_id
issue_number
artifact_type
author
timestamp
labels
assignees
source_url
text
```

Bot-generated comments (e.g. `github-actions[bot]`, `dependabot[bot]`) are flagged to prevent noise during extraction.

### Reproduce download

```bash
export GITHUB_TOKEN=<your_token>
python scripts/ingestion/fetch_github.py
```

---

# Architecture

```
GitHub API
     │
     ▼
fetch_github.py
     │
     ▼
normalize_artifacts.py
→ data/artifacts/artifacts.json
     │
     ▼
chunk_artifacts.py
→ data/chunks/chunks.json
     │
     ▼
extract_claims_ollama.py
→ data/extractions/raw_extractions.json
     │
     ▼
validate_extractions.py
→ data/extractions/validated_extractions.json
     │
     ├─────────────────────────────────────────┐
     ▼                                         ▼
resolve_entities.py                     deduplicate_claims.py
→ canonical_map.json                    → claims_deduped.json
→ alias_index.json                      → conflict_report.json
→ merge_log.json
     │                                         │
     └─────────────────────────────────────────┘
                        │
                        ▼
                   build_graph.py
            → nodes.json / edges.json
                        │
                        ▼
             normalize_entities.py
        → nodes_normalized.json / edges_normalized.json
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
      build_embeddings.py      FastAPI server
      → TF-IDF index           api/main.py
              │
              ▼
retrieve_context.py / graph_search.py
```

---

# Setup

```bash
# clone project
cd ddgop/layer10-fastapi-memory

# virtual environment
python -m venv venv
source venv/bin/activate

# dependencies
pip install -r requirements.txt

# install Ollama
https://ollama.com/download

ollama pull llama3
ollama serve
```

---

# Running the Pipeline

```bash
# 1. Fetch GitHub issues
GITHUB_TOKEN=<token> python scripts/ingestion/fetch_github.py

# 2. Normalize artifacts
python scripts/preprocessing/normalize_artifacts.py

# 3. Chunk text
python scripts/preprocessing/chunk_artifacts.py

# 4. Extract claims
python scripts/extraction/extract_claims_ollama.py

# 5. Validate extractions
python scripts/extraction/validate_extractions.py

# 6. Resolve entity aliases
python scripts/deduplication/resolve_entities.py

# 7. Deduplicate claims
python scripts/deduplication/deduplicate_claims.py

# 8. Build graph
python scripts/graph/build_graph.py
python scripts/graph/normalize_entities.py

# 9. Build retrieval index
python scripts/retrieval/build_embeddings.py
```

---

# Preprocessing

## Artifact Normalization

GitHub issues and PRs are flattened into a list of artifacts:

```
issue_body
comment
```

Each artifact preserves metadata such as:

* issue number
* author
* labels
* timestamps
* source URL

---

## Chunking

Artifacts are split into **overlapping sentence-aware chunks** to fit within LLM context limits.

Chunk properties:

```
chunk_id
artifact_id
issue_number
offset_start
offset_end
text
author
timestamp
source_url
```

Offsets allow claims to reference the **exact text span** supporting the extracted fact.

---

# Extraction

Claims and entities are extracted using a local LLM.

Model:

```
Llama-3 via Ollama
```

Inference settings:

```
temperature = 0
top_p = 1
```

This ensures **deterministic extraction**.

Extraction produces:

```
entities
claims
confidence scores
evidence excerpts
```

Example claim:

```
routing.py --imports--> starlette.routing
```

Each claim must include a verbatim evidence phrase.

---

# Validation & Repair

LLM output is validated against a strict JSON schema.

The validation layer performs:

* schema validation
* entity repair
* relation normalization
* automatic entity creation when referenced by claims
* confidence filtering

Outputs:

```
validated_extractions.json
validation_report.json
```

Metrics recorded:

* entities extracted
* claims extracted
* retention after validation
* validation issues

These metrics allow monitoring extraction quality.

---

# Deduplication

## Entity Canonicalization

Entity names are normalized and clustered using fuzzy similarity.

Algorithm:

```
token_set similarity (rapidfuzz)
threshold = ENTITY_SIM_THRESHOLD
```

Canonical name selection:

```
highest frequency
→ longest name
→ alphabetical
```

Outputs:

```
canonical_map.json
alias_index.json
merge_log.json
```

All merges are logged for auditability.

---

## Claim Deduplication

Claims are grouped by:

```
(subject, relation, object)
```

Evidence from all occurrences is merged.

Duplicate evidence snippets are removed using fuzzy similarity.

Each claim stores:

```
confidence
support_count
evidence list
first_seen
last_seen
```

Conflicting claims are recorded in:

```
conflict_report.json
```

Lower-support claims are marked:

```
superseded = true
```

rather than deleted.

### Artifact Deduplication

GitHub discussions may contain duplicated content due to quoting,
cross-posting, or repeated comments.

In this prototype pipeline artifacts are normalized but not aggressively
deduplicated because GitHub comments rarely appear as exact duplicates.

In a production deployment artifact deduplication would be implemented
using a two-stage approach:

1. **Exact duplicate detection**
   - SHA256 hash of normalized artifact text.

2. **Near-duplicate detection**
   - MinHash / Jaccard similarity over token shingles.
   - Used to detect quoted replies or repeated comments.

Duplicate artifacts would share a canonical `artifact_id` while preserving
their original source metadata for provenance.

---

# Memory Graph

Graph structure:

## Nodes

```
id
type
aliases
frequency
first_seen
last_seen
```

## Edges

```
subject
relation
object
confidence
support_count
evidence
first_seen
last_seen
superseded
conflict_with
```

Claim IDs are deterministically generated from:

```
SHA256(subject|relation|object)
```

ensuring reproducible graph construction.

---

# Retrieval

Query pipeline:

```
question
↓
TF-IDF search over chunks
↓
entity detection in top chunks
↓
graph expansion
↓
context pack
```

TF-IDF configuration:

```
unigrams + bigrams
sublinear TF
cosine similarity
```

Entity mentions are detected via substring matching against node IDs.

---

## Context Pack

Each query returns:

```
{
  question
  evidence_snippets
  graph_claims
  entities_found
  total_evidence
  total_claims
}
```

Every result contains **direct source URLs and offsets** for auditing.

---

# Querying

### CLI Retrieval

```bash
python scripts/retrieval/retrieve_context.py "Who reviews PRs about routing?"
python scripts/retrieval/retrieve_context.py "What does routing.py depend on?"
```

### Graph Exploration

```bash
python scripts/retrieval/graph_search.py "routing.py" --depth 2
python scripts/retrieval/graph_search.py "tiangolo" --relation reviews --reverse
```

### REST API

```bash
uvicorn api.main:app --reload --port 8000
```

Endpoints:

| Endpoint               | Description            |
| ---------------------- | ---------------------- |
| `/`                    | Health check           |
| `/stats`               | Graph statistics       |
| `/nodes`               | Node listing           |
| `/node/{entity}`       | Node + neighbors       |
| `/memory`              | Claims for entity      |
| `/neighbors`           | Edge lookup            |
| `/search`              | TF-IDF evidence search |
| `/context`             | Full context pack      |
| `/evidence/{claim_id}` | Evidence for claim     |
| `/conflicts`           | Conflicting claims     |

---

# Visualization

Interactive graph viewer:

```bash
python visualization/graph_viewer.py
```

Open:

```
visualization/memory_graph.html
```

Features:

* entity relationship graph
* filtering by relation / type
* minimum evidence support
* clickable evidence links

---

# Ontology

Defined in:

```
schema/ontology.json
```

### Entity Types

```
Person
Component
Feature
Bug
PullRequest
Issue
Library
Concept
Config
Test
```

### Relation Types

```
fixes
closes
depends_on
uses
defines
imports
reviews
authors
supersedes
tests
configures
requires
supports
mentions
relates_to
```

---

# Long-Term Correctness

The system models evolving knowledge:

```
first_seen
last_seen
superseded
conflict_with
```

These allow representing:

```
past facts
current facts
conflicting claims
decision reversals
```

---

# Adapting to Layer10's Target Environment

## Unstructured + Structured Fusion

For production environments (email, Slack, Jira):

New entity types:

```
Thread
Channel
Project
Sprint
Decision
```

New relations:

```
assigned_to
blocks
mentioned_in
decided_in
```

---

## Long-Term Memory

Durable memory:

```
claims with ≥3 supporting sources
```

Ephemeral memory:

```
low-confidence single-source claims
```

A decay process downgrades stale claims.

---

## Grounding & Safety

Every claim contains:

```
source_url
artifact_id
offset_start
offset_end
```

If a source is deleted:

```
claims become orphaned
```

and are excluded from retrieval.

---

## Permissions

Evidence objects reference artifact IDs.

An ACL layer filters evidence based on user access to underlying artifacts.

---

## Operational Considerations

Incremental ingestion:

```
updated_at watermark
```

Only new artifacts are processed.

Cost:

```
Llama-3 local inference
≈ 2–4 chunks/sec on laptop
```

Evaluation:

A golden question set verifies retrieval accuracy after pipeline runs.

Metrics stored in:

```
data/extractions/validation_report.json
```

---

# Reproducibility

The repository contains:

* ingestion scripts
* extraction pipeline
* graph construction
* retrieval API
* visualization tools

Running the steps above reproduces the full memory graph from raw GitHub data.
