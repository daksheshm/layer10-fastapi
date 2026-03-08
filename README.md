# Layer10 Take-Home — Grounded Long-Term Memory via Structured Extraction

A pipeline that turns GitHub issue/PR discussions into a queryable, grounded memory graph with evidence provenance, deduplication, and an interactive visualisation.

---

## Corpus

**Source:** `fastapi/fastapi` GitHub repository — closed issues and pull requests  
**Size:** ~100 issues/PRs with comments (~500–1 000 total artifacts)  
**Why:** Dense technical discussion with decisions, reversals, cross-references, and identity resolution challenges (bots, aliases, multiple maintainers).

**Reproduce the download:**
```bash
export GITHUB_TOKEN=<your_token>
python scripts/ingestion/fetch_github.py
```

---

## Architecture

```
Raw Issues (GitHub API)
        │
        ▼
normalize_artifacts.py   → data/artifacts/artifacts.json
        │
        ▼
chunk_artifacts.py        → data/chunks/chunks.json
        │
        ▼
extract_claims_ollama.py  → data/extractions/raw_extractions.json
        │
        ▼
validate_extractions.py   → data/extractions/validated_extractions.json
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
resolve_entities.py                     deduplicate_claims.py
→ data/resolved/canonical_map.json     → data/graph/claims_deduped.json
→ data/resolved/alias_index.json       → data/graph/conflict_report.json
→ data/resolved/merge_log.json
        │                                         │
        └─────────────────────────────────────────┘
                            │
                            ▼
                      build_graph.py
                → data/graph/nodes.json
                → data/graph/edges.json
                            │
                            ▼
                   normalize_entities.py
                → data/graph/nodes_normalized.json
                → data/graph/edges_normalized.json
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
          build_embeddings.py    api/main.py
          → data/embeddings/     (FastAPI server)
                   │
                   ▼
          retrieve_context.py
          graph_search.py
```

---

## Setup

```bash
# Clone / unzip the project
cd ddgop/layer10-fastapi-memory

# Create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama (for extraction)
# https://ollama.com/download
ollama pull llama3
ollama serve
```

---

## Running the Pipeline (end-to-end)

```bash
# 1. Fetch GitHub issues
GITHUB_TOKEN=<token> python scripts/ingestion/fetch_github.py

# 2. Normalise artifacts
python scripts/preprocessing/normalize_artifacts.py

# 3. Chunk text
python scripts/preprocessing/chunk_artifacts.py

# 4. Extract claims (requires Ollama running)
python scripts/extraction/extract_claims_ollama.py

# 5. Validate and repair extractions
python scripts/extraction/validate_extractions.py

# 6. Resolve entity aliases
python scripts/deduplication/resolve_entities.py

# 7. Deduplicate claims
python scripts/deduplication/deduplicate_claims.py

# 8. Build graph
python scripts/graph/build_graph.py
python scripts/graph/normalize_entities.py

# 9. Build TF-IDF embeddings for retrieval
python scripts/retrieval/build_embeddings.py
```

---

## Querying

**CLI search:**
```bash
python scripts/retrieval/retrieve_context.py "Who reviews PRs about routing?"
python scripts/retrieval/retrieve_context.py "What does routing.py depend on?"
```

**CLI graph exploration:**
```bash
python scripts/retrieval/graph_search.py "routing.py" --depth 2
python scripts/retrieval/graph_search.py "tiangolo" --relation reviews --reverse
```

**REST API:**
```bash
uvicorn api.main:app --reload --port 8000
```

| Endpoint | Description |
|---|---|
| `GET /` | Health check + counts |
| `GET /stats` | Node/edge statistics |
| `GET /nodes?type=Person` | List nodes (with type filter) |
| `GET /node/{entity}` | Full node record + neighbor summary |
| `GET /memory?entity=routing.py` | Outgoing claims for an entity |
| `GET /neighbors?entity=tiangolo` | All edges touching an entity |
| `GET /search?q=operationId+bug` | TF-IDF evidence search |
| `GET /context?q=Who+fixed+operationId` | Full grounded context pack |
| `GET /evidence/{claim_id}` | All evidence for one claim |
| `GET /conflicts` | Conflicting claim pairs |

---

## Visualisation

```bash
python visualization/graph_viewer.py
# Open visualization/memory_graph.html in any browser

# Filters:
python visualization/graph_viewer.py --type Person
python visualization/graph_viewer.py --relation fixes --min-support 2
python visualization/graph_viewer.py --limit 100
```

---

## Ontology

Defined in `schema/ontology.json`.

**Entity types:** `Person`, `Component`, `Feature`, `Bug`, `PullRequest`, `Issue`, `Library`, `Concept`, `Config`, `Test`

**Relation types:** `fixes`, `closes`, `depends_on`, `uses`, `defines`, `imports`, `reviews`, `authors`, `supersedes`, `tests`, `configures`, `requires`, `supports`, `mentions`, `relates_to`

---

## Design Decisions

### Extraction
- **Model:** Llama 3 via Ollama (free, local, no API key).
- **Prompt:** Zero-shot with strict JSON schema + controlled vocabulary for entity/relation types.
- **Confidence:** The model emits a 0–1 confidence per claim; claims below `EXTRACTION_MIN_CONFIDENCE` (default 0.5) are discarded at validation time.
- **Retries:** Up to 2 retries per chunk on parse failure; partial JSON is recovered with regex.

### Deduplication
- **Entity resolution:** Trigram-based Jaccard similarity (via `rapidfuzz`). Entities above `ENTITY_SIM_THRESHOLD` (default 0.82) are merged; the most-frequent form becomes canonical. All merges are logged in `merge_log.json` for auditability and reversal.
- **Claim dedup:** Claims grouped by `(canonical_subject, relation, canonical_object)`. Evidence lists from all matching chunks are merged; duplicate evidence is detected via `fuzz.token_set_ratio`.
- **Conflicts:** Claims sharing the same `(subject, object)` but with different relations are flagged in `conflict_report.json`. Low-support claims are marked `superseded: true` rather than deleted.

### Grounding
Every claim stores a list of `evidence` objects, each containing:
- `excerpt` — verbatim phrase from the source text
- `source_url` — direct link to the GitHub comment/issue
- `artifact_id`, `chunk_id` — internal traceback identifiers
- `offset_start`, `offset_end` — character offsets in the original artifact text
- `timestamp`, `author`

### Retrieval
- **Chunk search:** TF-IDF (unigrams + bigrams, sublinear TF) with cosine similarity. No GPU required.
- **Graph expansion:** After finding the top-K chunks, entity names found in those chunks are used to pull related claims from the graph.
- **Context pack:** Returns both ranked evidence snippets and graph claims in a single structured response.

### Long-term correctness
- `superseded` flag distinguishes "was true" from "is true now".
- `first_seen` / `last_seen` timestamps on nodes and edges.
- Extraction version tracking is embedded in file naming conventions; re-running any step produces a new output without overwriting the old one (add `--output` flag or timestamp suffix).

---

## Adapting to Layer10's Target Environment

### Unstructured + structured fusion
In production (email, Slack, Jira/Linear), the ontology would add entity types like `Thread`, `Channel`, `Project`, `Sprint`, `Decision`, and relation types like `blocks`, `assigned_to`, `mentioned_in`, `decided_in`. Jira ticket IDs and Slack thread permalinks would serve as grounding anchors.

### Long-term memory vs ephemeral context
- Durable memory: high-support (≥3 sources) claims that survive multiple extraction runs.
- Ephemeral: single-source claims with low confidence held in a "staging" store for human review.
- Decay: claims not re-evidenced within a configurable TTL window are downgraded to `stale`.

### Grounding & safety
- Every API response carries `source_url` and `offset` so any claim can be audited.
- Deletions/redactions: when a source is deleted, all claims whose *only* evidence points to that source are immediately marked `orphaned` and excluded from retrieval.

### Permissions
The `evidence` objects record `artifact_id` and `source_url`. An ACL layer maps users → readable artifact IDs. At query time, evidence items are filtered to the user's readable set before building the context pack.

### Operational reality
- Incremental ingestion: the pipeline is idempotent; run `fetch_github.py` again and only new/updated issues are re-processed (use `updated_at` watermark).
- Cost: Llama 3 local inference at ~2–4 chunks/sec on a consumer laptop; a GPU machine or batch API call would be used in production.
- Evaluation: a small golden-set of (question, expected_claim) pairs is checked after every pipeline run. Precision/recall on the golden set is logged to `data/extractions/validation_report.json`.