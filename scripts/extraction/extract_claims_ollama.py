"""
extract_claims_ollama.py

Runs structured extraction over every (non-bot) chunk using a locally-running
Ollama model.  Each extraction produces typed entities and grounded claims
with a confidence score.

Produces: data/extractions/raw_extractions.json

Run:
    python scripts/extraction/extract_claims_ollama.py

Prerequisites:
    ollama serve          # start the Ollama daemon
    ollama pull llama3    # pull the model once
"""

import sys
import json
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    CHUNKS_DIR, EXTRACTIONS_DIR,
    OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, MAX_OUTPUT_TOKENS,
    ensure_dirs
)

INPUT_FILE  = CHUNKS_DIR      / "chunks.json"
OUTPUT_FILE = EXTRACTIONS_DIR / "raw_extractions.json"

# Skip bot-authored chunks — no semantic content worth extracting
SKIP_BOTS = True

# Retry a chunk at most this many times on parse failure
MAX_RETRIES = 2


# ==============================
# Prompt
# ==============================

SYSTEM_PROMPT = """You are a precise knowledge-extraction engine for software engineering discussions.

Your task:
  1. Identify ENTITIES (technical nouns: files, libraries, features, bugs, people, concepts, configs, tests).
  2. Identify CLAIMS: grounded relationships between two entities.

Entity type vocabulary:
  Person | Component | Feature | Bug | PullRequest | Issue | Library | Concept | Config | Test | unknown

Relation type vocabulary (use ONLY these):
  fixes | closes | depends_on | uses | defines | imports | reviews | authors |
  supersedes | tests | configures | requires | supports | mentions | relates_to

Rules:
  - Every claim MUST have an evidence_excerpt — a verbatim short phrase (≤80 chars) from the input text.
  - confidence: float 0.0–1.0 reflecting how clearly the text supports this claim.
  - Do NOT invent claims not supported by the text.
  - Subject and object must both appear in the entities list.
  - Return ONLY valid JSON — no markdown fences, no prose.

Output format (strict):
{
  "entities": [
    {"name": "<name>", "type": "<EntityType>"}
  ],
  "claims": [
    {
      "subject":          "<entity name>",
      "relation":         "<relation>",
      "object":           "<entity name>",
      "evidence_excerpt": "<verbatim short phrase from text>",
      "confidence":       0.85,
      "subject_type":     "<EntityType>",
      "object_type":      "<EntityType>"
    }
  ]
}"""


def build_prompt(text):
    # Truncate to stay well within context window
    safe_text = text[:3000]
    return f"{SYSTEM_PROMPT}\n\nText to analyse:\n\"\"\"\n{safe_text}\n\"\"\""


# ==============================
# JSON extraction from model output
# ==============================

def extract_json(text):
    """
    Try several strategies to pull a JSON object out of the model response.
    Returns a dict or None on failure.
    """
    # Strategy 1: direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find outermost {...}
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: strip markdown fences then retry
    cleaned = re.sub(r'```(?:json)?', '', text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None


# ==============================
# Ollama call
# ==============================

def call_ollama(prompt, attempt=0):
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": MAX_OUTPUT_TOKENS,
            "top_p":       1.0
        }
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"]


# ==============================
# Single-chunk processing
# ==============================

def process_chunk(chunk):
    """
    Run extraction on one chunk with retries.
    Returns a result dict with 'extraction' or 'error'.
    """
    prompt = build_prompt(chunk["text"])

    parsed = None
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = call_ollama(prompt, attempt)
            parsed = extract_json(raw)
            if parsed is not None:
                break
            last_error = "json_parse_failed"
        except requests.exceptions.Timeout:
            last_error = "ollama_timeout"
            time.sleep(2)
        except Exception as e:
            last_error = str(e)
            time.sleep(1)

    if parsed is None:
        return {
            "chunk_id":     chunk["chunk_id"],
            "artifact_id":  chunk["artifact_id"],
            "issue_number": chunk["issue_number"],
            "source_url":   chunk["source_url"],
            "error":        last_error,
            "extraction":   {"entities": [], "claims": []}
        }

    # Ensure required fields exist
    parsed.setdefault("entities", [])
    parsed.setdefault("claims",   [])

    # Attach default confidence if missing
    for claim in parsed["claims"]:
        claim.setdefault("confidence", 0.7)

    return {
        "chunk_id":       chunk["chunk_id"],
        "artifact_id":    chunk["artifact_id"],
        "issue_number":   chunk["issue_number"],
        "source_url":     chunk["source_url"],
        "author":         chunk.get("author"),
        "timestamp":      chunk.get("timestamp"),
        "offset_start":   chunk.get("offset_start", 0),
        "offset_end":     chunk.get("offset_end",   0),
        "extraction":     parsed
    }


# ==============================
# Main pipeline
# ==============================

def main():
    ensure_dirs()

    print(f"Loading chunks from {INPUT_FILE} …")
    chunks = json.load(open(INPUT_FILE))
    print(f"  Loaded {len(chunks)} total chunks")

    if SKIP_BOTS:
        chunks = [c for c in chunks if not c.get("is_bot", False)]
        print(f"  After bot-filter : {len(chunks)} chunks to process\n")

    results       = []
    error_count   = 0

    for chunk in tqdm(chunks, desc="Extracting"):
        result = process_chunk(chunk)
        results.append(result)
        if result.get("error"):
            error_count += 1

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    total_entities = sum(len(r["extraction"].get("entities", [])) for r in results)
    total_claims   = sum(len(r["extraction"].get("claims",   [])) for r in results)

    print(f"\n  Processed chunks : {len(results)}")
    print(f"  Errors           : {error_count}")
    print(f"  Total entities   : {total_entities}")
    print(f"  Total claims     : {total_claims}")
    print(f"\nSaved {len(results)} extractions → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()