"""
chunk_artifacts.py

Splits artifact text into overlapping chunks suitable for embedding and
LLM extraction.  Chunking is sentence-aware: we never cut in the middle of
a sentence unless the sentence itself exceeds CHUNK_SIZE.

Bot artifacts are kept only as single-chunk references (their text is usually
noise) and are flagged so downstream steps can optionally filter them.

Produces: data/chunks/chunks.json

Run:
    python scripts/preprocessing/chunk_artifacts.py
"""

import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    ARTIFACTS_DIR, CHUNKS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
    ensure_dirs
)

INPUT_FILE  = ARTIFACTS_DIR / "artifacts.json"
OUTPUT_FILE = CHUNKS_DIR    / "chunks.json"


# ==============================
# Sentence splitter
# ==============================

SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text):
    """Split text into sentences using punctuation heuristics."""
    # Normalise newlines first
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)

    parts = SENTENCE_BOUNDARY.split(text)
    # Preserve double-newlines as paragraph breaks
    sentences = []
    for part in parts:
        for sub in part.split('\n\n'):
            sub = sub.strip()
            if sub:
                sentences.append(sub)
    return sentences


# ==============================
# Chunking logic
# ==============================

def chunk_text_smart(text, size, overlap):
    """
    Build overlapping chunks that respect sentence boundaries.

    Returns a list of dicts with keys: text, offset_start, offset_end.
    """
    if not text.strip():
        return []

    sentences    = split_sentences(text)
    chunks       = []
    current      = []
    current_len  = 0
    char_cursor  = 0

    chunk_start  = 0

    for sentence in sentences:
        s_len = len(sentence)

        # If adding this sentence would exceed CHUNK_SIZE, flush the buffer
        if current_len + s_len > size and current:
            chunk_text = " ".join(current)
            chunks.append({
                "text":         chunk_text,
                "offset_start": chunk_start,
                "offset_end":   chunk_start + len(chunk_text)
            })

            # Roll back by OVERLAP characters worth of sentences
            overlap_sentences = []
            overlap_len       = 0
            for sent in reversed(current):
                if overlap_len + len(sent) > overlap:
                    break
                overlap_sentences.insert(0, sent)
                overlap_len += len(sent)

            current     = overlap_sentences
            current_len = overlap_len
            chunk_start = char_cursor - overlap_len

        # If a single sentence is bigger than CHUNK_SIZE, hard-split it
        if s_len > size:
            for start in range(0, s_len, size - overlap):
                piece = sentence[start : start + size]
                chunks.append({
                    "text":         piece,
                    "offset_start": char_cursor + start,
                    "offset_end":   char_cursor + start + len(piece)
                })
        else:
            current.append(sentence)
            current_len += s_len

        char_cursor += s_len + 1   # +1 for space

    # Flush remaining
    if current:
        chunk_text = " ".join(current)
        chunks.append({
            "text":         chunk_text,
            "offset_start": chunk_start,
            "offset_end":   chunk_start + len(chunk_text)
        })

    return chunks


# ==============================
# Main pipeline
# ==============================

def main():
    ensure_dirs()

    print(f"Loading artifacts from {INPUT_FILE} …")
    artifacts = json.load(open(INPUT_FILE))
    print(f"  Loaded {len(artifacts)} artifacts\n")

    chunks        = []
    skipped_empty = 0

    for artifact in artifacts:
        text   = artifact.get("text", "").strip()
        is_bot = artifact.get("is_bot", False)

        if not text:
            skipped_empty += 1
            continue

        # For bot comments keep a single chunk flagged as bot
        if is_bot:
            chunks.append({
                "chunk_id":      f"{artifact['artifact_id']}_chunk_0",
                "artifact_id":   artifact["artifact_id"],
                "issue_number":  artifact["issue_number"],
                "artifact_type": artifact["artifact_type"],
                "issue_type":    artifact.get("issue_type", "unknown"),
                "state":         artifact.get("state", "unknown"),
                "labels":        artifact.get("labels", []),
                "author":        artifact["author"],
                "timestamp":     artifact["timestamp"],
                "source_url":    artifact["source_url"],
                "offset_start":  0,
                "offset_end":    len(text),
                "text":          text,
                "is_bot":        True
            })
            continue

        artifact_chunks = chunk_text_smart(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(artifact_chunks):
            chunks.append({
                "chunk_id":      f"{artifact['artifact_id']}_chunk_{i}",
                "artifact_id":   artifact["artifact_id"],
                "issue_number":  artifact["issue_number"],
                "artifact_type": artifact["artifact_type"],
                "issue_type":    artifact.get("issue_type", "unknown"),
                "state":         artifact.get("state", "unknown"),
                "labels":        artifact.get("labels", []),
                "author":        artifact["author"],
                "timestamp":     artifact["timestamp"],
                "source_url":    artifact["source_url"],
                "offset_start":  chunk["offset_start"],
                "offset_end":    chunk["offset_end"],
                "text":          chunk["text"],
                "is_bot":        False
            })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(chunks, f, indent=2)

    bot_chunks = sum(1 for c in chunks if c["is_bot"])
    print(f"  Chunks created    : {len(chunks)}")
    print(f"  Bot chunks        : {bot_chunks}")
    print(f"  Skipped (empty)   : {skipped_empty}")
    print(f"\nSaved {len(chunks)} chunks → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()