"""
normalize_artifacts.py

Flattens raw GitHub issues/PRs into a flat list of Artifact records.

Each artifact is either:
  - issue_body  : the title + body of an issue or PR
  - comment     : a single comment on an issue or PR

Produces: data/artifacts/artifacts.json

Run:
    python scripts/preprocessing/normalize_artifacts.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import RAW_DIR, ARTIFACTS_DIR, ensure_dirs

INPUT_FILE  = RAW_DIR      / "fastapi_issues.json"
OUTPUT_FILE = ARTIFACTS_DIR / "artifacts.json"

# Bot authors whose comments carry no semantic content
BOT_AUTHORS = {
    "github-actions[bot]",
    "codecov[bot]",
    "codspeed-hq[bot]",
    "dependabot[bot]",
    "pre-commit-ci[bot]",
    "allcontributors[bot]"
}


# ==============================
# Builders
# ==============================

def build_issue_body_artifact(issue):
    return {
        "artifact_id":   f"issue_{issue['number']}_body",
        "issue_number":  issue["number"],
        "artifact_type": "issue_body",
        "issue_type":    issue["type"],          # "PR" | "Issue"
        "state":         issue["state"],          # "open" | "closed"
        "labels":        issue.get("labels", []),
        "author":        issue["author"],
        "reviewers":     issue.get("reviewers", []),
        "assignees":     issue.get("assignees", []),
        "timestamp":     issue["created_at"],
        "closed_at":     issue.get("closed_at"),
        "updated_at":    issue.get("updated_at"),
        "text":          issue["title"] + "\n\n" + issue["body"],
        "source_url":    issue["url"],
        "is_bot":        False
    }


def build_comment_artifact(issue_number, issue_type, issue_state, comment):
    author = comment["author"]
    return {
        "artifact_id":   f"issue_{issue_number}_comment_{comment['comment_id']}",
        "issue_number":  issue_number,
        "artifact_type": "comment",
        "issue_type":    issue_type,
        "state":         issue_state,
        "labels":        [],
        "author":        author,
        "reviewers":     [],
        "assignees":     [],
        "timestamp":     comment["created_at"],
        "closed_at":     None,
        "updated_at":    comment.get("updated_at"),
        "text":          comment["body"],
        "source_url":    comment["url"],
        "is_bot":        author in BOT_AUTHORS
    }


# ==============================
# Stats helpers
# ==============================

def print_stats(artifacts):
    total      = len(artifacts)
    bots       = sum(1 for a in artifacts if a["is_bot"])
    issue_body = sum(1 for a in artifacts if a["artifact_type"] == "issue_body")
    comments   = sum(1 for a in artifacts if a["artifact_type"] == "comment")
    prs        = sum(1 for a in artifacts if a["issue_type"] == "PR" and a["artifact_type"] == "issue_body")
    issues     = sum(1 for a in artifacts if a["issue_type"] == "Issue" and a["artifact_type"] == "issue_body")

    print(f"  Total artifacts   : {total}")
    print(f"  Issue bodies      : {issue_body}  (PRs: {prs}, Issues: {issues})")
    print(f"  Comments          : {comments}  ({bots} from bots)")


# ==============================
# Main pipeline
# ==============================

def main():
    ensure_dirs()

    print(f"Loading raw issues from {INPUT_FILE} …")
    issues = json.load(open(INPUT_FILE))
    print(f"  Loaded {len(issues)} issue/PR records\n")

    artifacts = []

    for issue in issues:
        artifacts.append(build_issue_body_artifact(issue))

        for comment in issue.get("comments", []):
            artifacts.append(
                build_comment_artifact(
                    issue["number"],
                    issue["type"],
                    issue["state"],
                    comment
                )
            )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(artifacts, f, indent=2)

    print_stats(artifacts)
    print(f"\nSaved {len(artifacts)} artifacts → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()