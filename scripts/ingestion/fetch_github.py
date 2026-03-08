"""
fetch_github.py

Fetches issues and pull requests from a GitHub repository via the REST API.

Produces: data/raw/fastapi_issues.json

Run:
    GITHUB_TOKEN=<token> python scripts/ingestion/fetch_github.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path

# ==============================
# Load project config
# ==============================

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import (
    GITHUB_TOKEN, REPO_OWNER, REPO_NAME,
    FETCH_STATE, FETCH_LIMIT, RAW_DIR, ensure_dirs
)

BASE_URL   = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
OUTPUT_PATH = RAW_DIR / "fastapi_issues.json"

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"
else:
    print("[warn] GITHUB_TOKEN not set — requests will be rate-limited to 60/hour")


# ==============================
# Helpers
# ==============================

def github_get(url, params=None):
    """Make a GitHub API GET request with basic retry on 429."""
    for attempt in range(3):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429 or resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait  = max(reset - time.time(), 1)
            print(f"  [rate-limit] sleeping {wait:.0f}s …")
            time.sleep(wait)
            continue

        raise Exception(f"GitHub API error {resp.status_code}: {resp.text[:300]}")

    raise Exception(f"Exhausted retries for {url}")


def fetch_issues_page(page, per_page):
    params = {
        "state":     FETCH_STATE,
        "per_page":  per_page,
        "page":      page,
        "sort":      "updated",
        "direction": "desc"
    }
    return github_get(f"{BASE_URL}/issues", params)


def fetch_comments(comments_url):
    """Fetch all comments for an issue/PR (handles pagination)."""
    all_comments = []
    page = 1
    while True:
        batch = github_get(comments_url, {"per_page": 100, "page": page})
        if not batch:
            break
        all_comments.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return all_comments


def fetch_pr_reviews(pr_number):
    """Fetch review events for a PR (reviewer identities)."""
    url = f"{BASE_URL}/pulls/{pr_number}/reviews"
    try:
        return github_get(url, {"per_page": 100})
    except Exception:
        return []


def normalize_issue(issue, comments, reviews=None):
    """Flatten a GitHub API issue/PR object into our storage format."""
    is_pr = "pull_request" in issue

    artifact = {
        "source_id":  str(issue["id"]),
        "number":     issue["number"],
        "type":       "PR" if is_pr else "Issue",
        "url":        issue["html_url"],
        "title":      issue["title"],
        "body":       issue["body"] or "",
        "author":     issue["user"]["login"],
        "state":      issue["state"],

        # Label names — useful for classification later
        "labels": [lb["name"] for lb in issue.get("labels", [])],

        # Assignees
        "assignees": [a["login"] for a in issue.get("assignees", [])],

        "created_at": issue["created_at"],
        "closed_at":  issue.get("closed_at"),
        "updated_at": issue["updated_at"],

        "comments": [
            {
                "comment_id": str(c["id"]),
                "author":     c["user"]["login"],
                "body":       c["body"] or "",
                "created_at": c["created_at"],
                "updated_at": c.get("updated_at"),
                "url":        c["html_url"]
            }
            for c in comments
        ]
    }

    # Attach reviewer info for PRs
    if is_pr and reviews:
        artifact["reviewers"] = list({
            r["user"]["login"]
            for r in reviews
            if r.get("user") and r["state"] in ("APPROVED", "CHANGES_REQUESTED", "COMMENTED")
        })
    else:
        artifact["reviewers"] = []

    return artifact


# ==============================
# Main pipeline
# ==============================

def main():
    ensure_dirs()

    print(f"\nFetching {FETCH_LIMIT} {FETCH_STATE} issues/PRs from {REPO_OWNER}/{REPO_NAME}\n")

    issues  = []
    page    = 1

    while len(issues) < FETCH_LIMIT:
        per_page = min(100, FETCH_LIMIT - len(issues))
        batch    = fetch_issues_page(page, per_page)

        if not batch:
            break

        issues.extend(batch)
        print(f"  page {page}: fetched {len(batch)} items (total so far: {len(issues)})")
        page += 1

    issues = issues[:FETCH_LIMIT]
    print(f"\nProcessing {len(issues)} items …\n")

    artifacts = []

    for i, issue in enumerate(issues):
        is_pr      = "pull_request" in issue
        issue_type = "PR" if is_pr else "Issue"

        print(f"  [{i+1}/{len(issues)}] {issue_type} #{issue['number']} — {issue['title'][:60]}")

        comments = []
        if issue.get("comments", 0) > 0:
            comments = fetch_comments(issue["comments_url"])
            time.sleep(0.15)

        reviews = []
        if is_pr:
            reviews = fetch_pr_reviews(issue["number"])
            time.sleep(0.15)

        artifact = normalize_issue(issue, comments, reviews)
        artifacts.append(artifact)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(artifacts, f, indent=2)

    print(f"\nSaved {len(artifacts)} artifacts → {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()