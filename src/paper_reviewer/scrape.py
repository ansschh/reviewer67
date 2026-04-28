"""Stage 1 — pull all submissions/reviews/decisions from OpenReview.

Resumable: skips venues whose JSONL already exists.
Streaming: writes one line per paper as we go, so a crash doesn't lose work.
Unauthenticated: public data only — avoids burning your account's rate limit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .config import OPENREVIEW_BASE, PATHS, VENUES


def _client():
    import openreview  # imported lazily so unit tests don't need the dep installed

    return openreview.api.OpenReviewClient(baseurl=OPENREVIEW_BASE)


def _classify_reply(reply: dict) -> str:
    """Best-effort classifier for the reply's role on a submission forum."""
    invs = reply.get("invitations") or [reply.get("invitation", "")]
    for i in invs:
        if not isinstance(i, str):
            continue
        # Order matters: "Meta_Review" must be checked before "Review" because
        # both end with "Review".
        if i.endswith("Meta_Review"):
            return "meta_review"
        if i.endswith("Decision"):
            return "decision"
        if i.endswith("Official_Review") or i.endswith("Review"):
            return "review"
    return "other"


def scrape_venue(venue_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{venue_id.replace('/', '_')}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] {venue_id} (cached at {out_path})")
        return out_path

    client = _client()
    venue_group = client.get_group(venue_id)
    submission_name = venue_group.content.get("submission_name", {}).get("value", "Submission")

    submissions = client.get_all_notes(
        invitation=f"{venue_id}/-/{submission_name}",
        details="replies",
    )

    tmp_path = out_path.with_suffix(".jsonl.partial")
    with tmp_path.open("w", encoding="utf-8") as f:
        for sub in tqdm(submissions, desc=venue_id):
            reviews: list[dict] = []
            meta_review: dict | None = None
            decision: dict | None = None
            for r in sub.details.get("replies", []) if sub.details else []:
                kind = _classify_reply(r)
                if kind == "review":
                    reviews.append(r)
                elif kind == "meta_review":
                    meta_review = r
                elif kind == "decision":
                    decision = r

            record = {
                "venue": venue_id,
                "forum_id": sub.id,
                "title": (sub.content or {}).get("title", {}).get("value"),
                "abstract": (sub.content or {}).get("abstract", {}).get("value"),
                "keywords": (sub.content or {}).get("keywords", {}).get("value"),
                "pdf_url": f"https://openreview.net/pdf?id={sub.id}",
                "reviews": reviews,
                "meta_review": meta_review,
                "decision": decision,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)
    return out_path


def scrape_all(venues: Iterable[str] | None = None) -> list[Path]:
    PATHS.ensure()
    venues = list(venues) if venues else VENUES
    out_paths: list[Path] = []
    for v in venues:
        try:
            out_paths.append(scrape_venue(v, PATHS.raw))
        except Exception as e:
            print(f"[fail] {v}: {e!r}")
    return out_paths


def iter_papers(venues: Iterable[str] | None = None):
    """Yield paper dicts from cached JSONL across venues."""
    venues = list(venues) if venues else VENUES
    for v in venues:
        p = PATHS.raw / f"{v.replace('/', '_')}.jsonl"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
