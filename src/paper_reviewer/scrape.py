"""Stage 1 — pull all submissions/reviews/decisions from OpenReview.

Per-venue spec lives in venues.py. This module just executes the pull.

  - Resumable: skips venues whose JSONL already exists.
  - Streaming: writes one line per paper, so a crash doesn't lose work.
  - API v1 + v2: dispatched per spec.
  - Concurrent: scrapes multiple venues in parallel via thread pool.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .config import OPENREVIEW_BASE, PATHS
from .venues import VENUES, VenueSpec, venue_id_to_spec, venue_ids


def _client_v1():
    import openreview

    return openreview.Client(baseurl="https://api.openreview.net")


def _client_v2():
    import openreview

    return openreview.api.OpenReviewClient(baseurl=OPENREVIEW_BASE)


def _classify_reply(reply: dict, spec: VenueSpec) -> str:
    """Given a reply note, return 'review' / 'meta_review' / 'decision' / 'other'.

    v1 stores invitation as a single string; v2 stores it as `invitations` list.
    Order matters: Meta_Review must be checked before Review (both end with 'Review').
    """
    invs: list[str] = reply.get("invitations") or []
    if not invs:
        single = reply.get("invitation")
        if single:
            invs = [single]

    for i in invs:
        if not isinstance(i, str):
            continue
        if any(i.endswith(s) for s in spec.meta_review_suffixes):
            return "meta_review"
        if any(i.endswith(s) for s in spec.decision_suffixes):
            return "decision"
        if any(i.endswith(s) for s in spec.review_suffixes):
            return "review"
    return "other"


def _scrape_one(spec: VenueSpec, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.venue_id.replace('/', '_')}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    client = _client_v1() if spec.api_version == 1 else _client_v2()
    submissions = client.get_all_notes(invitation=spec.submission_invitation, details="replies")

    tmp_path = out_path.with_suffix(".jsonl.partial")
    with tmp_path.open("w", encoding="utf-8") as f:
        for sub in submissions:
            reviews: list[dict] = []
            meta_review: dict | None = None
            decision: dict | None = None
            for r in sub.details.get("replies", []) if sub.details else []:
                kind = _classify_reply(r, spec)
                if kind == "review":
                    reviews.append(r)
                elif kind == "meta_review":
                    meta_review = r
                elif kind == "decision":
                    decision = r

            content = sub.content or {}
            # v1 content is flat strings; v2 wraps in {"value": ...}.
            def _v(k):
                v = content.get(k)
                return v.get("value") if isinstance(v, dict) else v

            record = {
                "venue": spec.venue_id,
                "api_version": spec.api_version,
                "forum_id": sub.id,
                "title": _v("title"),
                "abstract": _v("abstract"),
                "keywords": _v("keywords"),
                "pdf_url": f"https://openreview.net/pdf?id={sub.id}",
                "reviews": reviews,
                "meta_review": meta_review,
                "decision": decision,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(out_path)
    return out_path


def scrape_venue(venue_id: str, out_dir: Path | None = None) -> Path:
    """Scrape one venue by id. Lookup against the registry."""
    spec = venue_id_to_spec(venue_id)
    if spec is None:
        raise KeyError(f"unknown venue: {venue_id} (add it to src/paper_reviewer/venues.py)")
    return _scrape_one(spec, out_dir or PATHS.raw)


def scrape_all(venues: Iterable[str] | None = None, max_workers: int = 2) -> list[Path]:
    """Scrape many venues concurrently. Failures are logged but don't kill the run."""
    PATHS.ensure()
    venue_list = list(venues) if venues else venue_ids()
    specs = [venue_id_to_spec(v) for v in venue_list]
    out_paths: list[Path] = []

    def _do(spec: VenueSpec):
        try:
            return spec.venue_id, _scrape_one(spec, PATHS.raw)
        except Exception as e:
            print(f"[fail] {spec.venue_id}: {type(e).__name__}: {str(e)[:120]}")
            return spec.venue_id, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do, s) for s in specs if s]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="venues"):
            vid, path = fut.result()
            if path is not None:
                out_paths.append(path)
                print(f"  ok: {vid} -> {path.name}")
    return out_paths


def iter_papers(venues: Iterable[str] | None = None):
    """Yield paper dicts from cached JSONL across venues."""
    venues = list(venues) if venues else venue_ids()
    for v in venues:
        p = PATHS.raw / f"{v.replace('/', '_')}.jsonl"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
