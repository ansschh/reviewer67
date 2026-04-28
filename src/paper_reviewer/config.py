"""Static config: venues, paths, model defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Venue list lives in venues.py as a registry of VenueSpec objects (per-venue
# API version, submission invitation, schema details). This shim keeps the
# legacy `VENUES: list[str]` import working — prefer importing from .venues.
from .venues import venue_ids as _venue_ids
VENUES: list[str] = _venue_ids()

DATA_DIR = Path(os.environ.get("PR_DATA_DIR", "./data")).resolve()


@dataclass(frozen=True)
class Paths:
    raw: Path = DATA_DIR / "raw"
    pdfs: Path = DATA_DIR / "pdfs"
    text: Path = DATA_DIR / "text"
    weaknesses: Path = DATA_DIR / "weaknesses"
    embeddings: Path = DATA_DIR / "embeddings"
    clusters: Path = DATA_DIR / "clusters"
    personas: Path = DATA_DIR / "personas"
    reviews: Path = DATA_DIR / "reviews"
    cache: Path = DATA_DIR / "cache"
    calibration: Path = DATA_DIR / "calibration"

    def ensure(self) -> None:
        for p in (
            self.raw, self.pdfs, self.text, self.weaknesses, self.embeddings,
            self.clusters, self.personas, self.reviews, self.cache, self.calibration,
        ):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


# Model defaults. Overridable via env. We deliberately use different model
# families for review vs. meta-review to reduce compounding bias.
REVIEW_MODEL = os.environ.get("PR_REVIEW_MODEL", "claude-opus-4-7")
PERSONA_MODEL = os.environ.get("PR_PERSONA_MODEL", "claude-sonnet-4-6")
META_MODEL = os.environ.get("PR_META_MODEL", "gpt-5")
EMBED_MODEL = os.environ.get("PR_EMBED_MODEL", "text-embedding-3-large")
WEAKNESS_EXTRACT_MODEL = os.environ.get("PR_WEAKNESS_MODEL", "claude-haiku-4-5-20251001")

# OpenReview API v2 base URL.
OPENREVIEW_BASE = "https://api2.openreview.net"
