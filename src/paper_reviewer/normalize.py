"""Normalize per-venue review schemas to a common shape.

OpenReview review notes vary in field names and rating scales by venue and
year (v1 vs v2, ICLR's `rating` vs NeurIPS's `recommendation` vs ICML 2025's
structured fields, etc.). This module hides those differences so downstream
code only sees a `NormalizedReview`.

The schema-specific bits (which fields to look at, structured-critique field
list) live in the venue registry — see `venues.py`. This module just executes
the spec.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .venues import VenueSpec, venue_id_to_spec


@dataclass
class NormalizedReview:
    venue: str
    forum_id: str
    review_id: str
    summary: str
    strengths: str
    weaknesses: str
    questions: str
    rating: float | None          # rescaled to 1-10
    confidence: float | None      # 1-5
    soundness: float | None       # 1-4 if present
    presentation: float | None
    contribution: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedDecision:
    accepted: bool | None
    raw: str


def _val(content: dict | None, *keys: str) -> Any:
    """Pull `content[key]` for the first key present.

    v1 stores content values directly: `content[key] = "..."`.
    v2 wraps them: `content[key] = {"value": "..."}`.
    Handle both.
    """
    if not content:
        return None
    for k in keys:
        if k in content:
            v = content[k]
            return v.get("value") if isinstance(v, dict) else v
    return None


_RATING_NUM = re.compile(r"^\s*(\d+(?:\.\d+)?)")


def _rating_to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _RATING_NUM.match(v)
        if m:
            return float(m.group(1))
    return None


def normalize_review(venue: str, forum_id: str, note: dict) -> NormalizedReview:
    spec = venue_id_to_spec(venue)
    content = note.get("content", {}) or {}
    rating = _rating_to_float(_val(content, *(spec.rating_fields if spec else (
        "rating", "recommendation", "overall_rating", "overall_recommendation",
    ))))

    # 1-5 scales (ICML 2025 'overall_recommendation', some older venues' 'overall_rating')
    # rescale to 1-10 so downstream comparisons are apples-to-apples.
    if rating is not None and rating <= 5:
        if "overall_recommendation" in content or "overall_rating" in content:
            rating = rating * 2
        elif "rating" not in content and "recommendation" not in content:
            rating = rating * 2

    weakness_keys = spec.weakness_fields if spec else ("weaknesses", "limitations", "strength_and_weaknesses")
    weaknesses = _val(content, *weakness_keys)
    if not weaknesses and spec and spec.structured_critique_fields:
        parts = [s for f in spec.structured_critique_fields if (s := _val(content, f))]
        weaknesses = "\n\n".join(parts)

    summary_keys = spec.summary_fields if spec else ("summary", "paper_summary", "summary_of_the_paper")
    questions_keys = spec.questions_fields if spec else ("questions", "questions_for_authors")
    confidence_keys = spec.confidence_fields if spec else ("confidence",)

    return NormalizedReview(
        venue=venue,
        forum_id=forum_id,
        review_id=note.get("id", ""),
        summary=_val(content, *summary_keys) or "",
        strengths=_val(content, "strengths", "strengths_and_weaknesses") or "",
        weaknesses=weaknesses or "",
        questions=_val(content, *questions_keys) or "",
        rating=rating,
        confidence=_rating_to_float(_val(content, *confidence_keys)),
        soundness=_rating_to_float(_val(content, "soundness")),
        presentation=_rating_to_float(_val(content, "presentation")),
        contribution=_rating_to_float(_val(content, "contribution")),
    )


_ACCEPT_TOKENS = ("accept", "oral", "spotlight", "poster")
_REJECT_TOKENS = ("reject", "withdraw", "desk reject")


def normalize_decision(note: dict | None) -> NormalizedDecision:
    if note is None:
        return NormalizedDecision(accepted=None, raw="")
    content = note.get("content", {}) or {}
    raw = _val(content, "decision", "recommendation") or ""
    raw_l = str(raw).lower()
    if any(t in raw_l for t in _REJECT_TOKENS):
        return NormalizedDecision(accepted=False, raw=raw)
    if any(t in raw_l for t in _ACCEPT_TOKENS):
        return NormalizedDecision(accepted=True, raw=raw)
    return NormalizedDecision(accepted=None, raw=raw)
