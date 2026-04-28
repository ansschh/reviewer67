"""Normalize per-venue review schemas to a common shape.

OpenReview review notes vary in field names and rating scales by venue and year:
ICLR uses `rating` 1-10, NeurIPS sometimes uses `recommendation`, ICML's recent
years use `soundness`/`presentation`/`contribution` (1-4), etc. This module
hides those details so downstream code only sees a `NormalizedReview`.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


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
    """Pull `content[key]['value']` for the first key that's present."""
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


_ICML_2025_CRITIQUE_FIELDS = (
    "methods_and_evaluation_criteria",
    "experimental_designs_or_analyses",
    "theoretical_claims",
    "essential_references_not_discussed",
    "other_strengths_and_weaknesses",
)


def normalize_review(venue: str, forum_id: str, note: dict) -> NormalizedReview:
    content = note.get("content", {})
    rating = _rating_to_float(
        _val(content, "rating", "recommendation", "overall_rating", "overall_recommendation")
    )
    # ICLR uses 1-10. NeurIPS sometimes 1-10. ICML's recent `recommendation` is 1-10.
    # ICML 2025's `overall_recommendation` is 1-5; we rescale below. Older 1-5 venues too.
    if rating is not None and rating <= 5:
        if "overall_recommendation" in content or "overall_rating" in content:
            rating = rating * 2  # 1-5 -> 2-10
        elif "rating" not in content and "recommendation" not in content:
            rating = rating * 2

    # ICML 2025 has no `weaknesses` field — critique is split across structured
    # subsections. Concatenate them as the weaknesses block so downstream
    # extraction sees one consistent field.
    weaknesses = _val(content, "weaknesses", "limitations")
    if not weaknesses:
        parts = [s for f in _ICML_2025_CRITIQUE_FIELDS if (s := _val(content, f))]
        weaknesses = "\n\n".join(parts)

    return NormalizedReview(
        venue=venue,
        forum_id=forum_id,
        review_id=note.get("id", ""),
        summary=_val(content, "summary", "paper_summary") or "",
        strengths=_val(content, "strengths", "strengths_and_weaknesses") or "",
        weaknesses=weaknesses or "",
        questions=_val(content, "questions", "questions_for_authors") or "",
        rating=rating,
        confidence=_rating_to_float(_val(content, "confidence")),
        soundness=_rating_to_float(_val(content, "soundness")),
        presentation=_rating_to_float(_val(content, "presentation")),
        contribution=_rating_to_float(_val(content, "contribution")),
    )


_ACCEPT_TOKENS = ("accept", "oral", "spotlight", "poster")
_REJECT_TOKENS = ("reject", "withdraw", "desk reject")


def normalize_decision(note: dict | None) -> NormalizedDecision:
    if note is None:
        return NormalizedDecision(accepted=None, raw="")
    content = note.get("content", {})
    raw = _val(content, "decision", "recommendation") or ""
    raw_l = str(raw).lower()
    if any(t in raw_l for t in _REJECT_TOKENS):
        return NormalizedDecision(accepted=False, raw=raw)
    if any(t in raw_l for t in _ACCEPT_TOKENS):
        return NormalizedDecision(accepted=True, raw=raw)
    return NormalizedDecision(accepted=None, raw=raw)
