"""Venue registry — declarative specs for every venue we scrape.

OpenReview has two API versions and a half-dozen review schema dialects. Rather
than hardcode the differences across scrape.py / normalize.py / personas.py,
each venue is a single `VenueSpec` here. Adding a new venue = adding one entry
(plus a normalizer entry if its schema is exotic).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VenueSpec:
    venue_id: str                       # e.g. "ICLR.cc/2024/Conference"
    api_version: int                    # 1 or 2
    submission_invitation: str          # e.g. "ICLR.cc/2024/Conference/-/Submission"

    # Reply classification (suffix of the invitation field on the reply note)
    review_suffixes: tuple[str, ...] = ("Official_Review",)
    meta_review_suffixes: tuple[str, ...] = ("Meta_Review",)
    decision_suffixes: tuple[str, ...] = ("Decision",)

    # Field names to try (in priority order) when normalizing a review note
    rating_fields: tuple[str, ...] = ("rating", "recommendation", "overall_rating", "overall_recommendation")
    confidence_fields: tuple[str, ...] = ("confidence",)
    weakness_fields: tuple[str, ...] = ("weaknesses", "limitations", "strength_and_weaknesses")
    summary_fields: tuple[str, ...] = ("summary", "paper_summary", "summary_of_the_paper")
    questions_fields: tuple[str, ...] = ("questions", "questions_for_authors")

    # ICML-2025-style structured critique fields concatenated to form weaknesses
    structured_critique_fields: tuple[str, ...] = ()


# ----------------------------------------------------------------------------

# Tier A: review-mined venues with verified scrape paths.

_ICML_2025_FIELDS = (
    "methods_and_evaluation_criteria",
    "experimental_designs_or_analyses",
    "theoretical_claims",
    "essential_references_not_discussed",
    "other_strengths_and_weaknesses",
)

_v1 = lambda v, year: VenueSpec(
    venue_id=v,
    api_version=1,
    submission_invitation=f"{v}/-/Blind_Submission",
)

_v2 = lambda v: VenueSpec(
    venue_id=v,
    api_version=2,
    submission_invitation=f"{v}/-/Submission",
)


VENUES: list[VenueSpec] = [
    # ---- ICLR (v1 = 2018-2023, v2 = 2024+) ----
    _v1("ICLR.cc/2018/Conference", 2018),
    _v1("ICLR.cc/2019/Conference", 2019),
    _v1("ICLR.cc/2020/Conference", 2020),
    _v1("ICLR.cc/2021/Conference", 2021),
    _v1("ICLR.cc/2022/Conference", 2022),
    _v1("ICLR.cc/2023/Conference", 2023),
    _v2("ICLR.cc/2024/Conference"),
    _v2("ICLR.cc/2025/Conference"),
    _v2("ICLR.cc/2026/Conference"),

    # ---- NeurIPS (v1 for 2021-2022, v2 from 2023 onward) ----
    _v1("NeurIPS.cc/2021/Conference", 2021),
    _v1("NeurIPS.cc/2022/Conference", 2022),
    _v2("NeurIPS.cc/2023/Conference"),
    _v2("NeurIPS.cc/2024/Conference"),
    _v2("NeurIPS.cc/2025/Conference"),

    # ---- ICML — only 2025 has public reviews on OR (with structured schema) ----
    VenueSpec(
        venue_id="ICML.cc/2025/Conference",
        api_version=2,
        submission_invitation="ICML.cc/2025/Conference/-/Submission",
        rating_fields=("overall_recommendation", "rating", "recommendation"),
        structured_critique_fields=_ICML_2025_FIELDS,
    ),

    # ---- COLM (Conference on Language Modeling) ----
    _v2("colmweb.org/COLM/2024/Conference"),
    _v2("colmweb.org/COLM/2025/Conference"),

    # ---- AISTATS ----
    _v2("aistats.org/AISTATS/2025/Conference"),
    # AISTATS 2024 not on OpenReview.

    # ---- UAI (Uncertainty in AI) ----
    _v2("auai.org/UAI/2024/Conference"),
    _v2("auai.org/UAI/2025/Conference"),

    # ---- CoRL (robotics) ----
    _v2("robot-learning.org/CoRL/2024/Conference"),
    _v2("robot-learning.org/CoRL/2025/Conference"),

    # ---- RLC (Reinforcement Learning Conference) ----
    _v2("rl-conference.cc/RLC/2024/Conference"),
    _v2("rl-conference.cc/RLC/2025/Conference"),

    # NeurIPS Datasets & Benchmarks doesn't seem to be a separate OR group.
]


def venue_id_to_spec(venue_id: str) -> VenueSpec | None:
    for v in VENUES:
        if v.venue_id == venue_id:
            return v
    return None


def venue_ids() -> list[str]:
    return [v.venue_id for v in VENUES]
