"""Build reviewer personas from real OpenReview reviews.

A persona is a reviewing-style profile: priorities, pet peeves, typical rating
calibration, what they consistently flag in your subfield. We sample 20-30
real reviews (from rejected papers, weighted toward your subfield), then have
an LLM extract the style. The output goes into the simulator's system prompt.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from . import llm
from .config import PATHS, PERSONA_MODEL
from .normalize import normalize_decision, normalize_review
from .scrape import iter_papers


@dataclass
class Persona:
    name: str
    venue: str
    style: str             # the LLM-generated profile
    priorities: list[str]  # short list of recurring criticism types
    avg_rating: float
    n_reviews_sampled: int


def _sample_reviews(
    venue_filter: str | None,
    rejected_only: bool,
    n: int,
    seed: int,
) -> list[dict]:
    """Collect (review, decision, paper) tuples matching filters."""
    rng = random.Random(seed)
    pool: list[dict] = []
    for paper in iter_papers():
        if venue_filter and venue_filter not in paper["venue"]:
            continue
        decision = normalize_decision(paper.get("decision"))
        if rejected_only and decision.accepted is not False:
            continue
        for rn in paper.get("reviews", []):
            nr = normalize_review(paper["venue"], paper["forum_id"], rn)
            if not nr.weaknesses:
                continue
            pool.append(
                {
                    "venue": nr.venue,
                    "title": paper.get("title"),
                    "review": nr.to_dict(),
                    "accepted": decision.accepted,
                }
            )
    rng.shuffle(pool)
    return pool[:n]


_PERSONA_PROMPT = """\
You are profiling a peer reviewer's style based on a sample of their reviews.

Read the reviews below and produce a reviewer persona that another LLM can
adopt to write reviews in this same style. The persona should capture:

- What this reviewer prioritizes (rigor, novelty, clarity, theoretical
  grounding, empirical breadth, etc.) — be specific to what shows up here.
- Recurring criticisms — the patterns this reviewer falls back on.
- Tone (terse vs verbose, hostile vs constructive, hedged vs decisive).
- Calibration: do they give middling scores often? Are they harsh? When do
  they cite specific lines/sections vs make vague critiques?

Return a JSON object:
{{
  "style": "<2-4 paragraph profile, written in the second person ('You ...')>",
  "priorities": ["<short phrase>", "<short phrase>", ...]
}}

REVIEWS:
{reviews}
"""


def build_persona(
    name: str,
    venue: str,
    rejected_only: bool = True,
    n: int = 25,
    seed: int = 0,
) -> Persona:
    samples = _sample_reviews(venue_filter=venue, rejected_only=rejected_only, n=n, seed=seed)
    if not samples:
        raise RuntimeError(f"No reviews matched venue={venue!r}, rejected_only={rejected_only}")

    rendered = []
    for s in samples:
        r = s["review"]
        rendered.append(
            f"[venue={s['venue']} accepted={s['accepted']} rating={r['rating']} conf={r['confidence']}]\n"
            f"SUMMARY: {r['summary'][:600]}\n"
            f"WEAKNESSES: {r['weaknesses'][:1500]}\n"
            f"QUESTIONS: {r['questions'][:600]}\n"
        )
    prompt = _PERSONA_PROMPT.format(reviews="\n---\n".join(rendered))
    out = llm.chat_json(PERSONA_MODEL, prompt, max_tokens=4000, reasoning_effort="low")

    ratings = [s["review"]["rating"] for s in samples if s["review"]["rating"] is not None]
    persona = Persona(
        name=name,
        venue=venue,
        style=out.get("style", ""),
        priorities=out.get("priorities", []),
        avg_rating=sum(ratings) / len(ratings) if ratings else 5.0,
        n_reviews_sampled=len(samples),
    )
    return persona


def default_persona_specs() -> list[dict]:
    """A decent default panel covering the rejection-archetype space.

    Each spec is consumed by build_persona via different (seed, rejected_only)
    so the LLM extracts a different slice of reviewer styles."""
    return [
        {"name": "rigor_hawk",      "venue": "ICLR.cc/2025",    "rejected_only": True,  "seed": 1},
        {"name": "novelty_skeptic", "venue": "NeurIPS.cc/2024", "rejected_only": True,  "seed": 2},
        {"name": "empirical_reviewer", "venue": "ICML.cc/2025", "rejected_only": True,  "seed": 3},
        {"name": "balanced_ac",     "venue": "NeurIPS.cc/2025", "rejected_only": False, "seed": 4},
        {"name": "writing_critic",  "venue": "ICLR.cc/2024",    "rejected_only": True,  "seed": 5},
    ]


def save_personas(personas: list[Persona], name: str = "panel") -> Path:
    PATHS.ensure()
    out = PATHS.personas / f"{name}.json"
    out.write_text(json.dumps([asdict(p) for p in personas], indent=2), encoding="utf-8")
    return out


def load_personas(name: str = "panel") -> list[Persona]:
    p = PATHS.personas / f"{name}.json"
    return [Persona(**r) for r in json.loads(p.read_text(encoding="utf-8"))]
