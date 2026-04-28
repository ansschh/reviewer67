"""Rebuttal simulator.

Two passes per reviewer:
  1. Generate the rebuttal you'd send to this reviewer.
  2. Have the reviewer agent decide whether the rebuttal would change their
     score, and by how much.

Calibrate against real post-rebuttal score changes if you have them — the
scraped data sometimes contains revised reviews after the rebuttal phase, and
the score deltas are real ground truth.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import llm
from .config import META_MODEL, REVIEW_MODEL
from .personas import Persona
from .review import ReviewerOutput


_REBUTTAL_PROMPT = """\
You are an author writing a rebuttal. Address this reviewer's concerns concisely.
For each weakness they raised, either (a) clarify a misunderstanding with
specific paper references, or (b) propose a concrete revision/experiment.
Do not hand-wave. Length: 250-400 words.

REVIEWER REVIEW:
{review}

PAPER (for context, do not paraphrase verbatim):
{paper}

Output your rebuttal as plain text.
"""


_REREVIEW_PROMPT = """\
You previously gave this review. The authors have now responded. Re-evaluate.
Be skeptical of hand-waving but credit concrete commitments and clarifications.

YOUR ORIGINAL REVIEW:
{review}

AUTHOR REBUTTAL:
{rebuttal}

Output JSON:
{{
  "score_change": <signed integer, e.g. -1, 0, +1>,
  "new_rating": <1-10>,
  "reasoning": "<2-3 sentences>"
}}
"""


@dataclass
class RebuttalOutcome:
    persona: str
    rebuttal: str
    score_change: int
    new_rating: float
    reasoning: str


def simulate_rebuttal(
    paper_text: str,
    reviewer_out: ReviewerOutput,
    persona: Persona,
    rebuttal_model: str = META_MODEL,
    rereview_model: str = REVIEW_MODEL,
) -> RebuttalOutcome:
    review_str = json.dumps(reviewer_out.review, indent=2)
    rebuttal = llm.chat(
        rebuttal_model,
        _REBUTTAL_PROMPT.format(review=review_str, paper=paper_text),
        system="You are a careful, specific paper author writing a rebuttal.",
        max_tokens=1500,
    )
    rer = llm.chat_json(
        rereview_model,
        _REREVIEW_PROMPT.format(review=review_str, rebuttal=rebuttal),
        system=f"You are reviewer '{persona.name}' for {persona.venue}. Stay in character.",
        max_tokens=600,
    )
    return RebuttalOutcome(
        persona=persona.name,
        rebuttal=rebuttal,
        score_change=int(rer.get("score_change", 0)),
        new_rating=float(rer.get("new_rating", reviewer_out.review.get("rating", 5))),
        reasoning=rer.get("reasoning", ""),
    )
