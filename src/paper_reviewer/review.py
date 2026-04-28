"""Stage 3 — the multi-agent review panel.

Pipeline:
  1. N persona reviewers, run in parallel, each conditioned on:
       - their persona profile,
       - the relevant cluster labels (rejection priors) for your subfield,
       - the paper text.
  2. A meta-reviewer (different model family — bias compounds otherwise)
     aggregates them and writes an AC-style decision-leaning summary.
  3. A rejection-risk pass: given the panel + meta, score the top-K rejection
     risks the author should fix before submission.
  4. (Optional) Iterate: revise the paper, re-run, track score deltas.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import llm
from .config import META_MODEL, PATHS, REVIEW_MODEL
from .mine import Cluster
from .personas import Persona


# --- prompts ------------------------------------------------------------------

_REVIEWER_SYS = """\
You are a peer reviewer for {venue}. Stay in character — your reviewing style is:

{style}

Your common priorities and pet peeves: {priorities}

You have access to a list of weaknesses that real reviewers in this subfield
frequently cite (mined from prior reviews). Lean on these where they apply,
but DO NOT invent issues that aren't in the paper. Cite specific section or
line numbers from the paper for every weakness you raise. No vague critiques.
"""


_REVIEWER_USER = """\
COMMON SUBFIELD WEAKNESSES (from real reviews, ranked by frequency):
{cluster_block}

PAPER TO REVIEW:
{paper}

Write your review. Output JSON ONLY, this exact schema:
{{
  "summary": "<paper summary in your own words>",
  "strengths": ["..."],
  "weaknesses": [{{"point": "<specific>", "evidence": "<section/line>", "severity": "minor|major|fatal"}}],
  "questions": ["..."],
  "soundness": <1-4>,
  "presentation": <1-4>,
  "contribution": <1-4>,
  "rating": <1-10>,
  "confidence": <1-5>
}}
"""


_META_PROMPT = """\
You are an Area Chair for {venue}. Below are {n} reviews of one paper. Write a
meta-review that:
- Reconciles disagreements (whose concerns are load-bearing?).
- Lists the issues that, if unfixed, would cause rejection.
- Estimates an acceptance probability in [0, 1] based on the panel's signal.

REVIEWS:
{reviews}

Output JSON:
{{
  "synthesis": "<2-3 paragraphs>",
  "fatal_issues": ["..."],
  "fixable_issues": ["..."],
  "accept_prob": <0..1>
}}
"""


_RISK_PROMPT = """\
Given this multi-reviewer panel and meta-review, list the top 5 changes the
authors should make before submission, ranked by how much they reduce rejection
risk. For each, cite the reviewer(s) raising it and what concrete fix would
satisfy them.

PANEL:
{panel}

META:
{meta}

Output JSON:
{{ "top_risks": [{{"risk": "...", "raised_by": ["..."], "fix": "..."}}, ...] }}
"""


# --- runtime ------------------------------------------------------------------

@dataclass
class ReviewerOutput:
    persona: str
    venue: str
    review: dict
    model: str
    elapsed_sec: float


@dataclass
class PanelResult:
    paper_id: str
    reviews: list[ReviewerOutput]
    meta_review: dict
    risks: dict
    avg_rating: float
    accept_prob: float


def _format_clusters(clusters: list[Cluster], top_k: int = 12) -> str:
    if not clusters:
        return "(none mined yet)"
    rows = []
    for c in clusters[:top_k]:
        ar = f"{c.accept_rate:.0%}" if c.accept_rate is not None else "n/a"
        rows.append(f"- {c.label}  (n={c.size}, accept_rate={ar})")
    return "\n".join(rows)


async def _run_reviewer(
    persona: Persona,
    paper: str,
    cluster_block: str,
    model: str,
) -> ReviewerOutput:
    sys = _REVIEWER_SYS.format(
        venue=persona.venue,
        style=persona.style,
        priorities=", ".join(persona.priorities),
    )
    user = _REVIEWER_USER.format(cluster_block=cluster_block, paper=paper)
    t0 = time.time()
    raw = await llm.chat_async(model, user, system=sys, max_tokens=4096)
    review = llm._parse_json(raw)
    return ReviewerOutput(
        persona=persona.name,
        venue=persona.venue,
        review=review,
        model=model,
        elapsed_sec=time.time() - t0,
    )


def _meta_review(personas_out: list[ReviewerOutput], venue: str) -> dict:
    rendered = []
    for r in personas_out:
        rendered.append(f"[{r.persona} @ {r.venue}]\n{json.dumps(r.review, indent=2)}")
    prompt = _META_PROMPT.format(venue=venue, n=len(personas_out), reviews="\n---\n".join(rendered))
    # Generous max_tokens to leave room for GPT-5 reasoning before output.
    return llm.chat_json(META_MODEL, prompt, max_tokens=8000, reasoning_effort="medium")


def _risk_pass(personas_out: list[ReviewerOutput], meta: dict) -> dict:
    panel = "\n---\n".join(
        f"[{r.persona}] {json.dumps(r.review, indent=2)}" for r in personas_out
    )
    prompt = _RISK_PROMPT.format(panel=panel, meta=json.dumps(meta, indent=2))
    return llm.chat_json(META_MODEL, prompt, max_tokens=8000, reasoning_effort="medium")


async def run_panel(
    paper_text: str,
    paper_id: str,
    personas: list[Persona],
    clusters: list[Cluster],
    venue: str = "ICLR.cc/2026",
    model: str = REVIEW_MODEL,
) -> PanelResult:
    cluster_block = _format_clusters(clusters)
    tasks = [_run_reviewer(p, paper_text, cluster_block, model) for p in personas]
    reviews = await asyncio.gather(*tasks)

    meta = _meta_review(reviews, venue=venue)
    risks = _risk_pass(reviews, meta)

    ratings = [r.review.get("rating") for r in reviews if isinstance(r.review.get("rating"), (int, float))]
    avg = sum(ratings) / len(ratings) if ratings else float("nan")

    return PanelResult(
        paper_id=paper_id,
        reviews=reviews,
        meta_review=meta,
        risks=risks,
        avg_rating=avg,
        accept_prob=float(meta.get("accept_prob", 0.0)),
    )


def save_panel(result: PanelResult, name: str | None = None) -> Path:
    PATHS.ensure()
    name = name or f"{result.paper_id}_{int(time.time())}"
    out = PATHS.reviews / f"{name}.json"
    out.write_text(
        json.dumps(
            {
                "paper_id": result.paper_id,
                "avg_rating": result.avg_rating,
                "accept_prob": result.accept_prob,
                "meta_review": result.meta_review,
                "risks": result.risks,
                "reviews": [asdict(r) for r in result.reviews],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out


def iterate(
    paper_text: str,
    paper_id: str,
    personas: list[Persona],
    clusters: list[Cluster],
    rounds: int,
    revise_fn,
    venue: str = "ICLR.cc/2026",
) -> list[PanelResult]:
    """Run review -> revise -> review for `rounds` iterations.

    `revise_fn(paper_text, panel_result) -> new_paper_text` is supplied by the
    caller (could be an LLM-driven revision or just a manual stop point).
    Track score deltas to know if revisions are actually helping.
    """
    history: list[PanelResult] = []
    text = paper_text
    for i in range(rounds):
        result = asyncio.run(run_panel(text, f"{paper_id}_r{i}", personas, clusters, venue=venue))
        save_panel(result, name=f"{paper_id}_r{i}")
        history.append(result)
        print(
            f"[round {i}] avg_rating={result.avg_rating:.2f}  "
            f"accept_prob={result.accept_prob:.2f}  "
            f"top_risk={(result.risks.get('top_risks') or [{}])[0].get('risk', '')[:80]}"
        )
        if i + 1 == rounds:
            break
        text = revise_fn(text, result)
    return history
