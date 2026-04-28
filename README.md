# paper-reviewer

> A pre-submission review simulator that learns from real OpenReview data.
> Mines reviewer rejection patterns, builds persona-conditioned LLM reviewers, runs them as a panel against your paper, and scores your rejection risk before you ever submit.

[![tests](https://img.shields.io/badge/tests-passing-brightgreen)](#) [![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![python](https://img.shields.io/badge/python-3.10%2B-blue)](#)

## What it actually does

1. **Scrape** — pulls all submissions, reviews, and accept/reject decisions from ICLR / NeurIPS / ICML on OpenReview into resumable JSONL.
2. **Mine** — extracts weakness sentences from real reviews, embeds, HDBSCAN-clusters them, and labels each cluster with the criticism it represents. Filtered to *your* subfield via abstract similarity.
3. **Simulate** — runs a panel of 5 LLM reviewer-personas (each derived from real reviewer styles in the data), then a different-model meta-reviewer, then a top-5 rejection-risk pass with concrete fixes.

Plus calibration that tells you how much to trust the panel.

## Calibration — does this actually work?

Held-out N=198 OpenReview papers, stratified across accept / reject / borderline:

| Metric | Value | 95% CI | Interpretation |
|---|---:|---:|---|
| Spearman ρ (rating)        | **+0.434** | [+0.32, +0.54] | Ranks papers ~as well as a moderate reviewer |
| AUC (accept/reject)        | **+0.811** | [+0.75, +0.86] | Genuinely predictive of acceptance |
| Jaccard top-3 weakness clusters | +0.681 | [+0.20, +1.00] | Surfaces same top concerns as real reviewers (bimodal — wide CI) |

Run with `claude-haiku-4-5-20251001`, 2 reviewers per panel, batch sync. Total cost: ~$4.

**Calibration caveats — read these:**
- The simulator is **conservatively biased** (mean −0.91 vs real reviewers, std compressed 2.6×). Treat simulated ratings as a *lower bound*.
- Persona panels are mined from the same venues we calibrate against → mild data leakage. A held-out year is on the roadmap.
- Production runs use `claude-opus-4-7`, which is likely better calibrated than the Haiku numbers above.

## Quick start

```bash
git clone https://github.com/<you>/paper-reviewer && cd paper-reviewer
python -m venv .venv && . .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -e .
cp .env.example .env                                # add ANTHROPIC_API_KEY and OPENAI_API_KEY

# Stage 1 — scrape (one-time, ~5-10 min for current 6-venue set)
paper-reviewer scrape

# Stage 2 — mine weaknesses, filtered to YOUR subfield
echo "<your paper's abstract>" > abstract.txt
paper-reviewer mine --abstract abstract.txt --min-cluster-size 10

# Stage 3 — build the persona panel (~$2)
paper-reviewer personas

# Stage 4 — review your paper (~$5 with default Opus reviewers)
paper-reviewer review --pdf paper.pdf --venue ICLR.cc/2026

# Optional — validate the simulator on real papers (~$5 cheap recipe)
paper-reviewer calibrate \
    --n 200 --reviewers 2 \
    --model claude-haiku-4-5-20251001 --no-batch
```

Outputs go to `./data/` (override with `PR_DATA_DIR`). LLM calls are cached by prompt hash so re-runs are ~free.

## What's covered

| Venue | Years | Reviews | Decisions | Meta-reviews |
|---|---|---:|---:|---:|
| ICLR        | 2024, 2025, 2026 | ✅ | ✅ | ✅ |
| NeurIPS     | 2024, 2025       | ✅ | ✅ | — (not public) |
| ICML        | 2025             | ✅ | ✅ | — (not public) |

ICML 2024 didn't release public reviews on OpenReview. ICML 2026 review cycle is still open. Tier-A expansion (ICLR 2018–2023, NeurIPS 2021–2023, COLM, AISTATS, UAI, CoRL, RLC, TMLR, ACL Rolling Review) is on the roadmap and gated by per-venue normalizer specs.

Conferences NOT on OpenReview (CVPR, AAAI, IJCAI, OSDI, SOSP, VLDB, SIGMOD, …) are intentionally **not** supported, even via "guideline-based" personas. We won't ship coverage we can't ground in real data.

## Architecture

```
src/paper_reviewer/
  config.py       venue list, paths, model defaults
  scrape.py       OpenReview pull (details='replies', resumable JSONL)
  normalize.py    per-venue review-field normalizer
  extract.py      PDF → text via pymupdf, with retry+backoff
  llm.py          cached Anthropic + OpenAI client (sqlite cache, prompt-hash dedupe)
  batch.py        Anthropic + OpenAI batch APIs (50% off, 24h SLA)
  mine.py         weakness sentence extraction, embeddings, PCA + HDBSCAN, cluster labelling
  personas.py     samples N real reviews per archetype → reviewer style profiles
  review.py       async multi-agent panel + meta-review + rejection-risk + iteration loop
  rebuttal.py     rebuttal generator + persona-conditioned re-review with score deltas
  predictor.py    logistic-regression accept/reject classifier on review embeddings
  calibrate.py    stratified hold-out validation: Spearman + AUC + Jaccard with bootstrap CIs
  cli.py          entry point: scrape | mine | personas | review | calibrate | train-predictor
```

## Models

The default mix uses **different model families** for review and meta-review to break compounding bias:

| Role | Default | Why |
|---|---|---|
| Reviewer panel    | `claude-opus-4-7`            | Deep critique |
| Persona builder   | `claude-sonnet-4-6`          | Style extraction |
| Meta-reviewer     | `gpt-5`                      | Different family — independent aggregation |
| Cluster labeler   | `claude-haiku-4-5-20251001` | Cheap, short outputs |
| Embeddings        | `text-embedding-3-large`     | OpenAI 3072-dim |

Override any of these via `.env` (`PR_REVIEW_MODEL`, etc.). `local:<sentence-transformer-name>` works for embeddings if you want to avoid OpenAI.

## Cost ballpark

| Run | Models | Cost |
|---|---|---:|
| Scrape (one-time)            | none           | $0 |
| Mine + cluster (filtered)    | embeddings + Haiku labels | $1-2 |
| Persona panel build          | Sonnet × 5     | $1-2 |
| Single paper review (Pro)    | 5× Opus + GPT-5 meta | $4-5 |
| Calibration (cheap recipe, N=200) | Haiku 4.5  | $4 |
| Calibration (full recipe, N=200)  | Opus 4.7   | $200+ |

LLM cache deduplicates by prompt hash — re-running with the same paper is free until you change the prompt or model.

## Limitations to know about

- Persona panel is generated from rejected papers, so it's *biased toward criticism* — a deliberate choice (you want this for pre-submission risk-finding), but don't read low ratings as a verdict.
- Calibration was run on the same venues that personas were trained from. There is some leakage. We expect ρ to drop a bit on a fully-clean held-out year.
- Top-3 weakness Jaccard is bimodal (papers tend to match all or none) so its CI is wide; treat the mean as the meaningful number.
- We don't yet do rebuttal calibration against real post-rebuttal score changes. The rebuttal sim works but isn't validated.
- Workshop venues, CVPR-track venues, and HotCRP venues (OSDI, SOSP) are not in scope. See "What's covered" above.

## Contributing

Tests must pass on every PR (`pytest -q`). Code lives under `src/paper_reviewer/`. New venue support means adding a `VenueSpec` and, if its review schema is exotic, a normalizer entry.

Issues and PRs welcome — especially:
- New OpenReview venue support
- Better persona prompts (compare to held-out reviews)
- Rebuttal calibration against real post-rebuttal score data
- Local embedding fallback that achieves > 0.8 cosine agreement with `text-embedding-3-large`

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use this for research, cite as:

```
@software{paper_reviewer_2026,
  title = {paper-reviewer: pre-submission review simulator grounded in OpenReview data},
  year = {2026},
  url = {https://github.com/<you>/paper-reviewer},
}
```
