# Changelog

## [0.1.0] — 2026-04-28

Initial release.

### Added
- OpenReview scraper (API v2) for ICLR / NeurIPS / ICML, with `details='replies'` and resumable JSONL output.
- Per-venue normalizer that handles ICLR's `rating`, NeurIPS's `recommendation`, and ICML 2025's structured review fields (`methods_and_evaluation_criteria`, `experimental_designs_or_analyses`, etc.).
- Weakness-sentence extraction + embedding + PCA + HDBSCAN clustering, with abstract-conditioned subfield filtering.
- LLM-labeled cluster names, accept-rate per cluster.
- 5-persona reviewer panel built from sampled real reviews (rigor_hawk / novelty_skeptic / empirical_reviewer / balanced_ac / writing_critic).
- Multi-agent review pipeline: parallel persona reviews → cross-family meta-reviewer → rejection-risk top-5 with concrete fixes.
- Rebuttal generator + persona-conditioned re-review with score-change estimate.
- Calibration: stratified hold-out, Spearman ρ + ROC-AUC + Jaccard top-3 cluster overlap, all with paired bootstrap 95% CIs.
- Anthropic + OpenAI batch-API support for offline calibration jobs (50% off).
- Concurrent thread-pool sync mode for calibration; configurable concurrency via `PR_CALIBRATE_CONCURRENCY`.
- SQLite-backed prompt-hash cache for chat + embeddings.
- CLI: `scrape | mine | personas | review | calibrate | train-predictor`.

### Calibration baselines

**Stage 2 (cheap, N=198, Haiku 4.5, 2 reviewers):**
Spearman ρ=+0.434 [95% CI +0.32, +0.54], AUC=+0.811 [95% CI +0.75, +0.86], Jaccard top-3=+0.681. Simulator conservatively biased (mean −0.91 vs real reviewers).

**Stage 3 (boundary upgrade, N=20 worst-disagreement, Opus 4.7, 5 reviewers):**
Production Opus panel closes 31% of the rating-bias gap (−3.40 → −2.33) on the hardest cases. AUC stays strong (+0.789) on this subset. Spearman drops to +0.17 due to range restriction (boundary papers cluster in real-rating [6.5, 9.0]) — not a real signal degradation, just a statistical artifact of the subset selection. Simulator is systematically pessimistic: every boundary case rated below its real score.

### Known limitations
- ICML 2024 not on OpenReview publicly — skipped.
- ICML 2026 review cycle still open — skipped.
- NeurIPS / ICML 2024-25 don't expose meta-reviews publicly.
- Persona panels were sampled from the same venues we calibrate against — mild data leakage.
- No rebuttal calibration against real post-rebuttal score changes yet.
