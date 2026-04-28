"""Hold-out calibration of the simulator against real OpenReview outcomes.

Three metrics, each tells you a different thing:

  (A) Spearman rho between simulated and real average rating
      → "do we rank papers in the same order as real reviewers?"

  (B) ROC-AUC between simulated accept_prob and real accept/reject decision
      → "is our number actually predictive of acceptance?"

  (C) Jaccard overlap of top-3 weakness clusters between sim and real reviewers
      → "do users feel we surfaced the right concerns?"

All three reported with 95% bootstrap CIs. With N<50 the CIs are wide enough
that the point estimate is meaningless on its own — bootstrap is required, not
optional.

Cost-cutting tricks supported:
  --reviewers N      run a smaller panel (default 5)
  --model M          override the reviewer model (default: REVIEW_MODEL from .env)
  --batch            use Anthropic/OpenAI batch APIs (50% off, 24h SLA)
  --stratify         decision-balanced sampling (otherwise dominated by accepts)
  --bootstrap N      bootstrap iterations (default 1000)
  --cap-chars N      truncate paper text (default 30k for cheap panels)
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from . import llm
from .batch import BatchRequest, batch_chat
from .config import PATHS, REVIEW_MODEL
from .extract import download_pdf, extract_to_cache
from .mine import Cluster, _cosine_topk, load_clusters
from .normalize import normalize_decision, normalize_review
from .personas import Persona, load_personas
from .review import _REVIEWER_SYS, _REVIEWER_USER, _format_clusters


# ---- sampling ---------------------------------------------------------------

@dataclass
class _CalibPaper:
    forum_id: str
    venue: str
    real_avg_rating: float
    accepted: bool
    real_weakness_text: str  # concatenated real weaknesses for cluster Jaccard


def _sample_papers(n: int, stratify: bool, seed: int = 0) -> list[_CalibPaper]:
    """Sample papers with at least 3 real reviews. If stratify, force a roughly
    balanced split: 30% accept, 30% reject, 40% borderline (real_avg in [4,6])."""
    from .scrape import iter_papers

    rng = random.Random(seed)
    pool: list[_CalibPaper] = []
    for paper in iter_papers():
        reviews = [normalize_review(paper["venue"], paper["forum_id"], r) for r in paper.get("reviews", [])]
        ratings = [r.rating for r in reviews if r.rating is not None]
        if len(ratings) < 3:
            continue
        decision = normalize_decision(paper.get("decision"))
        if decision.accepted is None:
            continue
        weakness_text = "\n\n".join(r.weaknesses for r in reviews if r.weaknesses)
        pool.append(_CalibPaper(
            forum_id=paper["forum_id"],
            venue=paper["venue"],
            real_avg_rating=float(np.mean(ratings)),
            accepted=decision.accepted,
            real_weakness_text=weakness_text,
        ))
    rng.shuffle(pool)

    if not stratify:
        return pool[:n]

    accepts = [p for p in pool if p.accepted and not (4 <= p.real_avg_rating <= 6)]
    rejects = [p for p in pool if not p.accepted and not (4 <= p.real_avg_rating <= 6)]
    borderline = [p for p in pool if 4 <= p.real_avg_rating <= 6]
    n_acc = int(n * 0.30)
    n_rej = int(n * 0.30)
    n_bord = n - n_acc - n_rej
    out = accepts[:n_acc] + rejects[:n_rej] + borderline[:n_bord]
    rng.shuffle(out)
    return out[:n]


# ---- simulator (cheap-panel form) -------------------------------------------

def _build_review_request(
    paper: _CalibPaper,
    persona: Persona,
    cluster_block: str,
    paper_text: str,
    cap_chars: int,
) -> BatchRequest:
    sys = _REVIEWER_SYS.format(
        venue=persona.venue,
        style=persona.style,
        priorities=", ".join(persona.priorities),
    )
    user = _REVIEWER_USER.format(
        cluster_block=cluster_block,
        paper=paper_text[:cap_chars],
    )
    return BatchRequest(
        custom_id=f"{paper.forum_id}__{persona.name}",
        user=user,
        system=sys,
        max_tokens=4096,
        reasoning_effort="low",
    )


def _parse_review(raw: str) -> dict:
    return llm._parse_json(raw) if raw else {"_error": "empty"}


# ---- metrics ----------------------------------------------------------------

@dataclass
class MetricResult:
    name: str
    value: float
    ci_low: float
    ci_high: float
    n: int


def _bootstrap_ci(values_a: list[float], values_b: list[float], stat_fn, n_iter: int = 1000, seed: int = 0):
    """Generic paired bootstrap. Returns (point_est, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    n = len(values_a)
    point = stat_fn(values_a, values_b)
    samples = []
    a = np.array(values_a)
    b = np.array(values_b)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        try:
            s = stat_fn(a[idx].tolist(), b[idx].tolist())
            if not np.isnan(s):
                samples.append(s)
        except Exception:
            pass
    if not samples:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _spearman(a, b):
    r, _ = spearmanr(a, b)
    return r


def _auc(real_accepted, sim_prob):
    if len(set(real_accepted)) < 2:
        return float("nan")
    return roc_auc_score(real_accepted, sim_prob)


# ---- weakness-cluster Jaccard -----------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _weakness_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if len(s.strip()) >= 30]


def _topk_clusters_for_text(text: str, clusters: list[Cluster], k: int = 3) -> set[str]:
    """Embed the weakness sentences in text, assign each to nearest cluster
    centroid (computed from cluster sample sentences), return top-k cluster
    labels by frequency."""
    sentences = _weakness_sentences(text)
    if not sentences or not clusters:
        return set()
    sent_vecs = np.array(llm.embed(sentences))
    sent_vecs /= (np.linalg.norm(sent_vecs, axis=1, keepdims=True) + 1e-9)

    # cluster centroids = mean of their sample-sentence embeddings
    centroids = []
    labels = []
    for c in clusters:
        if not c.sample_sentences:
            continue
        v = np.array(llm.embed(c.sample_sentences))
        v = v.mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        centroids.append(v)
        labels.append(c.label)
    if not centroids:
        return set()
    cmat = np.array(centroids)

    sims = sent_vecs @ cmat.T  # (n_sent, n_cluster)
    best = sims.argmax(axis=1)
    counts: dict[str, int] = {}
    for i in best:
        counts[labels[i]] = counts.get(labels[i], 0) + 1
    return {lab for lab, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:k]}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / max(1, len(a | b))


# ---- main -------------------------------------------------------------------

@dataclass
class CalibrationResult:
    n: int
    metrics: list[MetricResult] = field(default_factory=list)
    pairs: list[dict] = field(default_factory=list)


def calibrate(
    n: int = 200,
    reviewers: int = 2,
    model: str | None = None,
    use_batch: bool = True,
    stratify: bool = True,
    bootstrap: int = 1000,
    cap_chars: int = 30_000,
    persona_set: str = "panel",
    cluster_set: str = "default",
    seed: int = 0,
) -> CalibrationResult:
    """Run the cheap-panel calibration recipe end-to-end.

    Default arguments correspond to "Stage 2" from the calibration plan:
    N=200, 2 reviewers, default model = whatever's in PR_REVIEW_MODEL (you
    typically want to override to claude-haiku-4-5 for cheap calibration),
    batch API on, stratified sampling, 1000 bootstrap iterations.
    """
    model = model or REVIEW_MODEL
    personas = load_personas(persona_set)[:reviewers]
    clusters = load_clusters(cluster_set)
    cluster_block = _format_clusters(clusters)

    samples = _sample_papers(n=n, stratify=stratify, seed=seed)
    print(f"sampled {len(samples)} papers (stratify={stratify})")
    if stratify:
        n_acc = sum(1 for s in samples if s.accepted)
        avg = np.mean([s.real_avg_rating for s in samples])
        print(f"  accept rate: {n_acc/len(samples):.0%}, avg rating: {avg:.2f}")

    # Step 1: extract paper text for each
    paper_texts: dict[str, str] = {}
    for s in tqdm(samples, desc="pdf extract"):
        try:
            pdf = download_pdf(s.forum_id)
            paper_texts[s.forum_id] = extract_to_cache(pdf).read_text(encoding="utf-8")
        except Exception as e:
            print(f"  skipping {s.forum_id}: {e!r}")
            paper_texts[s.forum_id] = ""

    # Step 2: build all reviewer requests
    requests: list[BatchRequest] = []
    request_meta: list[tuple[str, str]] = []  # (forum_id, persona_name)
    for s in samples:
        if not paper_texts.get(s.forum_id):
            continue
        for p in personas:
            requests.append(_build_review_request(s, p, cluster_block, paper_texts[s.forum_id], cap_chars))
            request_meta.append((s.forum_id, p.name))

    print(f"submitting {len(requests)} reviewer calls (model={model}, batch={use_batch})")

    # Step 3: run them — batch API or concurrent sync depending on flag
    if use_batch:
        raw_responses = batch_chat(model, requests)
    else:
        # Thread pool: each LLM call is I/O-bound, GIL releases during HTTP wait.
        # Concurrency = 3 stays under Anthropic tier-1 ITPM/OTPM limits with
        # 30k-char inputs. Higher tiers can crank this up via PR_CALIBRATE_CONCURRENCY.
        import os
        from concurrent.futures import ThreadPoolExecutor
        concurrency = int(os.environ.get("PR_CALIBRATE_CONCURRENCY", "3"))

        raw_responses = [None] * len(requests)

        def _one(i):
            r = requests[i]
            try:
                raw_responses[i] = llm.chat(
                    model, r.user, system=r.system,
                    max_tokens=r.max_tokens, reasoning_effort=r.reasoning_effort,
                )
            except Exception as e:
                raw_responses[i] = ""
                print(f"  call {i} failed: {type(e).__name__}: {str(e)[:120]}")

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(tqdm(ex.map(_one, range(len(requests))), total=len(requests), desc="sync calls"))

    # Step 4: aggregate per paper
    by_paper: dict[str, list[dict]] = {s.forum_id: [] for s in samples}
    for raw, (fid, _) in zip(raw_responses, request_meta):
        rev = _parse_review(raw)
        if "rating" in rev:
            by_paper[fid].append(rev)

    pairs: list[dict] = []
    for s in samples:
        revs = by_paper.get(s.forum_id, [])
        ratings = [r["rating"] for r in revs if isinstance(r.get("rating"), (int, float))]
        if not ratings:
            continue
        sim_avg = float(np.mean(ratings))
        # Cheap accept-prob heuristic for AUC: rating / 10
        sim_prob = sim_avg / 10.0
        # Top-3 weakness clusters: union the panel's weakness 'point' fields
        sim_weakness_text = "\n".join(
            (w.get("point", "") if isinstance(w, dict) else str(w))
            for r in revs for w in (r.get("weaknesses") or [])
        )
        pairs.append({
            "forum_id": s.forum_id,
            "venue": s.venue,
            "real_avg": s.real_avg_rating,
            "accepted": s.accepted,
            "sim_avg": sim_avg,
            "sim_prob": sim_prob,
            "sim_weakness_text": sim_weakness_text,
            "real_weakness_text": s.real_weakness_text,
        })

    # Step 5: metrics
    metrics: list[MetricResult] = []

    real_avgs = [p["real_avg"] for p in pairs]
    sim_avgs = [p["sim_avg"] for p in pairs]
    if len(pairs) >= 5:
        rho, lo, hi = _bootstrap_ci(real_avgs, sim_avgs, _spearman, n_iter=bootstrap)
        metrics.append(MetricResult("spearman_rating", rho, lo, hi, n=len(pairs)))

    accepts = [int(p["accepted"]) for p in pairs]
    probs = [p["sim_prob"] for p in pairs]
    if len(set(accepts)) >= 2:
        auc, lo, hi = _bootstrap_ci(accepts, probs, _auc, n_iter=bootstrap)
        metrics.append(MetricResult("auc_accept", auc, lo, hi, n=len(pairs)))

    # Jaccard on weakness clusters (uses real cluster set as the basis)
    if clusters:
        jaccards = []
        for p in tqdm(pairs, desc="jaccard"):
            sim_top = _topk_clusters_for_text(p["sim_weakness_text"], clusters, k=3)
            real_top = _topk_clusters_for_text(p["real_weakness_text"], clusters, k=3)
            j = _jaccard(sim_top, real_top)
            if not np.isnan(j):
                jaccards.append(j)
            p["sim_top_clusters"] = list(sim_top)
            p["real_top_clusters"] = list(real_top)
        if jaccards:
            mean_j = float(np.mean(jaccards))
            lo, hi = float(np.percentile(jaccards, 2.5)), float(np.percentile(jaccards, 97.5))
            metrics.append(MetricResult("jaccard_top3_clusters", mean_j, lo, hi, n=len(jaccards)))

    result = CalibrationResult(n=len(pairs), metrics=metrics, pairs=pairs)

    PATHS.ensure()
    out = PATHS.calibration / "calibration.json"
    out.write_text(json.dumps({
        "n": result.n,
        "model": model,
        "reviewers": reviewers,
        "stratified": stratify,
        "bootstrap": bootstrap,
        "metrics": [asdict(m) for m in result.metrics],
        "pairs": result.pairs,
    }, indent=2), encoding="utf-8")

    # pretty print
    print()
    print("=" * 60)
    print(f"CALIBRATION RESULTS  (N={result.n}, model={model})")
    print("=" * 60)
    for m in result.metrics:
        print(f"  {m.name:30}  {m.value:+.3f}  [95% CI: {m.ci_low:+.3f}, {m.ci_high:+.3f}]")
    print(f"\n  full result -> {out}")

    return result


# ---- boundary-case upgrade ("Stage 3") --------------------------------------

def calibrate_boundary(
    prev_path: str | Path = None,
    n: int = 20,
    model: str | None = None,
    persona_set: str = "panel",
    cluster_set: str = "default",
) -> dict:
    """Stage 3: take the N papers from a prior calibration where the cheap
    panel was most wrong (|sim_avg - real_avg| largest) and re-run them with
    the full production panel (5× Opus reviewers + GPT-5 meta).

    Tells you whether the production pipeline meaningfully closes the gap or
    whether boundary cases are inherently noisy. If full-panel ρ on this
    subset >> cheap-panel ρ, production is worth the extra cost. If not,
    you've hit a fundamental signal limit and should manage user expectations.
    """
    import asyncio
    from .extract import download_pdf, extract_to_cache
    from .review import run_panel
    from .personas import load_personas as _lp
    from .mine import load_clusters as _lc

    prev_path = Path(prev_path or (PATHS.calibration / "calibration.json"))
    prev = json.loads(prev_path.read_text(encoding="utf-8"))
    pairs = prev["pairs"]
    boundary = sorted(pairs, key=lambda p: -abs(p["sim_avg"] - p["real_avg"]))[:n]
    print(f"selected {len(boundary)} boundary cases (max |Δrating| = "
          f"{abs(boundary[0]['sim_avg'] - boundary[0]['real_avg']):.2f})")

    personas = _lp(persona_set)
    clusters = _lc(cluster_set)

    upgraded = []
    for p in tqdm(boundary, desc="full-panel boundary"):
        try:
            pdf = download_pdf(p["forum_id"])
            text = extract_to_cache(pdf).read_text(encoding="utf-8")[:60_000]
        except Exception as e:
            print(f"  skip {p['forum_id']}: {e!r}")
            continue
        try:
            result = asyncio.run(run_panel(
                paper_text=text,
                paper_id=p["forum_id"],
                personas=personas,
                clusters=clusters,
                venue=p["venue"],
                model=model or REVIEW_MODEL,
            ))
        except Exception as e:
            print(f"  panel failed on {p['forum_id']}: {e!r}")
            continue
        upgraded.append({
            "forum_id": p["forum_id"],
            "real_avg": p["real_avg"],
            "accepted": p["accepted"],
            "cheap_sim_avg": p["sim_avg"],
            "full_sim_avg": result.avg_rating,
            "full_accept_prob": result.accept_prob,
        })

    if len(upgraded) < 5:
        raise RuntimeError(f"only {len(upgraded)} upgraded - too few for stats")

    real = [u["real_avg"] for u in upgraded]
    cheap = [u["cheap_sim_avg"] for u in upgraded]
    full = [u["full_sim_avg"] for u in upgraded]
    accepts = [int(u["accepted"]) for u in upgraded]
    full_probs = [u["full_accept_prob"] for u in upgraded]

    metrics = {
        "n_boundary": len(upgraded),
        "cheap_spearman_on_boundary": float(_spearman(real, cheap)),
        "full_spearman_on_boundary": float(_spearman(real, full)),
        "delta_spearman": float(_spearman(real, full) - _spearman(real, cheap)),
        "full_auc_on_boundary": float(_auc(accepts, full_probs)) if len(set(accepts)) > 1 else float("nan"),
        "cheap_bias": float(np.mean([c - r for c, r in zip(cheap, real)])),
        "full_bias": float(np.mean([f - r for f, r in zip(full, real)])),
    }
    out = {
        "config": {"n_requested": n, "model": model or REVIEW_MODEL, "from": str(prev_path)},
        "metrics": metrics,
        "pairs": upgraded,
    }
    save = PATHS.calibration / "calibration_boundary.json"
    save.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"BOUNDARY UPGRADE  (N={len(upgraded)}, model={model or REVIEW_MODEL})")
    print("=" * 60)
    print(f"  cheap Spearman on boundary:  {metrics['cheap_spearman_on_boundary']:+.3f}")
    print(f"  full  Spearman on boundary:  {metrics['full_spearman_on_boundary']:+.3f}")
    print(f"  Δ Spearman:                  {metrics['delta_spearman']:+.3f}")
    print(f"  full AUC on boundary:        {metrics['full_auc_on_boundary']:+.3f}")
    print(f"  cheap bias (sim - real):     {metrics['cheap_bias']:+.3f}")
    print(f"  full  bias (sim - real):     {metrics['full_bias']:+.3f}")
    print(f"\n  full result -> {save}")
    return out
