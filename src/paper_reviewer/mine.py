"""Stage 2 — extract weakness sentences from real reviews, embed, cluster, label.

The goal is a per-subfield prior over reviewer rejection patterns. Filter to
papers near your abstract before clustering, otherwise the clusters drown in
generic "writing is unclear" critiques.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import llm
from .config import PATHS, WEAKNESS_EXTRACT_MODEL
from .normalize import normalize_decision, normalize_review
from .scrape import iter_papers


# --- weakness sentence extraction --------------------------------------------

# Crude sentence splitter — fine because we cluster after, errors get absorbed.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if len(s.strip()) >= 30]


@dataclass
class WeaknessRow:
    venue: str
    forum_id: str
    review_id: str
    sentence: str
    rating: float | None
    confidence: float | None
    accepted: bool | None
    abstract: str
    title: str | None


def extract_weaknesses() -> Path:
    """Walk all scraped venues, dump one weakness sentence per row to JSONL."""
    PATHS.ensure()
    out = PATHS.weaknesses / "weaknesses.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] weaknesses already extracted at {out}")
        return out

    n_rows = 0
    with out.open("w", encoding="utf-8") as f:
        for paper in tqdm(iter_papers(), desc="weaknesses"):
            decision = normalize_decision(paper.get("decision"))
            for review_note in paper.get("reviews", []):
                nr = normalize_review(paper["venue"], paper["forum_id"], review_note)
                # Take both `weaknesses` and any "Cons:"/"Limitations:" sentences in summary.
                for sent in _split_sentences(nr.weaknesses):
                    row = WeaknessRow(
                        venue=nr.venue,
                        forum_id=nr.forum_id,
                        review_id=nr.review_id,
                        sentence=sent,
                        rating=nr.rating,
                        confidence=nr.confidence,
                        accepted=decision.accepted,
                        abstract=paper.get("abstract") or "",
                        title=paper.get("title"),
                    )
                    f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
                    n_rows += 1
    print(f"wrote {n_rows} weakness sentences -> {out}")
    return out


# --- abstract-conditioned filtering ------------------------------------------

def _load_weaknesses() -> list[dict]:
    p = PATHS.weaknesses / "weaknesses.jsonl"
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def _cosine_topk(query: np.ndarray, mat: np.ndarray, k: int) -> np.ndarray:
    q = query / (np.linalg.norm(query) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    return np.argsort(-sims)[:k]


def filter_by_abstract(your_abstract: str, top_k_papers: int = 500) -> list[dict]:
    """Return weakness rows whose paper's abstract is closest to yours.

    Embeds each unique abstract once, finds top_k_papers nearest, then keeps
    every weakness sentence from those papers.
    """
    rows = _load_weaknesses()
    abstracts: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        abstracts.setdefault(r["abstract"], []).append(i)

    abs_keys = list(abstracts.keys())
    abs_vecs = np.array(llm.embed(abs_keys))
    q_vec = np.array(llm.embed([your_abstract])[0])
    top_idx = _cosine_topk(q_vec, abs_vecs, top_k_papers)
    keep_row_idx: list[int] = []
    for i in top_idx:
        keep_row_idx.extend(abstracts[abs_keys[int(i)]])
    return [rows[i] for i in keep_row_idx]


# --- clustering --------------------------------------------------------------

@dataclass
class Cluster:
    label: str
    size: int
    accept_rate: float | None         # fraction of cluster's reviews on accepted papers
    avg_rating: float | None
    sample_sentences: list[str]
    member_indices: list[int]


def cluster_weaknesses(
    rows: list[dict],
    min_cluster_size: int = 25,
    label_with_llm: bool = True,
    pca_dim: int = 50,
) -> list[Cluster]:
    import hdbscan
    from sklearn.decomposition import PCA

    sentences = [r["sentence"] for r in rows]
    vecs = np.array(llm.embed(sentences))
    # Normalize so projected distance behaves like cosine.
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    # In raw 3072-dim space HDBSCAN labels almost everything as noise (curse of
    # dim). PCA-reduce to ~50 dims first — standard recipe for HDBSCAN+text.
    if pca_dim and vecs.shape[1] > pca_dim and len(vecs) > pca_dim:
        vecs = PCA(n_components=pca_dim, random_state=0).fit_transform(vecs)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(vecs)

    clusters: list[Cluster] = []
    for cid in sorted(set(labels)):
        if cid == -1:  # noise
            continue
        member_idx = [i for i, lab in enumerate(labels) if lab == cid]
        members = [rows[i] for i in member_idx]
        ratings = [m["rating"] for m in members if m.get("rating") is not None]
        accept = [m["accepted"] for m in members if m.get("accepted") is not None]
        sample = [m["sentence"] for m in members[:8]]

        label = "(unlabeled)"
        if label_with_llm:
            label = _label_cluster(sample)

        clusters.append(
            Cluster(
                label=label,
                size=len(members),
                accept_rate=(sum(1 for a in accept if a) / len(accept)) if accept else None,
                avg_rating=(sum(ratings) / len(ratings)) if ratings else None,
                sample_sentences=sample,
                member_indices=member_idx,
            )
        )
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def _label_cluster(samples: list[str]) -> str:
    bullets = "\n".join(f"- {s}" for s in samples)
    prompt = (
        "These are weakness sentences from peer reviews of ML papers. "
        "Give a short (<=8 word) label that captures the COMMON criticism they share. "
        "Reply with the label only, no quotes.\n\n"
        f"{bullets}"
    )
    # Generous max_tokens because GPT-5 family burns reasoning tokens before output.
    # The actual label is short; this is just headroom for hidden reasoning.
    try:
        out = llm.chat(WEAKNESS_EXTRACT_MODEL, prompt, max_tokens=1024).strip()
        return out.splitlines()[0] if out else "(unlabeled)"
    except Exception:
        return "(unlabeled)"


def save_clusters(clusters: list[Cluster], name: str = "default") -> Path:
    PATHS.ensure()
    out = PATHS.clusters / f"{name}.json"
    out.write_text(
        json.dumps([asdict(c) for c in clusters], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def load_clusters(name: str = "default") -> list[Cluster]:
    p = PATHS.clusters / f"{name}.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [Cluster(**r) for r in raw]
