"""Decision predictor: a logistic-regression baseline on
(review-text embeddings, scores) -> accept/reject. Honest: this is the *vibes
into a number* check on the simulator. If your panel's reviews + scores plug
into this and predict P(accept) high while every real reviewer would say no,
your simulator is broken.

Intentionally simple. The point is calibration, not state-of-the-art."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import llm
from .config import PATHS
from .normalize import normalize_decision, normalize_review
from .scrape import iter_papers


@dataclass
class _Feature:
    paper_id: str
    avg_rating: float
    avg_confidence: float
    review_text: str
    accepted: bool


def _build_features(max_papers: int | None = None) -> list[_Feature]:
    feats: list[_Feature] = []
    for paper in iter_papers():
        if max_papers and len(feats) >= max_papers:
            break
        decision = normalize_decision(paper.get("decision"))
        if decision.accepted is None:
            continue
        reviews = [normalize_review(paper["venue"], paper["forum_id"], r) for r in paper.get("reviews", [])]
        ratings = [r.rating for r in reviews if r.rating is not None]
        confs = [r.confidence for r in reviews if r.confidence is not None]
        if not ratings:
            continue
        text = "\n\n".join((r.summary or "") + " " + (r.weaknesses or "") for r in reviews)[:6000]
        feats.append(
            _Feature(
                paper_id=paper["forum_id"],
                avg_rating=float(np.mean(ratings)),
                avg_confidence=float(np.mean(confs)) if confs else 3.0,
                review_text=text,
                accepted=bool(decision.accepted),
            )
        )
    return feats


def train(max_papers: int = 4000, save_name: str = "predictor") -> Path:
    feats = _build_features(max_papers=max_papers)
    if len(feats) < 200:
        raise RuntimeError(f"Not enough labeled papers: {len(feats)}")
    embeds = np.array(llm.embed([f.review_text for f in feats]))
    extras = np.array([[f.avg_rating, f.avg_confidence] for f in feats])
    X = np.concatenate([embeds, extras], axis=1)
    y = np.array([1 if f.accepted else 0 for f in feats])

    clf = LogisticRegression(max_iter=2000, C=0.5)
    clf.fit(X, y)
    score = clf.score(X, y)
    print(f"train_acc={score:.3f}  n={len(feats)}  n_pos={int(y.sum())}")

    PATHS.ensure()
    out = PATHS.calibration / f"{save_name}.pkl"
    with out.open("wb") as f:
        pickle.dump({"clf": clf, "embed_dim": embeds.shape[1]}, f)
    return out


def predict_from_panel(panel_path: Path, model_name: str = "predictor") -> float:
    """Plug a saved PanelResult JSON into the trained classifier."""
    panel = json.loads(Path(panel_path).read_text(encoding="utf-8"))
    reviews = panel["reviews"]
    text = "\n\n".join(
        (r["review"].get("summary", "") + " " + " ".join(
            w.get("point", "") if isinstance(w, dict) else str(w)
            for w in (r["review"].get("weaknesses") or [])
        ))
        for r in reviews
    )[:6000]
    ratings = [r["review"].get("rating") for r in reviews if isinstance(r["review"].get("rating"), (int, float))]
    confs = [r["review"].get("confidence") for r in reviews if isinstance(r["review"].get("confidence"), (int, float))]
    avg_r = float(np.mean(ratings)) if ratings else 5.0
    avg_c = float(np.mean(confs)) if confs else 3.0
    emb = np.array(llm.embed([text]))
    X = np.concatenate([emb, np.array([[avg_r, avg_c]])], axis=1)

    with (PATHS.calibration / f"{model_name}.pkl").open("rb") as f:
        bundle = pickle.load(f)
    clf = bundle["clf"]
    return float(clf.predict_proba(X)[0, 1])
