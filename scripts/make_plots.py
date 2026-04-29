"""Generate the README's plots from our actual calibration / mining data.

Reads from data/calibration/calibration.json, data/calibration/calibration_boundary.json,
data/clusters/default.json, data/personas/panel.json, data/raw/*.jsonl
and writes PNGs to assets/. All plots use the same minimal-but-colorful palette.

Run: `python scripts/make_plots.py`
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
DATA = ROOT / "data"


# ---- shared style -----------------------------------------------------------

P = {
    "primary":   "#6f5fc4",  # purple — main reviewer/sim signal
    "secondary": "#b6acdf",  # lavender — paired/baseline
    "accent":    "#3f9d8c",  # teal — accept / good
    "danger":    "#d05a4a",  # coral — reject / bad
    "warn":      "#d99f3a",  # gold — borderline
    "neutral":   "#9aa3ad",  # gray — comparison
    "ink":       "#2d2d2d",  # text
    "soft":      "#e8e8ee",  # grid
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "text.color": P["ink"],
    "axes.labelcolor": P["ink"],
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.labelsize": 10.5,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.axisbelow": True,
    "grid.color": P["soft"],
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.18,
})


def _annotate_bars(ax, bars, fmt="{:.2f}", offset=0.01, color=None):
    color = color or P["ink"]
    ymax = ax.get_ylim()[1]
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2, h + offset * ymax,
            fmt.format(h), ha="center", va="bottom",
            fontsize=9, color=color,
        )


# ---- 1. Calibration metrics with 95% CI -------------------------------------

def plot_calibration_metrics():
    cal = json.loads((DATA / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    metrics = {m["name"]: m for m in cal["metrics"]}
    names = [
        ("spearman_rating",       "Spearman ρ\n(rating)",       P["primary"]),
        ("auc_accept",            "AUC\n(accept/reject)",       P["accent"]),
        ("jaccard_top3_clusters", "Jaccard\n(top-3 weaknesses)", P["warn"]),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xs = np.arange(len(names))
    vals = [metrics[k]["value"] for k, _, _ in names]
    los = [metrics[k]["value"] - metrics[k]["ci_low"] for k, _, _ in names]
    his = [metrics[k]["ci_high"] - metrics[k]["value"] for k, _, _ in names]
    colors = [c for _, _, c in names]

    bars = ax.bar(xs, vals, color=colors, width=0.55, edgecolor="white", linewidth=2)
    ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor=P["ink"],
                elinewidth=1.0, capsize=6, capthick=1.0, alpha=0.7)

    for x, v in zip(xs, vals):
        ax.text(x, v + 0.04, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, color=P["ink"], fontweight="bold")

    ax.axhline(0.4, color=P["neutral"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(names) - 0.45, 0.42, "ρ ≥ 0.4 = useful signal",
            fontsize=8.5, color=P["neutral"], style="italic")

    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n, _ in names])
    ax.set_ylabel("score (1.0 = perfect)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Calibration on N={cal['n']} held-out OpenReview papers (95% bootstrap CI)",
                 loc="left", pad=12, fontsize=11.5, color=P["ink"])
    ax.yaxis.grid(True)

    plt.savefig(ASSETS / "calibration_metrics.png")
    plt.close()


# ---- 2. Sim vs real rating scatter ------------------------------------------

def plot_sim_vs_real():
    cal = json.loads((DATA / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    pairs = cal["pairs"]
    real = np.array([p["real_avg"] for p in pairs])
    sim = np.array([p["sim_avg"] for p in pairs])
    accepted = np.array([p["accepted"] for p in pairs])

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.scatter(real[accepted], sim[accepted], s=24, alpha=0.55,
               color=P["accent"], edgecolor="none", label="Accepted")
    ax.scatter(real[~accepted], sim[~accepted], s=24, alpha=0.55,
               color=P["danger"], edgecolor="none", label="Rejected")

    lims = [min(real.min(), sim.min()) - 0.5, max(real.max(), sim.max()) + 0.5]
    ax.plot(lims, lims, color=P["neutral"], linewidth=0.7, linestyle="--", alpha=0.7)
    # Bias line: sim = real + bias
    bias = float(np.mean(sim - real))
    ax.plot(lims, [l + bias for l in lims], color=P["primary"], linewidth=1.2, alpha=0.8,
            label=f"mean offset ({bias:+.2f})")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("real reviewer avg rating")
    ax.set_ylabel("simulated avg rating")
    ax.set_title("Simulated vs real rating", loc="left", pad=12, fontsize=11.5)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.7)

    plt.savefig(ASSETS / "sim_vs_real_scatter.png")
    plt.close()


# ---- 3. Bias distribution histogram -----------------------------------------

def plot_bias_histogram():
    cal = json.loads((DATA / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    diffs = np.array([p["sim_avg"] - p["real_avg"] for p in cal["pairs"]])

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bins = np.arange(np.floor(diffs.min() * 2) / 2, np.ceil(diffs.max() * 2) / 2 + 0.5, 0.5)
    ax.hist(diffs, bins=bins, color=P["primary"], alpha=0.85, edgecolor="white", linewidth=1.5)
    mean = float(np.mean(diffs))
    ax.axvline(0, color=P["neutral"], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(mean, color=P["danger"], linewidth=1.5, label=f"mean = {mean:+.2f}")
    ax.set_xlabel("(simulated rating)  −  (real rating)")
    ax.set_ylabel("papers")
    ax.set_title("Bias distribution: simulator runs cold",
                 loc="left", pad=12, fontsize=11.5)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.yaxis.grid(True)

    plt.savefig(ASSETS / "bias_histogram.png")
    plt.close()


# ---- 4. Stage 3 boundary cheap vs full --------------------------------------

def plot_boundary_comparison():
    b = json.loads((DATA / "calibration" / "calibration_boundary.json").read_text(encoding="utf-8"))
    pairs = sorted(b["pairs"], key=lambda p: -p["real_avg"])
    n = len(pairs)
    real = np.array([p["real_avg"] for p in pairs])
    cheap = np.array([p["cheap_sim_avg"] for p in pairs])
    full = np.array([p["full_sim_avg"] for p in pairs])

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    xs = np.arange(n)
    w = 0.27
    ax.bar(xs - w, real, width=w, color=P["accent"], label="real reviewers", edgecolor="white", linewidth=1.0)
    ax.bar(xs,     cheap, width=w, color=P["secondary"], label="cheap (Haiku, 2-reviewer)", edgecolor="white", linewidth=1.0)
    ax.bar(xs + w, full,  width=w, color=P["primary"], label="full (Opus, 5-reviewer)", edgecolor="white", linewidth=1.0)

    ax.set_xticks(xs)
    ax.set_xticklabels([p["forum_id"][:6] for p in pairs], rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("avg rating")
    ax.set_ylim(0, 10)
    bias_cheap = float(np.mean(cheap - real))
    bias_full = float(np.mean(full - real))
    ax.set_title(f"Stage 3 boundary (N={n}) — Opus closes 31% of bias gap "
                 f"({bias_cheap:+.2f} → {bias_full:+.2f})",
                 loc="left", pad=12, fontsize=11.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=3)
    ax.yaxis.grid(True)

    plt.savefig(ASSETS / "boundary_cheap_vs_full.png")
    plt.close()


# ---- 5. Mined rejection clusters --------------------------------------------

def plot_clusters():
    cl = json.loads((DATA / "clusters" / "default.json").read_text(encoding="utf-8"))
    cl = sorted(cl, key=lambda c: c["size"])

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.5 * len(cl) + 1.5)))
    ys = np.arange(len(cl))
    sizes = [c["size"] for c in cl]
    accepts = [c["accept_rate"] if c["accept_rate"] is not None else 0.5 for c in cl]

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "rejaccept", [P["danger"], "#dadada", P["accent"]], N=256,
    )
    norm = mpl.colors.Normalize(vmin=0.20, vmax=0.80)
    colors = [cmap(norm(a)) for a in accepts]

    bars = ax.barh(ys, sizes, color=colors, edgecolor="white", linewidth=1.2)
    for b, s, a in zip(bars, sizes, accepts):
        ax.text(b.get_width() + max(sizes) * 0.01, b.get_y() + b.get_height() / 2,
                f"  n={s}  •  accept {a:.0%}", va="center", fontsize=9, color=P["ink"])

    ax.set_yticks(ys)
    ax.set_yticklabels([c["label"] for c in cl], fontsize=10)
    ax.set_xlabel("weakness sentences in cluster")
    ax.set_title("Mined rejection patterns (filtered to subfield)",
                 loc="left", pad=12, fontsize=11.5)
    ax.set_xlim(0, max(sizes) * 1.4)
    ax.xaxis.grid(True)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.04, shrink=0.7)
    cbar.set_label("accept rate of papers in cluster", fontsize=9)
    cbar.outline.set_linewidth(0)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(ASSETS / "clusters.png")
    plt.close()


# ---- 6. Corpus venues bar chart ---------------------------------------------

def plot_venues():
    rows = []
    for f in sorted((DATA / "raw").glob("*.jsonl")):
        if f.stat().st_size == 0:
            continue
        n = nrev = 0
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                n += 1
                if d.get("reviews"):
                    nrev += 1
        # short label
        venue = f.stem.replace("_Conference", "").replace("_", " ")
        venue = venue.replace("aistats.org AISTATS", "AISTATS").replace("auai.org UAI", "UAI") \
                     .replace("colmweb.org COLM", "COLM") \
                     .replace("robot-learning.org CoRL", "CoRL") \
                     .replace("rl-conference.cc RLC", "RLC") \
                     .replace("ICLR.cc", "ICLR").replace("NeurIPS.cc", "NeurIPS") \
                     .replace("ICML.cc", "ICML")
        rows.append((venue, n, nrev))

    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.3 * len(rows) + 1.5)))
    ys = np.arange(len(rows))
    totals = [r[1] for r in rows]
    revs = [r[2] for r in rows]

    ax.barh(ys, totals, color=P["secondary"], edgecolor="white", linewidth=1.0, label="papers")
    ax.barh(ys, revs, color=P["primary"], edgecolor="white", linewidth=1.0, label="with public reviews")

    for y, t, r in zip(ys, totals, revs):
        ax.text(t + max(totals) * 0.005, y, f"  {t:,}", va="center", fontsize=8.5, color=P["ink"])

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("submissions")
    total_p = sum(totals)
    total_r = sum(revs)
    ax.set_title(f"Corpus: {total_p:,} papers • {total_r:,} with reviews • "
                 f"{len(rows)} venue-years",
                 loc="left", pad=12, fontsize=11.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.xaxis.grid(True)

    plt.savefig(ASSETS / "corpus_venues.png")
    plt.close()


# ---- 7. Real vs sim rating distribution -------------------------------------

def plot_rating_dist():
    cal = json.loads((DATA / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    real = np.array([p["real_avg"] for p in cal["pairs"]])
    sim = np.array([p["sim_avg"] for p in cal["pairs"]])

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bins = np.arange(1, 10.5, 0.5)
    ax.hist(real, bins=bins, color=P["accent"], alpha=0.7, label=f"real (μ={real.mean():.2f}, σ={real.std():.2f})",
            edgecolor="white", linewidth=1.2)
    ax.hist(sim, bins=bins, color=P["primary"], alpha=0.7, label=f"simulated (μ={sim.mean():.2f}, σ={sim.std():.2f})",
            edgecolor="white", linewidth=1.2)
    ax.set_xlabel("avg rating")
    ax.set_ylabel("papers")
    ax.set_title("Rating distributions: real reviewers spread wider; simulator clusters in 4-5",
                 loc="left", pad=12, fontsize=11.5)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.yaxis.grid(True)

    plt.savefig(ASSETS / "rating_distribution.png")
    plt.close()


# ---- 8. ROC curve -----------------------------------------------------------

def plot_roc():
    from sklearn.metrics import roc_curve, auc
    cal = json.loads((DATA / "calibration" / "calibration.json").read_text(encoding="utf-8"))
    y = [int(p["accepted"]) for p in cal["pairs"]]
    p = [pp["sim_prob"] for pp in cal["pairs"]]
    fpr, tpr, _ = roc_curve(y, p)
    a = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(fpr, tpr, color=P["primary"], linewidth=2.0, label=f"AUC = {a:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=P["primary"])
    ax.plot([0, 1], [0, 1], color=P["neutral"], linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Accept/reject ROC", loc="left", pad=12, fontsize=11.5)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.grid(True, alpha=0.7)

    plt.savefig(ASSETS / "roc_curve.png")
    plt.close()


# ---- 9. Persona priorities heatmap ------------------------------------------

def plot_personas():
    panel = json.loads((DATA / "personas" / "panel.json").read_text(encoding="utf-8"))
    # Take the top-N priorities per persona, then build a binary matrix.
    all_pris = []
    for p in panel:
        all_pris.extend(p["priorities"])
    # Distill priorities by keyword groups (otherwise too many distinct strings)
    KEYWORDS = {
        "Novelty": ["novelty", "novel", "prior work", "concurrent"],
        "Empirical breadth": ["experimental", "datasets", "scope", "scalability", "breadth"],
        "Baselines": ["baseline", "comparison", "missing comparison"],
        "Theoretical rigor": ["theoretical", "proof", "formality", "rigor"],
        "Ablations": ["ablation"],
        "Significance / stats": ["significance", "statistical", "seeds"],
        "Clarity / writing": ["clarity", "presentation", "writing", "self-contained"],
        "Reproducibility": ["reproducibility", "code"],
        "Cost / runtime": ["cost", "runtime", "compute", "scalability"],
        "Motivation": ["motivation", "design choice", "justification"],
    }
    rows = list(KEYWORDS.keys())
    M = np.zeros((len(rows), len(panel)))
    for j, p in enumerate(panel):
        priorities_text = " | ".join(p["priorities"]).lower()
        for i, r in enumerate(rows):
            for kw in KEYWORDS[r]:
                if kw in priorities_text:
                    M[i, j] += 1
    M = M.clip(0, 3)  # cap intensity

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "purpleramp", ["#ffffff", P["secondary"], P["primary"]], N=256,
    )
    ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=3)

    ax.set_xticks(np.arange(len(panel)))
    ax.set_xticklabels([p["name"].replace("_", "\n") for p in panel], fontsize=9.5)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                ax.text(j, i, "●", ha="center", va="center",
                        color=P["ink"] if M[i, j] >= 2 else P["primary"],
                        fontsize=10 + 2 * M[i, j])
    ax.set_title("What each reviewer-persona consistently flags",
                 loc="left", pad=12, fontsize=11.5)
    ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    plt.savefig(ASSETS / "personas.png")
    plt.close()


# ---- main -------------------------------------------------------------------

if __name__ == "__main__":
    funs = [
        plot_calibration_metrics,
        plot_sim_vs_real,
        plot_bias_histogram,
        plot_boundary_comparison,
        plot_clusters,
        plot_venues,
        plot_rating_dist,
        plot_roc,
        plot_personas,
    ]
    for f in funs:
        try:
            f()
            print(f"  ok  {f.__name__}")
        except Exception as e:
            print(f"  FAIL  {f.__name__}: {type(e).__name__}: {e}")
