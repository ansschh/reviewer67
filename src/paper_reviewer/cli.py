"""Command-line entry point.

    paper-reviewer scrape
    paper-reviewer mine --abstract path/to/abstract.txt
    paper-reviewer personas
    paper-reviewer review --pdf paper.pdf
    paper-reviewer calibrate --n 50
    paper-reviewer train-predictor
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import calibrate as calibrate_mod
from . import mine as mine_mod
from . import personas as personas_mod
from . import predictor as predictor_mod
from . import review as review_mod
from . import scrape as scrape_mod
from .config import PATHS
from .extract import extract_to_cache
from .personas import build_persona


def _cmd_scrape(args: argparse.Namespace) -> int:
    venues = args.venues.split(",") if args.venues else None
    paths = scrape_mod.scrape_all(venues=venues)
    for p in paths:
        print(p)
    return 0


def _cmd_mine(args: argparse.Namespace) -> int:
    abstract = Path(args.abstract).read_text(encoding="utf-8") if args.abstract else None
    mine_mod.extract_weaknesses()
    rows = (
        mine_mod.filter_by_abstract(abstract, top_k_papers=args.top_k_papers)
        if abstract
        else mine_mod._load_weaknesses()
    )
    print(f"clustering {len(rows)} weakness sentences ...")
    clusters = mine_mod.cluster_weaknesses(rows, min_cluster_size=args.min_cluster_size)
    out = mine_mod.save_clusters(clusters, name=args.name)
    for c in clusters[:20]:
        ar = f"{c.accept_rate:.0%}" if c.accept_rate is not None else "n/a"
        print(f"  [{c.size:5d} | accept={ar}] {c.label}")
    print(f"\nsaved -> {out}")
    return 0


def _cmd_personas(args: argparse.Namespace) -> int:
    specs = personas_mod.default_persona_specs()
    personas = []
    for s in specs:
        print(f"building persona: {s['name']} ({s['venue']})")
        personas.append(build_persona(**s))
    out = personas_mod.save_personas(personas, name=args.name)
    print(f"saved -> {out}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    text_path = extract_to_cache(args.pdf)
    paper_text = text_path.read_text(encoding="utf-8")[:60_000]
    personas = personas_mod.load_personas(args.persona_set)
    clusters = mine_mod.load_clusters(args.cluster_set)

    if args.rounds <= 1:
        result = asyncio.run(review_mod.run_panel(
            paper_text=paper_text,
            paper_id=Path(args.pdf).stem,
            personas=personas,
            clusters=clusters,
            venue=args.venue,
        ))
        out = review_mod.save_panel(result, name=Path(args.pdf).stem)
        print(json.dumps({
            "avg_rating": result.avg_rating,
            "accept_prob": result.accept_prob,
            "top_risks": result.risks.get("top_risks", [])[:3],
            "saved": str(out),
        }, indent=2))
    else:
        # Manual revision loop: between rounds we wait for the user to update
        # the PDF in place, then re-extract. For fully automated rewriting,
        # call review.iterate() programmatically with your own revise_fn.
        def _manual(_text, _result):
            print("\n(Apply your revisions to the PDF in place, then press Enter to re-run.)")
            try:
                input()
            except EOFError:
                pass
            text_path = extract_to_cache(args.pdf)
            return text_path.read_text(encoding="utf-8")[:60_000]

        history = review_mod.iterate(
            paper_text=paper_text,
            paper_id=Path(args.pdf).stem,
            personas=personas,
            clusters=clusters,
            rounds=args.rounds,
            revise_fn=_manual,
            venue=args.venue,
        )
        print("\nrating trajectory:", [round(h.avg_rating, 2) for h in history])
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    calibrate_mod.calibrate(
        n=args.n,
        reviewers=args.reviewers,
        model=args.model,
        use_batch=args.batch,
        stratify=args.stratify,
        bootstrap=args.bootstrap,
        cap_chars=args.cap_chars,
        persona_set=args.persona_set,
        cluster_set=args.cluster_set,
        seed=args.seed,
    )
    return 0


def _cmd_calibrate_boundary(args: argparse.Namespace) -> int:
    calibrate_mod.calibrate_boundary(
        prev_path=args.from_path,
        n=args.n,
        model=args.model,
        persona_set=args.persona_set,
        cluster_set=args.cluster_set,
    )
    return 0


def _cmd_train_predictor(args: argparse.Namespace) -> int:
    out = predictor_mod.train(max_papers=args.max_papers)
    print(f"saved -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="paper-reviewer")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrape", help="Pull OpenReview venues into JSONL.")
    s.add_argument("--venues", default=None, help="Comma-separated venue IDs (default: all).")
    s.set_defaults(func=_cmd_scrape)

    s = sub.add_parser("mine", help="Extract + cluster weakness sentences.")
    s.add_argument("--abstract", default=None, help="Path to a text file with your paper's abstract (filters to subfield).")
    s.add_argument("--top-k-papers", type=int, default=500, dest="top_k_papers")
    s.add_argument("--min-cluster-size", type=int, default=25, dest="min_cluster_size")
    s.add_argument("--name", default="default")
    s.set_defaults(func=_cmd_mine)

    s = sub.add_parser("personas", help="Build the reviewer panel from sampled real reviews.")
    s.add_argument("--name", default="panel")
    s.set_defaults(func=_cmd_personas)

    s = sub.add_parser("review", help="Run the panel on your paper.")
    s.add_argument("--pdf", required=True)
    s.add_argument("--persona-set", default="panel", dest="persona_set")
    s.add_argument("--cluster-set", default="default", dest="cluster_set")
    s.add_argument("--venue", default="ICLR.cc/2026")
    s.add_argument("--rounds", type=int, default=1)
    s.set_defaults(func=_cmd_review)

    s = sub.add_parser("calibrate", help="Validate against held-out real papers.")
    s.add_argument("--n", type=int, default=200, help="Sample size. >=200 for tight 95%% CI.")
    s.add_argument("--reviewers", type=int, default=2, help="Reviewers per panel. 2 is the cheap-stage default.")
    s.add_argument("--model", default=None, help="Override review model (e.g. claude-haiku-4-5-20251001).")
    s.add_argument("--batch", action="store_true", default=True, help="Use Anthropic/OpenAI batch API (50%% off, 24h SLA).")
    s.add_argument("--no-batch", dest="batch", action="store_false")
    s.add_argument("--stratify", action="store_true", default=True, help="Decision-balanced sampling.")
    s.add_argument("--no-stratify", dest="stratify", action="store_false")
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--cap-chars", type=int, default=30_000, dest="cap_chars")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--persona-set", default="panel", dest="persona_set")
    s.add_argument("--cluster-set", default="default", dest="cluster_set")
    s.set_defaults(func=_cmd_calibrate)

    s = sub.add_parser("calibrate-boundary", help="Stage 3: upgrade worst-disagreement boundary cases to full Opus panel.")
    s.add_argument("--from", default=None, dest="from_path", help="Path to a prior calibration.json")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--model", default=None)
    s.add_argument("--persona-set", default="panel", dest="persona_set")
    s.add_argument("--cluster-set", default="default", dest="cluster_set")
    s.set_defaults(func=_cmd_calibrate_boundary)

    s = sub.add_parser("train-predictor", help="Train accept/reject classifier on scraped data.")
    s.add_argument("--max-papers", type=int, default=4000, dest="max_papers")
    s.set_defaults(func=_cmd_train_predictor)

    args = p.parse_args(argv)
    PATHS.ensure()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
