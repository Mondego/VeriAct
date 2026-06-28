#!/usr/bin/env python3
"""Aggregate a run's spec-harness scores: pass@{0.25,0.5,0.75,1.0} + mean metrics.

Run from the out-root. Reads per-task scores from (in priority):
  1. harness_scores.json  (written by score_all.py)
  2. each <task>/submission.json  (the agent's submitted scores)

pass@T = fraction of tasks with post_correctness >= T AND post_completeness >= T.
Writes aggregate.json and prints a summary. Self-contained (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS = ["post_correctness", "post_completeness", "pre_correctness", "pre_completeness"]
THRESHOLDS = [0.25, 0.5, 0.75, 1.0]


def _row(task_id, scores):
    return {"task_id": task_id, **{m: scores.get(m) for m in METRICS}}


def load_rows(root: str) -> list[dict]:
    hs = os.path.join(root, "harness_scores.json")
    if os.path.exists(hs):
        with open(hs) as fh:
            data = json.load(fh)
        return [_row(r.get("task_id", "?"), r) for r in data if not r.get("error")]
    rows = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name, "submission.json")
        if not os.path.exists(sub):
            continue
        with open(sub) as fh:
            d = json.load(fh)
        scores = d.get("scores")
        if isinstance(scores, dict) and scores:
            rows.append(_row(d.get("task_id", name), scores))
    return rows


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    means = {
        m: round(sum((r[m] or 0.0) for r in rows) / n, 4) if n else 0.0
        for m in METRICS
    }
    passes = {}
    for t in THRESHOLDS:
        c = sum(
            1
            for r in rows
            if (r["post_correctness"] or 0.0) >= t and (r["post_completeness"] or 0.0) >= t
        )
        passes[f"{t:g}"] = {"count": c, "rate": round(c / n, 4) if n else 0.0}
    return {"n_tasks": n, "mean_metrics": means, "pass_at": passes}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="pass@{0.25,0.5,0.75,1.0} + mean metrics.")
    p.add_argument("--root", default=ROOT, help="out-root dir (default: this dir)")
    p.add_argument("--out", default=None, help="output json (default: <root>/aggregate.json)")
    args = p.parse_args(argv)

    rows = load_rows(args.root)
    if not rows:
        print("No scores found (need harness_scores.json or */submission.json).", file=sys.stderr)
        return 1
    agg = aggregate(rows)

    out = args.out or os.path.join(args.root, "aggregate.json")
    with open(out, "w") as fh:
        json.dump(agg, fh, indent=2)

    print(f"tasks scored: {agg['n_tasks']}")
    print("mean metrics:")
    for m, v in agg["mean_metrics"].items():
        print(f"  {m:18s} {v}")
    print("pass@T (post_correctness >= T AND post_completeness >= T):")
    for t, info in agg["pass_at"].items():
        print(f"  @{t:<4} {info['rate']:.3f}  ({info['count']}/{agg['n_tasks']})")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
