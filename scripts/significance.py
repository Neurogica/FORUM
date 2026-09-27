#!/usr/bin/env python3
"""Paired bootstrap significance tests (FORUM vs best single, FORUM vs plain ensemble).

    python scripts/significance.py --benchmark refadv --lineup balanced

The per-instance score is the mean of the Acc@{0.5, 0.75, 0.9} indicators; the bootstrap
resamples instances with replacement (10,000 draws) and reports the mean difference,
its 95% percentile interval, and a two-sided p-value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import THRESHOLDS, candidates, load_lineup, primary_box  # noqa: E402

from forum.boxes import iou  # noqa: E402
from forum.fusion import fuse  # noqa: E402


def score_vector(indices, box_fn, gt) -> np.ndarray:
    return np.array(
        [sum(iou(box_fn(i), gt[i]) >= t for t in THRESHOLDS) / len(THRESHOLDS) for i in indices]
    )


def paired_bootstrap(a: np.ndarray, b: np.ndarray, draws: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(draws)])
    low, high = np.percentile(means, [2.5, 97.5])
    p = max(2 * min((means <= 0).mean(), (means >= 0).mean()), 1 / draws)
    return 100 * diff.mean(), 100 * low, 100 * high, p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="refadv")
    parser.add_argument("--lineup", default="balanced")
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()

    members, indices, gt = load_lineup(args.predictions, args.benchmark, args.lineup)
    forum = score_vector(indices, lambda i: fuse(candidates(members, i), tau=args.tau), gt)
    plain = score_vector(
        indices, lambda i: fuse(candidates(members, i), rule="average", tau=args.tau), gt
    )
    singles = {
        name: score_vector(indices, lambda i, r=rows, f=fields: primary_box(r, f, i), gt)
        for name, rows, fields in members
    }
    best_name = max(singles, key=lambda name: singles[name].mean())

    print(f"## {args.benchmark} — {args.lineup} lineup (paired bootstrap, 10k resamples)\n")
    for label, baseline in [
        (f"best single ({best_name})", singles[best_name]),
        ("plain ensemble", plain),
    ]:
        delta, low, high, p = paired_bootstrap(forum, baseline)
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        print(f"FORUM vs {label}: delta={delta:+.1f}  95% CI [{low:+.1f}, {high:+.1f}]  {p_text}")


if __name__ == "__main__":
    main()
