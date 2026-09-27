#!/usr/bin/env python3
"""Agreement-versus-confidence analysis on Ref-Adv-s (the selection-signal study).

    python scripts/agreement_analysis.py --lineup balanced

Reports:
1. the 2x2 accuracy table of self-consistency (k stochastic samples reproducing the
   primary box) versus cross-model agreement (>=2 members in one cluster) — the
   "confident yet wrong" distractor trap sits in the confident/disagreeing cell;
2. accuracy as a function of the number of agreeing members;
3. the pairwise error correlation between members.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import load_lineup, primary_box  # noqa: E402

from forum.boxes import iou, valid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineup", default="balanced")
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument(
        "--confident-at",
        type=int,
        default=7,
        help="samples reproducing the primary box to count as confident",
    )
    args = parser.parse_args()

    members, indices, gt = load_lineup(args.predictions, "refadv", args.lineup)
    primary_name, primary_rows, primary_fields = members[0]

    def agreement(i: int) -> int:
        boxes = [primary_box(rows, fields, i) for _, rows, fields in members]
        boxes = [b for b in boxes if valid(b)]
        return max(1 + sum(iou(a, b) >= 0.5 for b in boxes if b is not a) for a in boxes)

    # 1) 2x2: self-consistency x cross-model agreement (primary member = first in lineup)
    cells: dict[tuple[bool, bool], list[int]] = {}
    for i in indices:
        box = primary_rows[i].get("greedy_box")
        samples = [s for s in primary_rows[i].get("sample_boxes") or [] if valid(s)]
        if not valid(box) or not samples:
            continue
        support = sum(iou(s, box) >= 0.5 for s in samples)
        key = (support >= args.confident_at, agreement(i) >= 2)
        cells.setdefault(key, []).append(int(iou(box, gt[i]) >= 0.5))
    total = sum(len(v) for v in cells.values())
    print(f"## 2x2: self-consistency x agreement — primary {primary_name} (n={total})\n")
    print("| | agree >= 2 | agree < 2 |")
    print("|---|---|---|")
    for confident, label in [(True, "confident"), (False, "not confident")]:
        row = []
        for agree in (True, False):
            values = cells.get((confident, agree), [])
            row.append(f"{100 * np.mean(values):.1f} ({len(values)})" if values else "-")
        print(f"| {label} | {row[0]} | {row[1]} |")

    # 2) accuracy by number of agreeing members (primary-box correctness)
    by_agree: dict[int, list[int]] = {}
    for i in indices:
        box = primary_box(primary_rows, primary_fields, i)
        if not valid(box):
            continue
        by_agree.setdefault(agreement(i), []).append(int(iou(box, gt[i]) >= 0.5))
    print("\n## Accuracy by number of agreeing members\n")
    for count in sorted(by_agree):
        values = by_agree[count]
        print(f"- {count} member(s): {100 * np.mean(values):.1f}% (n={len(values)})")

    # 3) pairwise error correlation between members
    errors = {
        name: np.array([int(iou(primary_box(rows, fields, i), gt[i]) < 0.5) for i in indices])
        for name, rows, fields in members
    }
    print("\n## Pairwise error correlation\n")
    names = list(errors)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            corr = np.corrcoef(errors[names[a]], errors[names[b]])[0, 1]
            print(f"- {names[a]} x {names[b]}: {corr:.3f}")


if __name__ == "__main__":
    main()
