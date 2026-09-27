#!/usr/bin/env python3
"""Evaluate FORUM against singles and the plain coordinate-averaging ensemble.

    python scripts/evaluate.py --benchmark refadv --lineup balanced
    python scripts/evaluate.py --benchmark refcocoplus_testB --lineup balanced --loo
    python scripts/evaluate.py --benchmark refadv --lineup balanced --ablation --tau-sweep

Reads member prediction JSONL files from predictions/<benchmark>/<member>.jsonl.
The primary metric is the mean of Acc@{0.5, 0.75, 0.9}; each threshold is also reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forum.boxes import iou, valid
from forum.fusion import fuse

THRESHOLDS = (0.5, 0.75, 0.9)

# lineup -> [(member file stem, display name, candidate fields per benchmark kind)]
# On Ref-Adv-s the candidate pool optionally includes the zoom re-ask box; on the RefCOCO
# family every member contributes its greedy box only.
LINEUPS = {
    "balanced": [
        ("qwen35_27b", "Qwen3.5-27B", {"refadv": ["greedy_box"]}),
        ("gemma4_31b", "Gemma-4-31B", {"refadv": ["zoom_box", "greedy_box"]}),
        ("glm46v", "GLM-4.6V", {"refadv": ["zoom_box", "greedy_box"]}),
    ],
    "dominant": [
        ("qwen36_35b_a3b", "Qwen3.6-35B-A3B", {"refadv": ["zoom_box", "greedy_box"]}),
        ("gemma4_31b", "Gemma-4-31B", {"refadv": ["zoom_box", "greedy_box"]}),
        ("nemotron_30b", "Nemotron-Omni-30B", {"refadv": ["zoom_box", "greedy_box"]}),
    ],
}
DEFAULT_FIELDS = ["greedy_box"]


def load_predictions(path: Path) -> dict[int, dict]:
    return {int(json.loads(line)["row_idx"]): json.loads(line) for line in path.open()}


def load_lineup(predictions_dir: Path, benchmark: str, lineup: str):
    kind = benchmark.split("_")[0].split(":")[0]
    members = []
    for stem, name, fields_by_kind in LINEUPS[lineup]:
        path = predictions_dir / benchmark / f"{stem}.jsonl"
        members.append((name, load_predictions(path), fields_by_kind.get(kind, DEFAULT_FIELDS)))
    indices = sorted(set.intersection(*[set(rows) for _, rows, _ in members]))
    gt = {i: members[0][1][i]["gt_bbox_xyxy"] for i in indices}
    return members, indices, gt


def candidates(members, i, subset=None):
    return [
        (j, rows[i][field])
        for j, (_, rows, fields) in enumerate(members)
        if (subset is None or j in subset) and i in rows
        for field in fields
        if valid(rows[i].get(field))
    ]


def primary_box(rows: dict, fields: list[str], i: int):
    for field in fields:
        if valid(rows[i].get(field)):
            return rows[i][field]
    return None


def metrics(indices, box_fn, gt):
    accs = [
        100 * sum(iou(box_fn(i), gt[i]) >= t for i in indices) / len(indices) for t in THRESHOLDS
    ]
    return (sum(accs) / len(accs), *accs)


def fmt(values) -> str:
    return " | ".join(f"{v:.1f}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="refadv")
    parser.add_argument("--lineup", choices=sorted(LINEUPS), default="balanced")
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--loo", action="store_true", help="leave-one-member-out")
    parser.add_argument("--tau-sweep", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="localization-rule ablation")
    parser.add_argument("--facets", action="store_true", help="negation / distractor breakdown")
    args = parser.parse_args()

    members, indices, gt = load_lineup(args.predictions, args.benchmark, args.lineup)
    full = set(range(len(members)))

    def forum(i, subset=None, rule="medoid", tau=None, weights=None):
        return fuse(
            candidates(members, i, subset), rule=rule, tau=tau or args.tau, member_weights=weights
        )

    print(f"## {args.benchmark} — {args.lineup} lineup (n={len(indices)})\n")
    print("| System | mean | @0.5 | @0.75 | @0.9 |")
    print("|---|---|---|---|---|")
    for name, rows, fields in members:
        m = metrics(indices, lambda i, r=rows, f=fields: primary_box(r, f, i), gt)
        print(f"| {name} (single) | {fmt(m)} |")
    print(f"| Plain ensemble | {fmt(metrics(indices, lambda i: forum(i, rule='average'), gt))} |")
    print(f"| **FORUM** | **{fmt(metrics(indices, forum, gt)).replace(' | ', '** | **')}** |")

    if args.loo:
        print("\n### Leave-one-member-out (mean)\n")
        for j, (name, _, _) in enumerate(members):
            m = metrics(indices, lambda i, s=full - {j}: forum(i, s), gt)
            print(f"- drop {name}: {m[0]:.1f}")

    if args.tau_sweep:
        print("\n### Clustering-threshold sensitivity (FORUM mean)\n")
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
            m = metrics(indices, lambda i, t=tau: forum(i, tau=t), gt)
            print(f"- tau={tau}: {m[0]:.1f}")

    if args.ablation:
        calib = indices[::2]  # precision weights fitted on the even-index half
        weights = {
            j: max(
                1e-3,
                sum(iou(primary_box(rows, fields, i), gt[i]) >= 0.75 for i in calib) / len(calib),
            )
            for j, (_, rows, fields) in enumerate(members)
        }
        print("\n### Localization ablation (same agreement-selected clusters)\n")
        for label, kwargs in [
            ("Coordinate average (plain ensemble)", {"rule": "average"}),
            ("Inverse-variance weighted", {"rule": "invvar", "weights": weights}),
            ("Medoid (FORUM)", {"rule": "medoid"}),
        ]:
            m = metrics(indices, lambda i, k=kwargs: forum(i, **k), gt)
            print(f"- {label}: {fmt(m)}")

    if args.facets:
        rows0 = members[0][1]
        facets = [
            ("negation", lambda r: r.get("use_negation")),
            ("no negation", lambda r: not r.get("use_negation")),
            ("2-3 distractors", lambda r: 2 <= (r.get("distractor_count") or 0) <= 3),
            ("4-6 distractors", lambda r: 4 <= (r.get("distractor_count") or 0) <= 6),
            (">=7 distractors", lambda r: (r.get("distractor_count") or 0) >= 7),
        ]
        print("\n### Facet breakdown (FORUM mean)\n")
        for label, predicate in facets:
            subset = [i for i in indices if predicate(rows0[i])]
            if not subset:
                continue
            m = metrics(subset, forum, gt)
            print(f"- {label} (n={len(subset)}): {m[0]:.1f}")


if __name__ == "__main__":
    main()
