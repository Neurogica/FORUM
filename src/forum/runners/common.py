"""Shared runner loop: greedy grounding, optional stochastic samples, optional zoom candidate.

Each model runner supplies an `ask(image, prompt, greedy, n)` callback returning generated
text(s). This module handles data loading, box parsing, the optional zoom re-ask (crop the
region around the greedy box, upscale, re-ask, map the answer back), and JSONL output.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from ..boxes import iou, parse_boxes, valid
from ..data import load_rows

AskFn = Callable[[Image.Image, str, bool, int], list[str]]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", default="refadv", help='e.g. "refadv", "refcocoplus:testB"')
    parser.add_argument("--sample", type=int, default=0, help="deterministic subset size (0=all)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--k", type=int, default=0, help="stochastic samples beyond greedy")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--zoom", action="store_true", help="add a zoomed re-ask candidate")
    parser.add_argument("--zoom-size", type=int, default=1280)
    parser.add_argument("--zoom-margin", type=float, default=0.6)
    parser.add_argument("--zoom-min-frac", type=float, default=0.15)
    parser.add_argument("--output", required=True, help="output predictions JSONL path")


def expand_roi(
    box: list[float], width: int, height: int, margin: float, min_frac: float
) -> list[float]:
    x1, y1, x2, y2 = box
    bw = max((x2 - x1) * (1 + 2 * margin), width * min_frac)
    bh = max((y2 - y1) * (1 + 2 * margin), height * min_frac)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [
        max(0.0, cx - bw / 2),
        max(0.0, cy - bh / 2),
        min(float(width), cx + bw / 2),
        min(float(height), cy + bh / 2),
    ]


def zoom_candidate(
    ask: AskFn,
    image: Image.Image,
    prompt: str,
    greedy_box: list[float],
    fmt: str,
    size: int,
    margin: float,
    min_frac: float,
) -> list[float] | None:
    """Crop an expanded region around the greedy box, re-ask, and map the box back."""
    width, height = image.size
    rx1, ry1, rx2, ry2 = (
        int(round(v)) for v in expand_roi(greedy_box, width, height, margin, min_frac)
    )
    rx2, ry2 = max(rx2, rx1 + 1), max(ry2, ry1 + 1)
    crop = image.convert("RGB").crop((rx1, ry1, rx2, ry2))
    scale = size / max(crop.size)
    if scale > 1.0:
        crop = crop.resize((round(crop.size[0] * scale), round(crop.size[1] * scale)))
    text = ask(crop, prompt, True, 1)[0]
    boxes = parse_boxes(text, rx2 - rx1, ry2 - ry1, fmt)
    if not boxes:
        return None
    b = boxes[0]
    return [b[0] + rx1, b[1] + ry1, b[2] + rx1, b[3] + ry1]


def run(args: argparse.Namespace, ask: AskFn, fmt: str, prompt_template: str) -> None:
    rows = load_rows(args.benchmark, sample=args.sample, seed=args.seed, limit=args.limit)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hits, start = 0, time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for n, row in enumerate(rows, 1):
            image, width, height = row["image"], row["width"], row["height"]
            prompt = prompt_template.format(ref=row["expression"])
            greedy = parse_boxes(ask(image, prompt, True, 1)[0], width, height, fmt)
            greedy_box = greedy[0] if greedy else None
            samples = []
            if args.k > 0:
                for text in ask(image, prompt, False, args.k):
                    boxes = parse_boxes(text, width, height, fmt)
                    samples.append(boxes[0] if boxes else None)
            zoom = None
            if args.zoom and valid(greedy_box):
                zoom = zoom_candidate(
                    ask,
                    image,
                    prompt,
                    greedy_box,
                    fmt,
                    args.zoom_size,
                    args.zoom_margin,
                    args.zoom_min_frac,
                )
            record = {
                "row_idx": row["row_idx"],
                "expression": row["expression"],
                "gt_bbox_xyxy": row["gt_bbox_xyxy"],
                "use_negation": row["use_negation"],
                "distractor_count": row["distractor_count"],
                "greedy_box": greedy_box,
            }
            if zoom is not None:
                record["zoom_box"] = zoom
            if samples:
                record["sample_boxes"] = samples
            f.write(json.dumps(record) + "\n")
            f.flush()
            hits += int(iou(greedy_box, row["gt_bbox_xyxy"]) >= 0.5)
            if n % 25 == 0 or n == len(rows):
                rate = n / max(1e-9, time.time() - start)
                print(f"[{n}/{len(rows)}] greedy@0.5={hits / n:.3f}  {rate:.2f} rows/s", flush=True)
    print(f"DONE n={len(rows)} greedy@0.5={hits / len(rows):.4f} -> {out_path}", flush=True)
