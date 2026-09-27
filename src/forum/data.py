"""Benchmark loading: Ref-Adv-s and the RefCOCO family, mapped to one row schema.

Every row provides: row_idx, image (PIL), expression, gt_bbox_xyxy (absolute pixels),
width, height, use_negation, distractor_count.
"""

from __future__ import annotations

import json
import random

from datasets import Image as HFImage
from datasets import load_dataset

# benchmark name -> (hf repo, {split: hf split})
BENCHMARKS = {
    "refadv": ("dddraxxx/ref-adv-s", {"": "train"}),
    "refcoco": ("lmms-lab/RefCOCO", {"val": "val", "testA": "testA", "testB": "testB"}),
    "refcocoplus": ("lmms-lab/RefCOCOplus", {"val": "val", "testA": "testA", "testB": "testB"}),
    "refcocog": ("lmms-lab/RefCOCOg", {"val": "val", "test": "test"}),
}


def _distractor_count(value: object) -> int:
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return 0
        return len(parsed) if isinstance(parsed, list) else int(parsed)
    return 0


def load_rows(
    benchmark: str,
    sample: int = 0,
    seed: int = 7,
    limit: int = 0,
) -> list[dict]:
    """Load benchmark rows. `benchmark` is e.g. "refadv" or "refcocoplus:testB".

    sample: deterministic random subset size (0 = all rows). limit: hard cap after sampling.
    """
    name, _, split = benchmark.partition(":")
    repo, splits = BENCHMARKS[name]
    ds = load_dataset(repo, split=splits[split]).cast_column("image", HFImage(decode=True))
    indices = list(range(len(ds)))
    if sample and sample < len(indices):
        indices = sorted(random.Random(seed).sample(indices, sample))
    rows = []
    for i in indices:
        r = ds[i]
        image = r["image"]
        width, height = image.size
        if name == "refadv":
            gt = [float(v) for v in r["solution"]]
            expression = str(r.get("normal_caption", ""))
            negation = bool(r.get("use_negation", False))
            distractors = _distractor_count(r.get("distractors"))
        else:
            x, y, w, h = (float(v) for v in r["bbox"])  # COCO xywh
            gt = [x, y, x + w, y + h]
            answer = r.get("answer")
            expression = str(answer[0] if isinstance(answer, list) and answer else answer or "")
            negation, distractors = False, 0
        rows.append(
            {
                "row_idx": i,
                "image": image,
                "expression": expression,
                "gt_bbox_xyxy": gt,
                "width": int(width),
                "height": int(height),
                "use_negation": negation,
                "distractor_count": distractors,
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows
