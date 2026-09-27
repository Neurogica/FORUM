"""Bounding-box utilities: validation, IoU, coordinate conversion, and text parsing."""

from __future__ import annotations

import re

Box = list[float]

_BOX_WRAPPER_RE = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", flags=re.S)
_KEYED_RE = re.compile(
    r'"(?:bboxes|bbox_2d|box_2d|bbox|box)"\s*:\s*\[+\s*'
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
)
_QUAD_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def valid(box: object) -> bool:
    """A box is a length-4 [x1, y1, x2, y2] list with positive width and height."""
    return isinstance(box, (list, tuple)) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def iou(a: object, b: object) -> float:
    if not valid(a) or not valid(b):
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def to_abs_xyxy(quad: list[float], width: int, height: int, fmt: str) -> Box:
    """Convert a raw 4-number model output to absolute [x1, y1, x2, y2] pixels.

    fmt: "norm1000_xyxy" | "norm1000_yxyx" | "pixel"
    """
    a, b, c, d = (float(v) for v in quad)
    if fmt == "norm1000_xyxy":
        return [a / 1000 * width, b / 1000 * height, c / 1000 * width, d / 1000 * height]
    if fmt == "norm1000_yxyx":
        return [b / 1000 * width, a / 1000 * height, d / 1000 * width, c / 1000 * height]
    if fmt == "pixel":
        return [a, b, c, d]
    raise ValueError(f"unknown box format: {fmt}")


def parse_boxes(text: str, width: int, height: int, fmt: str) -> list[Box]:
    """Extract bounding boxes from generated text.

    Prefers boxes attached to a JSON key (bboxes / box_2d / bbox / box); falls back to any
    bracketed 4-number group. Returns absolute-pixel xyxy boxes, invalid boxes filtered out.
    """
    if not text or not text.strip():
        return []
    wrapped = _BOX_WRAPPER_RE.search(text)
    if wrapped:
        text = wrapped.group(1)
    quads = _KEYED_RE.findall(text) or _QUAD_RE.findall(text)
    out = []
    for quad in quads:
        box = to_abs_xyxy(list(quad), width, height, fmt)
        if valid(box):
            out.append(box)
    return out
