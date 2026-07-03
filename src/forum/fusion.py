"""FORUM test-time fusion: agreement-based region selection + medoid localization.

Candidates are (member_id, box) pairs pooled from a set of frozen MLLMs. Stage A clusters
the candidates by IoU (single link) and keeps the cluster supported by the most distinct
members, breaking ties by cluster size. Stage B returns the medoid of the selected cluster:
the actual member box closest to all the others, rather than a synthetic coordinate average.
"""

from __future__ import annotations

from .boxes import Box, iou

Candidate = tuple[int, Box]


def single_link_clusters(candidates: list[Candidate], tau: float = 0.5) -> list[list[int]]:
    """Single-link clustering of candidate boxes: two boxes link when IoU >= tau."""
    clusters: list[list[int]] = []
    for k, (_, box) in enumerate(candidates):
        for cluster in clusters:
            if any(iou(box, candidates[t][1]) >= tau for t in cluster):
                cluster.append(k)
                break
        else:
            clusters.append([k])
    return clusters


def select_region(candidates: list[Candidate], tau: float = 0.5) -> list[int]:
    """Stage A: keep the cluster supported by the most distinct members (ties -> size)."""
    clusters = single_link_clusters(candidates, tau)
    return max(clusters, key=lambda c: (len({candidates[t][0] for t in c}), len(c)))


def localize(boxes: list[Box], rule: str = "medoid", weights: list[float] | None = None) -> Box:
    """Stage B: produce the final box from the selected cluster.

    medoid : the member box closest to the others (maximum total IoU) — an actual prediction.
    average: plain coordinate mean (the standard test-time box-ensemble baseline).
    invvar : precision-weighted mean (weights supplied per box).
    """
    if rule == "medoid":
        return max(boxes, key=lambda b: sum(iou(b, other) for other in boxes))
    if rule == "average":
        return [sum(b[k] for b in boxes) / len(boxes) for k in range(4)]
    if rule == "invvar":
        if weights is None:
            raise ValueError("invvar localization requires per-box weights")
        total = sum(weights) or 1.0
        return [
            sum(b[k] * w for b, w in zip(boxes, weights, strict=True)) / total for k in range(4)
        ]
    raise ValueError(f"unknown localization rule: {rule}")


def fuse(
    candidates: list[Candidate],
    rule: str = "medoid",
    tau: float = 0.5,
    member_weights: dict[int, float] | None = None,
) -> Box | None:
    """Full FORUM prediction from pooled candidates; None when no candidate is available."""
    if not candidates:
        return None
    selected = select_region(candidates, tau)
    boxes = [candidates[t][1] for t in selected]
    weights = None
    if member_weights is not None:
        weights = [member_weights[candidates[t][0]] for t in selected]
    return localize(boxes, rule=rule, weights=weights)
