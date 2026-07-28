"""Bird's-Eye-View (BEV) IoU between axis-aligned 3-D boxes, projected to the
XY plane.

Boxes are treated as axis-aligned rectangles in the XY plane (yaw ignored
by design). Two implementations are provided:
- ``bev_iou``        – single-pair (used in unit tests, NMS)
- ``bev_iou_matrix`` – batch (used in feature extraction)

HISTORY (see docs/EXPERIMENT_LOG.md):
- Originally axis-aligned only (yaw ignored by design) to keep the CPU
  path simple after the §17 GPU-path corruption bug (a CUDA
  `mmcv.ops.boxes_iou_bev` fast path was fed the wrong box encoding and
  produced nonsensical rotated-IoU values).
- §38 found the axis-aligned convention systematically *understates* true
  overlap for elongated objects at non-trivial headings (trailer, truck,
  bus, barrier) — routine for turning/angled vehicles, not an edge case.
  §40 replaced the geometry with a correct rotated computation (numpy
  distance pre-filter + shapely exact polygon intersection, vectorized via
  shapely 2.x's array ufuncs) and rebuilt the full pipeline on it.
- §40's result: rotated geometry produced a real, measurable
  orientation/attribute-error improvement (mAOE, mAAE) but a net *mAP*
  regression (−0.55pp) versus the axis-aligned baseline. §41 tested and
  refuted the hypothesis that this was an NMS-suppression artifact
  recoverable by retuning `iou_threshold` — the regression (specifically
  bus's) was completely NMS-threshold-invariant, ruling that out.
- **Reverted back to this axis-aligned implementation** as the project's
  validated production convention — rotated IoU is not adopted. Kept as
  a pure-numpy vectorized implementation (fast: full-dataset rebuilds run
  in ~15s), no shapely dependency needed for this convention.
"""

from __future__ import annotations

import numpy as np

from src.io.schemas import DetectionBox


def bev_iou(a: DetectionBox, b: DetectionBox) -> float:
    """Axis-aligned BEV IoU between two DetectionBox objects.

    Box convention: size = [width, length, height]. Axis-aligned means
    each box spans [cx - length/2, cx + length/2] in x and
    [cy - width/2, cy + width/2] in y (yaw ignored) — matches the
    heading-along-x convention used elsewhere in this project when yaw=0.
    """
    ax, ay = a.translation[0], a.translation[1]
    aw, al = a.size[0], a.size[1]
    bx, by = b.translation[0], b.translation[1]
    bw, bl = b.size[0], b.size[1]

    a_xmin, a_xmax = ax - al / 2.0, ax + al / 2.0
    a_ymin, a_ymax = ay - aw / 2.0, ay + aw / 2.0
    b_xmin, b_xmax = bx - bl / 2.0, bx + bl / 2.0
    b_ymin, b_ymax = by - bw / 2.0, by + bw / 2.0

    ix = max(0.0, min(a_xmax, b_xmax) - max(a_xmin, b_xmin))
    iy = max(0.0, min(a_ymax, b_ymax) - max(a_ymin, b_ymin))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0

    union = (aw * al) + (bw * bl) - inter
    return inter / union if union > 0.0 else 0.0


def _boxes_to_array(boxes: list[DetectionBox]) -> np.ndarray:
    """Convert boxes to [N, 4] array: [cx, cy, w, l]. A 5th (yaw) column is
    tolerated and ignored by _iou_from_arrays for call-site compatibility
    with code that still carries yaw (e.g. GT boxes) — see raw_iou_matrix.
    """
    arr = np.empty((len(boxes), 4), dtype=np.float64)
    for i, b in enumerate(boxes):
        arr[i, 0] = b.translation[0]
        arr[i, 1] = b.translation[1]
        arr[i, 2] = b.size[0]
        arr[i, 3] = b.size[1]
    return arr


def _iou_from_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised axis-aligned BEV IoU. Returns [N, M] float64 matrix.

    Accepts either [N,4] ([cx,cy,w,l]) or [N,5] ([cx,cy,w,l,yaw]) arrays --
    any yaw column is ignored (kept for compatibility with callers that
    still carry yaw, e.g. router_dataset.py's GT matching, which built the
    5-column format while rotated IoU was in production; see HISTORY).
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    a_xmin = a[:, 0] - a[:, 3] / 2.0
    a_xmax = a[:, 0] + a[:, 3] / 2.0
    a_ymin = a[:, 1] - a[:, 2] / 2.0
    a_ymax = a[:, 1] + a[:, 2] / 2.0
    b_xmin = b[:, 0] - b[:, 3] / 2.0
    b_xmax = b[:, 0] + b[:, 3] / 2.0
    b_ymin = b[:, 1] - b[:, 2] / 2.0
    b_ymax = b[:, 1] + b[:, 2] / 2.0

    ix = np.maximum(
        0.0,
        np.minimum(a_xmax[:, None], b_xmax[None, :]) - np.maximum(a_xmin[:, None], b_xmin[None, :]),
    )
    iy = np.maximum(
        0.0,
        np.minimum(a_ymax[:, None], b_ymax[None, :]) - np.maximum(a_ymin[:, None], b_ymin[None, :]),
    )
    inter = ix * iy

    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    union = area_a + area_b - inter

    result = np.zeros((n, m), dtype=np.float64)
    valid = union > 0.0
    result[valid] = inter[valid] / union[valid]
    return result


def bev_iou_matrix(
    boxes_a: list[DetectionBox],
    boxes_b: list[DetectionBox],
) -> np.ndarray:
    """Return [N, M] float64 axis-aligned BEV IoU matrix."""
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    return _iou_from_arrays(
        _boxes_to_array(boxes_a),
        _boxes_to_array(boxes_b),
    )


def raw_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Public entry point for _iou_from_arrays -- lets other modules (e.g.
    src/moe/router_dataset.py, which matches DetectionBox against GtBox,
    two different types) reuse the same vectorized computation instead of
    re-deriving it or calling the single-pair bev_iou() in a naive per-pair
    loop inside a greedy-matching loop (the original §39/§40 performance
    bug when this path was rotated -- see HISTORY above).

    a, b: [N, 4] or [N, 5] arrays of [cx, cy, w, l] or [cx, cy, w, l, yaw]
    (any yaw column is ignored -- see _iou_from_arrays).
    """
    return _iou_from_arrays(a, b)
