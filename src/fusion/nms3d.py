"""Class-wise greedy Non-Maximum Suppression over DetectionBox lists.

Algorithm:
  1. Sort boxes by detection_score descending.
  2. Greedily keep the highest-score box.
  3. Suppress any remaining box whose BEV IoU with a kept box exceeds
     iou_threshold.
  4. Repeat until the candidate list is empty.

Class-aware mode (default): NMS is applied independently per detection class
so boxes from different classes never suppress each other.
"""

from __future__ import annotations

from src.fusion.bev_iou import bev_iou_matrix
from src.io.schemas import DetectionBox
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def nms3d(
    boxes: list[DetectionBox],
    iou_threshold: float = 0.3,
    class_aware: bool = True,
    max_boxes: int = 500,
) -> list[DetectionBox]:
    """Apply greedy NMS and return surviving boxes sorted by score.

    Args:
        boxes: Input boxes (any order, any mix of classes).
        iou_threshold: Suppress if BEV IoU >= this value.
        class_aware: If True, run NMS per class (recommended).
        max_boxes: Hard cap on returned boxes per call.

    Returns:
        Filtered list of DetectionBox, sorted by score descending.
    """
    if not boxes:
        return []

    if class_aware:
        kept: list[DetectionBox] = []
        classes = {b.detection_name for b in boxes}
        for cls in sorted(classes):
            cls_boxes = [b for b in boxes if b.detection_name == cls]
            kept.extend(_greedy_nms(cls_boxes, iou_threshold))
        kept.sort(key=lambda b: b.detection_score, reverse=True)
    else:
        kept = _greedy_nms(boxes, iou_threshold)

    if len(kept) > max_boxes:
        log.debug("NMS capped output from %d to %d boxes", len(kept), max_boxes)
        kept = kept[:max_boxes]

    return kept


def _greedy_nms(boxes: list[DetectionBox], iou_threshold: float) -> list[DetectionBox]:
    """Single-class greedy NMS — internal helper.

    Precomputes the full pairwise IoU matrix ONCE via the vectorized
    `bev_iou_matrix` instead of calling the single-pair `bev_iou` inside
    the O(n^2) suppression loop (rebuilding shapely polygons from scratch
    on every call) -- same perf fix pattern as §40's
    `_match_predictions_to_gt` (see docs/EXPERIMENT_LOG.md). Was the
    remaining bottleneck after §40: a 1,804-token rotated-IoU inference
    pass took ~10min with the per-pair version.
    """
    sorted_boxes = sorted(boxes, key=lambda b: b.detection_score, reverse=True)
    n = len(sorted_boxes)
    kept: list[DetectionBox] = []
    suppressed = [False] * n
    if n == 0:
        return kept

    iou_mat = bev_iou_matrix(sorted_boxes, sorted_boxes)

    for i in range(n):
        if suppressed[i]:
            continue
        kept.append(sorted_boxes[i])
        row = iou_mat[i]
        for j in range(i + 1, n):
            if not suppressed[j] and row[j] >= iou_threshold:
                suppressed[j] = True

    return kept
