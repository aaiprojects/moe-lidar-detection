"""Router feature extraction from DetectionBox objects.

Features per box (all numeric, ready for gradient boosting):

  Static (per-box):
    expert_id          int   0=centerpoint, 1=pointpillars, 2=centerpoint_pillar,
                             3=ssn, 4=voxelnext, 5=transfusion_l,
                             6=bevfusion_lidar
    detection_score    float model confidence
    dist_from_ego      float √(x²+y²) in metres
    box_width          float metres
    box_length         float metres
    box_height         float metres
    vel_magnitude      float √(vx²+vy²) m/s

  Cross-expert peer features (any class, BEV IoU ≥ 0.1).
  These encode *presence* agreement — a car/truck pair at the same location
  still counts, because something is likely there:

    n_peer_overlaps    int   peer boxes with BEV IoU ≥ 0.1 (any class)
    max_peer_iou       float highest BEV IoU with any peer box
    mean_peer_score    float mean detection_score of overlapping peer boxes
                             (0 if no overlapping peers)

  Uncertainty / disagreement features:
    score_variance     float variance of scores of overlapping peer boxes
                             (0 if ≤1 overlapping peers)
    expert_agreement   float fraction of other active experts that have a
                             matching box (IoU ≥ 0.1, any class)
    n_spatial_overlaps int   count of spatially overlapping peer boxes
                             (same as n_peer_overlaps under class-agnostic
                             matching; exposed next to class_agreement so
                             the model can condition on peer density)
    class_agreement    float among spatial peers, fraction with SAME class.
                             -1.0 = no spatial peers (missing / unknown —
                             HistGradientBoosting routes this separately)
                             0.0  = peers present but all disagree on class
                             1.0  = all spatial peers agree on class
    max_class_score    float highest detection_score for this class in this
                             sample across ALL experts
    n_active_experts   int   number of experts that contributed ≥1 box

Note: LiDAR occupancy features (n_points_inside, point_density) were tested
and hurt held-out mAP (single-sweep vs multi-sweep mismatch). They are NOT
part of FEATURE_NAMES. Infrastructure remains in src/utils/lidar_utils.py
for future multi-sweep work.

  Map prior (optional, see docs/EXPERIMENT_LOG.md):
    dist_to_drivable_area  float  approximate distance (m) from the box's
                                  BEV center to the nearest drivable-area
                                  boundary in nuScenes' rasterized semantic-
                                  prior map mask (0.0 = on drivable area).
                                  -1.0 sentinel when no map mask is available
                                  for this token (e.g. live/non-nuScenes
                                  inference) -- same missing-value convention
                                  as class_agreement's CLASS_AGREEMENT_MISSING.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.fusion.bev_iou import bev_iou, bev_iou_matrix
from src.io.schemas import DetectionBox

# Canonical class order matches nuscenes_infos_val.pkl metainfo
_CLASS_TO_ID: dict[str, int] = {
    "car": 0,
    "truck": 1,
    "trailer": 2,
    "bus": 3,
    "construction_vehicle": 4,
    "bicycle": 5,
    "motorcycle": 6,
    "pedestrian": 7,
    "traffic_cone": 8,
    "barrier": 9,
}

# Expert name → integer id (extend as new experts are added)
_EXPERT_TO_ID: dict[str, int] = {
    "centerpoint": 0,
    "pointpillars": 1,
    "centerpoint_pillar": 2,
    "ssn": 3,
    "voxelnext": 4,
    "transfusion_l": 5,
    "bevfusion_lidar": 6,
}

# Model input features. class_id is deliberately excluded: each per-class
# router is fit on rows already filtered to one class_id, so it's a
# constant within every training call and carries zero split information
# (confirmed empirically -- gain-based importance is exactly 0.0 across all
# 10 classes). It's still tracked as row metadata for filtering, just not
# fed to the model -- see BoxFeatures.class_id and router_dataset.py, which
# adds it back to the CSV as its own column, same as label/sample_token.
FEATURE_NAMES: list[str] = [
    "expert_id",
    "detection_score",
    "dist_from_ego",
    "box_width",
    "box_length",
    "box_height",
    "vel_magnitude",
    "n_peer_overlaps",
    "max_peer_iou",
    "mean_peer_score",
    "score_variance",
    "expert_agreement",
    "n_spatial_overlaps",
    "class_agreement",
    "max_class_score",
    "n_active_experts",
    "dist_to_drivable_area",
]

# FT-Transformer input features: FEATURE_NAMES minus the map-derived
# dist_to_drivable_area (XGBoost-only; see docs/EXPERIMENT_LOG.md). Named
# rather than a positional X[:, :N] slice, so it stays correct regardless
# of where dist_to_drivable_area sits in FEATURE_NAMES or how many other
# features are added/removed around it.
NN_FEATURE_NAMES: list[str] = [f for f in FEATURE_NAMES if f != "dist_to_drivable_area"]

# Sentinel for class_agreement when no spatial peers exist.
# Distinct from 0.0 (= peers present but unanimous class disagreement).
CLASS_AGREEMENT_MISSING: float = -1.0

# Sentinel for dist_to_drivable_area when no map mask is available for this
# token (e.g. live Ouster inference outside nuScenes' mapped locations).
# Distinct from 0.0 (= on drivable area) and any positive off-road distance.
DIST_TO_DRIVABLE_AREA_MISSING: float = -1.0

# Approximate off-road distance via a small number of vectorized dilation
# queries against the rasterized mask (see _dist_to_drivable_area_batch) --
# cheap without needing the (absent) nuScenes map-expansion vector pack.
#
# IMPORTANT: nuscenes.utils.map_mask.MapMask.mask(dilation) is decorated
# with @cached(cache=LRUCache(maxsize=3)) -- each *distinct* dilation value
# recomputes a full-image cv2.distanceTransform (expensive: this made a
# 50-token smoke test take ~20 min before this was found, one call could
# take minutes). Together with the always-queried dilation=0.0, this ladder
# must have <=2 entries so all 3 distinct dilations fit in that cache and
# get computed once per map (4 maps total), not re-thrashed every token.
_DRIVABLE_AREA_DILATION_LEVELS_M: tuple[float, ...] = (2.0, 8.0)
_DRIVABLE_AREA_OFF_ROAD_CAP_M: float = 16.0

PEER_IOU_THRESHOLD: float = 0.1


@dataclass
class BoxFeatures:
    expert_id: int
    class_id: int
    detection_score: float
    dist_from_ego: float
    box_width: float
    box_length: float
    box_height: float
    vel_magnitude: float
    n_peer_overlaps: int
    max_peer_iou: float
    mean_peer_score: float
    score_variance: float
    expert_agreement: float
    n_spatial_overlaps: int
    class_agreement: float
    max_class_score: float
    n_active_experts: int
    dist_to_drivable_area: float = DIST_TO_DRIVABLE_AREA_MISSING

    def to_list(self) -> list[float]:
        """Model input vector, in FEATURE_NAMES order. class_id is kept as
        a field on this dataclass (row metadata, used for filtering by
        router_dataset.py) but deliberately excluded here -- see the
        FEATURE_NAMES comment in this module for why."""
        return [
            float(self.expert_id),
            self.detection_score,
            self.dist_from_ego,
            self.box_width,
            self.box_length,
            self.box_height,
            self.vel_magnitude,
            float(self.n_peer_overlaps),
            self.max_peer_iou,
            self.mean_peer_score,
            self.score_variance,
            self.expert_agreement,
            float(self.n_spatial_overlaps),
            self.class_agreement,
            self.max_class_score,
            float(self.n_active_experts),
            self.dist_to_drivable_area,
        ]


def _dist_to_drivable_area_batch(xs: np.ndarray, ys: np.ndarray, mask) -> np.ndarray:
    """Approximate distance (m) from each (x, y) point to the nearest
    drivable-area boundary, via a handful of vectorized dilation queries
    against nuScenes' rasterized semantic-prior mask (`nuscenes.utils.
    map_mask.MapMask`, part of the base dataset -- no map-expansion vector
    pack required). 0.0 if already on drivable area; capped at
    _DRIVABLE_AREA_OFF_ROAD_CAP_M for points still off-mask at the largest
    dilation tested. Not exact geometry (unlike bev_iou's polygon
    intersection) -- a cheap, discretized estimate, sufficient for a
    router feature.
    """
    n = len(xs)
    dist = np.full(n, _DRIVABLE_AREA_OFF_ROAD_CAP_M, dtype=np.float64)
    on0 = mask.is_on_mask(xs, ys, dilation=0.0).astype(bool)
    dist[on0] = 0.0
    remaining = ~on0
    for d in _DRIVABLE_AREA_DILATION_LEVELS_M:
        if not remaining.any():
            break
        idx = np.nonzero(remaining)[0]
        on_d = mask.is_on_mask(xs[idx], ys[idx], dilation=d).astype(bool)
        newly_on = idx[on_d]
        dist[newly_on] = d
        remaining[newly_on] = False
    return dist


def extract_features(
    box: DetectionBox,
    peer_boxes: list[DetectionBox],
    n_active_experts: int,
    n_other_experts: int,
    max_class_score: float,
    peer_ious: np.ndarray | None = None,
    dist_to_drivable_area: float = DIST_TO_DRIVABLE_AREA_MISSING,
) -> BoxFeatures:
    """Compute router features for one predicted box.

    Peer presence features (n_peer_overlaps, expert_agreement, …) are
    class-agnostic.  ``class_agreement`` separately reports what fraction of
    those spatial peers share this box's class.  When no spatial peers exist,
    class_agreement is CLASS_AGREEMENT_MISSING (-1.0), not 0.0 — so "isolated
    detection" is not conflated with "surrounded by class disagreement."
    """
    x, y = box.translation[0], box.translation[1]
    vx, vy = box.velocity[0], box.velocity[1]

    expert_id = _EXPERT_TO_ID.get(box.model_name, len(_EXPERT_TO_ID))
    class_id = _CLASS_TO_ID.get(box.detection_name, -1)

    overlapping_peer_scores: list[float] = []
    overlapping_peer_experts: set[str] = set()
    max_iou = 0.0
    n_same_class_spatial = 0

    def _accumulate(peer: DetectionBox, iou: float) -> None:
        nonlocal max_iou, n_same_class_spatial
        if iou < PEER_IOU_THRESHOLD:
            return
        # Presence signal — any class
        overlapping_peer_scores.append(peer.detection_score)
        overlapping_peer_experts.add(peer.model_name)
        if iou > max_iou:
            max_iou = iou
        # Class-agreement numerator
        if peer.detection_name == box.detection_name:
            n_same_class_spatial += 1

    if peer_ious is not None and len(peer_boxes) > 0:
        for j, peer in enumerate(peer_boxes):
            _accumulate(peer, float(peer_ious[j]))
    else:
        for peer in peer_boxes:
            _accumulate(peer, bev_iou(box, peer))

    n_overlaps = len(overlapping_peer_scores)
    mean_peer = sum(overlapping_peer_scores) / n_overlaps if n_overlaps else 0.0

    if n_overlaps >= 2:
        mu = mean_peer
        score_variance = sum((s - mu) ** 2 for s in overlapping_peer_scores) / n_overlaps
    else:
        score_variance = 0.0

    expert_agreement = (
        len(overlapping_peer_experts) / n_other_experts
        if n_other_experts > 0
        else 0.0
    )

    # -1.0 = no peers (unknown); 0.0 = peers present but all disagree on class
    if n_overlaps == 0:
        class_agreement = CLASS_AGREEMENT_MISSING
    else:
        class_agreement = n_same_class_spatial / n_overlaps

    return BoxFeatures(
        expert_id=expert_id,
        class_id=class_id,
        detection_score=box.detection_score,
        dist_from_ego=math.sqrt(x * x + y * y),
        box_width=box.size[0],
        box_length=box.size[1],
        box_height=box.size[2],
        vel_magnitude=math.sqrt(vx * vx + vy * vy),
        n_peer_overlaps=n_overlaps,
        max_peer_iou=max_iou,
        mean_peer_score=mean_peer,
        score_variance=score_variance,
        expert_agreement=expert_agreement,
        n_spatial_overlaps=n_overlaps,
        class_agreement=class_agreement,
        max_class_score=max_class_score,
        n_active_experts=n_active_experts,
        dist_to_drivable_area=dist_to_drivable_area,
    )


def _max_class_scores(
    expert_boxes: dict[str, list[DetectionBox]],
) -> dict[str, float]:
    """Return the max detection_score per class across all experts for one sample."""
    best: dict[str, float] = {}
    for boxes in expert_boxes.values():
        for b in boxes:
            cls = b.detection_name
            if cls not in best or b.detection_score > best[cls]:
                best[cls] = b.detection_score
    return best


def extract_features_for_sample(
    expert_boxes: dict[str, list[DetectionBox]],
    mask=None,
) -> dict[str, list[BoxFeatures]]:
    """Extract features for all boxes in one sample across all experts.

    Uses a vectorised BEV IoU matrix (NumPy) instead of per-pair Python
    calls — roughly 50-100x faster than the naive loop.

    Args:
        expert_boxes: model_name → list[DetectionBox] for one sample token.
        mask: Optional `nuscenes.utils.map_mask.MapMask` for this token's
            scene (see router_dataset.build_token_to_mask /
            infer_router's mask_by_token). When None, dist_to_drivable_area
            is DIST_TO_DRIVABLE_AREA_MISSING for every box (e.g. live
            inference outside nuScenes' mapped locations).

    Returns:
        model_name → list[BoxFeatures] in the same order as input boxes.
    """
    n_active = sum(1 for boxes in expert_boxes.values() if boxes)
    class_max = _max_class_scores(expert_boxes)

    # One batched map query across every box in the sample (all models),
    # instead of per-box calls -- mirrors bev_iou_matrix's vectorization.
    drivable_dist_by_id: dict[int, float] = {}
    if mask is not None:
        all_boxes = [b for boxes in expert_boxes.values() for b in boxes]
        if all_boxes:
            xs = np.array([b.translation[0] for b in all_boxes], dtype=np.float64)
            ys = np.array([b.translation[1] for b in all_boxes], dtype=np.float64)
            dists = _dist_to_drivable_area_batch(xs, ys, mask)
            drivable_dist_by_id = {id(b): float(d) for b, d in zip(all_boxes, dists)}

    result: dict[str, list[BoxFeatures]] = {}
    for model_name, boxes in expert_boxes.items():
        peer_boxes: list[DetectionBox] = []
        for other_name, other_boxes in expert_boxes.items():
            if other_name != model_name:
                peer_boxes.extend(other_boxes)

        n_other = sum(
            1 for name, bxs in expert_boxes.items()
            if name != model_name and bxs
        )

        iou_mat: np.ndarray | None = None
        if boxes and peer_boxes:
            iou_mat = bev_iou_matrix(boxes, peer_boxes)  # [N, M]

        result[model_name] = [
            extract_features(
                b,
                peer_boxes,
                n_active_experts=n_active,
                n_other_experts=n_other,
                max_class_score=class_max.get(b.detection_name, b.detection_score),
                peer_ious=iou_mat[i] if iou_mat is not None else None,
                dist_to_drivable_area=drivable_dist_by_id.get(
                    id(b), DIST_TO_DRIVABLE_AREA_MISSING
                ),
            )
            for i, b in enumerate(boxes)
        ]
    return result
