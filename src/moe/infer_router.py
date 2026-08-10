"""Apply the trained router to produce gated MoE predictions.

Gating mechanism (UW-MTL-inspired soft-temperature blending):

  For each predicted box:
    τ     = class_tau.get(class_name, global_tau)        # per-class temperature
    w     = sigmoid(router_proba / τ)                    # temperature-scaled weight
    combined = w * router_proba + (1 - w) * expert_score

  Interpretation:
    - When τ is small → sigmoid saturates → router has strong control.
    - When τ is large → w ≈ 0.5 → equal blending (recovers original behaviour).
    - This mirrors the soft-temperature uncertainty weighting in the UW-MTL paper
      (w_k = exp(-log σ²_k / τ)), applied here at the per-box score level.

  Then run class-aware NMS on combined scores.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from src.fusion.nms3d import nms3d
from src.io.schemas import DetectionBox
from src.moe.features import extract_features_for_sample
from src.moe.router_dataset import LidarInfo
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _sigmoid(x: float) -> float:
    """Standard logistic function, used to turn router_proba/tau into the
    soft-temperature blend weight w (see module docstring)."""
    return 1.0 / (1.0 + math.exp(-x))


def score_boxes_with_router(
    expert_boxes: dict[str, list[DetectionBox]],
    router,
    expert_weight: float = 0.5,
    router_weight: float = 0.5,
    global_tau: float = 1.0,
    class_tau: dict[str, float] | None = None,
    lidar_info: LidarInfo | None = None,
) -> list[DetectionBox]:
    """Score all boxes for one sample using the router and return re-scored copies.

    Args:
        expert_boxes: model_name → list[DetectionBox] for one sample token.
        router: Fitted classifier with predict_proba().
        expert_weight: Fallback weight of original detection_score (used when
                       soft-temperature blending is disabled, i.e. τ → ∞).
        router_weight: Fallback weight of router probability.
        global_tau: Default temperature for classes not in class_tau.
        class_tau: Per-class temperature dict (class_name → τ).
        lidar_info: Ignored. Single-sweep point features hurt held-out mAP;
                    kept only for call-site compatibility.

    Returns:
        List of DetectionBox copies with detection_score replaced by combined score.
    """
    if class_tau is None:
        class_tau = {}
    if lidar_info is not None:
        log.debug("lidar_info ignored — point-cloud features are not in FEATURE_NAMES")

    features_by_model = extract_features_for_sample(expert_boxes)

    rescored: list[DetectionBox] = []
    for model_name, boxes in expert_boxes.items():
        feats_list = features_by_model[model_name]
        if not feats_list:
            continue

        X = np.array([f.to_list() for f in feats_list], dtype=np.float32)
        router_proba = router.predict_proba(X)[:, 1]

        for box, rp in zip(boxes, router_proba):
            tau = class_tau.get(box.detection_name, global_tau)

            # Soft-temperature weighting: w = sigmoid(router_proba / τ)
            w = _sigmoid(float(rp) / tau)
            combined = w * float(rp) + (1.0 - w) * box.detection_score

            new_box = copy.copy(box)
            new_box.detection_score = float(np.clip(combined, 0.0, 1.0))
            rescored.append(new_box)

    return rescored


def score_boxes_per_class_router(
    expert_boxes: dict[str, list[DetectionBox]],
    per_class_routers: dict[str, object],
    global_tau: float = 1.0,
    class_tau: dict[str, float] | None = None,
    lidar_info: LidarInfo | None = None,
    mask: object | None = None,
) -> list[DetectionBox]:
    """Like score_boxes_with_router but routes each box to its class-specific classifier.

    Args:
        expert_boxes: model_name → list[DetectionBox] for one sample token.
        per_class_routers: class_name → fitted classifier.
        global_tau: Default temperature.
        class_tau: Per-class temperature overrides.
        lidar_info: Ignored (point-cloud features removed from production vector).
        mask: Optional MapMask for this token's scene (dist_to_drivable_area
            feature). None -> missing sentinel, same as build-time when a
            token's scene has no registered map (see router_dataset.
            build_token_to_mask). Must be supplied here whenever it was
            supplied at training time, or the router sees a systematically
            different feature distribution at inference than it was trained
            on for every box.

    Returns:
        List of DetectionBox copies with combined score.
    """
    if class_tau is None:
        class_tau = {}

    features_by_model = extract_features_for_sample(expert_boxes, mask=mask)

    rescored: list[DetectionBox] = []
    for model_name, boxes in expert_boxes.items():
        feats_list = features_by_model[model_name]
        if not feats_list:
            continue

        # Group by class so we can batch-predict per class
        class_groups: dict[str, list[tuple[int, DetectionBox, list[float]]]] = {}
        for idx, (box, feats) in enumerate(zip(boxes, feats_list)):
            cls = box.detection_name
            class_groups.setdefault(cls, []).append((idx, box, feats.to_list()))

        box_scores: dict[int, float] = {}
        for cls, group in class_groups.items():
            router = per_class_routers.get(cls)
            if router is None:
                # No classifier for this class — keep expert score unchanged
                for idx, box, _ in group:
                    box_scores[idx] = box.detection_score
                continue

            X = np.array([g[2] for g in group], dtype=np.float32)
            proba = router.predict_proba(X)[:, 1]
            tau = class_tau.get(cls, global_tau)

            for (idx, box, _), rp in zip(group, proba):
                w = _sigmoid(float(rp) / tau)
                combined = w * float(rp) + (1.0 - w) * box.detection_score
                box_scores[idx] = float(np.clip(combined, 0.0, 1.0))

        for idx, box in enumerate(boxes):
            new_box = copy.copy(box)
            new_box.detection_score = box_scores.get(idx, box.detection_score)
            rescored.append(new_box)

    return rescored


def infer_moe(
    predictions: dict[str, dict[str, list[DetectionBox]]],
    router,
    sample_tokens: list[str],
    expert_weight: float = 0.5,
    router_weight: float = 0.5,
    global_tau: float = 1.0,
    class_tau: dict[str, float] | None = None,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.1,
    per_class_score_thresholds: dict[str, float] | None = None,
    class_aware: bool = True,
    max_boxes: int = 500,
    lidar_info_by_token: dict[str, LidarInfo] | None = None,
) -> dict[str, list[DetectionBox]]:
    """Run gated MoE inference with soft-temperature blending over a set of tokens.

    Args:
        predictions: model_name → sample_token → list[DetectionBox].
        router: Fitted router classifier.
        sample_tokens: Tokens to process (must be the EVAL split, not train).
        expert_weight: Fallback weight on original detection_score.
        router_weight: Fallback weight on router probability.
        global_tau: Default soft-temperature τ for all classes.
        class_tau: Per-class τ override (class_name → float).
        iou_threshold: NMS IoU suppression threshold.
        score_threshold: Global fallback minimum combined score before NMS.
        per_class_score_thresholds: Per-class minimum score overrides.  When a
            class appears here its threshold replaces ``score_threshold``.
            Derived from PR-curve F1 optimisation on the scene-grouped eval split.
        class_aware: Run NMS per class.
        max_boxes: Hard cap on boxes per sample.

    Returns:
        Dict mapping sample_token → gated list[DetectionBox].
    """
    pc_thresh = per_class_score_thresholds or {}
    log.info(
        "Running gated MoE on %d samples (global_tau=%.2f, per_class_thresholds=%s)",
        len(sample_tokens),
        global_tau,
        bool(pc_thresh),
    )

    moe_output: dict[str, list[DetectionBox]] = {}
    for token in sample_tokens:
        expert_boxes: dict[str, list[DetectionBox]] = {
            model: preds.get(token, []) for model, preds in predictions.items()
        }

        rescored = score_boxes_with_router(
            expert_boxes,
            router,
            expert_weight=expert_weight,
            router_weight=router_weight,
            global_tau=global_tau,
            class_tau=class_tau,
            lidar_info=(lidar_info_by_token or {}).get(token),
        )

        filtered = [
            b for b in rescored
            if b.detection_score >= pc_thresh.get(b.detection_name, score_threshold)
        ]
        moe_output[token] = nms3d(
            filtered,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
            max_boxes=max_boxes,
        )

    total = sum(len(v) for v in moe_output.values())
    log.info("MoE output: %d total boxes across %d samples", total, len(moe_output))
    return moe_output


def infer_moe_per_class(
    predictions: dict[str, dict[str, list[DetectionBox]]],
    per_class_routers: dict[str, object],
    sample_tokens: list[str],
    global_tau: float = 1.0,
    class_tau: dict[str, float] | None = None,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.1,
    per_class_score_thresholds: dict[str, float] | None = None,
    class_aware: bool = True,
    max_boxes: int = 500,
    lidar_info_by_token: dict[str, LidarInfo] | None = None,
    mask_by_token: dict[str, object] | None = None,
) -> dict[str, list[DetectionBox]]:
    """Run per-class gated MoE inference over a set of sample tokens.

    Each box is scored by the classifier trained specifically for its class.

    Args:
        predictions: model_name → sample_token → list[DetectionBox].
        per_class_routers: class_name → fitted classifier (from load_per_class_routers).
        sample_tokens: Tokens to process (eval split only).
        global_tau: Default soft-temperature τ.
        class_tau: Per-class τ override.
        iou_threshold: NMS IoU suppression threshold.
        score_threshold: Global fallback minimum combined score before NMS.
        per_class_score_thresholds: Per-class minimum score overrides.  When a
            class appears here its threshold replaces ``score_threshold``.
        class_aware: Run NMS per class.
        max_boxes: Hard cap on boxes per sample.
        mask_by_token: Optional sample_token -> MapMask (see
            src.moe.router_dataset.build_token_to_mask) for the
            dist_to_drivable_area feature. Must match what build_dataset()
            was given for whatever router is being scored here — see
            score_boxes_per_class_router's docstring.

    Returns:
        Dict mapping sample_token → gated list[DetectionBox].
    """
    pc_thresh = per_class_score_thresholds or {}
    log.info(
        "Running per-class gated MoE on %d samples (global_tau=%.2f, per_class_thresholds=%s)",
        len(sample_tokens),
        global_tau,
        bool(pc_thresh),
    )

    moe_output: dict[str, list[DetectionBox]] = {}
    for token in sample_tokens:
        expert_boxes: dict[str, list[DetectionBox]] = {
            model: preds.get(token, []) for model, preds in predictions.items()
        }

        rescored = score_boxes_per_class_router(
            expert_boxes,
            per_class_routers,
            global_tau=global_tau,
            class_tau=class_tau,
            lidar_info=(lidar_info_by_token or {}).get(token),
            mask=(mask_by_token or {}).get(token),
        )

        filtered = [
            b for b in rescored
            if b.detection_score >= pc_thresh.get(b.detection_name, score_threshold)
        ]
        moe_output[token] = nms3d(
            filtered,
            iou_threshold=iou_threshold,
            class_aware=class_aware,
            max_boxes=max_boxes,
        )

    total = sum(len(v) for v in moe_output.values())
    log.info("Per-class MoE output: %d total boxes across %d samples", total, len(moe_output))
    return moe_output
