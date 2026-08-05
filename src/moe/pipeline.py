"""End-to-end inference recipe for the final, adopted MoE configuration.

This module encapsulates the full pipeline validated in the project's
experiment log (mAP=0.6152, NDS=0.6694 on the 45 held-out test scenes):

  1. Run all five frozen expert detectors (not part of this module — their
     predictions are loaded from JSON, see src/io/load_predictions.py).
  2. For each predicted box, blend two per-class scorers into a single
     probability: combined = lambda * p_xgboost + (1 - lambda) * p_ft_transformer
     (see EnsembleRouter below). Nine of the ten classes use an uncalibrated
     XGBoost model; bicycle uses a sigmoid-calibrated one (see
     src/moe/xgboost_router.py for why).
  3. Blend that combined probability with the box's own expert score using
     a soft-temperature gate (src/moe/infer_router.py), filter by per-class
     score thresholds, and run class-aware NMS (src/fusion/nms3d.py).
  4. Apply temporal post-processing (src/fusion/tracker.py): an orphan-score
     penalty and single-frame interpolation, for every class except car and
     bicycle, where this stage was found to be net-neutral or harmful.

Every numeric constant below (lambda, thresholds, tau, orphan penalty,
interpolation discount, excluded classes) is loaded from
configs/moe_final.yaml, which is the single source of truth for the
adopted configuration. This module is deliberately free of any training
code -- see src/moe/xgboost_router.py and src/moe/nn_router.py for that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.fusion.tracker import (
    apply_track_rescoring,
    get_scene_ordered_tokens,
    interpolate_missed_frames,
    track_scene,
)
from src.io.schemas import DetectionBox
from src.moe.features import FEATURE_NAMES, NN_FEATURE_NAMES
from src.moe.infer_router import infer_moe_per_class
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Column indices of NN_FEATURE_NAMES within FEATURE_NAMES -- computed once,
# by name, so this stays correct regardless of where dist_to_drivable_area
# (the one feature the FT-Transformer excludes) sits in FEATURE_NAMES.
_NN_FEATURE_IDX = [FEATURE_NAMES.index(f) for f in NN_FEATURE_NAMES]


class EnsembleRouter:
    """Per-class convex blend of an XGBoost router and an FT-Transformer router.

    combined_score = lambda * p_xgboost + (1 - lambda) * p_ft_transformer

    Exposes predict_proba(X) so it's a drop-in classifier for
    src.moe.infer_router.infer_moe_per_class, which expects one such
    object per class in its ``per_class_routers`` dict. X's columns must
    be in FEATURE_NAMES order.
    """

    def __init__(self, xgb_router, nn_router, lam: float) -> None:
        self.xgb_router = xgb_router
        self.nn_router = nn_router
        self.lam = lam

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb_router.predict_proba(X)[:, 1]
        p_nn = self.nn_router.predict_proba(X[:, _NN_FEATURE_IDX])[:, 1]
        p = self.lam * p_xgb + (1 - self.lam) * p_nn
        return np.stack([1 - p, p], axis=1)


def load_final_config(config_path: str | Path) -> dict:
    """Load configs/moe_final.yaml into a plain dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_ensemble_routers(
    xgb_routers: dict[str, object],
    nn_routers: dict[str, object],
    lambdas: dict[str, float],
) -> dict[str, EnsembleRouter]:
    """Wrap per-class XGBoost + FT-Transformer routers into EnsembleRouters."""
    return {
        cls: EnsembleRouter(xgb_routers[cls], nn_routers[cls], lambdas[cls])
        for cls in lambdas
        if xgb_routers.get(cls) is not None and nn_routers.get(cls) is not None
    }


def apply_temporal_refinement(
    moe_result: dict[str, list[DetectionBox]],
    nusc,
    sample_tokens: list[str],
    orphan_penalty: float,
    interpolation_discount: float,
    excluded_classes: set[str],
    dedup_iou_threshold: float = 0.3,
) -> dict[str, list[DetectionBox]]:
    """Apply orphan-penalty rescoring + single-frame interpolation per scene.

    Detections belonging to a class in ``excluded_classes`` (car, bicycle in
    the adopted config) are left untouched -- their original, NMS'd boxes
    are kept as-is, since this stage was found to be net-neutral or harmful
    for those two classes specifically.

    Args:
        moe_result: sample_token -> list[DetectionBox], the output of
            infer_moe_per_class (already NMS'd).
        nusc: A NuScenes devkit instance, used only to order tokens by scene
            and timestamp (see src.fusion.tracker.get_scene_ordered_tokens).
        sample_tokens: The tokens being processed (must match moe_result).
        orphan_penalty: Score multiplier for detections with no temporal
            corroboration (e.g. 0.7).
        interpolation_discount: Score multiplier applied to the weaker
            neighbor's score when synthesizing an interpolated box (e.g. 0.9).
        excluded_classes: Classes to exclude from both corrections.
        dedup_iou_threshold: Same "same object" BEV IoU threshold NMS uses
            (ensemble.iou_threshold) -- drops an interpolation candidate if
            it would duplicate a box a different track already placed in
            that frame (can happen when one real object's trajectory gets
            fragmented into two tracks; see interpolate_missed_frames).

    Returns:
        sample_token -> list[DetectionBox], the final pipeline output.
    """
    scenes = get_scene_ordered_tokens(nusc, set(sample_tokens))

    all_frames = []
    all_track_info = {}
    orig_boxes_by_token = {}
    for scene_tokens in scenes:
        frames = [(t, moe_result.get(t, [])) for t in scene_tokens]
        track_info = track_scene(frames)
        all_frames.extend(frames)
        all_track_info.update(track_info)
        for token, boxes in frames:
            orig_boxes_by_token[token] = boxes

    rescored = apply_track_rescoring(all_frames, all_track_info, orphan_penalty=orphan_penalty)
    interpolated = interpolate_missed_frames(
        all_frames, all_track_info, score_discount=interpolation_discount,
        dedup_iou_threshold=dedup_iou_threshold,
    )

    combined: dict[str, list[DetectionBox]] = {}
    for token, orig_boxes in orig_boxes_by_token.items():
        n_orig = len(orig_boxes)
        new_interp_boxes = interpolated[token][n_orig:]

        kept = []
        for orig_box, rescored_box in zip(orig_boxes, rescored[token]):
            kept.append(orig_box if orig_box.detection_name in excluded_classes else rescored_box)
        for interp_box in new_interp_boxes:
            if interp_box.detection_name not in excluded_classes:
                kept.append(interp_box)
        combined[token] = kept

    return combined


def run_final_pipeline(
    predictions: dict[str, dict[str, list[DetectionBox]]],
    ensemble_routers: dict[str, EnsembleRouter],
    sample_tokens: list[str],
    cfg: dict,
    nusc,
    mask_by_token: dict[str, object] | None = None,
) -> dict[str, list[DetectionBox]]:
    """Run the complete adopted MoE pipeline: fusion + NMS + temporal refinement.

    Args:
        predictions: model_name -> sample_token -> list[DetectionBox], the
            five experts' raw predictions (see src/io/load_predictions.py).
        ensemble_routers: class_name -> EnsembleRouter (see
            build_ensemble_routers).
        sample_tokens: Tokens to run inference on.
        cfg: Parsed configs/moe_final.yaml (see load_final_config).
        nusc: A NuScenes devkit instance (needed for scene/timestamp
            ordering during temporal refinement).
        mask_by_token: Optional sample_token -> MapMask for the
            dist_to_drivable_area feature (see
            src.moe.router_dataset.build_token_to_mask). Required if
            training used map masks, since XGBoost sees a fixed-length
            feature vector.

    Returns:
        sample_token -> list[DetectionBox], ready to pass to
        src.io.save_predictions.save_nuscenes_predictions or
        src.evaluation.evaluate_nuscenes.
    """
    ensemble = cfg["ensemble"]
    router = cfg["router"]
    tracking = cfg["tracking"]

    moe_result = infer_moe_per_class(
        predictions=predictions,
        per_class_routers=ensemble_routers,
        sample_tokens=sample_tokens,
        global_tau=router["global_tau"],
        class_tau=router["class_tau"],
        iou_threshold=ensemble["iou_threshold"],
        score_threshold=ensemble["score_threshold"],
        per_class_score_thresholds=ensemble["per_class_score_thresholds"],
        class_aware=ensemble["class_aware"],
        max_boxes=ensemble["max_boxes_per_sample"],
        lidar_info_by_token=None,
        mask_by_token=mask_by_token,
    )
    log.info("Fusion + NMS done for %d samples", len(sample_tokens))

    final = apply_temporal_refinement(
        moe_result,
        nusc,
        sample_tokens,
        orphan_penalty=tracking["orphan_penalty"],
        interpolation_discount=tracking["interpolation_discount"],
        excluded_classes=set(tracking["excluded_classes"]),
        dedup_iou_threshold=ensemble["iou_threshold"],
    )
    log.info("Temporal refinement done")
    return final
