"""Build the labeled router training dataset.

For each predicted box in the training split:
  - Match it to a ground-truth box (axis-aligned BEV IoU >= 0.5, same class,
    greedy -- see src/fusion/bev_iou.py's HISTORY for the §40/§41/§44
    rotated-IoU detour and revert)
  - Label: 1 if matched, 0 if not
  - Extract router features

Output: a pandas DataFrame with columns = FEATURE_NAMES + ['class_id', 'label', 'sample_token', 'model_name']

GT boxes live in nuscenes_infos_val.pkl under 'instances' in ego frame.
They are converted to global frame using the per-sample ego2global matrix.

bbox_3d format: [x, y, z, width, length, height, yaw]  (ego frame)
ego2global: 4×4 row-major list  [[r00,r01,r02,t0],[r10,r11,r12,t1],...]
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusion.bev_iou import raw_iou_matrix
from src.io.schemas import DetectionBox
from src.moe.features import FEATURE_NAMES, extract_features_for_sample
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

GT_IOU_THRESHOLD: float = 0.5

# nuScenes' own TP-metric matching distance (DIST_TH_TP in the official
# devkit). Used as an alternative label-matching criterion for classes whose
# footprint is small/narrow relative to a BEV-IoU>=0.5 bar (see
# docs/EXPERIMENT_LOG.md — narrow-class label mismatch investigation).
GT_CENTER_DIST_THRESHOLD_M: float = 2.0

# Label → class name mapping from nuscenes_infos_val.pkl metainfo
_LABEL_TO_CLASS: dict[int, str] = {
    0: "car",
    1: "truck",
    2: "trailer",
    3: "bus",
    4: "construction_vehicle",
    5: "bicycle",
    6: "motorcycle",
    7: "pedestrian",
    8: "traffic_cone",
    9: "barrier",
}


@dataclass
class LidarInfo:
    """Per-token metadata needed to load and transform a LiDAR sweep."""
    lidar_path: Path                   # absolute path to .pcd.bin file
    lidar2ego: list[list[float]]       # 4×4 LiDAR-sensor → ego transform
    ego2global: list[list[float]]      # 4×4 ego → global transform
    num_pts_feats: int = 5             # floats per point in .pcd.bin


@dataclass
class GtBox:
    """Ground-truth box in global frame."""
    translation: list[float]   # [x, y, z] global
    size: list[float]          # [width, length, height]
    detection_name: str
    yaw: float = 0.0            # global-frame yaw, radians (§40: rotated IoU)


def _transform_point(mat4x4: list[list[float]], x: float, y: float, z: float) -> tuple[float, float, float]:
    """Apply a 4×4 row-major homogeneous transform to a 3-D point."""
    R = mat4x4
    ox = R[0][0]*x + R[0][1]*y + R[0][2]*z + R[0][3]
    oy = R[1][0]*x + R[1][1]*y + R[1][2]*z + R[1][3]
    oz = R[2][0]*x + R[2][1]*y + R[2][2]*z + R[2][3]
    return ox, oy, oz


def _compose_yaw(
    local_yaw: float,
    lidar2ego: list[list[float]],
    ego2global: list[list[float]],
) -> float:
    """Compose a LiDAR-frame yaw through two 4x4 transforms' rotation
    submatrices into a global-frame yaw (§40). Mirrors _transform_point's
    translation-only chain, but for orientation: compose the local
    z-rotation with each transform's own rotation submatrix and extract
    the resulting yaw. Valid for ground-plane boxes (negligible roll/pitch
    in lidar2ego/ego2global), same assumption _quat_to_yaw already makes
    for predictions.
    """
    c, s = math.cos(local_yaw), math.sin(local_yaw)
    r_local = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_l2e = np.array(lidar2ego)[:3, :3]
    r_e2g = np.array(ego2global)[:3, :3]
    r_global = r_e2g @ r_l2e @ r_local
    return float(math.atan2(r_global[1, 0], r_global[0, 0]))


def load_gt_by_token(pkl_path: Path) -> dict[str, list[GtBox]]:
    """Load ground-truth boxes from nuscenes_infos_val.pkl.

    GT bbox_3d is stored in **LiDAR sensor frame**.  We apply two transforms:
      1. lidar2ego  (sensor → vehicle body)
      2. ego2global (vehicle body → world/map)

    Returns:
        Dict mapping sample_token → list[GtBox] in global frame.
    """
    log.info("Loading GT from %s", pkl_path)
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    gt_by_token: dict[str, list[GtBox]] = {}
    for info in data["data_list"]:
        token: str = info["token"]
        ego2global: list[list[float]] = info["ego2global"]        # 4×4
        lidar2ego: list[list[float]] = info["lidar_points"]["lidar2ego"]  # 4×4

        gt_boxes: list[GtBox] = []
        for inst in info.get("instances", []):
            if not inst.get("bbox_3d_isvalid", True):
                continue
            label: int = inst["bbox_label_3d"]
            class_name = _LABEL_TO_CLASS.get(label)
            if class_name is None:
                continue

            # mmdet3d NuScenes pkl stores bbox_3d as [x, y, z, l, w, h, yaw]
            # (length before width), but nuScenes submission uses [width, length, height].
            # Swap here so GtBox.size follows the same [width, length, height] convention
            # as DetectionBox.size, making _gt_iou directly comparable.
            bbox = inst["bbox_3d"]  # [x, y, z, length, width, height, yaw] in LiDAR frame
            lx, ly, lz = bbox[0], bbox[1], bbox[2]
            box_length, w, h = bbox[3], bbox[4], bbox[5]
            local_yaw = bbox[6]

            # LiDAR → ego
            ex, ey, ez = _transform_point(lidar2ego, lx, ly, lz)
            # ego → global
            gx, gy, gz = _transform_point(ego2global, ex, ey, ez)
            gyaw = _compose_yaw(local_yaw, lidar2ego, ego2global)

            gt_boxes.append(GtBox(
                translation=[gx, gy, gz],
                size=[w, box_length, h],   # [width, length, height] — matches DetectionBox
                detection_name=class_name,
                yaw=gyaw,
            ))

        gt_by_token[token] = gt_boxes

    log.info("Loaded GT for %d samples", len(gt_by_token))
    return gt_by_token


def build_token_to_mask(nusc, tokens) -> dict[str, object]:
    """sample_token -> nuscenes.utils.map_mask.MapMask for that sample's
    scene, for the dist_to_drivable_area feature (src/moe/features.py).

    Uses only the base nuScenes tables (nusc.map[i]['mask'], already
    loaded as a MapMask from the rasterized semantic-prior PNG) -- no
    map-expansion vector pack required. Caller must have already loaded
    ``nusc = NuScenes(...)`` (mirrors src/fusion/tracker.py's
    get_scene_ordered_tokens, which takes the same pre-loaded ``nusc``).

    Args:
        nusc: A loaded NuScenes instance.
        tokens: Sample tokens to look up (only these are resolved).

    Returns:
        sample_token -> MapMask. Tokens whose scene has no registered map
        are simply absent (callers should treat missing lookups as "no
        mask available", same as passing mask=None to feature extraction).
    """
    log_to_mask: dict[str, object] = {}
    for m in nusc.map:
        for lt in m["log_tokens"]:
            log_to_mask[lt] = m["mask"]

    scene_to_mask: dict[str, object] = {}
    for scene in nusc.scene:
        mask = log_to_mask.get(scene["log_token"])
        if mask is not None:
            scene_to_mask[scene["token"]] = mask

    result: dict[str, object] = {}
    for token in tokens:
        sample = nusc.get("sample", token)
        mask = scene_to_mask.get(sample["scene_token"])
        if mask is not None:
            result[token] = mask
    return result


def warm_map_masks(mask_by_token: dict[str, object], dilation_levels_m: tuple[float, ...]) -> None:
    """Force-compute and cache each distinct MapMask's base mask and dilated
    variants ONCE in the calling process, before handing mask_by_token to a
    multiprocessing.Pool.

    Each dilated mask is a full-resolution rasterized array (~300MB) built
    via a full-image cv2.distanceTransform; the underlying MapMask.mask()
    is memoized (LRUCache), but that memoization is per-process. If workers
    are fork()ed *before* this is called, each worker lazily (and
    independently) computes and holds its own copy -- 8 workers x 4 maps x
    3 dilation levels of ~300MB each blew past 100GB and crashed the
    machine before this fix. Calling this first means fork()'s
    copy-on-write actually applies: workers inherit references to the same
    already-computed arrays and only ever read them (is_on_mask never
    mutates), so the ~10GB steady-state cost is paid once, not once per
    worker. gc.collect() between maps bounds the transient
    cv2.distanceTransform peak (~12GB per map) instead of letting multiple
    maps' peaks stack within one process's lifetime.
    """
    import gc

    seen: set[int] = set()
    for mask in mask_by_token.values():
        if id(mask) in seen:
            continue
        seen.add(id(mask))
        mask.mask(0.0)
        for d in dilation_levels_m:
            mask.mask(d)
        gc.collect()


def load_lidar_info_by_token(
    pkl_path: Path,
    nuscenes_dir: Path,
) -> dict[str, LidarInfo]:
    """Load per-token LiDAR sweep metadata from nuscenes_infos_val.pkl.

    Args:
        pkl_path:     Path to nuscenes_infos_val.pkl.
        nuscenes_dir: Root nuScenes directory (contains ``samples/LIDAR_TOP/``).

    Returns:
        Dict mapping sample_token → LidarInfo.
    """
    lidar_top = nuscenes_dir / "samples" / "LIDAR_TOP"
    log.info("Loading LiDAR info from %s", pkl_path)
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    info_by_token: dict[str, LidarInfo] = {}
    for info in data["data_list"]:
        token: str = info["token"]
        lp = info["lidar_points"]
        lidar_path = lidar_top / Path(lp["lidar_path"]).name
        info_by_token[token] = LidarInfo(
            lidar_path=lidar_path,
            lidar2ego=lp["lidar2ego"],
            ego2global=info["ego2global"],
            num_pts_feats=int(lp.get("num_pts_feats", 5)),
        )

    log.info("Loaded LiDAR info for %d samples", len(info_by_token))
    return info_by_token


def _gt_iou(pred: DetectionBox, gt: GtBox) -> float:
    """Axis-aligned BEV IoU between a predicted box and a GT box.

    §40 tried a rotated (yaw-aware) alternative here; §41 confirmed its mAP
    regression was not an NMS-suppression artifact. Reverted back to
    axis-aligned as the project's validated production convention -- see
    docs/EXPERIMENT_LOG.md §38-41 and src/fusion/bev_iou.py's HISTORY.
    Same convention as bev_iou.bev_iou (kept as a separate copy here to
    avoid a fusion<->moe import cycle).
    """
    px, py = pred.translation[0], pred.translation[1]
    gx, gy = gt.translation[0], gt.translation[1]
    pw, pl = pred.size[0], pred.size[1]
    gw, gl = gt.size[0], gt.size[1]

    p_xmin, p_xmax = px - pl / 2.0, px + pl / 2.0
    p_ymin, p_ymax = py - pw / 2.0, py + pw / 2.0
    g_xmin, g_xmax = gx - gl / 2.0, gx + gl / 2.0
    g_ymin, g_ymax = gy - gw / 2.0, gy + gw / 2.0

    ix = max(0.0, min(p_xmax, g_xmax) - max(p_xmin, g_xmin))
    iy = max(0.0, min(p_ymax, g_ymax) - max(p_ymin, g_ymin))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0

    union = (pw * pl) + (gw * gl) - inter
    return inter / union if union > 0.0 else 0.0


def _center_dist(pred: DetectionBox, gt: GtBox) -> float:
    """2-D (BEV) center distance between a predicted box and a GT box."""
    px, py = pred.translation[0], pred.translation[1]
    gx, gy = gt.translation[0], gt.translation[1]
    return ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5


def _match_predictions_to_gt(
    pred_boxes: list[DetectionBox],
    gt_boxes: list[GtBox],
    iou_threshold: float = GT_IOU_THRESHOLD,
    metric: str = "iou",
    dist_threshold_m: float = GT_CENTER_DIST_THRESHOLD_M,
) -> list[int]:
    """Greedy matching of predictions to GT (score-descending).

    Args:
        metric: "iou" (default, BEV IoU >= iou_threshold, best-IoU-wins) or
            "center_distance" (nuScenes-style: center distance <= dist_threshold_m,
            nearest-GT-wins). The IoU criterion is a strict geometric-overlap bar
            that penalises narrow/small objects (motorcycle, bicycle, trailer)
            much more harshly than nuScenes' own center-distance matching does —
            see docs/EXPERIMENT_LOG.md for the investigation that motivated this.

    Returns:
        List of labels (1=TP, 0=FP) in the same order as pred_boxes.
    """
    labels = [0] * len(pred_boxes)
    if not gt_boxes:
        return labels

    sorted_indices = sorted(
        range(len(pred_boxes)),
        key=lambda i: pred_boxes[i].detection_score,
        reverse=True,
    )
    matched_gt: set[int] = set()

    # Same-class compatibility mask, computed once (cheap — no geometry).
    pred_classes = [p.detection_name for p in pred_boxes]
    gt_classes = [g.detection_name for g in gt_boxes]

    if metric == "iou":
        # §40 perf fix: precompute the full pairwise IoU matrix ONCE
        # (cached-polygon vectorized path in bev_iou.raw_iou_matrix)
        # instead of calling _gt_iou per (pred, gt) pair inside the greedy
        # loop below, which rebuilt shapely polygons from scratch on every
        # call and made a full dataset build ~1hr instead of ~15s.
        pred_arr = np.array(
            [[p.translation[0], p.translation[1], p.size[0], p.size[1], p.yaw] for p in pred_boxes],
            dtype=np.float64,
        )
        gt_arr = np.array(
            [[g.translation[0], g.translation[1], g.size[0], g.size[1], g.yaw] for g in gt_boxes],
            dtype=np.float64,
        )
        iou_mat = raw_iou_matrix(pred_arr, gt_arr)  # [N_pred, N_gt]

    for pred_idx in sorted_indices:
        pred = pred_boxes[pred_idx]
        best_gt_idx = -1

        if metric == "center_distance":
            best_dist = dist_threshold_m
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if gt_classes[gt_idx] != pred_classes[pred_idx]:
                    continue
                dist = _center_dist(pred, gt)
                if dist <= best_dist:
                    best_dist = dist
                    best_gt_idx = gt_idx
        else:
            best_iou = iou_threshold
            row = iou_mat[pred_idx]
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                if gt_classes[gt_idx] != pred_classes[pred_idx]:
                    continue
                iou = row[gt_idx]
                if iou >= best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            labels[pred_idx] = 1
            matched_gt.add(best_gt_idx)

    return labels


# Module-level, set by build_dataset() BEFORE the worker Pool is created so
# forked children inherit it via copy-on-write -- avoids re-pickling
# MapMask (wraps a decoded raster array) through the task queue on every
# single token, since only 4 distinct masks exist for the whole dataset.
_MASK_BY_TOKEN: dict[str, object] = {}


def _process_token(args: tuple) -> list[dict]:
    """Process a single sample token — designed for multiprocessing.Pool.

    Args tuple: (token, expert_boxes_for_token, gt_boxes)
    """
    token, expert_boxes_for_token, gt_boxes = args

    mask = _MASK_BY_TOKEN.get(token)
    features_by_model = extract_features_for_sample(expert_boxes_for_token, mask=mask)
    rows: list[dict] = []

    for model_name, boxes in expert_boxes_for_token.items():
        labels = _match_predictions_to_gt(boxes, gt_boxes)
        feats_list = features_by_model[model_name]
        for box, label, feats in zip(boxes, labels, feats_list):
            row = dict(zip(FEATURE_NAMES, feats.to_list()))
            row["class_id"] = feats.class_id
            row["label"] = label
            row["sample_token"] = token
            row["model_name"] = model_name
            rows.append(row)

    return rows


def build_dataset(
    predictions: dict[str, dict[str, list[DetectionBox]]],
    gt_by_token: dict[str, list[GtBox]],
    sample_tokens: list[str],
    lidar_info_by_token: dict[str, LidarInfo] | None = None,
    mask_by_token: dict[str, object] | None = None,
    n_workers: int = 8,
) -> pd.DataFrame:
    """Build a labeled feature DataFrame for a given set of sample tokens.

    Args:
        predictions:          model_name → sample_token → list[DetectionBox].
        gt_by_token:          sample_token → list[GtBox] (global frame).
        sample_tokens:        Which tokens to include.
        lidar_info_by_token:  Ignored. Point-cloud features are not in FEATURE_NAMES
                              (single-sweep occupancy hurt held-out mAP).
        mask_by_token:        Optional sample_token -> MapMask (see
                              build_token_to_mask) for dist_to_drivable_area.
                              Tokens absent from this dict get the missing
                              sentinel for that one feature.
        n_workers:            Number of parallel worker processes (default 8).

    Returns:
        DataFrame with columns FEATURE_NAMES + ['class_id','label','sample_token','model_name'].
    """
    import multiprocessing as mp
    from tqdm import tqdm

    global _MASK_BY_TOKEN
    _MASK_BY_TOKEN = mask_by_token or {}

    if lidar_info_by_token is not None:
        log.warning(
            "lidar_info_by_token ignored — n_points_inside/point_density are not "
            "in the production FEATURE_NAMES (see Experiment 5b)."
        )

    task_args = []
    missing_gt = 0
    for token in sample_tokens:
        if token not in gt_by_token:
            missing_gt += 1
            continue
        expert_boxes_for_token: dict[str, list[DetectionBox]] = {
            model: preds.get(token, [])
            for model, preds in predictions.items()
        }
        task_args.append((token, expert_boxes_for_token, gt_by_token[token]))

    if missing_gt > 0:
        log.warning("Skipped %d tokens with no GT in pkl", missing_gt)

    all_rows: list[dict] = []
    effective_workers = min(n_workers, mp.cpu_count(), len(task_args))
    log.info("Processing %d tokens with %d workers...", len(task_args), effective_workers)

    if effective_workers <= 1:
        for args in tqdm(task_args, desc="Building dataset", unit="token"):
            all_rows.extend(_process_token(args))
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=effective_workers) as pool:
            for rows in tqdm(
                pool.imap_unordered(_process_token, task_args, chunksize=20),
                total=len(task_args),
                desc="Building dataset",
                unit="token",
            ):
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        # pool.imap_unordered's completion order is nondeterministic across
        # rebuilds even from byte-identical inputs, which silently made every
        # downstream CalibratedClassifierCV fold assignment (and therefore
        # calibration curve) nondeterministic too -- see EXPERIMENT_LOG.md
        # §30. Sort to a stable, content-derived key so the CSV is
        # byte-reproducible regardless of worker scheduling.
        df = df.sort_values(["sample_token", "model_name"], kind="stable").reset_index(drop=True)
        log.info(
            "Dataset: %d rows, %d positives (%.1f%%), %d tokens, %d experts",
            len(df),
            df["label"].sum(),
            100.0 * df["label"].mean(),
            df["sample_token"].nunique(),
            df["model_name"].nunique(),
        )
    return df


def make_token_split(
    all_tokens: list[str],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split sample tokens into train and eval sets deterministically.

    Args:
        all_tokens: All available sample tokens (sorted for reproducibility).
        train_ratio: Fraction for router training.
        seed: Random seed (from moe.yaml).

    Returns:
        (train_tokens, eval_tokens)
    """
    import random

    rng = random.Random(seed)
    tokens = sorted(all_tokens)
    rng.shuffle(tokens)
    split = int(len(tokens) * train_ratio)
    return tokens[:split], tokens[split:]
