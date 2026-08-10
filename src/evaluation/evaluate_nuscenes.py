"""Official nuScenes detection evaluation wrapper.

Runs the nuScenes devkit evaluator on a submission JSON file and returns
a structured result dict with mAP, NDS, and per-class AP.

Requires: nuscenes-devkit  (pip install nuscenes-devkit)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _metrics_to_result(metrics) -> dict[str, Any]:
    """Convert a nuScenes DetectionMetrics object to this module's result dict."""
    mean_ap = metrics.mean_ap
    nds = metrics.nd_score
    per_class_ap: dict[str, float] = dict(metrics.mean_dist_aps)

    summary = {
        "mAP": mean_ap,
        "NDS": nds,
        "mATE": metrics.tp_errors.get("trans_err", float("nan")),
        "mASE": metrics.tp_errors.get("scale_err", float("nan")),
        "mAOE": metrics.tp_errors.get("orient_err", float("nan")),
        "mAVE": metrics.tp_errors.get("vel_err", float("nan")),
        "mAAE": metrics.tp_errors.get("attr_err", float("nan")),
    }
    summary.update({f"AP_{cls}": ap for cls, ap in per_class_ap.items()})

    log.info(
        "Evaluation complete | mAP=%.4f | NDS=%.4f",
        mean_ap,
        nds,
    )

    return {
        "map": mean_ap,
        "nds": nds,
        "metrics": metrics,
        "per_class_ap": per_class_ap,
        "summary": summary,
    }


def _evaluate_on_sample_tokens(
    nusc,
    submission_path: Path,
    eval_config,
    sample_tokens: list[str],
    verbose: bool,
):
    """Run official nuScenes detection metrics on an explicit token subset."""
    from nuscenes.eval.common.loaders import (
        add_center_dist,
        filter_eval_boxes,
        load_gt_of_sample_tokens,
        load_prediction_of_sample_tokens,
    )
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import (
        DetectionBox,
        DetectionMetricDataList,
        DetectionMetrics,
    )

    pred_boxes, _meta = load_prediction_of_sample_tokens(
        str(submission_path),
        eval_config.max_boxes_per_sample,
        DetectionBox,
        sample_tokens=sample_tokens,
        verbose=verbose,
    )
    gt_boxes = load_gt_of_sample_tokens(
        nusc, sample_tokens, DetectionBox, verbose=verbose
    )

    pred_tokens = set(pred_boxes.sample_tokens)
    gt_tokens = set(gt_boxes.sample_tokens)
    if pred_tokens != gt_tokens:
        missing = sorted(gt_tokens - pred_tokens)
        extra = sorted(pred_tokens - gt_tokens)
        raise AssertionError(
            "Samples in split doesn't match samples in predictions. "
            f"Missing {len(missing)} GT tokens in submission; "
            f"submission has {len(extra)} unexpected tokens."
        )

    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    pred_boxes = filter_eval_boxes(
        nusc, pred_boxes, eval_config.class_range, verbose=verbose
    )
    gt_boxes = filter_eval_boxes(
        nusc, gt_boxes, eval_config.class_range, verbose=verbose
    )

    start_time = time.time()
    metric_data_list = DetectionMetricDataList()
    for class_name in eval_config.class_names:
        for dist_th in eval_config.dist_ths:
            md = accumulate(
                gt_boxes,
                pred_boxes,
                class_name,
                eval_config.dist_fcn_callable,
                dist_th,
            )
            metric_data_list.set(class_name, dist_th, md)

    metrics = DetectionMetrics(eval_config)
    for class_name in eval_config.class_names:
        for dist_th in eval_config.dist_ths:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, eval_config.min_recall, eval_config.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)

        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, eval_config.dist_th_tp)]
            # Matches the official devkit's own exclusions: traffic cones
            # have no meaningful orientation/velocity/attribute labels in
            # nuScenes (they're static and rotationally symmetric), and
            # barriers have no attribute or velocity labels either. These
            # error types are undefined for these classes, not merely hard
            # to estimate, so they're reported as NaN rather than computed.
            if class_name in ["traffic_cone"] and metric_name in [
                "attr_err",
                "vel_err",
                "orient_err",
            ]:
                tp = float("nan")
            elif class_name in ["barrier"] and metric_name in ["attr_err", "vel_err"]:
                tp = float("nan")
            else:
                tp = calc_tp(metric_data, eval_config.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    metrics.add_runtime(time.time() - start_time)
    return metrics, metric_data_list


def evaluate_submission(
    submission_path: Path,
    nuscenes_root: Path | None = None,
    version: str = "v1.0-trainval",
    split: str = "val",
    sample_tokens: list[str] | None = None,
    verbose: bool = False,
    nusc=None,
) -> dict[str, Any]:
    """Evaluate a nuScenes submission JSON using the official devkit.

    Args:
        submission_path: Path to the nuScenes submission JSON (results + meta).
        nuscenes_root: Path to the nuScenes dataset root (contains v1.0-trainval/).
            Optional only when ``nusc`` is supplied.
        version: nuScenes dataset version ('v1.0-trainval' or 'v1.0-mini').
        split: Evaluation split ('val' or 'test'). Ignored when ``sample_tokens``
            is provided — use that parameter to score a custom token subset
            (e.g. the 804-scene calibration split in ``eval.csv``).
        sample_tokens: Optional explicit sample tokens to evaluate. When set,
            GT and predictions are loaded only for these tokens instead of the
            full predefined ``split``.
        verbose: Print devkit progress to stdout.
        nusc: Optional already-loaded NuScenes instance to reuse. The trainval
            tables cost ~8 GB and tens of seconds to parse, so callers scoring
            several submissions in a row (e.g. one per expert) should load once
            and pass it here. ``version``/``nuscenes_root`` are then unused.

    Returns:
        Dict with keys:
            'map'         : mean AP across all classes
            'nds'         : nuScenes Detection Score
            'metrics'     : full NuScenesMetrics object
            'per_class_ap': dict mapping class name → AP
            'summary'     : flat dict of all scalar metrics
    """
    try:
        from nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval
    except ImportError as e:
        raise ImportError(
            "nuscenes-devkit is required for evaluation. "
            "Install with: pip install nuscenes-devkit"
        ) from e

    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    if nusc is None:
        if nuscenes_root is None:
            raise ValueError("Pass nuscenes_root, or an already-loaded nusc to reuse.")
        if not nuscenes_root.exists():
            raise FileNotFoundError(f"nuScenes root not found: {nuscenes_root}")
        log.info("Loading nuScenes %s from %s", version, nuscenes_root)
        nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=verbose)
    else:
        log.info("Reusing caller-supplied NuScenes instance (%s)", nusc.version)

    eval_config = config_factory("detection_cvpr_2019")

    log.info("Running official nuScenes evaluation on %s", submission_path)
    if sample_tokens is not None:
        log.info(
            "Evaluating custom subset of %d sample tokens (not full '%s' split)",
            len(sample_tokens),
            split,
        )
        metrics, _metric_data_list = _evaluate_on_sample_tokens(
            nusc,
            submission_path,
            eval_config,
            sample_tokens=sample_tokens,
            verbose=verbose,
        )
    else:
        with tempfile.TemporaryDirectory() as output_dir:
            evaluator = NuScenesEval(
                nusc,
                config=eval_config,
                result_path=str(submission_path),
                eval_set=split,
                output_dir=output_dir,
                verbose=verbose,
            )
            metrics, _metric_data_list = evaluator.evaluate()

    return _metrics_to_result(metrics)


def print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    """Print a side-by-side comparison table for multiple model results.

    Args:
        results: Dict mapping model_name → result dict from evaluate_submission().
    """
    scalar_keys = ["mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE"]
    col_w = 10
    name_w = 22

    header = f"{'Model':<{name_w}}" + "".join(f"{k:>{col_w}}" for k in scalar_keys)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, result in results.items():
        s = result["summary"]
        row = f"{model_name:<{name_w}}"
        for k in scalar_keys:
            val = s.get(k, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * len(header) + "\n")
