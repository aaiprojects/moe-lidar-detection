"""Official nuScenes detection evaluation wrapper.

Runs the nuScenes devkit evaluator on a submission JSON file and returns
a structured result dict with mAP, NDS, and per-class AP.

Requires: nuscenes-devkit  (pip install nuscenes-devkit)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def evaluate_submission(
    submission_path: Path,
    nuscenes_root: Path,
    version: str = "v1.0-trainval",
    split: str = "val",
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate a nuScenes submission JSON using the official devkit.

    Args:
        submission_path: Path to the nuScenes submission JSON (results + meta).
        nuscenes_root: Path to the nuScenes dataset root (contains v1.0-trainval/).
        version: nuScenes dataset version ('v1.0-trainval' or 'v1.0-mini').
        split: Evaluation split ('val' or 'test').
        verbose: Print devkit progress to stdout.

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
    if not nuscenes_root.exists():
        raise FileNotFoundError(f"nuScenes root not found: {nuscenes_root}")

    log.info("Loading nuScenes %s from %s", version, nuscenes_root)
    nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=verbose)

    eval_config = config_factory("detection_cvpr_2019")

    with tempfile.TemporaryDirectory() as output_dir:
        log.info("Running official nuScenes evaluation on %s", submission_path)
        evaluator = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=str(submission_path),
            eval_set=split,
            output_dir=output_dir,
            verbose=verbose,
        )
        metrics, metric_data_list = evaluator.evaluate()

    mean_ap = metrics.mean_ap
    nds = metrics.nd_score

    # mean_dist_aps is a dict: class_name → mean AP across distance thresholds
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
