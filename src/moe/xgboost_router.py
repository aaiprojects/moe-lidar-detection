"""XGBoost per-class router models for the MoE LiDAR detection pipeline.

Trains one gradient-boosted tree classifier per nuScenes detection class,
using the same 18-feature vector as the FT-Transformer router (see
src/moe/features.py). XGBoost's ``scale_pos_weight`` corrects for the heavy
class imbalance in the training data (e.g. ~2% positive rate for bicycle),
but this reweighting distorts the model's raw probability scale, which
matters for this pipeline because downstream scores are blended, thresholded,
and rank-ordered by nuScenes' AP metric.

For most classes this distortion doesn't change the final ranking enough to
matter. For bicycle specifically -- the class with the most extreme
imbalance -- an additional sigmoid (Platt-scaling) calibration step is
applied via ``CalibratedClassifierCV``, using 3-fold stratified
cross-validation followed by a refit on the full training partition. This
was found empirically: calibrating every class made 8/9 non-bicycle classes
regress slightly, while calibrating bicycle alone gained +1.4pp AP with no
effect on any other class (nuScenes' class-aware NMS makes each class's
scoring fully independent of the others).

Overfitting control. A fixed 200-round, depth-6, no-regularization recipe
across all ten classes was found (via a train-vs-eval AP comparison, see
notebook/model_training_evaluation.ipynb Step 5) to overfit most classes to
some degree, and badly for a few (e.g. traffic_cone, barrier). Two fixes
address this:

  1. Row/feature subsampling (``subsample``, ``colsample_bytree``) and L2
     regularization (``reg_lambda``), applied uniformly -- standard variance
     reduction for gradient-boosted trees, cheap and rarely harmful.
  2. Per-class early stopping directly against ``val_df`` (the externally
     supplied calibration split): each class's boosting-round count is
     chosen to maximize validation AUCPR on that class's own eval rows,
     then the final model is refit on the full training partition using
     that round count. Using ``val_df`` for this is intentional, not
     leakage -- this project already treats that partition as the
     designated split "for calibrating post-hoc hyperparameters" (see
     configs/moe_final.yaml / the paper's Methodology section), exactly
     what a boosting-round count is. An internal, training-only probe split
     was tried first and rejected: it consistently picked far too many
     rounds (near its ceiling for nearly every class) because a random or
     even scene-grouped slice of the SAME 85 training scenes is still more
     similar to the rest of training than the genuinely different 20
     calibration scenes are -- the real gap is driven by a scene-level
     distribution shift between the two partitions, not pure sample-level
     overfitting, so only a validation set drawn from the actually-different
     distribution can detect it.

     One consequence: construction_vehicle's gap does NOT close under this
     fix (train AP ~0.73, eval AP ~0.09 either way). Its calibration split
     has only 56 positive rows for this class, and BEVFusion -- by far its
     strongest expert (59.6% train positive rate) -- contributes only 32
     rows there vs. 840 in training. No boosting-round choice or
     regularization strength can fix an evaluation partition that doesn't
     have enough of the right examples to evaluate against; this is a
     small-sample/distribution-shift limit of the data split, not a model
     capacity problem (confirmed by the FT-Transformer showing the same
     collapse independently -- see src/moe/nn_router.py).

Training runs on GPU (``device="cuda"``) for speed. Inference intentionally
runs on CPU: this pipeline scores small per-token batches of CPU-resident
numpy arrays, and XGBoost's GPU predict path falls back to a slower
mismatched-device code path on inputs like that (verified empirically), so
the fitted booster's device is explicitly reset to "cpu" before saving.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# A list (not src.utils.class_mapping.NUSCENES_CLASSES, which is an
# unordered frozenset): training and loading iterate this in order, so a
# stable, deterministic order is needed here even though set membership
# would suffice for validation elsewhere.
NUSCENES_CLASSES: list[str] = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

_CLASS_TO_ID: dict[str, int] = {
    "car": 0, "truck": 1, "trailer": 2, "bus": 3, "construction_vehicle": 4,
    "bicycle": 5, "motorcycle": 6, "pedestrian": 7, "traffic_cone": 8, "barrier": 9,
}

# Classes for which the final pipeline applies sigmoid calibration.
# Everything else uses the raw scale_pos_weight-corrected XGBoost output.
CALIBRATED_CLASSES: set[str] = {"bicycle"}

# XGBoost hyperparameters (identical across all ten per-class models).
# n_estimators is NOT fixed here -- it's chosen per class by early stopping
# (see train_xgboost_per_class_routers), since a single round count is a poor
# fit across classes ranging from ~2,900 to ~120,000 positive examples.
XGB_PARAMS: dict = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "tree_method": "hist",
}

MAX_N_ESTIMATORS = 500
# Deliberately generous: a lower patience (e.g. 20) was found to stop some
# classes (bicycle) prematurely on a noisy validation-metric plateau, before
# it recovered and kept improving -- see the module docstring.
EARLY_STOPPING_ROUNDS = 40
FALLBACK_N_ESTIMATORS = 150

SEED = 42


def _reset_device_to_cpu(clf) -> None:
    """Force a fitted XGBoost estimator (calibrated or not) back to CPU.

    After GPU training, XGBoost's predict path on small CPU numpy arrays
    silently falls back to a slower mismatched-device DMatrix path unless
    the booster's device is explicitly reset. ``set_params`` alone doesn't
    always propagate to an already-fit booster, so both the estimator's
    params and the underlying booster's device attribute are updated.
    """
    if isinstance(clf, CalibratedClassifierCV):
        estimators = [c.estimator for c in clf.calibrated_classifiers_]
    else:
        estimators = [clf]

    for estimator in estimators:
        estimator.set_params(device="cpu")
        try:
            estimator.get_booster().set_param({"device": "cpu"})
        except Exception:
            pass


def _pick_n_estimators(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    device: str,
    seed: int,
    max_n_estimators: int,
    early_stopping_rounds: int,
    fallback_n_estimators: int,
) -> int:
    """Pick a per-class boosting-round count via early stopping against
    this class's own rows in the externally-supplied calibration split
    (see module docstring for why this, rather than an internal probe).

    Falls back to ``fallback_n_estimators`` (no early stopping) if no
    validation data is available for this class -- e.g. ``val_df`` wasn't
    supplied, or this class has zero positives there.
    """
    if X_val is None or y_val is None or y_val.sum() == 0:
        log.warning(
            "No validation positives for early stopping — falling back to n_estimators=%d",
            fallback_n_estimators,
        )
        return fallback_n_estimators

    n_pos, n_total = int(y_train.sum()), len(y_train)
    scale_pos_weight = (n_total - n_pos) / max(n_pos, 1)

    probe = xgb.XGBClassifier(
        **XGB_PARAMS,
        n_estimators=max_n_estimators,
        scale_pos_weight=scale_pos_weight,
        device=device,
        random_state=seed,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="aucpr",
    )
    probe.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return int(probe.best_iteration) + 1


def train_xgboost_per_class_routers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    calibrated_classes: set[str] | None = None,
    use_gpu: bool = True,
    min_positives: int = 20,
    seed: int = SEED,
    max_n_estimators: int = MAX_N_ESTIMATORS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    fallback_n_estimators: int = FALLBACK_N_ESTIMATORS,
) -> dict[str, object]:
    """Train one XGBoost router per nuScenes detection class.

    Each class's boosting-round count is chosen via early stopping against
    that class's own rows in ``val_df``, then the final model is refit on
    the class's full training partition using that round count -- see the
    module docstring for why using ``val_df`` this way is intentional
    (it's this project's designated calibration/hyperparameter-tuning
    split), not leakage relative to the project's separately held-out test
    set.

    Args:
        train_df: Training DataFrame from build_dataset() (must contain
            ``class_id``, ``label``, and every column in ``feature_names``).
        val_df: Held-out calibration DataFrame -- used both to pick each
            class's boosting-round count (see above) and to report AUC/AP.
            If omitted, every class falls back to ``fallback_n_estimators``
            with no early stopping.
        feature_names: Feature columns to use (defaults to
            ``src.moe.features.FEATURE_NAMES``).
        calibrated_classes: Classes to wrap in ``CalibratedClassifierCV``
            (defaults to ``CALIBRATED_CLASSES`` = {"bicycle"}).
        use_gpu: Train with ``device="cuda"`` (falls back to CPU if no GPU
            is available). Inference always uses CPU regardless of this flag.
        min_positives: Skip classes with fewer positive training labels.
        seed: Random seed for reproducibility.
        max_n_estimators: Upper bound on boosting rounds during the early-
            stopping probe.
        early_stopping_rounds: Rounds of no validation-metric improvement
            before the probe stops.
        fallback_n_estimators: Round count used when ``val_df`` has no
            positives for a class (see _pick_n_estimators).

    Returns:
        Dict mapping class_name -> fitted classifier (or None if skipped).
        Each classifier exposes sklearn's ``predict_proba(X)``.
    """
    from src.moe.features import FEATURE_NAMES as DEFAULT_FEATURES

    if feature_names is None:
        feature_names = DEFAULT_FEATURES
    if calibrated_classes is None:
        calibrated_classes = CALIBRATED_CLASSES

    import torch
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    log.info("Training XGBoost router per class on %s | features=%d", device, len(feature_names))

    routers: dict[str, object] = {}

    for cls in NUSCENES_CLASSES:
        cls_id = _CLASS_TO_ID[cls]
        sub_train = train_df[train_df["class_id"] == cls_id]

        n_pos = int(sub_train["label"].sum())
        n_total = len(sub_train)
        if n_pos < min_positives:
            log.warning("Class %-22s: %d positives / %d total — skipping", cls, n_pos, n_total)
            routers[cls] = None
            continue

        X_train = sub_train[feature_names].values.astype(np.float32)
        y_train = sub_train["label"].values.astype(int)

        X_val = y_val = None
        if val_df is not None:
            sub_val = val_df[val_df["class_id"] == cls_id]
            if len(sub_val) > 0:
                X_val = sub_val[feature_names].values.astype(np.float32)
                y_val = sub_val["label"].values.astype(int)

        best_n_estimators = _pick_n_estimators(
            X_train, y_train, X_val, y_val, device, seed,
            max_n_estimators, early_stopping_rounds, fallback_n_estimators,
        )

        n_neg = n_total - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        base_clf = xgb.XGBClassifier(
            **XGB_PARAMS,
            n_estimators=best_n_estimators,
            scale_pos_weight=scale_pos_weight,
            device=device,
            random_state=seed,
        )

        if cls in calibrated_classes:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=cv, ensemble=False)
        else:
            clf = base_clf

        clf.fit(X_train, y_train)
        _reset_device_to_cpu(clf)

        log.info(
            "Trained %-22s: %d rows, %.2f%% positive, n_estimators=%d%s",
            cls, n_total, 100.0 * n_pos / n_total, best_n_estimators,
            " (calibrated)" if cls in calibrated_classes else "",
        )

        if y_val is not None and y_val.sum() > 0:
            proba = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            ap = average_precision_score(y_val, proba)
            log.info("  └─ val AUC=%.4f  AP=%.4f", auc, ap)

        routers[cls] = clf

    return routers


def save_xgboost_per_class_routers(routers: dict[str, object], output_dir: Path) -> None:
    """Save each per-class XGBoost router to output_dir as router_<class>.pkl.

    Uses pickle (not XGBoost's native ``save_model``) because calibrated
    classes are wrapped in a ``CalibratedClassifierCV``, which isn't a plain
    Booster. This also matches the loading convention used by
    ``src/moe/infer_router.py``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for cls, clf in routers.items():
        if clf is None:
            continue
        with (output_dir / f"router_{cls}.pkl").open("wb") as f:
            pickle.dump(clf, f)
        saved.append(cls)

    log.info("Saved XGBoost per-class routers for %d classes -> %s", len(saved), output_dir)


def load_xgboost_per_class_routers(input_dir: Path) -> dict[str, object]:
    """Load per-class XGBoost routers saved by save_xgboost_per_class_routers."""
    input_dir = Path(input_dir)
    routers: dict[str, object] = {}
    for cls in NUSCENES_CLASSES:
        path = input_dir / f"router_{cls}.pkl"
        if path.exists():
            with path.open("rb") as f:
                routers[cls] = pickle.load(f)
        else:
            routers[cls] = None
    return routers
