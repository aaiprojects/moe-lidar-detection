"""Load expert prediction files in nuScenes submission format."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import AbstractSet

from src.io.schemas import DetectionBox
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

REQUIRED_BOX_FIELDS: tuple[str, ...] = (
    "sample_token",
    "translation",
    "size",
    "rotation",
    "velocity",
    "detection_name",
    "detection_score",
)


def load_nuscenes_predictions(
    path: Path,
    model_name: str,
    frame: str = "global",
    score_threshold: float = 0.0,
    sample_tokens: AbstractSet[str] | list[str] | None = None,
) -> dict[str, list[DetectionBox]]:
    """Load a nuScenes submission JSON and return boxes keyed by sample token.

    Args:
        path: Path to the JSON file.
        model_name: Identifier for this expert (used in DetectionBox.model_name).
        frame: Coordinate frame the predictions are in ('global', 'lidar', 'ego').
        score_threshold: Discard boxes with score strictly below this value.
        sample_tokens: If set, keep only these sample tokens. Strongly recommended
            when running on a subset of keyframes — each full val JSON is hundreds
            of MB on disk and several GB once parsed into Python objects.

    Returns:
        Dict mapping sample_token → list of DetectionBox.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required fields are missing or values are invalid.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found for model '{model_name}': {path}"
        )

    token_filter: set[str] | None = None
    if sample_tokens is not None:
        token_filter = set(sample_tokens)

    log.info(
        "Loading predictions from %s (model=%s%s)",
        path,
        model_name,
        f", {len(token_filter)} tokens" if token_filter else ", all tokens",
    )
    with path.open() as f:
        data = json.load(f)

    if "results" not in data:
        raise ValueError(
            f"Prediction file {path} has no 'results' key. "
            "Expected standard nuScenes submission format."
        )

    raw_results = data["results"]
    # A full-val prediction file is hundreds of MB on disk; explicitly drop
    # the parsed-JSON dict once its 'results' payload is extracted so peak
    # memory doesn't briefly hold both the raw dict and the DetectionBox
    # objects being built from it.
    del data
    gc.collect()

    results: dict[str, list[DetectionBox]] = {}
    total_loaded = 0
    total_skipped = 0

    for sample_token, raw_boxes in raw_results.items():
        if token_filter is not None and sample_token not in token_filter:
            continue
        if not sample_token:
            raise ValueError("Encountered empty sample_token in prediction file.")

        boxes: list[DetectionBox] = []
        for raw in raw_boxes:
            _check_required_fields(raw, sample_token, path)

            if raw["detection_score"] < score_threshold:
                total_skipped += 1
                continue

            try:
                box = DetectionBox(
                    sample_token=sample_token,
                    model_name=model_name,
                    translation=list(raw["translation"]),
                    size=list(raw["size"]),
                    rotation=list(raw["rotation"]),
                    velocity=list(raw.get("velocity", [0.0, 0.0])),
                    detection_name=raw["detection_name"],
                    detection_score=float(raw["detection_score"]),
                    attribute_name=raw.get("attribute_name") or None,
                    frame=frame,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid box in sample {sample_token!r} from {path}: {exc}"
                ) from exc

            boxes.append(box)
            total_loaded += 1

        results[sample_token] = boxes

    del raw_results
    gc.collect()

    log.info(
        "Loaded %d boxes across %d samples (skipped %d below score threshold)",
        total_loaded,
        len(results),
        total_skipped,
    )
    return results


def _check_required_fields(raw: dict, sample_token: str, path: Path) -> None:
    """Raise ValueError naming any REQUIRED_BOX_FIELDS missing from a raw box dict.

    Checked before constructing DetectionBox so a malformed box is reported
    with its sample token and source file, not a bare KeyError.
    """
    missing = [f for f in REQUIRED_BOX_FIELDS if f not in raw]
    if missing:
        raise ValueError(
            f"Box in sample {sample_token!r} from {path} is missing fields: {missing}"
        )
