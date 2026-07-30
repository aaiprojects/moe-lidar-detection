"""Save fused predictions to nuScenes submission format JSON."""

from __future__ import annotations

import json
from pathlib import Path

from src.io.schemas import DetectionBox
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def save_nuscenes_predictions(
    boxes_by_token: dict[str, list[DetectionBox]],
    output_path: Path,
    use_lidar: bool = True,
    use_camera: bool = False,
    use_radar: bool = False,
    use_map: bool = False,
    use_external: bool = False,
) -> None:
    """Write fused boxes to a nuScenes-compatible submission JSON.

    Args:
        boxes_by_token: Dict mapping sample_token → list of DetectionBox.
        output_path: Destination path for the JSON file.
        use_lidar: Set True for LiDAR-based experts (default).
        use_camera/use_radar/use_map/use_external: Sensor flags for meta block.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {}
    total_boxes = 0
    for token, boxes in boxes_by_token.items():
        results[token] = [b.to_nuscenes_dict() for b in boxes]
        total_boxes += len(boxes)

    submission = {
        "meta": {
            "use_lidar": use_lidar,
            "use_camera": use_camera,
            "use_radar": use_radar,
            "use_map": use_map,
            "use_external": use_external,
        },
        "results": results,
    }

    with output_path.open("w") as f:
        json.dump(submission, f)

    log.info(
        "Saved %d boxes across %d samples → %s",
        total_boxes,
        len(results),
        output_path,
    )
