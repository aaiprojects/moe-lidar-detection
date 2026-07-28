"""Detection box schema for the lidar-moe-detector pipeline.

All predictions are normalized into DetectionBox before any fusion step.
Internal convention (Version 1):
  frame       : global  (nuScenes world frame — all submission JSONs use this)
  translation : [x, y, z]  metres
  size        : [width, length, height]  metres  (all positive)
  rotation    : [w, x, y, z]  quaternion
  yaw         : float  radians  (derived from rotation for 2-D fusion)
  velocity    : [vx, vy]  m/s
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.utils.class_mapping import validate_class

VALID_FRAMES: frozenset[str] = frozenset(["global", "lidar", "ego"])


@dataclass
class DetectionBox:
    sample_token: str
    model_name: str
    translation: list[float]          # [x, y, z]
    size: list[float]                 # [width, length, height]
    rotation: list[float]             # quaternion [w, x, y, z]
    velocity: list[float]             # [vx, vy]
    detection_name: str
    detection_score: float
    attribute_name: str | None
    frame: str = "global"
    yaw: float = field(init=False)

    def __post_init__(self) -> None:
        self._validate()
        self.yaw = _quat_to_yaw(self.rotation)

    def _validate(self) -> None:
        if not self.sample_token:
            raise ValueError("sample_token must not be empty")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if self.frame not in VALID_FRAMES:
            raise ValueError(
                f"Unknown coordinate frame {self.frame!r}. "
                f"Valid frames: {sorted(VALID_FRAMES)}"
            )
        if len(self.translation) != 3:
            raise ValueError(f"translation must have 3 elements, got {self.translation}")
        if len(self.size) != 3:
            raise ValueError(f"size must have 3 elements, got {self.size}")
        for dim in self.size:
            if dim <= 0:
                raise ValueError(f"All size dimensions must be positive, got {self.size}")
        if len(self.rotation) != 4:
            raise ValueError(f"rotation must have 4 elements (quaternion), got {self.rotation}")
        if len(self.velocity) != 2:
            raise ValueError(f"velocity must have 2 elements, got {self.velocity}")
        if not math.isfinite(self.detection_score):
            raise ValueError(f"detection_score must be finite, got {self.detection_score}")
        if not (0.0 <= self.detection_score <= 1.0):
            raise ValueError(
                f"detection_score must be in [0, 1], got {self.detection_score}"
            )
        # Validate and normalise class name (raises ValueError on unknown class)
        self.detection_name = validate_class(self.detection_name)

    def to_nuscenes_dict(self) -> dict:
        """Serialise to nuScenes submission format (one box entry)."""
        return {
            "sample_token": self.sample_token,
            "translation": list(self.translation),
            "size": list(self.size),
            "rotation": list(self.rotation),
            "velocity": list(self.velocity),
            "detection_name": self.detection_name,
            "detection_score": float(self.detection_score),
            "attribute_name": self.attribute_name or "",
        }


def _quat_to_yaw(q: list[float]) -> float:
    """Convert a [w, x, y, z] quaternion to a yaw angle in radians.

    Assumes rotation is primarily around the z-axis (valid for ground-plane boxes).
    """
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
