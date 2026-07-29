"""Canonical nuScenes detection class names and validation utilities."""

from __future__ import annotations

NUSCENES_CLASSES: frozenset[str] = frozenset(
    [
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    ]
)

# Aliases from common model outputs → canonical name.
# Add entries here only; never silently apply them in core logic.
CLASS_ALIASES: dict[str, str] = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.rigid": "bus",
    "vehicle.bus.bendy": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "barrier",
    "movable_object.debris": "barrier",
    "static_object.bicycle_rack": "bicycle",
}


def validate_class(name: str) -> str:
    """Return the canonical class name or raise ValueError.

    Tries an exact match first, then the alias table.
    """
    if name in NUSCENES_CLASSES:
        return name
    if name in CLASS_ALIASES:
        return CLASS_ALIASES[name]
    raise ValueError(
        f"Unknown detection class {name!r}. "
        f"Valid classes: {sorted(NUSCENES_CLASSES)}. "
        f"Known aliases: {sorted(CLASS_ALIASES)}."
    )
