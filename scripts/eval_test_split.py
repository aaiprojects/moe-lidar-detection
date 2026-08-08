"""Score the fused MoE submission and every single expert on the held-out test scenes.

Writes one JSON per model to outputs/test_metrics/ as each finishes, so a crash
part-way through still leaves the completed evaluations on disk.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nuscenes import NuScenes  # noqa: E402

from src.evaluation.evaluate_nuscenes import evaluate_submission  # noqa: E402

# Defaults to data/nuscenes inside the repo; override with the NUSCENES_ROOT
# environment variable if the dataset lives elsewhere on your machine.
NUSCENES_ROOT = Path(os.environ.get("NUSCENES_ROOT", REPO / "data" / "nuscenes"))
OUT_DIR = REPO / "outputs" / "test_metrics"

SUBMISSIONS = {
    "moe_fused": REPO / "outputs" / "submission.json",
    "centerpoint": REPO / "predictions/test/centerpoint/predictions/pred_instances_3d/results_nusc.json",
    "centerpoint_pillar": REPO / "predictions/test/centerpoint_pillar/predictions/pred_instances_3d/results_nusc.json",
    "pointpillars": REPO / "predictions/test/pointpillars/predictions/pred_instances_3d/results_nusc.json",
    "ssn": REPO / "predictions/test/ssn/predictions/pred_instances_3d/results_nusc.json",
    "bevfusion_lidar": REPO / "predictions/test/bevfusion_lidar/predictions.json",
}


def main() -> None:
    if not NUSCENES_ROOT.exists():
        raise SystemExit(
            f"nuScenes dataset not found at {NUSCENES_ROOT}.\n"
            "Place it at data/nuscenes/ or set NUSCENES_ROOT to point at your copy:\n"
            "  NUSCENES_ROOT=/path/to/nuscenes python scripts/eval_test_split.py"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = json.loads((REPO / "training_data" / "token_split_3way.json").read_text())
    sample_tokens = split["test_tokens"]
    print(f"[eval] {len(sample_tokens):,} held-out test keyframes", flush=True)

    t0 = time.time()
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(NUSCENES_ROOT), verbose=False)
    print(f"[eval] nuScenes metadata loaded in {time.time() - t0:.0f}s", flush=True)

    for name, path in SUBMISSIONS.items():
        if not path.exists():
            print(f"[eval] SKIP {name}: missing {path}", flush=True)
            continue
        print(f"\n[eval] ===== {name} =====", flush=True)
        try:
            t1 = time.time()
            result = evaluate_submission(
                path, sample_tokens=sample_tokens, nusc=nusc, verbose=False
            )
            payload = {
                "model": name,
                "source": str(path.relative_to(REPO)),
                "n_sample_tokens": len(sample_tokens),
                "map": float(result["map"]),
                "nds": float(result["nds"]),
                "per_class_ap": {k: float(v) for k, v in result["per_class_ap"].items()},
                "summary": {k: float(v) for k, v in result["summary"].items()},
                "tp_errors": {k: float(v) for k, v in result["metrics"].tp_errors.items()},
                "eval_seconds": round(time.time() - t1, 1),
            }
            (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
            print(
                f"[eval] {name}: mAP={payload['map']:.4f} NDS={payload['nds']:.4f} "
                f"({payload['eval_seconds']}s)",
                flush=True,
            )
        except Exception as exc:  # keep going; a bad expert file must not kill the run
            print(f"[eval] FAILED {name}: {type(exc).__name__}: {exc}", flush=True)

    print(f"\n[eval] all done in {time.time() - t0:.0f}s -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
