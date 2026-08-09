"""Round-trip tests for load_predictions and save_predictions."""

import json
import tempfile
from pathlib import Path

import pytest

from src.io.load_predictions import load_nuscenes_predictions
from src.io.save_predictions import save_nuscenes_predictions


def _make_submission(n_samples: int = 3, boxes_per_sample: int = 2) -> dict:
    """Create a minimal synthetic nuScenes submission dict."""
    results = {}
    for i in range(n_samples):
        token = f"token_{i:04d}"
        boxes = []
        for j in range(boxes_per_sample):
            boxes.append(
                {
                    "sample_token": token,
                    "translation": [float(j), float(i), 0.5],
                    "size": [2.0, 4.0, 1.5],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0],
                    "detection_name": "car",
                    "detection_score": 0.8,
                    "attribute_name": "vehicle.moving",
                }
            )
        results[token] = boxes
    return {"meta": {"use_lidar": True}, "results": results}


def test_load_returns_correct_sample_count():
    """Every sample token and its full box count is loaded from the submission JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(_make_submission(n_samples=5, boxes_per_sample=3), f)
        path = Path(f.name)

    loaded = load_nuscenes_predictions(path, model_name="test")
    assert len(loaded) == 5
    for token, boxes in loaded.items():
        assert len(boxes) == 3


def test_load_assigns_model_name():
    """Every loaded DetectionBox gets the model_name passed to the loader."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(_make_submission(), f)
        path = Path(f.name)

    loaded = load_nuscenes_predictions(path, model_name="my_expert")
    for boxes in loaded.values():
        for box in boxes:
            assert box.model_name == "my_expert"


def test_load_score_threshold_filters():
    """A box scoring below score_threshold is dropped during loading."""
    submission = _make_submission()
    # Lower one box's score below threshold
    token = list(submission["results"].keys())[0]
    submission["results"][token][0]["detection_score"] = 0.05

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(submission, f)
        path = Path(f.name)

    loaded = load_nuscenes_predictions(path, model_name="m", score_threshold=0.1)
    # That sample should have one fewer box
    assert len(loaded[token]) == 1


def test_load_missing_file_raises():
    """Loading a nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_nuscenes_predictions(Path("/nonexistent/path.json"), model_name="m")


def test_load_missing_results_key_raises():
    """A JSON file without a top-level 'results' key is rejected as malformed."""
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"meta": {}}, f)
        path = Path(f.name)

    with pytest.raises(ValueError, match="no 'results' key"):
        load_nuscenes_predictions(path, model_name="m")


def test_round_trip_preserves_box_count():
    """load -> save round-trips every sample token and its box count unchanged."""
    submission = _make_submission(n_samples=4, boxes_per_sample=5)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(submission, f)
        input_path = Path(f.name)

    loaded = load_nuscenes_predictions(input_path, model_name="x")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "out.json"
        save_nuscenes_predictions(loaded, output_path)

        with output_path.open() as f:
            saved = json.load(f)

    assert set(saved["results"].keys()) == set(submission["results"].keys())
    for token in saved["results"]:
        assert len(saved["results"][token]) == len(submission["results"][token])


def test_save_creates_parent_dirs():
    """save_nuscenes_predictions creates any missing parent directories."""
    submission = _make_submission(n_samples=1)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(submission, f)
        input_path = Path(f.name)

    loaded = load_nuscenes_predictions(input_path, model_name="x")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "deep" / "nested" / "out.json"
        save_nuscenes_predictions(loaded, output_path)
        assert output_path.exists()


def test_saved_json_has_meta_and_results_keys():
    """The saved submission JSON has both a 'meta' and a 'results' top-level key."""
    submission = _make_submission(n_samples=2)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(submission, f)
        input_path = Path(f.name)

    loaded = load_nuscenes_predictions(input_path, model_name="x")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "out.json"
        save_nuscenes_predictions(loaded, output_path)
        saved = json.loads(output_path.read_text())

    assert "meta" in saved
    assert "results" in saved
    assert saved["meta"]["use_lidar"] is True
