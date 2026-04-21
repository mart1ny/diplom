import json
from pathlib import Path

import numpy as np
import pytest

from scripts.scene_calibration import SceneCalibration, load_scene_calibration


def test_scene_calibration_scales_pixels_to_meters() -> None:
    calibration = SceneCalibration(name="scaled", meters_per_pixel=0.5)

    distance = calibration.distance_between(np.array([0.0, 0.0]), np.array([6.0, 8.0]))

    assert distance == pytest.approx(5.0)
    assert calibration.project_displacement(np.array([0.0, 0.0]), np.array([2.0, 4.0])).tolist() == [
        1.0,
        2.0,
    ]


def test_scene_calibration_projects_points_via_homography() -> None:
    calibration = SceneCalibration(
        name="homography",
        homography=np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 3.0, 2.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    projected = calibration.project_point(np.array([4.0, 5.0]))

    assert projected.tolist() == pytest.approx([9.0, 17.0])


def test_load_scene_calibration_from_json(tmp_path: Path) -> None:
    payload = {
        "name": "intersection-a",
        "meters_per_pixel": 0.2,
        "distance_threshold_meters": 12.5,
    }
    config_path = tmp_path / "scene.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    calibration = load_scene_calibration(config_path)

    assert calibration is not None
    assert calibration.name == "intersection-a"
    assert calibration.as_metadata()["meters_per_pixel"] == pytest.approx(0.2)


def test_load_scene_calibration_requires_metric_transform(tmp_path: Path) -> None:
    config_path = tmp_path / "scene.json"
    config_path.write_text(json.dumps({"name": "broken"}), encoding="utf-8")

    with pytest.raises(ValueError, match="meters_per_pixel or homography"):
        load_scene_calibration(config_path)
