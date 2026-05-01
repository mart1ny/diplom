from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.scene_calibration import SceneCalibration
from tests.support import build_stub_pipeline, write_test_video


def test_pipeline_integration_emits_metric_near_miss_events(tmp_path: Path) -> None:
    video_path = write_test_video(tmp_path / "near_miss.avi", frame_count=4, fps=10.0)
    pipeline = build_stub_pipeline(
        frame_positions=[
            {1: (40.0, 50.0), 2: (60.0, 50.0)},
            {1: (45.0, 50.0), 2: (55.0, 50.0)},
            {1: (48.0, 50.0), 2: (52.0, 50.0)},
            {1: (49.0, 50.0), 2: (51.0, 50.0)},
        ],
        scene_calibration=SceneCalibration(meters_per_pixel=0.25, distance_threshold_meters=8.0),
        risk_threshold=0.01,
        distance_threshold=25.0,
        distance_threshold_meters=8.0,
    )

    result = pipeline.process_video(
        source=str(video_path),
        output_dir=tmp_path / "out-near-miss",
        mode="api",
        collect_metrics=True,
        events_filename="events.jsonl",
        write_video=False,
        save_txt=False,
    )

    assert result["frames_processed"] == 4
    assert result["tracking_summary"]["unique_tracks"] == 2
    assert result["scene_calibration"]["is_calibrated"] is True
    assert result["total_events"] >= 1

    event = result["events"][-1]
    assert event["risk_score"] >= 0.01
    assert event["physics"]["unit_system"] == "metric"
    assert event["physics"]["distance_m"] == pytest.approx(0.5, rel=1e-3)
    assert event["physics"]["ttc_seconds"] is not None
    assert event["physics"]["closing_speed_mps"] is not None

    events_path = Path(result["events_file"])
    lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["physics"]["distance_m"] == pytest.approx(0.5, rel=1e-3)


def test_pipeline_integration_regresses_lp_plan_for_queue_pressure(tmp_path: Path) -> None:
    video_path = write_test_video(tmp_path / "queues.avi", frame_count=5, fps=2.0)
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(
        json.dumps(
            {
                "north": [[35, 0], [65, 0], [65, 30], [35, 30]],
                "east": [[70, 35], [100, 35], [100, 65], [70, 65]],
            }
        ),
        encoding="utf-8",
    )
    pipeline = build_stub_pipeline(
        frame_positions=[
            {1: (42.0, 10.0), 2: (50.0, 14.0), 3: (58.0, 18.0), 4: (88.0, 50.0)},
            {1: (42.0, 10.0), 2: (50.0, 14.0), 3: (58.0, 18.0), 4: (88.0, 50.0)},
            {1: (42.0, 10.0), 2: (50.0, 14.0), 3: (58.0, 18.0), 4: (88.0, 50.0)},
            {1: (42.0, 10.0), 2: (50.0, 14.0), 3: (58.0, 18.0), 4: (88.0, 50.0)},
            {1: (42.0, 10.0), 2: (50.0, 14.0), 3: (58.0, 18.0), 4: (88.0, 50.0)},
        ],
        roi_config=str(roi_path),
        risk_threshold=0.95,
        distance_threshold=10.0,
        cycle_bounds=(20.0, 80.0),
    )

    result = pipeline.process_video(
        source=str(video_path),
        output_dir=tmp_path / "out-lp",
        mode="api",
        collect_metrics=True,
        events_filename="events.jsonl",
        write_video=False,
        save_txt=False,
    )

    latest_plan = result["latest_plan"]
    assert latest_plan["optimizer"] == "lp"
    assert latest_plan["solver_status"] in {"optimal", "optimal_inaccurate"}
    assert latest_plan["objective_value"] is not None
    assert latest_plan["durations"]["north"] > latest_plan["durations"]["east"]
    assert latest_plan["cycle"] >= 20.0

    assert len(result["plan_history"]) >= 2
    assert (
        result["plan_history"][1]["queues"]["north"] > result["plan_history"][1]["queues"]["east"]
    )
    assert result["performance_metrics"]["lp_iterations"] >= 2


def test_pipeline_integration_uses_demand_forecast_in_lp_plan(tmp_path: Path) -> None:
    video_path = write_test_video(tmp_path / "forecast.avi", frame_count=4, fps=2.0)
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(
        json.dumps(
            {
                "north": [[35, 0], [65, 0], [65, 30], [35, 30]],
                "east": [[70, 35], [100, 35], [100, 65], [70, 65]],
            }
        ),
        encoding="utf-8",
    )

    class FakeForecaster:
        def predict(self, history):
            approach = history[-1]["approach"]
            return [4.0] if approach == "north" else [18.0]

    pipeline = build_stub_pipeline(
        frame_positions=[
            {1: (42.0, 10.0), 2: (50.0, 14.0), 3: (88.0, 50.0)},
            {1: (42.0, 10.0), 2: (50.0, 14.0), 3: (88.0, 50.0)},
            {1: (42.0, 10.0), 2: (50.0, 14.0), 3: (88.0, 50.0)},
            {1: (42.0, 10.0), 2: (50.0, 14.0), 3: (88.0, 50.0)},
        ],
        roi_config=str(roi_path),
        risk_threshold=0.95,
        distance_threshold=10.0,
        cycle_bounds=(20.0, 80.0),
    )
    pipeline.demand_forecast_settings.enabled = True
    pipeline.demand_forecast_settings.alpha = 0.2
    pipeline.demand_forecast_settings.window_size = 1
    pipeline.demand_forecaster = FakeForecaster()

    result = pipeline.process_video(
        source=str(video_path),
        output_dir=tmp_path / "out-forecast",
        mode="api",
        collect_metrics=True,
        events_filename="events.jsonl",
        write_video=False,
        save_txt=False,
    )

    latest_plan = result["latest_plan"]
    assert result["demand_forecast"]["enabled"] is True
    assert result["demand_forecast"]["model_loaded"] is True
    assert result["performance_metrics"]["forecast_iterations"] >= 1
    assert latest_plan["forecast_queue"]["east"] == pytest.approx(18.0)
    assert latest_plan["effective_queue"]["east"] > latest_plan["effective_queue"]["north"]
    assert latest_plan["durations"]["east"] > latest_plan["durations"]["north"]
    assert result["plan_history"][0]["forecast"]["east"] == pytest.approx(18.0)
