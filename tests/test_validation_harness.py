from __future__ import annotations

import csv
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.validation_harness import (
    _dotted_get,
    _evaluate_expectation,
    build_validation_pipeline,
    evaluate_validation_case,
    extract_validation_metrics,
    load_validation_manifest,
    main,
    parse_args,
    run_validation_suite,
)


class FakePipeline:
    def process_video(
        self,
        source: str,
        output_dir: Path,
        show,
        save_txt,
        events_filename: str,
        collect_metrics,
        mode: str,
        write_video,
    ) -> dict[str, object]:
        assert output_dir.name in {"case-a", "case-b"}
        return {
            "frames_processed": 120,
            "total_events": 3,
            "events": [
                {"severity": "high"},
                {"severity": "medium"},
                {"severity": "medium"},
            ],
            "queue_history": [
                {"frame": 0, "queues": {"north": 2, "east": 1}},
                {"frame": 1, "queues": {"north": 4, "east": 3}},
            ],
            "latest_plan": {
                "optimizer": "lp",
                "solver_status": "optimal",
                "cycle": 64.0,
                "objective_value": 9.5,
                "residual_queue": {"north": 1.0, "east": 2.0},
                "effective_demand": {"north": 6.0, "east": 4.0},
            },
            "tracking_summary": {
                "unique_tracks": 9,
                "avg_active_tracks": 3.5,
                "peak_active_tracks": 5,
            },
            "scene_calibration": {"name": "intersection-a", "is_calibrated": True},
        }


def test_extract_validation_metrics_builds_tracking_and_lp_summary() -> None:
    metrics = extract_validation_metrics(
        FakePipeline().process_video("", Path("case-a"), None, None, "", None, "", None)
    )

    assert metrics["tracking"]["unique_tracks"] == 9
    assert metrics["queues"]["max_peak"] == 4.0
    assert metrics["near_miss"]["events_per_100_frames"] == 2.5
    assert metrics["lp"]["estimated_served_load"] == 7.0


def test_run_validation_suite_writes_json_and_csv_reports(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "smoke-suite",
                "cases": [
                    {
                        "id": "case-a",
                        "source": "video-a.mp4",
                        "expectations": {
                            "tracking.unique_tracks": {"min": 5},
                            "near_miss.total_events": {"max": 4},
                            "lp.optimizer": {"equals": "lp"},
                        },
                    },
                    {
                        "id": "case-b",
                        "source": "video-b.mp4",
                        "expectations": {
                            "lp.latest_cycle": {"target": 64.0, "tolerance": 0.1},
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_validation_suite(FakePipeline(), manifest_path, tmp_path / "reports")

    assert report["summary"] == {"total_cases": 2, "passed_cases": 2, "failed_cases": 0}
    json_path = tmp_path / "reports" / "validation_report.json"
    csv_path = tmp_path / "reports" / "validation_report.csv"
    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["cases"][0]["checks"][0]["metric"] == "tracking.unique_tracks"

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["case_id"] == "case-a"
    assert rows[0]["checks_passed"] == "3"


def test_validation_harness_helpers_cover_edge_cases(tmp_path: Path, monkeypatch, capsys) -> None:
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(json.dumps({"name": "oops"}), encoding="utf-8")
    with pytest.raises(ValueError, match="cases list"):
        load_validation_manifest(bad_manifest)

    assert _dotted_get({"a": {"b": 1}}, "a.b") == 1
    assert _dotted_get({"a": {"b": 1}}, "a.c") is None
    assert _evaluate_expectation(5, {"equals": 5}) == (True, "equals=5")
    assert _evaluate_expectation(5, {"min": 4, "max": 6}) == (True, "min=4, max=6")
    assert _evaluate_expectation(5.2, {"target": 5.0, "tolerance": 0.3}) == (
        True,
        "target=5.0, tolerance=0.3",
    )
    with pytest.raises(ValueError, match="unsupported expectation"):
        _evaluate_expectation(1, {"weird": 1})

    failed_case = evaluate_validation_case(
        {
            "id": "case-x",
            "source": "video.mp4",
            "expectations": {"tracking.unique_tracks": {"min": 99}},
        },
        FakePipeline().process_video("", Path("case-a"), None, None, "", None, "", None),
    )
    assert failed_case["status"] == "failed"
    assert failed_case["checks"][0]["passed"] is False

    fake_pipeline_module = types.ModuleType("scripts.pipeline_runner")

    class StubPipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_pipeline_module.TrafficPipeline = StubPipeline
    monkeypatch.setitem(sys.modules, "scripts.pipeline_runner", fake_pipeline_module)

    args = Namespace(
        model="model.pt",
        device="cpu",
        roi_config="roi.json",
        risk_threshold=0.5,
        distance_threshold=55.0,
        distance_threshold_meters=11.0,
        scene_calibration="scene.json",
        manifest="manifest.json",
        output_dir="out",
    )
    pipeline = build_validation_pipeline(args)
    assert pipeline.kwargs["model_path"] == "model.pt"
    assert pipeline.kwargs["scene_calibration_path"] == "scene.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validation_harness.py",
            "--manifest",
            "manifest.json",
            "--output-dir",
            "out",
        ],
    )
    parsed = parse_args()
    assert parsed.manifest == "manifest.json"
    assert parsed.output_dir == "out"

    monkeypatch.setattr("scripts.validation_harness.parse_args", lambda: args)
    monkeypatch.setattr(
        "scripts.validation_harness.build_validation_pipeline", lambda _: FakePipeline()
    )
    monkeypatch.setattr(
        "scripts.validation_harness.run_validation_suite",
        lambda pipeline, manifest_path, output_dir: {
            "summary": {"passed_cases": 1, "total_cases": 1},
            "cases": [],
        },
    )
    main()
    assert "1/1 cases passed" in capsys.readouterr().out
