from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.validation_harness import (
    extract_validation_metrics,
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
    metrics = extract_validation_metrics(FakePipeline().process_video("", Path("case-a"), None, None, "", None, "", None))

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
