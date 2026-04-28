from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol

try:  # pragma: no cover
    from run_modes import PipelineRunMode
except ImportError:  # pragma: no cover
    from scripts.run_modes import PipelineRunMode
try:  # pragma: no cover
    from scripts.settings import get_settings
except ImportError:  # pragma: no cover
    from settings import get_settings

if TYPE_CHECKING:  # pragma: no cover
    try:
        from scripts.pipeline_runner import TrafficPipeline
    except ImportError:  # pragma: no cover
        from pipeline_runner import TrafficPipeline


class PipelineProtocol(Protocol):
    def process_video(
        self,
        source: str,
        output_dir: Path,
        show: Optional[bool] = None,
        save_txt: Optional[bool] = None,
        events_filename: Optional[str] = None,
        collect_metrics: Optional[bool] = None,
        mode: Optional[str] = None,
        write_video: Optional[bool] = None,
    ) -> dict[str, Any]: ...


def load_validation_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("validation manifest must be a JSON object with a cases list")
    return payload


def _dotted_get(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _evaluate_expectation(actual: Any, rule: dict[str, Any]) -> tuple[bool, str]:
    if "equals" in rule:
        return actual == rule["equals"], f"equals={rule['equals']!r}"
    if "min" in rule or "max" in rule:
        passed = True
        bounds: list[str] = []
        if "min" in rule:
            passed = passed and actual is not None and actual >= rule["min"]
            bounds.append(f"min={rule['min']}")
        if "max" in rule:
            passed = passed and actual is not None and actual <= rule["max"]
            bounds.append(f"max={rule['max']}")
        return passed, ", ".join(bounds)
    if "target" in rule:
        tolerance = float(rule.get("tolerance", 0.0))
        passed = actual is not None and abs(float(actual) - float(rule["target"])) <= tolerance
        return passed, f"target={rule['target']}, tolerance={tolerance}"
    raise ValueError(f"unsupported expectation rule: {rule}")


def extract_validation_metrics(result: dict[str, Any]) -> dict[str, Any]:
    frames_processed = int(result.get("frames_processed") or 0)
    events = list(result.get("events") or [])
    queue_history = list(result.get("queue_history") or [])
    latest_plan = dict(result.get("latest_plan") or {})
    tracking_summary = dict(result.get("tracking_summary") or {})

    queue_peaks = [
        max((float(value) for value in entry.get("queues", {}).values()), default=0.0)
        for entry in queue_history
    ]
    severity_counts = {"low": 0, "medium": 0, "high": 0}
    for event in events:
        severity = str(event.get("severity") or "medium")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    residual_queue = dict(latest_plan.get("residual_queue") or {})
    effective_demand = dict(latest_plan.get("effective_demand") or {})
    residual_total = round(sum(float(value) for value in residual_queue.values()), 3)
    effective_total = round(sum(float(value) for value in effective_demand.values()), 3)
    estimated_served = round(max(effective_total - residual_total, 0.0), 3)

    return {
        "frames_processed": frames_processed,
        "tracking": {
            "unique_tracks": int(tracking_summary.get("unique_tracks") or 0),
            "avg_active_tracks": float(tracking_summary.get("avg_active_tracks") or 0.0),
            "peak_active_tracks": int(tracking_summary.get("peak_active_tracks") or 0),
        },
        "queues": {
            "mean_peak": round(sum(queue_peaks) / len(queue_peaks), 3) if queue_peaks else 0.0,
            "max_peak": round(max(queue_peaks, default=0.0), 3),
        },
        "near_miss": {
            "total_events": int(result.get("total_events") or len(events)),
            "events_per_100_frames": (
                round((len(events) / frames_processed) * 100.0, 3) if frames_processed else 0.0
            ),
            "severity_counts": severity_counts,
        },
        "lp": {
            "optimizer": latest_plan.get("optimizer"),
            "solver_status": latest_plan.get("solver_status"),
            "latest_cycle": latest_plan.get("cycle"),
            "objective_value": latest_plan.get("objective_value"),
            "estimated_served_load": estimated_served,
            "residual_queue_total": residual_total,
            "effective_demand_total": effective_total,
        },
        "scene_calibration": result.get("scene_calibration"),
    }


def evaluate_validation_case(
    case: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    metrics = extract_validation_metrics(result)
    expectations = case.get("expectations") or {}
    checks = []
    for dotted_path, rule in expectations.items():
        actual = _dotted_get(metrics, dotted_path)
        passed, expected = _evaluate_expectation(actual, rule)
        checks.append(
            {
                "metric": dotted_path,
                "actual": actual,
                "expected": expected,
                "passed": passed,
            }
        )
    return {
        "id": case["id"],
        "source": case["source"],
        "metrics": metrics,
        "checks": checks,
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
    }


def write_validation_report(
    report: dict[str, Any],
    output_dir: str | Path,
    report_name: str = "validation_report",
) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{report_name}.json"
    csv_path = target_dir / f"{report_name}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_id",
                "status",
                "frames_processed",
                "unique_tracks",
                "avg_active_tracks",
                "peak_active_tracks",
                "total_events",
                "events_per_100_frames",
                "max_queue_peak",
                "mean_queue_peak",
                "optimizer",
                "solver_status",
                "latest_cycle",
                "objective_value",
                "estimated_served_load",
                "residual_queue_total",
                "checks_passed",
                "checks_total",
            ],
        )
        writer.writeheader()
        for case in report["cases"]:
            metrics = case["metrics"]
            checks = case["checks"]
            writer.writerow(
                {
                    "case_id": case["id"],
                    "status": case["status"],
                    "frames_processed": metrics["frames_processed"],
                    "unique_tracks": metrics["tracking"]["unique_tracks"],
                    "avg_active_tracks": metrics["tracking"]["avg_active_tracks"],
                    "peak_active_tracks": metrics["tracking"]["peak_active_tracks"],
                    "total_events": metrics["near_miss"]["total_events"],
                    "events_per_100_frames": metrics["near_miss"]["events_per_100_frames"],
                    "max_queue_peak": metrics["queues"]["max_peak"],
                    "mean_queue_peak": metrics["queues"]["mean_peak"],
                    "optimizer": metrics["lp"]["optimizer"],
                    "solver_status": metrics["lp"]["solver_status"],
                    "latest_cycle": metrics["lp"]["latest_cycle"],
                    "objective_value": metrics["lp"]["objective_value"],
                    "estimated_served_load": metrics["lp"]["estimated_served_load"],
                    "residual_queue_total": metrics["lp"]["residual_queue_total"],
                    "checks_passed": sum(1 for check in checks if check["passed"]),
                    "checks_total": len(checks),
                }
            )
    return {"json": json_path, "csv": csv_path}


def run_validation_suite(
    pipeline: PipelineProtocol,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest = load_validation_manifest(manifest_path)
    cases = []
    for case in manifest["cases"]:
        case_output_dir = Path(output_dir) / str(case["id"])
        result = pipeline.process_video(
            source=str(case["source"]),
            output_dir=case_output_dir,
            mode=PipelineRunMode.RESEARCH.value,
            show=False,
            save_txt=False,
            events_filename=f"{case['id']}_events.jsonl",
            collect_metrics=True,
            write_video=False,
        )
        cases.append(evaluate_validation_case(case, result))

    report = {
        "suite": manifest.get("name", "validation"),
        "cases": cases,
        "summary": {
            "total_cases": len(cases),
            "passed_cases": sum(1 for case in cases if case["status"] == "passed"),
            "failed_cases": sum(1 for case in cases if case["status"] == "failed"),
        },
    }
    write_validation_report(report, output_dir)
    return report


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Validation harness for traffic pipeline runs.")
    parser.add_argument("--manifest", required=True, help="Path to validation manifest JSON.")
    parser.add_argument("--output-dir", default="results/validation", help="Directory for reports.")
    parser.add_argument(
        "--model", default=str(settings.model_paths.yolo_model_path), help="YOLO model path."
    )
    parser.add_argument("--device", default="cpu", help="Inference device.")
    parser.add_argument(
        "--roi-config",
        default=(
            str(settings.model_paths.roi_config_path)
            if settings.model_paths.roi_config_path is not None
            else None
        ),
        help="ROI config JSON path.",
    )
    parser.add_argument(
        "--scene-calibration",
        default=(
            str(settings.model_paths.scene_calibration_path)
            if settings.model_paths.scene_calibration_path is not None
            else None
        ),
        help="Scene calibration JSON path with meters_per_pixel or homography.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=settings.thresholds.distance_threshold_px,
        help="Candidate threshold in pixels for near-miss detection.",
    )
    parser.add_argument(
        "--distance-threshold-meters",
        type=float,
        default=settings.thresholds.distance_threshold_meters,
        help="Optional candidate threshold in meters after calibration.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=settings.thresholds.risk_threshold,
        help="Near-miss risk threshold.",
    )
    return parser.parse_args()


def build_validation_pipeline(args: argparse.Namespace) -> "TrafficPipeline":
    try:  # pragma: no cover
        from scripts.pipeline_runner import TrafficPipeline
    except ImportError:  # pragma: no cover
        from pipeline_runner import TrafficPipeline

    return TrafficPipeline(
        model_path=args.model,
        device=args.device,
        roi_config=args.roi_config,
        risk_threshold=args.risk_threshold,
        distance_threshold=args.distance_threshold,
        distance_threshold_meters=args.distance_threshold_meters,
        scene_calibration_path=args.scene_calibration,
    )


def main() -> None:
    args = parse_args()
    pipeline = build_validation_pipeline(args)
    report = run_validation_suite(pipeline, args.manifest, args.output_dir)
    summary = report["summary"]
    print(
        "Validation completed:",
        f"{summary['passed_cases']}/{summary['total_cases']} cases passed",
    )


if __name__ == "__main__":
    main()
