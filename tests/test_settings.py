from __future__ import annotations

from pathlib import Path

from scripts.settings import BASE_DIR, AppSettings


def test_settings_supports_legacy_flat_environment_names() -> None:
    settings = AppSettings(
        **{
            "YOLO_MODEL_PATH": str(BASE_DIR / "alt.pt"),
            "RISK_THRESHOLD": "0.7",
            "DISTANCE_THRESHOLD": "72.5",
            "TRACKER_BACKEND": "simple",
            "CYCLE_MIN": "40",
            "CYCLE_MAX": "95",
            "MAX_UPLOAD_SIZE_BYTES": "1024",
            "VIDEO_JOB_WORKERS": "3",
            "PEDESTRIAN_PHASE_ENABLED": "true",
        }
    )

    assert settings.model_paths.yolo_model_path == Path(BASE_DIR / "alt.pt")
    assert settings.thresholds.risk_threshold == 0.7
    assert settings.thresholds.distance_threshold_px == 72.5
    assert settings.tracker.backend == "simple"
    assert settings.optimizer.cycle_min == 40.0
    assert settings.optimizer.cycle_max == 95.0
    assert settings.api.max_upload_size_bytes == 1024
    assert settings.api.video_job_workers == 3
    assert settings.pedestrian_phase.enabled is True


def test_settings_supports_nested_aliases_and_derived_paths() -> None:
    settings = AppSettings(
        **{
            "MODEL_PATHS__SCENE_CALIBRATION_PATH": "config/scene.json",
            "THRESHOLDS__DISTANCE_THRESHOLD_METERS": "12.5",
            "OPTIMIZER__QUEUE_WEIGHT": "1.5",
            "OPTIMIZER__RISK_WEIGHT": "6.5",
            "PATHS__RESULTS_DIR": "custom-results",
            "LOGGING__FORMAT": "plain",
            "DEMAND_FORECAST__ENABLED": "true",
            "DEMAND_FORECAST__MODEL_PATH": "models/demand_lstm.pt",
            "DEMAND_FORECAST__SCALER_PATH": "models/demand_lstm_scaler.json",
        }
    )

    assert settings.model_paths.scene_calibration_path == Path("config/scene.json")
    assert settings.thresholds.distance_threshold_meters == 12.5
    assert settings.optimizer_weights.queue_weight == 1.5
    assert settings.optimizer_weights.risk_weight == 6.5
    assert settings.paths.results_dir == Path("custom-results")
    assert settings.paths.jobs_dir == Path("custom-results") / "jobs"
    assert settings.logging.fmt == "plain"
    assert settings.demand_forecast.enabled is True
    assert settings.demand_forecast.model_path == Path("models/demand_lstm.pt")
    assert settings.demand_forecast.scaler_path == Path("models/demand_lstm_scaler.json")
