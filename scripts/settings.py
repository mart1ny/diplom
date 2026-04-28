from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class ThresholdSettings(BaseModel):
    risk_threshold: float
    distance_threshold_px: float
    distance_threshold_meters: Optional[float]
    proximity_threshold_meters: float


class TrackerSettings(BaseModel):
    backend: Literal["bytetrack", "simple"]


class OptimizerSettings(BaseModel):
    cycle_min: float
    cycle_max: float
    target_cycle: int
    lambda_risk: float
    min_phase_duration: int
    max_phase_duration: int


class OptimizerWeights(BaseModel):
    queue_weight: float
    risk_weight: float


class ApiSettings(BaseModel):
    max_upload_size_bytes: int
    max_video_duration_seconds: int
    max_response_items: int
    video_job_workers: int


class ModelPathSettings(BaseModel):
    yolo_model_path: Path
    lstm_model_path: Optional[Path]
    roi_config_path: Optional[Path]
    scene_calibration_path: Optional[Path]


class PathSettings(BaseModel):
    base_dir: Path
    upload_dir: Path
    results_dir: Path
    jobs_dir: Path


class LoggingSettings(BaseModel):
    level: str
    fmt: Literal["json", "plain"]


class PedestrianPhaseSettings(BaseModel):
    enabled: bool
    name: str
    min_green: float
    max_green: float
    service_rate: float
    delay_weight: float
    queue_weight: float
    risk_weight: float
    base_demand: float


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    yolo_model_path: Path = Field(
        default=BASE_DIR / "yolov8n.pt",
        validation_alias=AliasChoices("YOLO_MODEL_PATH", "MODEL_PATHS__YOLO_MODEL_PATH"),
    )
    lstm_model_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("LSTM_MODEL_PATH", "MODEL_PATHS__LSTM_MODEL_PATH"),
    )
    roi_config_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("ROI_CONFIG_PATH", "MODEL_PATHS__ROI_CONFIG_PATH"),
    )
    scene_calibration_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices(
            "SCENE_CALIBRATION_PATH",
            "MODEL_PATHS__SCENE_CALIBRATION_PATH",
        ),
    )

    risk_threshold: float = Field(
        default=0.6,
        validation_alias=AliasChoices("RISK_THRESHOLD", "THRESHOLDS__RISK_THRESHOLD"),
    )
    distance_threshold_px: float = Field(
        default=60.0,
        validation_alias=AliasChoices(
            "DISTANCE_THRESHOLD_PX",
            "DISTANCE_THRESHOLD",
            "THRESHOLDS__DISTANCE_THRESHOLD_PX",
        ),
    )
    distance_threshold_meters: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices(
            "DISTANCE_THRESHOLD_METERS",
            "THRESHOLDS__DISTANCE_THRESHOLD_METERS",
        ),
    )
    proximity_threshold_meters: float = Field(
        default=35.0,
        validation_alias=AliasChoices(
            "PROXIMITY_THRESHOLD",
            "THRESHOLDS__PROXIMITY_THRESHOLD_METERS",
        ),
    )

    tracker_backend: Literal["bytetrack", "simple"] = Field(
        default="bytetrack",
        validation_alias=AliasChoices("TRACKER_BACKEND", "TRACKER__BACKEND"),
    )

    cycle_min: float = Field(
        default=50.0,
        validation_alias=AliasChoices("CYCLE_MIN", "OPTIMIZER__CYCLE_MIN"),
    )
    cycle_max: float = Field(
        default=90.0,
        validation_alias=AliasChoices("CYCLE_MAX", "OPTIMIZER__CYCLE_MAX"),
    )
    target_cycle: int = Field(
        default=90,
        validation_alias=AliasChoices("CYCLE_TIME", "OPTIMIZER__TARGET_CYCLE"),
    )
    lambda_risk: float = Field(
        default=5.0,
        validation_alias=AliasChoices("LAMBDA_RISK", "OPTIMIZER__LAMBDA_RISK"),
    )
    min_phase_duration: int = Field(
        default=10,
        validation_alias=AliasChoices("MIN_PHASE_DURATION", "OPTIMIZER__MIN_PHASE_DURATION"),
    )
    max_phase_duration: int = Field(
        default=60,
        validation_alias=AliasChoices("MAX_PHASE_DURATION", "OPTIMIZER__MAX_PHASE_DURATION"),
    )
    optimizer_queue_weight: float = Field(
        default=1.0,
        validation_alias=AliasChoices(
            "OPTIMIZER_QUEUE_WEIGHT",
            "OPTIMIZER__QUEUE_WEIGHT",
        ),
    )
    optimizer_risk_weight: float = Field(
        default=5.0,
        validation_alias=AliasChoices(
            "OPTIMIZER_RISK_WEIGHT",
            "OPTIMIZER__RISK_WEIGHT",
        ),
    )

    max_upload_size_bytes: int = Field(
        default=250 * 1024 * 1024,
        validation_alias=AliasChoices(
            "MAX_UPLOAD_SIZE_BYTES",
            "API__MAX_UPLOAD_SIZE_BYTES",
        ),
    )
    max_video_duration_seconds: int = Field(
        default=180,
        validation_alias=AliasChoices(
            "MAX_VIDEO_DURATION_SECONDS",
            "API__MAX_VIDEO_DURATION_SECONDS",
        ),
    )
    max_response_items: int = Field(
        default=200,
        validation_alias=AliasChoices(
            "MAX_RESPONSE_ITEMS",
            "API__MAX_RESPONSE_ITEMS",
        ),
    )
    video_job_workers: int = Field(
        default=1,
        validation_alias=AliasChoices(
            "VIDEO_JOB_WORKERS",
            "API__VIDEO_JOB_WORKERS",
        ),
    )

    upload_dir: Path = Field(
        default=BASE_DIR / "uploads",
        validation_alias=AliasChoices("UPLOAD_DIR", "PATHS__UPLOAD_DIR"),
    )
    results_dir: Path = Field(
        default=BASE_DIR / "results",
        validation_alias=AliasChoices("RESULTS_DIR", "PATHS__RESULTS_DIR"),
    )
    jobs_dir: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("JOBS_DIR", "PATHS__JOBS_DIR"),
    )

    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "LOGGING__LEVEL"),
    )
    log_format: Literal["json", "plain"] = Field(
        default="json",
        validation_alias=AliasChoices("LOG_FORMAT", "LOGGING__FORMAT"),
    )

    pedestrian_phase_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "PEDESTRIAN_PHASE_ENABLED",
            "PEDESTRIAN_PHASE__ENABLED",
        ),
    )
    pedestrian_phase_name: str = Field(
        default="pedestrian",
        validation_alias=AliasChoices(
            "PEDESTRIAN_PHASE_NAME",
            "PEDESTRIAN_PHASE__NAME",
        ),
    )
    pedestrian_min_green: float = Field(
        default=8.0,
        validation_alias=AliasChoices(
            "PEDESTRIAN_MIN_GREEN",
            "PEDESTRIAN_PHASE__MIN_GREEN",
        ),
    )
    pedestrian_max_green: float = Field(
        default=18.0,
        validation_alias=AliasChoices(
            "PEDESTRIAN_MAX_GREEN",
            "PEDESTRIAN_PHASE__MAX_GREEN",
        ),
    )
    pedestrian_service_rate: float = Field(
        default=0.5,
        validation_alias=AliasChoices(
            "PEDESTRIAN_SERVICE_RATE",
            "PEDESTRIAN_PHASE__SERVICE_RATE",
        ),
    )
    pedestrian_delay_weight: float = Field(
        default=0.6,
        validation_alias=AliasChoices(
            "PEDESTRIAN_DELAY_WEIGHT",
            "PEDESTRIAN_PHASE__DELAY_WEIGHT",
        ),
    )
    pedestrian_queue_weight: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "PEDESTRIAN_QUEUE_WEIGHT",
            "PEDESTRIAN_PHASE__QUEUE_WEIGHT",
        ),
    )
    pedestrian_risk_weight: float = Field(
        default=1.0,
        validation_alias=AliasChoices(
            "PEDESTRIAN_RISK_WEIGHT",
            "PEDESTRIAN_PHASE__RISK_WEIGHT",
        ),
    )
    pedestrian_base_demand: float = Field(
        default=1.0,
        validation_alias=AliasChoices(
            "PEDESTRIAN_BASE_DEMAND",
            "PEDESTRIAN_PHASE__BASE_DEMAND",
        ),
    )

    @property
    def thresholds(self) -> ThresholdSettings:
        return ThresholdSettings(
            risk_threshold=self.risk_threshold,
            distance_threshold_px=self.distance_threshold_px,
            distance_threshold_meters=self.distance_threshold_meters,
            proximity_threshold_meters=self.proximity_threshold_meters,
        )

    @property
    def tracker(self) -> TrackerSettings:
        return TrackerSettings(backend=self.tracker_backend)

    @property
    def optimizer(self) -> OptimizerSettings:
        return OptimizerSettings(
            cycle_min=self.cycle_min,
            cycle_max=self.cycle_max,
            target_cycle=self.target_cycle,
            lambda_risk=self.lambda_risk,
            min_phase_duration=self.min_phase_duration,
            max_phase_duration=self.max_phase_duration,
        )

    @property
    def optimizer_weights(self) -> OptimizerWeights:
        return OptimizerWeights(
            queue_weight=self.optimizer_queue_weight,
            risk_weight=self.optimizer_risk_weight,
        )

    @property
    def api(self) -> ApiSettings:
        return ApiSettings(
            max_upload_size_bytes=self.max_upload_size_bytes,
            max_video_duration_seconds=self.max_video_duration_seconds,
            max_response_items=self.max_response_items,
            video_job_workers=self.video_job_workers,
        )

    @property
    def model_paths(self) -> ModelPathSettings:
        return ModelPathSettings(
            yolo_model_path=self.yolo_model_path,
            lstm_model_path=self.lstm_model_path,
            roi_config_path=self.roi_config_path,
            scene_calibration_path=self.scene_calibration_path,
        )

    @property
    def paths(self) -> PathSettings:
        jobs_dir = self.jobs_dir or (self.results_dir / "jobs")
        return PathSettings(
            base_dir=BASE_DIR,
            upload_dir=self.upload_dir,
            results_dir=self.results_dir,
            jobs_dir=jobs_dir,
        )

    @property
    def logging(self) -> LoggingSettings:
        return LoggingSettings(level=self.log_level, fmt=self.log_format)

    @property
    def pedestrian_phase(self) -> PedestrianPhaseSettings:
        return PedestrianPhaseSettings(
            enabled=self.pedestrian_phase_enabled,
            name=self.pedestrian_phase_name,
            min_green=self.pedestrian_min_green,
            max_green=self.pedestrian_max_green,
            service_rate=self.pedestrian_service_rate,
            delay_weight=self.pedestrian_delay_weight,
            queue_weight=self.pedestrian_queue_weight,
            risk_weight=self.pedestrian_risk_weight,
            base_demand=self.pedestrian_base_demand,
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
