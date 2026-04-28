from __future__ import annotations

from scripts.settings import get_settings

_settings = get_settings()

MIN_PHASE_DURATION = _settings.optimizer.min_phase_duration
MAX_PHASE_DURATION = _settings.optimizer.max_phase_duration
CYCLE_TIME = _settings.optimizer.target_cycle
PROXIMITY_THRESHOLD = _settings.thresholds.proximity_threshold_meters

__all__ = [
    "MIN_PHASE_DURATION",
    "MAX_PHASE_DURATION",
    "CYCLE_TIME",
    "PROXIMITY_THRESHOLD",
]
