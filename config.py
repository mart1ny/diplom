"""
Global configuration flags for SUMO-driven optimization experiments.

The defaults are conservative enough for most SUMO junctions, but each value can
be overridden through environment variables if a deployment requires tuning:

- MIN_PHASE_DURATION: minimal green phase length (seconds).
- MAX_PHASE_DURATION: hard cap for any single phase (seconds).
- CYCLE_TIME: desired full signal cycle (seconds).
- PROXIMITY_THRESHOLD: distance filter for near-miss detection (SUMO meters).
"""

from __future__ import annotations

import os

MIN_PHASE_DURATION = int(os.getenv("MIN_PHASE_DURATION", 10))
MAX_PHASE_DURATION = int(os.getenv("MAX_PHASE_DURATION", 60))
CYCLE_TIME = int(os.getenv("CYCLE_TIME", 90))
PROXIMITY_THRESHOLD = float(os.getenv("PROXIMITY_THRESHOLD", 35.0))

__all__ = [
    "MIN_PHASE_DURATION",
    "MAX_PHASE_DURATION",
    "CYCLE_TIME",
    "PROXIMITY_THRESHOLD",
]
