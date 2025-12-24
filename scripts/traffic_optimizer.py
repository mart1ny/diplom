from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CYCLE_TIME, MAX_PHASE_DURATION, MIN_PHASE_DURATION


DEFAULT_PHASE_CONFIG = {
    "north": {"min_green": MIN_PHASE_DURATION, "max_green": MAX_PHASE_DURATION},
    "south": {"min_green": MIN_PHASE_DURATION, "max_green": MAX_PHASE_DURATION},
    "east": {"min_green": MIN_PHASE_DURATION, "max_green": MAX_PHASE_DURATION},
    "west": {"min_green": MIN_PHASE_DURATION, "max_green": MAX_PHASE_DURATION},
}


class PhaseOptimizer:
    """
    Адаптивное перераспределение зелёного по тем же принципам, что и SUMO-оптимизатор:
    - используем очереди (и риск) как показатель загрузки подхода;
    - сглаживаем изменения;
    - фиксируем длину цикла и ограничиваем длительности фаз.
    """

    def __init__(
        self,
        phase_config: Dict[str, Dict[str, float]],
        cycle_bounds: Tuple[float, float] = (40.0, 90.0),
        lambda_risk: float = 5.0,
        delay_weights: Optional[Dict[str, float]] = None,
        smoothing_alpha: float = 0.2,
        fixed_loss_per_cycle: float = 10.0,
        max_change_ratio: float = 0.3,
    ):
        self.phase_config = phase_config
        self.approaches = list(phase_config.keys())
        self.cycle_min, self.cycle_max = cycle_bounds
        self.lambda_risk = lambda_risk
        self.delay_weights = delay_weights or {a: 1.0 for a in self.approaches}
        self.smoothing_alpha = smoothing_alpha
        self.fixed_loss_per_cycle = fixed_loss_per_cycle
        self.max_change_ratio = max_change_ratio

        self.target_cycle = float(
            min(max(CYCLE_TIME, self.cycle_min), self.cycle_max)
            if self.cycle_min > 0 and self.cycle_max > 0
            else CYCLE_TIME
        )
        self._smoothed_loads: Optional[np.ndarray] = None
        self._prev_durations: Optional[np.ndarray] = None

    def _build_vectors(self, queues: Dict[str, float], risks: Dict[str, float]):
        q = np.array([queues.get(a, 0.0) for a in self.approaches], dtype=np.float32)
        r = np.array([risks.get(a, 0.0) for a in self.approaches], dtype=np.float32)
        w = np.array([self.delay_weights.get(a, 1.0) for a in self.approaches], dtype=np.float32)
        mins = np.array(
            [self.phase_config[a].get("min_green", MIN_PHASE_DURATION) for a in self.approaches],
            dtype=np.float32,
        )
        maxs = np.array(
            [self.phase_config[a].get("max_green", MAX_PHASE_DURATION) for a in self.approaches],
            dtype=np.float32,
        )
        return q, r, w, mins, maxs

    def _adjust_to_sum(self, values: np.ndarray, target_sum: float, mins: np.ndarray, maxs: np.ndarray):
        values = np.clip(values, mins, maxs).astype(np.float32)
        if target_sum <= 0:
            total = values.sum()
            return values if total == 0 else values * 0.0

        for _ in range(50):
            current = float(values.sum())
            diff = target_sum - current
            if abs(diff) < 1e-2:
                break
            if diff > 0:
                free = np.where(values < maxs - 1e-3)[0]
            else:
                free = np.where(values > mins + 1e-3)[0]
            if len(free) == 0:
                break
            allocation = diff / len(free)
            values[free] += allocation
            values = np.clip(values, mins, maxs)
        return values

    def optimize(self, queues: Dict[str, float], risks: Optional[Dict[str, float]] = None):
        risks = risks or {}
        q, r, w, mins, maxs = self._build_vectors(queues, risks)
        n = len(self.approaches)

        loads = (q * w) + self.lambda_risk * r
        loads = np.maximum(loads, 0.1)

        if self._smoothed_loads is None or len(self._smoothed_loads) != n:
            smoothed = loads.copy()
        else:
            smoothed = (1 - self.smoothing_alpha) * self._smoothed_loads + self.smoothing_alpha * loads
        self._smoothed_loads = smoothed

        total_demand = float(smoothed.sum())
        if total_demand <= 0:
            smoothed = np.ones_like(smoothed)
            total_demand = float(smoothed.sum())

        available_green = self.target_cycle - self.fixed_loss_per_cycle
        min_total = float(np.sum(mins))
        if available_green < min_total:
            available_green = min_total

        effective_green = (smoothed / total_demand) * available_green
        effective_green = np.clip(effective_green, mins, maxs)
        effective_green = self._adjust_to_sum(effective_green, available_green, mins, maxs)

        if self._prev_durations is not None and len(self._prev_durations) == n:
            lower_bounds = self._prev_durations * (1 - self.max_change_ratio)
            upper_bounds = self._prev_durations * (1 + self.max_change_ratio)
            effective_green = np.minimum(upper_bounds, np.maximum(lower_bounds, effective_green))
            effective_green = np.clip(effective_green, mins, maxs)
            effective_green = self._adjust_to_sum(effective_green, available_green, mins, maxs)

        self._prev_durations = effective_green.copy()

        greens = {
            approach: float(effective_green[i] / self.target_cycle)
            for i, approach in enumerate(self.approaches)
        }
        residual = {approach: float(max(q[i] - effective_green[i], 0.0)) for i, approach in enumerate(self.approaches)}

        return {
            "greens": greens,
            "cycle": float(self.target_cycle),
            "durations": {approach: float(effective_green[i]) for i, approach in enumerate(self.approaches)},
            "residual_queue": residual,
            "status": "adaptive",
        }
