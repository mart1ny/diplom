from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np

from config import CYCLE_TIME, MAX_PHASE_DURATION, MIN_PHASE_DURATION
from scripts.observability import metric_counter, metric_histogram


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    phase_type: str = "vehicle"
    min_green: float = float(MIN_PHASE_DURATION)
    max_green: float = float(MAX_PHASE_DURATION)
    service_rate: float = 1.0
    delay_weight: float = 1.0
    queue_weight: float = 1.0
    risk_weight: float = 1.0
    base_demand: float = 0.0


DEFAULT_PHASE_CONFIG = {
    "north": {
        "phase_type": "vehicle",
        "min_green": MIN_PHASE_DURATION,
        "max_green": MAX_PHASE_DURATION,
        "service_rate": 1.0,
        "delay_weight": 1.0,
        "queue_weight": 1.0,
        "risk_weight": 1.0,
    },
    "south": {
        "phase_type": "vehicle",
        "min_green": MIN_PHASE_DURATION,
        "max_green": MAX_PHASE_DURATION,
        "service_rate": 1.0,
        "delay_weight": 1.0,
        "queue_weight": 1.0,
        "risk_weight": 1.0,
    },
    "east": {
        "phase_type": "vehicle",
        "min_green": MIN_PHASE_DURATION,
        "max_green": MAX_PHASE_DURATION,
        "service_rate": 1.0,
        "delay_weight": 1.0,
        "queue_weight": 1.0,
        "risk_weight": 1.0,
    },
    "west": {
        "phase_type": "vehicle",
        "min_green": MIN_PHASE_DURATION,
        "max_green": MAX_PHASE_DURATION,
        "service_rate": 1.0,
        "delay_weight": 1.0,
        "queue_weight": 1.0,
        "risk_weight": 1.0,
    },
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
        queue_weight: float = 1.0,
        risk_weight: Optional[float] = None,
        smoothing_alpha: float = 0.2,
        fixed_loss_per_cycle: float = 10.0,
        max_change_ratio: float = 0.3,
        switch_penalty: float = 0.35,
        phase_hold_steps: int = 2,
        active_phase_min_share: float = 0.7,
    ):
        self.phase_specs = self._normalize_phase_specs(phase_config)
        self.phase_config = {
            spec.name: {
                "phase_type": spec.phase_type,
                "min_green": spec.min_green,
                "max_green": spec.max_green,
                "service_rate": spec.service_rate,
                "delay_weight": spec.delay_weight,
                "queue_weight": spec.queue_weight,
                "risk_weight": spec.risk_weight,
                "base_demand": spec.base_demand,
            }
            for spec in self.phase_specs
        }
        self.approaches = [spec.name for spec in self.phase_specs]
        self.cycle_min, self.cycle_max = cycle_bounds
        self.lambda_risk = lambda_risk
        self.delay_weights = delay_weights or {a: 1.0 for a in self.approaches}
        self.queue_weight = queue_weight
        self.risk_weight = lambda_risk if risk_weight is None else risk_weight
        self.smoothing_alpha = smoothing_alpha
        self.fixed_loss_per_cycle = fixed_loss_per_cycle
        self.max_change_ratio = max_change_ratio
        self.switch_penalty = switch_penalty
        self.phase_hold_steps = max(0, int(phase_hold_steps))
        self.active_phase_min_share = float(np.clip(active_phase_min_share, 0.0, 1.0))

        self.target_cycle = float(
            min(max(CYCLE_TIME, self.cycle_min), self.cycle_max)
            if self.cycle_min > 0 and self.cycle_max > 0
            else CYCLE_TIME
        )
        self.cycle_penalty = 0.15
        self.stability_penalty = 0.05
        self._smoothed_loads: Optional[np.ndarray] = None
        self._prev_durations: Optional[np.ndarray] = None
        self._active_phase_index: Optional[int] = None
        self._phase_hold_remaining = 0
        self._phase_switches = 0

    def _current_active_hold_target(self, mins: np.ndarray, maxs: np.ndarray) -> Optional[float]:
        if (
            self._active_phase_index is None
            or self._prev_durations is None
            or self._active_phase_index >= len(self._prev_durations)
        ):
            return None
        previous_duration = float(self._prev_durations[self._active_phase_index])
        bounded = min(previous_duration, float(maxs[self._active_phase_index]))
        hold_target = max(float(mins[self._active_phase_index]), bounded * self.active_phase_min_share)
        return hold_target

    def _update_phase_state(self, effective_green: np.ndarray) -> tuple[Optional[str], bool]:
        if len(effective_green) == 0:
            self._active_phase_index = None
            self._phase_hold_remaining = 0
            return None, False

        next_active_index = int(np.argmax(effective_green))
        switched = (
            self._active_phase_index is not None and next_active_index != self._active_phase_index
        )
        if switched:
            self._phase_switches += 1
            self._phase_hold_remaining = self.phase_hold_steps
        elif self._phase_hold_remaining > 0:
            self._phase_hold_remaining -= 1

        self._active_phase_index = next_active_index
        return self.approaches[next_active_index], switched

    def _normalize_phase_specs(self, phase_config: Dict[str, Dict[str, float]]) -> list[PhaseSpec]:
        specs: list[PhaseSpec] = []
        for name, raw in phase_config.items():
            specs.append(
                PhaseSpec(
                    name=name,
                    phase_type=str(raw.get("phase_type", "vehicle")),
                    min_green=float(raw.get("min_green", MIN_PHASE_DURATION)),
                    max_green=float(raw.get("max_green", MAX_PHASE_DURATION)),
                    service_rate=float(raw.get("service_rate", 1.0)),
                    delay_weight=float(raw.get("delay_weight", 1.0)),
                    queue_weight=float(raw.get("queue_weight", 1.0)),
                    risk_weight=float(raw.get("risk_weight", 1.0)),
                    base_demand=float(raw.get("base_demand", 0.0)),
                )
            )
        return specs

    def _build_effective_demand(
        self,
        queues: Dict[str, float],
        risks: Dict[str, float],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        q, r, delay_weights, mins, maxs, service_rates, queue_weights, risk_weights, base_demand = (
            self._build_vectors(queues, risks)
        )
        loads = (
            (q * delay_weights * queue_weights * self.queue_weight)
            + (r * risk_weights * self.risk_weight)
            + base_demand
        )
        return (
            q,
            r,
            delay_weights,
            mins,
            maxs,
            service_rates,
            np.maximum(loads, 0.1),
            base_demand,
        )

    def _smooth_loads(self, loads: np.ndarray) -> np.ndarray:
        if self._smoothed_loads is None or len(self._smoothed_loads) != len(loads):
            smoothed = loads.copy()
        else:
            smoothed = (
                1 - self.smoothing_alpha
            ) * self._smoothed_loads + self.smoothing_alpha * loads
        self._smoothed_loads = smoothed
        return smoothed

    def _solve_problem(self, problem: cp.Problem) -> str:
        for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
            try:
                problem.solve(solver=solver, warm_start=True, verbose=False)
            except Exception:
                continue
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return str(problem.status)
        return str(problem.status)

    def _build_vectors(self, queues: Dict[str, float], risks: Dict[str, float]):
        q = np.array([queues.get(a, 0.0) for a in self.approaches], dtype=np.float32)
        r = np.array([risks.get(a, 0.0) for a in self.approaches], dtype=np.float32)
        delay_weights = np.array(
            [
                self.phase_config[a].get("delay_weight", self.delay_weights.get(a, 1.0))
                for a in self.approaches
            ],
            dtype=np.float32,
        )
        mins = np.array(
            [self.phase_config[a].get("min_green", MIN_PHASE_DURATION) for a in self.approaches],
            dtype=np.float32,
        )
        maxs = np.array(
            [self.phase_config[a].get("max_green", MAX_PHASE_DURATION) for a in self.approaches],
            dtype=np.float32,
        )
        service_rates = np.array(
            [self.phase_config[a].get("service_rate", 1.0) for a in self.approaches],
            dtype=np.float32,
        )
        queue_weights = np.array(
            [self.phase_config[a].get("queue_weight", 1.0) for a in self.approaches],
            dtype=np.float32,
        )
        risk_weights = np.array(
            [self.phase_config[a].get("risk_weight", 1.0) for a in self.approaches],
            dtype=np.float32,
        )
        base_demand = np.array(
            [self.phase_config[a].get("base_demand", 0.0) for a in self.approaches],
            dtype=np.float32,
        )
        return q, r, delay_weights, mins, maxs, service_rates, queue_weights, risk_weights, base_demand

    def _adjust_to_sum(
        self, values: np.ndarray, target_sum: float, mins: np.ndarray, maxs: np.ndarray
    ) -> np.ndarray:
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

    def _heuristic_optimize(
        self,
        queues: Dict[str, float],
        risks: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        risks = risks or {}
        q, _, _, mins, maxs, _, loads, base_demand = self._build_effective_demand(queues, risks)
        smoothed = self._smooth_loads(loads)

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

        if self._prev_durations is not None and len(self._prev_durations) == len(self.approaches):
            lower_bounds = self._prev_durations * (1 - self.max_change_ratio)
            upper_bounds = self._prev_durations * (1 + self.max_change_ratio)
            effective_green = np.minimum(
                upper_bounds,
                np.maximum(lower_bounds, effective_green),
            )
            effective_green = np.clip(effective_green, mins, maxs)
            effective_green = self._adjust_to_sum(effective_green, available_green, mins, maxs)

        switch_penalty_applied = 0.0
        hold_target = self._current_active_hold_target(mins, maxs)
        if (
            hold_target is not None
            and self._active_phase_index is not None
            and self._phase_hold_remaining > 0
            and self._active_phase_index < len(effective_green)
        ):
            current_value = float(effective_green[self._active_phase_index])
            if current_value < hold_target:
                switch_penalty_applied = hold_target - current_value
                effective_green[self._active_phase_index] = hold_target
                effective_green = np.clip(effective_green, mins, maxs)
                effective_green = self._adjust_to_sum(effective_green, available_green, mins, maxs)

        self._prev_durations = effective_green.copy()
        active_phase, switched = self._update_phase_state(effective_green)

        greens = {
            approach: float(effective_green[i] / self.target_cycle)
            for i, approach in enumerate(self.approaches)
        }
        residual = {
            approach: float(max(q[i] - effective_green[i], 0.0))
            for i, approach in enumerate(self.approaches)
        }

        return {
            "greens": greens,
            "cycle": float(self.target_cycle),
            "durations": {
                approach: float(effective_green[i]) for i, approach in enumerate(self.approaches)
            },
            "residual_queue": residual,
            "phase_types": {spec.name: spec.phase_type for spec in self.phase_specs},
            "base_demand": {
                approach: float(base_demand[i]) for i, approach in enumerate(self.approaches)
            },
            "weights": {
                "queue_weight": float(self.queue_weight),
                "risk_weight": float(self.risk_weight),
            },
            "active_phase": active_phase,
            "phase_switches": int(self._phase_switches),
            "hold_steps_remaining": int(self._phase_hold_remaining),
            "switch_penalty_applied": float(switch_penalty_applied),
            "phase_switched": switched,
            "status": "adaptive_heuristic",
        }

    def optimize(self, queues: Dict[str, float], risks: Optional[Dict[str, float]] = None):
        risks = risks or {}
        q, r, _, mins, maxs, service_rates, loads, base_demand = self._build_effective_demand(
            queues, risks
        )
        smoothed = self._smooth_loads(loads)
        optimization_started_at = time.perf_counter()

        min_cycle = max(self.cycle_min, float(np.sum(mins)) + self.fixed_loss_per_cycle)
        max_cycle = min(self.cycle_max, float(np.sum(maxs)) + self.fixed_loss_per_cycle)
        if max_cycle < min_cycle:
            max_cycle = min_cycle

        n = len(self.approaches)
        green = cp.Variable(n)
        residual = cp.Variable(n, nonneg=True)
        cycle = cp.Variable()
        constraints = [
            green >= mins,
            green <= maxs,
            cycle >= min_cycle,
            cycle <= max_cycle,
            cp.sum(green) + self.fixed_loss_per_cycle == cycle,
            residual >= smoothed - cp.multiply(service_rates, green),
        ]

        objective_terms = [
            cp.sum(residual),
            self.cycle_penalty * cycle,
        ]
        switch_slack = None

        if self._prev_durations is not None and len(self._prev_durations) == n:
            lower_bounds = np.maximum(mins, self._prev_durations * (1 - self.max_change_ratio))
            upper_bounds = np.minimum(maxs, self._prev_durations * (1 + self.max_change_ratio))
            constraints.extend(
                [
                    green >= lower_bounds,
                    green <= upper_bounds,
                ]
            )
            deviation = cp.Variable(n, nonneg=True)
            constraints.extend(
                [
                    deviation >= green - self._prev_durations,
                    deviation >= self._prev_durations - green,
                ]
            )
            objective_terms.append(self.stability_penalty * cp.sum(deviation))

        hold_target = self._current_active_hold_target(mins, maxs)
        if (
            hold_target is not None
            and self._active_phase_index is not None
            and self._phase_hold_remaining > 0
            and self._active_phase_index < n
        ):
            switch_slack = cp.Variable(nonneg=True)
            constraints.append(switch_slack >= hold_target - green[self._active_phase_index])
            objective_terms.append(
                (self.switch_penalty * (1 + self._phase_hold_remaining)) * switch_slack
            )

        problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
        solver_status = self._solve_problem(problem)
        solve_duration = time.perf_counter() - optimization_started_at
        metric_histogram(
            "traffic_lp_solve_duration_seconds",
            "LP solve duration for traffic phase optimization.",
            solve_duration,
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
            labels={"solver_status": solver_status or "unknown"},
        )
        metric_counter(
            "traffic_lp_optimizations_total",
            "Total number of traffic optimization attempts.",
            labels={"solver_status": solver_status or "unknown"},
        )
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or green.value is None:
            fallback = self._heuristic_optimize(queues, risks)
            fallback["status"] = "fallback_heuristic"
            fallback["optimizer"] = "heuristic"
            fallback["solver_status"] = solver_status
            fallback["solve_duration_seconds"] = solve_duration
            return fallback

        effective_green = np.clip(np.array(green.value).reshape(-1), mins, maxs)
        optimized_cycle = float(cycle.value)
        self._prev_durations = effective_green.copy()
        active_phase, switched = self._update_phase_state(effective_green)

        greens = {
            approach: float(effective_green[i] / optimized_cycle)
            for i, approach in enumerate(self.approaches)
        }
        residual_values = np.maximum(np.array(residual.value).reshape(-1), 0.0)
        residual_queue = {
            approach: float(residual_values[i]) for i, approach in enumerate(self.approaches)
        }
        effective_demand = {
            approach: float(smoothed[i]) for i, approach in enumerate(self.approaches)
        }
        risk_breakdown = {approach: float(r[i]) for i, approach in enumerate(self.approaches)}

        return {
            "greens": greens,
            "cycle": optimized_cycle,
            "durations": {
                approach: float(effective_green[i]) for i, approach in enumerate(self.approaches)
            },
            "residual_queue": residual_queue,
            "effective_demand": effective_demand,
            "risk": risk_breakdown,
            "objective_value": float(problem.value) if problem.value is not None else None,
            "solver_status": solver_status,
            "solve_duration_seconds": solve_duration,
            "target_cycle": self.target_cycle,
            "optimizer": "lp",
            "phase_types": {spec.name: spec.phase_type for spec in self.phase_specs},
            "base_demand": {
                approach: float(base_demand[i]) for i, approach in enumerate(self.approaches)
            },
            "weights": {
                "queue_weight": float(self.queue_weight),
                "risk_weight": float(self.risk_weight),
            },
            "active_phase": active_phase,
            "phase_switches": int(self._phase_switches),
            "hold_steps_remaining": int(self._phase_hold_remaining),
            "switch_penalty_applied": (
                float(switch_slack.value) if switch_slack is not None and switch_slack.value is not None else 0.0
            ),
            "phase_switched": switched,
            "status": "optimal_lp",
        }
