from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np


DEFAULT_PHASE_CONFIG = {
    "north": {"min_green": 0.05, "max_green": 0.4, "saturation_flow": 0.25},
    "south": {"min_green": 0.05, "max_green": 0.4, "saturation_flow": 0.25},
    "east": {"min_green": 0.05, "max_green": 0.4, "saturation_flow": 0.25},
    "west": {"min_green": 0.05, "max_green": 0.4, "saturation_flow": 0.25},
}


class PhaseOptimizer:
    """
    Простая LP-модель: распределяем доли зелёного между подходами и длину цикла.
    Цель — минимизировать остаток очереди + штраф за риск.
    """

    def __init__(
        self,
        phase_config: Dict[str, Dict[str, float]],
        cycle_bounds: Tuple[float, float] = (40.0, 90.0),
        lambda_risk: float = 5.0,
        delay_weights: Optional[Dict[str, float]] = None,
    ):
        self.phase_config = phase_config
        self.approaches = list(phase_config.keys())
        self.cycle_min, self.cycle_max = cycle_bounds
        self.lambda_risk = lambda_risk
        self.delay_weights = delay_weights or {a: 1.0 for a in self.approaches}

    def _build_vectors(self, queues: Dict[str, float], risks: Dict[str, float]):
        q = np.array([queues.get(a, 0.0) for a in self.approaches], dtype=np.float32)
        r = np.array([risks.get(a, 0.0) for a in self.approaches], dtype=np.float32)
        w = np.array([self.delay_weights.get(a, 1.0) for a in self.approaches], dtype=np.float32)
        mins = np.array([self.phase_config[a]["min_green"] for a in self.approaches], dtype=np.float32)
        maxs = np.array([self.phase_config[a]["max_green"] for a in self.approaches], dtype=np.float32)
        service = np.array([self.phase_config[a]["saturation_flow"] for a in self.approaches], dtype=np.float32)
        return q, r, w, mins, maxs, service

    def optimize(self, queues: Dict[str, float], risks: Optional[Dict[str, float]] = None):
        risks = risks or {}
        q, r, w, mins, maxs, service = self._build_vectors(queues, risks)
        n = len(self.approaches)

        g = cp.Variable(n)
        s = cp.Variable(n)
        cycle = cp.Variable()

        constraints = [
            cp.sum(g) == 1.0,
            cycle >= self.cycle_min,
            cycle <= self.cycle_max,
            s >= 0,
        ]
        for i in range(n):
            constraints += [
                g[i] >= float(mins[i]),
                g[i] <= float(maxs[i]),
                s[i] >= q[i] - service[i] * g[i] * cycle,
            ]

        objective = cp.Minimize(cp.sum(cp.multiply(w, s)) + self.lambda_risk * cp.sum(cp.multiply(r, g)))
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS, warm_start=True)
        except cp.SolverError:
            problem.solve(solver=cp.SCS, warm_start=True)

        if g.value is None or cycle.value is None:
            # Фоллбек: равномерное распределение
            equal_share = 1.0 / n
            greens = {a: equal_share for a in self.approaches}
            return {
                "greens": greens,
                "cycle": float(self.cycle_min),
                "residual_queue": {a: float(q[i]) for i, a in enumerate(self.approaches)},
                "status": "fallback",
            }

        greens = {a: float(max(min(g.value[i], maxs[i]), mins[i])) for i, a in enumerate(self.approaches)}
        residual = {a: float(max(s.value[i], 0.0)) for i, a in enumerate(self.approaches)}
        return {
            "greens": greens,
            "cycle": float(cycle.value),
            "residual_queue": residual,
            "status": problem.status,
        }

