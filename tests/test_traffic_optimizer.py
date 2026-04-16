import pytest

from scripts.traffic_optimizer import PhaseOptimizer


def make_optimizer() -> PhaseOptimizer:
    phase_config = {
        "north": {
            "min_green": 5.0,
            "max_green": 60.0,
            "service_rate": 1.0,
            "delay_weight": 1.0,
        },
        "east": {
            "min_green": 5.0,
            "max_green": 60.0,
            "service_rate": 1.0,
            "delay_weight": 1.0,
        },
    }
    return PhaseOptimizer(
        phase_config,
        cycle_bounds=(20.0, 80.0),
        lambda_risk=5.0,
        fixed_loss_per_cycle=10.0,
    )


def test_lp_optimizer_returns_feasible_solution() -> None:
    optimizer = make_optimizer()

    plan = optimizer.optimize({"north": 12.0, "east": 6.0}, {"north": 0.1, "east": 0.0})

    total_green = sum(plan["durations"].values())
    assert plan["optimizer"] == "lp"
    assert plan["status"] == "optimal_lp"
    assert plan["solver_status"] in {"optimal", "optimal_inaccurate"}
    assert total_green + optimizer.fixed_loss_per_cycle == pytest.approx(plan["cycle"], abs=1e-3)
    assert plan["durations"]["north"] >= 5.0
    assert plan["durations"]["east"] >= 5.0


def test_lp_optimizer_prefers_heavier_queue() -> None:
    optimizer = make_optimizer()

    plan = optimizer.optimize({"north": 20.0, "east": 4.0}, {"north": 0.0, "east": 0.0})

    assert plan["durations"]["north"] > plan["durations"]["east"]
    assert plan["residual_queue"]["north"] >= 0.0


def test_lp_optimizer_uses_risk_signal_in_allocation() -> None:
    optimizer = make_optimizer()

    plan = optimizer.optimize({"north": 8.0, "east": 8.0}, {"north": 0.0, "east": 0.8})

    assert plan["durations"]["east"] > plan["durations"]["north"]
    assert plan["risk"]["east"] == pytest.approx(0.8)


def test_lp_optimizer_increases_cycle_for_heavier_load() -> None:
    low_load = make_optimizer().optimize({"north": 2.0, "east": 2.0}, {"north": 0.0, "east": 0.0})
    high_load = make_optimizer().optimize(
        {"north": 30.0, "east": 30.0}, {"north": 0.0, "east": 0.0}
    )

    assert high_load["cycle"] > low_load["cycle"]
