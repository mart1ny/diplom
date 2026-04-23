import pytest

from scripts.traffic_optimizer import PhaseOptimizer


def make_optimizer(**kwargs) -> PhaseOptimizer:
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
        **kwargs,
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


def test_lp_optimizer_respects_separate_queue_and_risk_weights() -> None:
    optimizer = make_optimizer(queue_weight=0.25, risk_weight=12.0)

    plan = optimizer.optimize({"north": 20.0, "east": 8.0}, {"north": 0.0, "east": 0.8})

    assert plan["weights"]["queue_weight"] == pytest.approx(0.25)
    assert plan["weights"]["risk_weight"] == pytest.approx(12.0)
    assert plan["durations"]["east"] > plan["durations"]["north"]


def test_lp_optimizer_damps_switching_between_iterations() -> None:
    optimizer = make_optimizer(
        switch_penalty=5.0,
        phase_hold_steps=2,
        active_phase_min_share=0.9,
    )

    first_plan = optimizer.optimize({"north": 24.0, "east": 2.0}, {"north": 0.0, "east": 0.0})
    second_plan = optimizer.optimize({"north": 10.0, "east": 11.0}, {"north": 0.0, "east": 0.0})

    assert first_plan["active_phase"] == "north"
    assert second_plan["active_phase"] == "north"
    assert second_plan["phase_switched"] is False
    assert second_plan["switch_penalty_applied"] >= 0.0


def test_lp_optimizer_supports_pedestrian_phase() -> None:
    optimizer = PhaseOptimizer(
        {
            "north": {
                "phase_type": "vehicle",
                "min_green": 5.0,
                "max_green": 50.0,
                "service_rate": 1.0,
                "delay_weight": 1.0,
            },
            "pedestrian": {
                "phase_type": "pedestrian",
                "min_green": 8.0,
                "max_green": 20.0,
                "service_rate": 0.5,
                "delay_weight": 0.5,
                "queue_weight": 0.0,
                "risk_weight": 0.0,
                "base_demand": 6.0,
            },
        },
        cycle_bounds=(20.0, 80.0),
        fixed_loss_per_cycle=10.0,
    )

    plan = optimizer.optimize({"north": 10.0}, {"north": 0.0})

    assert plan["phase_types"]["pedestrian"] == "pedestrian"
    assert plan["base_demand"]["pedestrian"] == pytest.approx(6.0)
    assert plan["durations"]["pedestrian"] >= 8.0
