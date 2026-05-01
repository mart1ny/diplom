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
    cycle_bounds = kwargs.pop("cycle_bounds", (20.0, 80.0))
    fixed_loss_per_cycle = kwargs.pop("fixed_loss_per_cycle", 10.0)
    lambda_risk = kwargs.pop("lambda_risk", 5.0)
    return PhaseOptimizer(
        phase_config,
        cycle_bounds=cycle_bounds,
        lambda_risk=lambda_risk,
        fixed_loss_per_cycle=fixed_loss_per_cycle,
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


def test_optimizer_internal_helpers_cover_smoothing_and_phase_state() -> None:
    optimizer = make_optimizer()
    loads = optimizer._smooth_loads(optimizer._build_effective_demand({"north": 4.0}, {})[6])
    assert loads.shape[0] == 2
    assert (
        optimizer._current_active_hold_target(
            mins=optimizer._build_vectors({}, {})[3],
            maxs=optimizer._build_vectors({}, {})[4],
        )
        is None
    )

    active_phase, switched = optimizer._update_phase_state(
        optimizer._adjust_to_sum(
            values=pytest.importorskip("numpy").array([6.0, 5.0], dtype="float32"),
            target_sum=11.0,
            mins=pytest.importorskip("numpy").array([5.0, 5.0], dtype="float32"),
            maxs=pytest.importorskip("numpy").array([10.0, 10.0], dtype="float32"),
        )
    )
    assert active_phase == "north"
    assert switched is False

    active_phase, switched = optimizer._update_phase_state(
        pytest.importorskip("numpy").array([5.0, 9.0], dtype="float32")
    )
    assert active_phase == "east"
    assert switched is True

    empty_active, empty_switched = optimizer._update_phase_state(
        pytest.importorskip("numpy").array([], dtype="float32")
    )
    assert empty_active is None
    assert empty_switched is False


def test_optimizer_adjust_to_sum_and_heuristic_paths() -> None:
    np = pytest.importorskip("numpy")
    optimizer = make_optimizer(cycle_bounds=(5.0, 20.0), fixed_loss_per_cycle=15.0)
    adjusted_zero = optimizer._adjust_to_sum(
        values=np.array([4.0, 5.0], dtype=np.float32),
        target_sum=0.0,
        mins=np.array([1.0, 1.0], dtype=np.float32),
        maxs=np.array([10.0, 10.0], dtype=np.float32),
    )
    assert adjusted_zero.tolist() == [0.0, 0.0]

    plan = optimizer._heuristic_optimize({"north": 0.0, "east": 0.0}, {})
    assert plan["status"] == "adaptive_heuristic"
    assert plan["cycle"] >= sum(plan["durations"].values())


def test_optimizer_falls_back_when_lp_problem_is_unresolved(monkeypatch) -> None:
    optimizer = make_optimizer()

    monkeypatch.setattr(optimizer, "_solve_problem", lambda problem: "solver_failed")

    plan = optimizer.optimize({"north": 3.0, "east": 2.0}, {"north": 0.0, "east": 0.0})

    assert plan["optimizer"] == "heuristic"
    assert plan["status"] == "fallback_heuristic"
    assert plan["solver_status"] == "solver_failed"


def test_optimizer_handles_inverted_cycle_bounds() -> None:
    optimizer = make_optimizer(cycle_bounds=(80.0, 20.0))

    plan = optimizer.optimize({"north": 8.0, "east": 8.0}, {"north": 0.0, "east": 0.0})

    assert plan["cycle"] >= 20.0


def test_optimizer_uses_forecast_queue_when_enabled() -> None:
    optimizer = make_optimizer(demand_forecast_alpha=0.4)

    plan = optimizer.optimize(
        {"north": 8.0, "east": 8.0},
        {"north": 0.0, "east": 0.0},
        forecast_queues={"north": 6.0, "east": 18.0},
    )

    assert plan["forecast_queue"]["east"] == pytest.approx(18.0)
    assert plan["effective_queue"]["east"] > plan["effective_queue"]["north"]
    assert plan["durations"]["east"] > plan["durations"]["north"]
    assert plan["weights"]["forecast_alpha"] == pytest.approx(0.4)
