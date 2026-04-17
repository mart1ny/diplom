import pytest

from scripts.rajectory_analysis import RiskAnalyzer, TrajectoryAnalyzer


def make_analyzer(fps: float = 10.0) -> RiskAnalyzer:
    trajectory_analyzer = TrajectoryAnalyzer()
    trajectory_analyzer.add_position(1, 0, 0.0, 0.0)
    trajectory_analyzer.add_position(1, 1, 1.0, 0.0)
    trajectory_analyzer.add_position(2, 0, 10.0, 0.0)
    trajectory_analyzer.add_position(2, 1, 9.0, 0.0)
    return RiskAnalyzer(
        trajectory_analyzer,
        fps=fps,
        use_lstm=False,
        min_conflict_speed=0.01,
        time_horizon=5.0,
    )


def test_compute_ttc_pet_uses_seconds() -> None:
    analyzer = make_analyzer(fps=10.0)

    ttc, pet, _ = analyzer.compute_ttc_pet(1, 2)

    assert ttc == pytest.approx(0.4, rel=1e-3)
    assert pet == pytest.approx(0.4, rel=1e-3)


def test_event_payload_separates_physics_and_risk() -> None:
    analyzer = make_analyzer(fps=10.0)

    events = analyzer.analyze_and_get_events(distance_threshold=20.0, risk_threshold=0.01)

    assert len(events) == 1
    event = events[0]
    assert event["risk_score"] == pytest.approx(event["risk_score_heuristic"])
    assert event["risk_score_model"] is None
    assert event["risk_score_source"] == "heuristic"
    assert event["physics"]["ttc_seconds"] == pytest.approx(0.4, rel=1e-3)
    assert event["physics"]["closing_speed_px_s"] == pytest.approx(20.0, rel=1e-3)
