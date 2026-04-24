import pytest

import scripts.rajectory_analysis as trajectory_module
from scripts.rajectory_analysis import RiskAnalyzer, TrajectoryAnalyzer
from scripts.scene_calibration import SceneCalibration


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


def test_metric_calibration_projects_near_miss_into_meters() -> None:
    trajectory_analyzer = TrajectoryAnalyzer()
    trajectory_analyzer.add_position(1, 0, 0.0, 0.0)
    trajectory_analyzer.add_position(1, 1, 1.0, 0.0)
    trajectory_analyzer.add_position(2, 0, 8.0, 0.0)
    trajectory_analyzer.add_position(2, 1, 7.0, 0.0)
    analyzer = RiskAnalyzer(
        trajectory_analyzer,
        fps=10.0,
        use_lstm=False,
        min_conflict_speed=0.01,
        time_horizon=5.0,
        scene_calibration=SceneCalibration(meters_per_pixel=0.5, distance_threshold_meters=6.0),
    )

    events = analyzer.analyze_and_get_events(
        distance_threshold=20.0,
        distance_threshold_meters=6.0,
        risk_threshold=0.01,
    )

    assert len(events) == 1
    event = events[0]
    assert event["physics"]["distance_m"] == pytest.approx(3.0, rel=1e-3)
    assert event["physics"]["closing_speed_mps"] == pytest.approx(10.0, rel=1e-3)
    assert event["physics"]["unit_system"] == "metric"


def test_trajectory_analyzer_helpers_cover_motion_and_pruning() -> None:
    analyzer = TrajectoryAnalyzer()
    analyzer.add_position(1, 0, 0.0, 0.0)
    analyzer.add_position(1, 1, 3.0, 4.0)
    analyzer.add_position(1, 2, 6.0, 8.0)
    analyzer.add_position(2, 0, 7.0, 8.0)
    analyzer.add_position(2, 2, 7.0, 8.0)
    analyzer.trajectories[3] = []

    assert analyzer.get_speed(1) == pytest.approx([5.0, 5.0])
    assert analyzer.get_direction(1)[0] == pytest.approx(0.927295, rel=1e-5)
    assert analyzer.get_last_position(1) == (6.0, 8.0)
    assert analyzer.get_last_position(99) is None

    candidates = analyzer.get_conflict_candidates(threshold=2.0)
    assert candidates == [(1, 2, pytest.approx(1.0))]

    velocity, position, last_frame = analyzer.estimate_velocity(1, window=2)
    assert velocity.tolist() == pytest.approx([3.0, 4.0])
    assert position.tolist() == pytest.approx([6.0, 8.0])
    assert last_frame == 2

    analyzer.prune(current_frame=3, max_age_frames=1, max_history=2, active_ids={1})
    assert 2 not in analyzer.trajectories
    assert 3 not in analyzer.trajectories
    assert len(analyzer.trajectories[1]) == 2


def test_metric_distance_threshold_filters_out_far_conflicts() -> None:
    trajectory_analyzer = TrajectoryAnalyzer()
    trajectory_analyzer.add_position(1, 0, 0.0, 0.0)
    trajectory_analyzer.add_position(1, 1, 1.0, 0.0)
    trajectory_analyzer.add_position(2, 0, 8.0, 0.0)
    trajectory_analyzer.add_position(2, 1, 7.0, 0.0)
    analyzer = RiskAnalyzer(
        trajectory_analyzer,
        fps=10.0,
        use_lstm=False,
        min_conflict_speed=0.01,
        time_horizon=5.0,
        scene_calibration=SceneCalibration(meters_per_pixel=1.0, distance_threshold_meters=2.0),
    )

    events = analyzer.analyze_and_get_events(
        distance_threshold=20.0,
        distance_threshold_meters=2.0,
        risk_threshold=0.01,
    )

    assert events == []


def test_use_lstm_requires_torch(monkeypatch) -> None:
    monkeypatch.setattr(trajectory_module, "torch", None)

    with pytest.raises(RuntimeError, match="PyTorch is required"):
        RiskAnalyzer(TrajectoryAnalyzer(), use_lstm=True)


def test_risk_analyzer_misc_helpers_and_non_conflict_paths() -> None:
    trajectory_analyzer = TrajectoryAnalyzer()
    trajectory_analyzer.add_position(1, 0, 0.0, 0.0)
    trajectory_analyzer.add_position(1, 0, 1.0, 0.0)
    trajectory_analyzer.add_position(1, 1, 2.0, 0.0)
    analyzer = RiskAnalyzer(trajectory_analyzer, fps=5.0, use_lstm=False)

    assert analyzer._frames_to_seconds(10) == pytest.approx(2.0)
    assert analyzer._project_position([1.0, 2.0]).tolist() == pytest.approx([1.0, 2.0])
    assert analyzer._project_velocity(
        trajectory_module.np.array([0.0, 0.0], dtype=trajectory_module.np.float32),
        trajectory_module.np.array([1.0, 0.0], dtype=trajectory_module.np.float32),
    ).tolist() == pytest.approx([5.0, 0.0])
    assert (
        analyzer._metric_distance(
            trajectory_module.np.array([0.0, 0.0]), trajectory_module.np.array([1.0, 1.0])
        )
        is None
    )
    assert analyzer._build_sequence(1, 99) is None
    assert analyzer.compute_ttc_pet(1, 99)[0] == float("inf")
    assert analyzer.risk_score(1, 99) == (0.0, None, None)
