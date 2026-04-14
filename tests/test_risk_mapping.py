import pytest

from scripts.risk_mapping import aggregate_risk_by_approach


class FakeQueueCounter:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_track_approach(self, track_id):
        return self.mapping.get(track_id)


def test_aggregate_risk_by_approach_works_without_metric_collection() -> None:
    queue_counter = FakeQueueCounter({1: "north", 2: "east", 3: "north"})
    events = [
        {"id1": 1, "id2": 2, "risk_score": 0.8},
        {"id1": 3, "id2": 99, "risk_score": 0.4},
    ]

    risk = aggregate_risk_by_approach(events, queue_counter, ["north", "east", "south"])

    assert risk["north"] == pytest.approx(0.6)
    assert risk["east"] == pytest.approx(0.4)
    assert risk["south"] == pytest.approx(0.0)
