from scripts.tracker_backends import (
    TrackerBackend,
    detection_centers,
    normalize_tracker_backend,
    tracked_centers,
)


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    def __init__(self, cls, xyxy, track_id=None):
        self.cls = cls
        self.xyxy = [xyxy]
        self.id = track_id


class FakeResults:
    def __init__(self, boxes):
        self.boxes = boxes


def test_normalize_tracker_backend() -> None:
    assert normalize_tracker_backend("bytetrack") is TrackerBackend.BYTETRACK
    assert normalize_tracker_backend(TrackerBackend.SIMPLE) is TrackerBackend.SIMPLE


def test_detection_centers_filters_by_class() -> None:
    results = FakeResults(
        [
            FakeBox(cls=2, xyxy=[0, 0, 10, 20]),
            FakeBox(cls=0, xyxy=[100, 100, 120, 120]),
        ]
    )

    assert detection_centers(results, class_id=2) == [(5.0, 10.0)]


def test_tracked_centers_uses_bytetrack_ids() -> None:
    results = FakeResults(
        [
            FakeBox(cls=2, xyxy=[0, 0, 10, 20], track_id=FakeScalar(7)),
            FakeBox(cls=2, xyxy=[10, 10, 30, 30]),
            FakeBox(cls=0, xyxy=[100, 100, 120, 120], track_id=FakeScalar(99)),
        ]
    )

    assert tracked_centers(results, class_id=2) == {7: (5.0, 10.0)}
