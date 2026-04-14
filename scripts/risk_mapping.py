from __future__ import annotations

from typing import Iterable, Mapping


def aggregate_risk_by_approach(
    events: list[Mapping[str, object]],
    queue_counter,
    approaches: Iterable[str],
) -> dict[str, float]:
    risk_by_approach = {name: 0.0 for name in approaches}
    for event in events:
        risk_score = float(event.get("risk_score", 0.0) or 0.0)
        if risk_score <= 0:
            continue
        approach1 = queue_counter.get_track_approach(event["id1"])
        approach2 = queue_counter.get_track_approach(event["id2"])
        if approach1:
            risk_by_approach[approach1] += risk_score * 0.5
        if approach2:
            risk_by_approach[approach2] += risk_score * 0.5
    return risk_by_approach
