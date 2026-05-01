from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from .demand_forecaster import save_queue_records
except ImportError:
    from demand_forecaster import save_queue_records  # type: ignore

APPROACHES = ["north", "east", "south", "west"]
WEATHER_STATES = ["clear", "clouds", "rain", "snow"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic traffic demand dataset.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--summary", type=str, default=None, help="Optional summary JSON path.")
    parser.add_argument("--lights", type=int, default=3, help="Number of synthetic intersections.")
    parser.add_argument("--days", type=int, default=45, help="How many days to generate.")
    parser.add_argument("--step-minutes", type=int, default=5, help="Sampling period in minutes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def gaussian_peak(x: float, center: float, width: float, amplitude: float) -> float:
    return amplitude * math.exp(-((x - center) ** 2) / (2.0 * width * width))


def choose_weather(rng: random.Random, month_bias: int) -> str:
    roll = rng.random()
    if month_bias in {12, 1, 2}:
        if roll < 0.15:
            return "snow"
        if roll < 0.35:
            return "clouds"
        if roll < 0.5:
            return "rain"
        return "clear"
    if roll < 0.12:
        return "rain"
    if roll < 0.35:
        return "clouds"
    return "clear"


def regime_multiplier(minute_of_day: int, weekend: bool) -> float:
    hour = minute_of_day / 60.0
    morning = gaussian_peak(hour, center=8.2, width=1.2, amplitude=9.0)
    evening = gaussian_peak(hour, center=17.8, width=1.5, amplitude=10.5)
    midday = gaussian_peak(hour, center=12.8, width=2.0, amplitude=4.0)
    night = gaussian_peak(hour, center=2.0, width=2.5, amplitude=-3.5)
    base = 4.0 + morning + evening + midday + night
    if weekend:
        base *= 0.75
        base += gaussian_peak(hour, center=14.0, width=2.8, amplitude=2.5)
    return max(base, 0.5)


def approach_bias(approach: str) -> float:
    return {
        "north": 1.15,
        "east": 0.95,
        "south": 0.85,
        "west": 1.0,
    }[approach]


def generate_records(
    *,
    lights: int,
    days: int,
    step_minutes: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    steps = int((24 * 60 / step_minutes) * days)
    records: List[Dict[str, Any]] = []
    previous_queue: Dict[str, float] = {}

    for light_idx in range(lights):
        light_id = f"intersection_{chr(ord('A') + light_idx)}"
        light_scale = rng.uniform(0.85, 1.35)
        incident_step = rng.randint(steps // 8, max(steps // 2, 1))
        incident_length = rng.randint(3, 12)

        for approach in APPROACHES:
            previous_queue[f"{light_id}:{approach}"] = rng.uniform(0.5, 4.0)

        for step in range(steps):
            ts = start + timedelta(minutes=step * step_minutes)
            weekend = ts.weekday() >= 5
            holiday = weekend and rng.random() < 0.03
            weather = choose_weather(rng, ts.month)
            minute_of_day = ts.hour * 60 + ts.minute
            common_regime = regime_multiplier(minute_of_day, weekend)
            weather_factor = {
                "clear": 1.0,
                "clouds": 1.05,
                "rain": 1.18,
                "snow": 1.28,
            }[weather]
            incident_active = incident_step <= step < incident_step + incident_length

            for approach in APPROACHES:
                key = f"{light_id}:{approach}"
                demand = common_regime * light_scale * approach_bias(approach) * weather_factor
                if incident_active and approach in {"north", "west"}:
                    demand *= 1.55
                if holiday:
                    demand *= 0.82

                noise = rng.gauss(0.0, 1.2)
                prev = previous_queue[key]
                queue = max(0.0, 0.62 * prev + demand + noise - rng.uniform(2.0, 5.0))
                queue = min(queue, 70.0)
                previous_queue[key] = queue

                risk = 0.04 + 0.012 * queue
                if weather in {"rain", "snow"}:
                    risk += 0.08
                if incident_active:
                    risk += 0.12
                risk += rng.uniform(-0.03, 0.03)
                risk = max(0.0, min(risk, 1.0))

                records.append(
                    {
                        "light_id": light_id,
                        "timestamp": ts.isoformat().replace("+00:00", "Z"),
                        "approach": approach,
                        "queue_len": round(queue, 3),
                        "risk_score": round(risk, 4),
                        "risk": round(risk, 4),
                        "weekday": ts.weekday(),
                        "hour": ts.hour,
                        "minute": ts.minute,
                        "is_weekend": weekend,
                        "is_holiday": holiday,
                        "weather": weather,
                        "weather_code": weather,
                        "incident_active": incident_active,
                    }
                )
    return records


def build_summary(records: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    unique_series = sorted({f"{rec['light_id']}::{rec['approach']}" for rec in records})
    queue_values = [float(rec["queue_len"]) for rec in records]
    risk_values = [float(rec["risk_score"]) for rec in records]
    weather_counts: Dict[str, int] = {}
    for rec in records:
        weather_counts[rec["weather"]] = weather_counts.get(rec["weather"], 0) + 1
    return {
        "output": str(output_path.resolve()),
        "records": len(records),
        "series": len(unique_series),
        "queue": {
            "min": round(min(queue_values), 3),
            "max": round(max(queue_values), 3),
            "mean": round(sum(queue_values) / len(queue_values), 3),
        },
        "risk": {
            "min": round(min(risk_values), 4),
            "max": round(max(risk_values), 4),
            "mean": round(sum(risk_values) / len(risk_values), 4),
        },
        "weather_counts": weather_counts,
    }


def main() -> None:
    args = parse_args()
    records = generate_records(
        lights=args.lights,
        days=args.days,
        step_minutes=args.step_minutes,
        seed=args.seed,
    )
    output_path = Path(args.output)
    save_queue_records(output_path, records)
    summary = build_summary(records, output_path)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} records to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
