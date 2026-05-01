from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lstm.demand_forecaster import (  # noqa: E402
    DemandForecaster,
    StandardScaler,
    TrafficDemandLSTM,
    WindowedTrafficDataset,
    build_windowed_samples,
    load_queue_records,
    save_queue_records,
    save_scaler,
)
from lstm.generate_synthetic_dataset import build_summary, generate_records  # noqa: E402


def test_generate_synthetic_records_and_summary(tmp_path: Path) -> None:
    records = generate_records(lights=2, days=2, step_minutes=15, seed=7)
    assert records
    assert records[0]["approach"] in {"north", "east", "south", "west"}
    assert "risk_score" in records[0]

    output_path = tmp_path / "synthetic.jsonl"
    save_queue_records(output_path, records)
    loaded = load_queue_records(output_path)
    assert len(loaded) == len(records)

    summary = build_summary(loaded, output_path)
    assert summary["records"] == len(records)
    assert summary["series"] == 8


def test_csv_roundtrip_preserves_boolean_and_numeric_fields(tmp_path: Path) -> None:
    records = generate_records(lights=1, days=1, step_minutes=60, seed=5)
    csv_path = tmp_path / "synthetic.csv"
    save_queue_records(csv_path, records)

    loaded = load_queue_records(csv_path)
    assert len(loaded) == len(records)
    assert isinstance(loaded[0]["queue_len"], float)
    assert isinstance(loaded[0]["is_weekend"], bool)
    assert isinstance(loaded[0]["incident_active"], bool)
    assert loaded[0]["timestamp"].endswith("Z")


def test_windowed_samples_scaler_and_predictor(tmp_path: Path) -> None:
    records = generate_records(lights=1, days=3, step_minutes=30, seed=9)
    samples = build_windowed_samples(records, seq_len=4, horizon=2)
    assert samples

    scaler = StandardScaler.fit(samples)
    dataset = WindowedTrafficDataset(samples[:8], scaler)
    x, y = dataset[0]
    assert x.shape[0] == 4
    assert y.shape[0] == 2

    model = TrafficDemandLSTM(input_size=x.shape[-1], hidden_size=16, num_layers=1, horizon=2)
    checkpoint_path = tmp_path / "demand_lstm.pt"
    scaler_path = tmp_path / "demand_lstm_scaler.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"input_size": x.shape[-1], "hidden_size": 16, "num_layers": 1, "horizon": 2},
        },
        checkpoint_path,
    )
    save_scaler(scaler_path, scaler)

    predictor = DemandForecaster(
        model_path=checkpoint_path,
        scaler_path=scaler_path,
        input_size=x.shape[-1],
        hidden_size=16,
        num_layers=1,
        horizon=2,
    )
    history = records[:4]
    prediction = predictor.predict(history)
    assert prediction.shape == (2,)
    assert float(prediction[0]) == pytest.approx(float(prediction[0]))
