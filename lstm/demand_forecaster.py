from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

WEATHER_CODES = ["clear", "clouds", "rain", "snow", "other"]


def _one_hot(value: int, num_classes: int) -> List[float]:
    vec = [0.0] * num_classes
    if 0 <= value < num_classes:
        vec[value] = 1.0
    return vec


def encode_weather(code: str) -> List[float]:
    idx = WEATHER_CODES.index(code) if code in WEATHER_CODES else len(WEATHER_CODES) - 1
    return _one_hot(idx, len(WEATHER_CODES))


def _parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _cyclic_pair(value: float, period: float) -> List[float]:
    angle = 2.0 * math.pi * (value / period)
    return [math.sin(angle), math.cos(angle)]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)
    timestamp = str(normalized["timestamp"])
    dt = _parse_timestamp(timestamp)
    normalized["timestamp"] = dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    normalized["queue_len"] = float(normalized["queue_len"])
    normalized["risk_score"] = float(normalized.get("risk_score", normalized.get("risk", 0.0)))
    normalized["risk"] = normalized["risk_score"]
    normalized["weekday"] = int(normalized.get("weekday", dt.weekday()))
    normalized["hour"] = int(normalized.get("hour", dt.hour))
    normalized["minute"] = int(normalized.get("minute", dt.minute))
    normalized["is_weekend"] = _as_bool(normalized.get("is_weekend", normalized["weekday"] >= 5))
    normalized["is_holiday"] = _as_bool(normalized.get("is_holiday", False))
    normalized["weather"] = str(normalized.get("weather", normalized.get("weather_code", "other")))
    normalized["incident_active"] = _as_bool(normalized.get("incident_active", False))
    return normalized


def load_queue_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8", newline="") as f:
        if suffix == ".csv":
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                records.append(normalize_record(row))
            return records
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(normalize_record(json.loads(line)))
    return records


def save_queue_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        fieldnames = [
            "light_id",
            "timestamp",
            "approach",
            "queue_len",
            "risk_score",
            "risk",
            "weekday",
            "hour",
            "minute",
            "is_weekend",
            "is_holiday",
            "weather",
            "weather_code",
            "incident_active",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                normalized = normalize_record(record)
                writer.writerow(
                    {
                        "light_id": normalized["light_id"],
                        "timestamp": normalized["timestamp"],
                        "approach": normalized["approach"],
                        "queue_len": normalized["queue_len"],
                        "risk_score": normalized["risk_score"],
                        "risk": normalized["risk"],
                        "weekday": normalized["weekday"],
                        "hour": normalized["hour"],
                        "minute": normalized["minute"],
                        "is_weekend": normalized["is_weekend"],
                        "is_holiday": normalized["is_holiday"],
                        "weather": normalized["weather"],
                        "weather_code": normalized.get("weather_code", normalized["weather"]),
                        "incident_active": bool(normalized.get("incident_active", False)),
                    }
                )
        return
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def group_by_series(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    series: Dict[str, List[Dict[str, Any]]] = {}
    for raw in records:
        rec = normalize_record(raw)
        key = f"{rec['light_id']}::{rec['approach']}"
        series.setdefault(key, []).append(rec)
    for key in series:
        series[key].sort(key=lambda x: x["timestamp"])
    return series


def feature_vector(record: Dict[str, Any]) -> List[float]:
    rec = normalize_record(record)
    minute_of_day = rec["hour"] * 60 + rec["minute"]
    features = [
        float(rec["queue_len"]),
        float(rec["risk_score"]),
        *_cyclic_pair(float(minute_of_day), 24.0 * 60.0),
        *_cyclic_pair(float(rec["weekday"]), 7.0),
        1.0 if rec["is_weekend"] else 0.0,
        1.0 if rec["is_holiday"] else 0.0,
        *encode_weather(rec["weather"]),
    ]
    return features


@dataclass(frozen=True)
class WindowedSample:
    x: np.ndarray
    y: np.ndarray
    series_key: str
    target_timestamps: tuple[str, ...]


def build_windowed_samples(
    records: Sequence[Dict[str, Any]],
    *,
    seq_len: int = 12,
    horizon: int = 3,
) -> List[WindowedSample]:
    samples: List[WindowedSample] = []
    for key, seq in group_by_series(records).items():
        if len(seq) < seq_len + horizon:
            continue
        for idx in range(len(seq) - seq_len - horizon + 1):
            window = seq[idx : idx + seq_len]
            target = seq[idx + seq_len : idx + seq_len + horizon]
            x = np.asarray([feature_vector(item) for item in window], dtype=np.float32)
            y = np.asarray([float(item["queue_len"]) for item in target], dtype=np.float32)
            samples.append(
                WindowedSample(
                    x=x,
                    y=y,
                    series_key=key,
                    target_timestamps=tuple(str(item["timestamp"]) for item in target),
                )
            )
    return samples


@dataclass
class StandardScaler:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, samples: Sequence[WindowedSample]) -> "StandardScaler":
        if not samples:
            raise ValueError("Cannot fit scaler without samples.")
        features = np.concatenate(
            [sample.x.reshape(-1, sample.x.shape[-1]) for sample in samples], axis=0
        )
        targets = np.concatenate([sample.y.reshape(-1, 1) for sample in samples], axis=0)
        feature_mean = features.mean(axis=0)
        feature_std = features.std(axis=0)
        feature_std[feature_std < 1e-6] = 1.0
        target_mean = float(targets.mean())
        target_std = float(targets.std())
        if target_std < 1e-6:
            target_std = 1.0
        return cls(
            feature_mean=feature_mean.astype(np.float32),
            feature_std=feature_std.astype(np.float32),
            target_mean=target_mean,
            target_std=target_std,
        )

    def transform_features(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.feature_mean) / self.feature_std).astype(np.float32)

    def transform_targets(self, value: np.ndarray) -> np.ndarray:
        return ((value - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_transform_targets(self, value: np.ndarray) -> np.ndarray:
        return (value * self.target_std + self.target_mean).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StandardScaler":
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
            target_mean=float(payload["target_mean"]),
            target_std=float(payload["target_std"]),
        )


class WindowedTrafficDataset(Dataset):
    def __init__(self, samples: Sequence[WindowedSample], scaler: StandardScaler):
        self.samples = list(samples)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        x = self.scaler.transform_features(sample.x)
        y = self.scaler.transform_targets(sample.y)
        return torch.from_numpy(x), torch.from_numpy(y)


class TrafficDemandLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.head(last_step)


def save_scaler(path: Path, scaler: StandardScaler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_scaler(path: Path) -> StandardScaler:
    return StandardScaler.from_dict(json.loads(path.read_text(encoding="utf-8")))


class DemandForecaster:
    def __init__(
        self,
        *,
        model_path: str | Path,
        scaler_path: str | Path,
        input_size: int,
        horizon: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.scaler = load_scaler(Path(scaler_path))
        self.model = TrafficDemandLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
        ).to(self.device)
        checkpoint = torch.load(Path(model_path), map_location=self.device)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.horizon = horizon

    def predict(self, history: Sequence[Dict[str, Any]]) -> np.ndarray:
        features = np.asarray([feature_vector(item) for item in history], dtype=np.float32)
        normalized = self.scaler.transform_features(features)
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(tensor).cpu().numpy()[0]
        return self.scaler.inverse_transform_targets(prediction)
