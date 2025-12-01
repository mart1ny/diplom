import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def _one_hot(value: int, num_classes: int) -> List[float]:
    vec = [0.0] * num_classes
    if 0 <= value < num_classes:
        vec[value] = 1.0
    return vec


def encode_weather(code: str) -> List[float]:
    codes = ["clear", "clouds", "rain", "snow", "other"]
    idx = codes.index(code) if code in codes else len(codes) - 1
    return _one_hot(idx, len(codes))


def load_queue_records(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_by_series(records: List[Dict]) -> Dict[str, List[Dict]]:
    series = {}
    for rec in records:
        key = f"{rec['light_id']}::{rec['approach']}"
        series.setdefault(key, []).append(rec)
    for key in series:
        series[key].sort(key=lambda x: x["timestamp"])
    return series


class TrafficDemandDataset(Dataset):
    def __init__(self, records: List[Dict], seq_len: int = 8, horizon: int = 3):
        self.samples = []
        series = group_by_series(records)
        for key, seq in series.items():
            if len(seq) < seq_len + horizon:
                continue
            for i in range(len(seq) - seq_len - horizon + 1):
                window = seq[i : i + seq_len]
                target = seq[i + seq_len : i + seq_len + horizon]
                x = []
                for item in window:
                    weekday = _one_hot(item["weekday"], 7)
                    hour = _one_hot(item["hour"], 24)
                    weekend = [1.0 if item.get("is_weekend") else 0.0]
                    holiday = [1.0 if item.get("is_holiday") else 0.0]
                    weather = encode_weather(item.get("weather", "other"))
                    x.append(
                        [
                            float(item["queue_len"]),
                            float(item["risk"]),
                        ]
                        + weekday
                        + hour
                        + weekend
                        + holiday
                        + weather
                    )
                y = [float(t["queue_len"]) for t in target]
                self.samples.append((np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


class TrafficDemandLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, horizon: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        pred = self.head(last_step)
        return pred

