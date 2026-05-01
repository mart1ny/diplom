from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .demand_forecaster import (
        StandardScaler,
        TrafficDemandLSTM,
        WindowedTrafficDataset,
        build_windowed_samples,
        load_queue_records,
        save_scaler,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    sys.path.append(str(CURRENT_DIR))
    from demand_forecaster import (  # type: ignore
        StandardScaler,
        TrafficDemandLSTM,
        WindowedTrafficDataset,
        build_windowed_samples,
        load_queue_records,
        save_scaler,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM forecaster for traffic queues.")
    parser.add_argument("--data", type=str, required=True, help="Path to queues.jsonl file.")
    parser.add_argument("--seq-len", type=int, default=12, help="Input sequence length.")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon in steps.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size of LSTM.")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout between LSTM layers.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/demand_forecaster",
        help="Directory to save model artifacts.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_samples(samples: Sequence, val_split: float, seed: int):
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_split))
    val_indices = set(indices[:val_size])
    train_samples = [samples[idx] for idx in indices if idx not in val_indices]
    val_samples = [samples[idx] for idx in indices if idx in val_indices]
    if not train_samples:
        raise ValueError("Training split is empty. Generate more data or reduce val-split.")
    return train_samples, val_samples


def evaluate(model, loader, criterion, scaler, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_items = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item() * x.size(0)

            preds_denorm = scaler.inverse_transform_targets(preds.cpu().numpy())
            y_denorm = scaler.inverse_transform_targets(y.cpu().numpy())
            total_mae += float(np.abs(preds_denorm - y_denorm).mean()) * x.size(0)
            total_items += x.size(0)
    return {
        "loss": total_loss / max(total_items, 1),
        "mae": total_mae / max(total_items, 1),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    records = load_queue_records(Path(args.data))
    samples = build_windowed_samples(records, seq_len=args.seq_len, horizon=args.horizon)
    if len(samples) < 2:
        raise ValueError("Dataset is too small. Generate more records before training.")

    train_samples, val_samples = split_samples(samples, args.val_split, args.seed)
    scaler = StandardScaler.fit(train_samples)
    train_set = WindowedTrafficDataset(train_samples, scaler)
    val_set = WindowedTrafficDataset(val_samples, scaler)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    input_size = train_samples[0].x.shape[-1]
    model = TrafficDemandLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        horizon=args.horizon,
        dropout=args.dropout,
    ).to(args.device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "demand_lstm.pt"
    scaler_path = output_dir / "demand_lstm_scaler.json"
    summary_path = output_dir / "training_summary.json"

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_items = 0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_items += x.size(0)

        train_metrics = {"loss": train_loss / max(train_items, 1)}
        val_metrics = evaluate(model, val_loader, criterion, scaler, args.device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_mae_queue": round(val_metrics["mae"], 6),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={row['train_loss']:.6f} | "
            f"val_loss={row['val_loss']:.6f} | "
            f"val_mae_queue={row['val_mae_queue']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "input_size": input_size,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.layers,
                        "horizon": args.horizon,
                        "seq_len": args.seq_len,
                        "dropout": args.dropout,
                    },
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            save_scaler(scaler_path, scaler)

    summary = {
        "data_path": str(Path(args.data).resolve()),
        "num_records": len(records),
        "num_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_loss": round(best_val_loss, 6),
        "output_dir": str(output_dir.resolve()),
        "artifacts": {
            "checkpoint": str(checkpoint_path.resolve()),
            "scaler": str(scaler_path.resolve()),
        },
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model to {checkpoint_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
