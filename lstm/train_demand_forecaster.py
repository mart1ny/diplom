import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

try:
    from .demand_forecaster import (
        TrafficDemandDataset,
        TrafficDemandLSTM,
        load_queue_records,
    )
except ImportError:  # script launched directly
    CURRENT_DIR = Path(__file__).resolve().parent
    sys.path.append(str(CURRENT_DIR))
    from demand_forecaster import (  # type: ignore
        TrafficDemandDataset,
        TrafficDemandLSTM,
        load_queue_records,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM forecaster for traffic queues.")
    parser.add_argument("--data", type=str, required=True, help="Path to queues.jsonl file.")
    parser.add_argument("--seq-len", type=int, default=8, help="Input sequence length.")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon (steps).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size of LSTM.")
    parser.add_argument("--layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device (cpu/cuda).")
    parser.add_argument("--save", type=str, default="models/lstm_forecaster.pt", help="Output model path.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    records = load_queue_records(data_path)
    dataset = TrafficDemandDataset(records, seq_len=args.seq_len, horizon=args.horizon)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Provide more records.")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    input_size = dataset[0][0].shape[-1]
    model = TrafficDemandLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        horizon=args.horizon,
    ).to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(args.device)
                y = y.to(args.device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()

