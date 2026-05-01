from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset and train the demand forecaster in one step."
    )
    parser.add_argument(
        "--dataset-output",
        default="data/lstm/synthetic_queues.csv",
        help="Where to save the generated synthetic dataset.",
    )
    parser.add_argument(
        "--summary-output",
        default="data/lstm/synthetic_queues.summary.json",
        help="Where to save the dataset summary.",
    )
    parser.add_argument(
        "--model-output-dir",
        default="models/demand_forecaster",
        help="Where to save model artifacts.",
    )
    parser.add_argument("--lights", type=int, default=4)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    run_step(
        [
            sys.executable,
            str(root / "generate_synthetic_dataset.py"),
            "--output",
            args.dataset_output,
            "--summary",
            args.summary_output,
            "--lights",
            str(args.lights),
            "--days",
            str(args.days),
            "--step-minutes",
            str(args.step_minutes),
            "--seed",
            str(args.seed),
        ]
    )

    run_step(
        [
            sys.executable,
            str(root / "train_demand_forecaster.py"),
            "--data",
            args.dataset_output,
            "--output-dir",
            args.model_output_dir,
            "--seq-len",
            str(args.seq_len),
            "--horizon",
            str(args.horizon),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--hidden-size",
            str(args.hidden_size),
            "--layers",
            str(args.layers),
            "--dropout",
            str(args.dropout),
            "--device",
            args.device,
        ]
    )


if __name__ == "__main__":
    main()
