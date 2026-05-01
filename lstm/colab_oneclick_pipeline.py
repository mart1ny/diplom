from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

DATASET_CANDIDATES = [
    Path("synthetic_queues_dataset.csv"),
    Path("synthetic_queues.csv"),
    Path("/content/synthetic_queues_dataset.csv"),
    Path("/content/synthetic_queues.csv"),
    Path("lstm/data/synthetic_queues_dataset.csv"),
    Path("data/lstm/synthetic_queues.csv"),
]

MODEL_OUTPUT_DIR = Path("models/demand_forecaster")
SEQ_LEN = 12
HORIZON = 3
EPOCHS = 25
BATCH_SIZE = 128
HIDDEN_SIZE = 96
LAYERS = 2
DROPOUT = 0.1
SEED = 42


def resolve_dataset_path() -> Path:
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    candidates = "\n".join(f"- {path}" for path in DATASET_CANDIDATES)
    raise FileNotFoundError(
        "CSV dataset not found. Upload the file to one of these paths:\n"
        f"{candidates}"
    )


def resolve_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_step(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset_path = resolve_dataset_path()
    device = resolve_device()
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using dataset: {dataset_path}")
    print(f"Using device: {device}")
    print(f"Artifacts will be saved to: {MODEL_OUTPUT_DIR.resolve()}")

    run_step(
        [
            sys.executable,
            str(root / "train_demand_forecaster.py"),
            "--data",
            str(dataset_path),
            "--output-dir",
            str(MODEL_OUTPUT_DIR),
            "--seq-len",
            str(SEQ_LEN),
            "--horizon",
            str(HORIZON),
            "--epochs",
            str(EPOCHS),
            "--batch-size",
            str(BATCH_SIZE),
            "--hidden-size",
            str(HIDDEN_SIZE),
            "--layers",
            str(LAYERS),
            "--dropout",
            str(DROPOUT),
            "--seed",
            str(SEED),
            "--device",
            device,
        ]
    )

    print("\nDone.")
    print("Generated files:")
    print(f"- {MODEL_OUTPUT_DIR / 'demand_lstm.pt'}")
    print(f"- {MODEL_OUTPUT_DIR / 'demand_lstm_scaler.json'}")
    print(f"- {MODEL_OUTPUT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    main()
