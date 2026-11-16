"""Step 6a: train the network that maps (H, eta, rho, xi0) to the volatility surface.

Usage:
    python scripts/03_train_network.py
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt

from rbergomi_nn.dataset import SurfaceDataset
from rbergomi_nn.project import DATASET_PATH, MODEL_PATH, save_figure, save_report
from rbergomi_nn.training import TrainingConfig, train_network


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--max-epochs", type=int, default=TrainingConfig.max_epochs)
    args = parser.parse_args()

    dataset = SurfaceDataset.load(args.dataset)
    train, val = dataset.subset("train"), dataset.subset("val")
    config = TrainingConfig(max_epochs=args.max_epochs)
    print(f"training on {len(train)} surfaces, validating on {len(val)}")

    start = time.perf_counter()
    model, history = train_network(train, val, config)
    elapsed = time.perf_counter() - start
    model.save(args.output)

    best = min(history, key=lambda row: row["val_rmse"])
    n_weights = sum(p.numel() for p in model.parameters())
    print(f"\nbest epoch {best['epoch']}: val RMSE {best['val_rmse']:.4f} vol pts")
    print(f"{n_weights} weights, trained in {elapsed:.0f}s -> {args.output}")

    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(6.2, 4))
    axis.semilogy(epochs, [row["train_rmse"] for row in history], label="train")
    axis.semilogy(epochs, [row["val_rmse"] for row in history], label="validation")
    axis.axvline(best["epoch"], color="k", ls=":", lw=0.8, label="selected weights")
    axis.set(title="Training history", xlabel="epoch", ylabel="RMSE (vol points)")
    axis.legend()
    save_figure(figure, "training_history")

    save_report(
        {
            "config": asdict(config),
            "n_weights": n_weights,
            "n_train": len(train),
            "n_val": len(val),
            "training_seconds": round(elapsed, 1),
            "best_epoch": best,
            "epochs_run": len(history),
        },
        "training",
    )


if __name__ == "__main__":
    main()
