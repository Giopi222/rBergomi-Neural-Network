"""Step 5: sample parameters, simulate their volatility surfaces and save the dataset.

Usage:
    python scripts/02_generate_dataset.py              # full dataset, about 5 CPU-core-hours
    python scripts/02_generate_dataset.py --smoke      # tiny dataset to check that everything runs
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from rbergomi_nn.dataset import SPLITS, DatasetConfig, build_dataset
from rbergomi_nn.project import DATASET_PATH
from rbergomi_nn.surface import MonteCarloConfig, SurfaceGrid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--smoke", action="store_true", help="generate a tiny dataset")
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    config = DatasetConfig(n_train_box=256, n_outer_box=64) if args.smoke else DatasetConfig()
    mc = MonteCarloConfig(n_paths=2_048) if args.smoke else MonteCarloConfig()

    start = time.perf_counter()
    dataset = build_dataset(config, SurfaceGrid(), mc, n_workers=args.workers)
    elapsed = time.perf_counter() - start
    dataset.save(args.output)

    print(f"\n{len(dataset)} surfaces in {elapsed / 60:.1f} min -> {args.output}")
    for split in SPLITS:
        subset = dataset.subset(split)
        noise = 100 * np.sqrt(np.mean(subset.implied_vol_stderr**2))
        print(f"  {split:5s} {len(subset):6d} surfaces   MC noise (RMS stderr) {noise:.3f} vol pts")
    n_invalid = int(np.isnan(dataset.implied_vol).any(axis=(1, 2)).sum())
    print(f"  surfaces with undefined implied vols: {n_invalid}")


if __name__ == "__main__":
    main()
