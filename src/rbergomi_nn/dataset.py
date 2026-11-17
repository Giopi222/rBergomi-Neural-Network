"""Dataset of rough Bergomi implied volatility surfaces.

Parameters ``theta = (H, eta, rho, xi0)`` are drawn from two Sobol sequences:

* one over ``TRAIN_BOX``, the domain the network is meant to cover, and
* one over the larger ``OUTER_BOX``, of which only the points *outside*
  ``TRAIN_BOX`` are kept.

Every sample is assigned to exactly one split:

=========  ==================================================================
``train``  inside ``TRAIN_BOX``, used to fit the network
``val``    inside ``TRAIN_BOX``, used for early stopping
``test``   inside ``TRAIN_BOX``, same distribution as ``train`` (interpolation)
``gap``    inside ``TRAIN_BOX`` but with ``H`` in ``HURST_GAP``, a slab removed
           from training: a region surrounded by data that was never seen
``ood``    outside ``TRAIN_BOX`` (extrapolation)
=========  ==================================================================

check dataset_explorer.ipynb to see the result.
"""

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from rbergomi_nn.simulator import RBergomiParams
from rbergomi_nn.surface import MonteCarloConfig, SurfaceGrid, implied_vol_surface

SPLITS = ("train", "val", "test", "gap", "ood")


class ParameterBox:
    """Axis-aligned box in parameter space, ordered as ``(H, eta, rho, xi0)``."""

    def __init__(self, lower, upper):
        self.lower, self.upper = np.array(lower), np.array(upper)
        self.width = self.upper - self.lower
        self.centre = 0.5 * (self.lower + self.upper)

    def sobol(self, n_points: int, seed: int) -> np.ndarray:
        """Scrambled Sobol points filling the box, shape ``(n_points, 4)``."""
        engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
        return self.lower + engine.draw(n_points, dtype=torch.float64).numpy() * self.width

    def normalise(self, theta: np.ndarray) -> np.ndarray:
        """Map the box onto ``[-1, 1]^4``."""
        return 2.0 * (theta - self.lower) / self.width - 1.0

    def contains(self, theta: np.ndarray) -> np.ndarray:
        return np.all((theta >= self.lower) & (theta <= self.upper), axis=-1)

    def distance_outside(self, theta: np.ndarray) -> np.ndarray:
        """Zero inside the box; outside, the norm of how far each coordinate lies
        beyond the box, in half-widths (1 = "one half-width outside")."""
        excess = np.maximum(np.abs(self.normalise(theta)) - 1.0, 0.0)
        return np.linalg.norm(excess, axis=-1)


#                             H     eta    rho    xi0
TRAIN_BOX = ParameterBox((0.05, 1.0, -0.90, 0.02), (0.40, 3.0, -0.20, 0.12))
OUTER_BOX = ParameterBox((0.02, 0.5, -0.98, 0.01), (0.50, 4.0, 0.00, 0.20))
HURST_GAP = (0.20, 0.25)


@dataclass
class DatasetConfig:
    n_train_box: int = 16_384  # Sobol points in TRAIN_BOX (train + val + test + gap)
    n_outer_box: int = 2_048  # Sobol points in OUTER_BOX (those outside TRAIN_BOX become ood)
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 2024


@dataclass
class SurfaceDataset:
    theta: np.ndarray  # (n, 4)
    implied_vol: np.ndarray  # (n, n_maturities, n_strikes)
    implied_vol_stderr: np.ndarray  # (n, n_maturities, n_strikes) Monte Carlo standard error
    split: np.ndarray  # (n,) one of SPLITS
    grid: SurfaceGrid
    mc: MonteCarloConfig

    def __len__(self) -> int:
        return len(self.theta)

    def subset(self, split: str) -> "SurfaceDataset":
        keep = self.split == split
        arrays = (self.theta, self.implied_vol, self.implied_vol_stderr, self.split)
        return SurfaceDataset(*(array[keep] for array in arrays), self.grid, self.mc)

    @property
    def flat_implied_vol(self) -> np.ndarray:
        """Surfaces flattened to ``(n, n_maturities * n_strikes)``, the network's output layout."""
        return self.implied_vol.reshape(len(self), -1)

    def save(self, path):
        content = asdict(self)
        content["grid"], content["mc"] = json.dumps(content["grid"]), json.dumps(content["mc"])
        np.savez_compressed(path, **content)

    @classmethod
    def load(cls, path) -> "SurfaceDataset":
        with np.load(path) as f:
            grid = SurfaceGrid(**{k: tuple(v) for k, v in json.loads(str(f["grid"])).items()})
            mc = MonteCarloConfig(**json.loads(str(f["mc"])))
            return cls(f["theta"], f["implied_vol"], f["implied_vol_stderr"], f["split"], grid, mc)


def sample_parameters(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    """Draw all parameter vectors and assign each one to a split."""
    inside = TRAIN_BOX.sobol(config.n_train_box, seed=config.seed)

    # Random train / val / test assignment, then the points in the gap are taken out.
    u = np.random.default_rng(config.seed).random(len(inside))
    val, val_or_test = config.val_fraction, config.val_fraction + config.test_fraction
    inside_split = np.select([u < val, u < val_or_test], ["val", "test"], "train")
    hurst = inside[:, 0]
    inside_split[(hurst >= HURST_GAP[0]) & (hurst <= HURST_GAP[1])] = "gap"

    outer = OUTER_BOX.sobol(config.n_outer_box, seed=config.seed + 1)
    outside = outer[~TRAIN_BOX.contains(outer)]

    theta = np.concatenate([inside, outside])
    split = np.concatenate([inside_split, np.full(len(outside), "ood")])
    return theta, split


def _simulate_one(theta, seed, grid, mc):
    with torch.no_grad():
        estimate = implied_vol_surface(RBergomiParams(*theta), grid, mc, seed)
    return estimate.implied_vol.numpy(), estimate.implied_vol_stderr.numpy()


def simulate_surfaces(theta, grid, mc, base_seed, n_workers):
    """Compute one Monte Carlo surface per row of ``theta``, in parallel.

    Surface ``i`` uses the seed ``base_seed + i``, so the result does not depend
    on the number of workers and the Monte Carlo noise is independent across
    surfaces. Each worker is single-threaded: one process per core is faster.
    """
    seeds = range(base_seed, base_seed + len(theta))
    with ProcessPoolExecutor(n_workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        results = pool.map(partial(_simulate_one, grid=grid, mc=mc), theta, seeds, chunksize=4)
        implied_vols, stderrs = zip(*tqdm(results, total=len(theta), desc="surfaces"))
    return np.stack(implied_vols), np.stack(stderrs)


def build_dataset(config, grid, mc, n_workers) -> SurfaceDataset:
    theta, split = sample_parameters(config)
    implied_vol, stderr = simulate_surfaces(theta, grid, mc, config.seed, n_workers)
    return SurfaceDataset(theta, implied_vol, stderr, split, grid, mc)
