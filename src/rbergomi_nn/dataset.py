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
"""

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rbergomi_nn.simulator import RBergomiParams
from rbergomi_nn.surface import MonteCarloConfig, SurfaceGrid, implied_vol_surface

SPLITS = ("train", "val", "test", "gap", "ood")


@dataclass(frozen=True)
class ParameterBox:
    """Axis-aligned box in parameter space, ordered as ``(H, eta, rho, xi0)``."""

    lower: tuple[float, float, float, float]
    upper: tuple[float, float, float, float]

    @property
    def lower_array(self) -> np.ndarray:
        return np.array(self.lower)

    @property
    def upper_array(self) -> np.ndarray:
        return np.array(self.upper)

    @property
    def centre(self) -> np.ndarray:
        return 0.5 * (self.lower_array + self.upper_array)

    def sobol(self, n_points: int, seed: int) -> np.ndarray:
        """Scrambled Sobol points filling the box, shape ``(n_points, 4)``."""
        engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
        unit_cube = engine.draw(n_points, dtype=torch.float64).numpy()
        return self.lower_array + unit_cube * (self.upper_array - self.lower_array)

    def normalise(self, theta: np.ndarray) -> np.ndarray:
        """Map the box onto ``[-1, 1]^4``."""
        return 2.0 * (theta - self.lower_array) / (self.upper_array - self.lower_array) - 1.0

    def contains(self, theta: np.ndarray) -> np.ndarray:
        return np.all((theta >= self.lower_array) & (theta <= self.upper_array), axis=-1)

    def excess(self, theta: np.ndarray) -> np.ndarray:
        """How far each coordinate lies outside the box, in normalised units.

        Zero for coordinates inside the box; 1 means "one half-width outside".
        """
        return np.clip(np.abs(self.normalise(theta)) - 1.0, 0.0, None)

    def distance_outside(self, theta: np.ndarray) -> np.ndarray:
        """Euclidean norm of ``excess``: zero inside the box, growing outside."""
        return np.linalg.norm(self.excess(theta), axis=-1)


#                             H     eta    rho    xi0
TRAIN_BOX = ParameterBox((0.05, 1.0, -0.90, 0.02), (0.40, 3.0, -0.20, 0.12))
OUTER_BOX = ParameterBox((0.02, 0.5, -0.98, 0.01), (0.50, 4.0, 0.00, 0.20))
HURST_GAP = (0.20, 0.25)


@dataclass(frozen=True)
class DatasetConfig:
    n_train_box: int = 16_384  # Sobol points in TRAIN_BOX (train + val + test + gap)
    n_outer_box: int = 2_048  # Sobol points in OUTER_BOX (those outside TRAIN_BOX become ood)
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 2024


@dataclass(frozen=True)
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
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
        keep = self.split == split
        return SurfaceDataset(
            self.theta[keep],
            self.implied_vol[keep],
            self.implied_vol_stderr[keep],
            self.split[keep],
            self.grid,
            self.mc,
        )

    @property
    def flat_implied_vol(self) -> np.ndarray:
        """Surfaces flattened to ``(n, n_maturities * n_strikes)``, the network's output layout."""
        return self.implied_vol.reshape(len(self), -1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta=self.theta,
            implied_vol=self.implied_vol,
            implied_vol_stderr=self.implied_vol_stderr,
            split=self.split,
            grid=json.dumps(asdict(self.grid)),
            mc=json.dumps(asdict(self.mc)),
        )

    @classmethod
    def load(cls, path: Path) -> "SurfaceDataset":
        with np.load(path) as archive:
            grid_fields = {k: tuple(v) for k, v in json.loads(str(archive["grid"])).items()}
            return cls(
                theta=archive["theta"],
                implied_vol=archive["implied_vol"],
                implied_vol_stderr=archive["implied_vol_stderr"],
                split=archive["split"],
                grid=SurfaceGrid(**grid_fields),
                mc=MonteCarloConfig(**json.loads(str(archive["mc"]))),
            )


def sample_parameters(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    """Draw all parameter vectors and assign each one to a split."""
    rng = np.random.default_rng(config.seed)

    inside = TRAIN_BOX.sobol(config.n_train_box, seed=config.seed)
    hurst = inside[:, 0]
    in_gap = (hurst >= HURST_GAP[0]) & (hurst <= HURST_GAP[1])

    # Random train / val / test assignment of the points that are not in the gap.
    uniform = rng.random(len(inside))
    inside_split = np.where(
        uniform < config.val_fraction,
        "val",
        np.where(uniform < config.val_fraction + config.test_fraction, "test", "train"),
    )
    inside_split[in_gap] = "gap"

    outer = OUTER_BOX.sobol(config.n_outer_box, seed=config.seed + 1)
    outside = outer[~TRAIN_BOX.contains(outer)]

    theta = np.concatenate([inside, outside])
    split = np.concatenate([inside_split, np.full(len(outside), "ood")])
    return theta, split


def _limit_worker_threads() -> None:
    # One process per core is faster than letting every process spawn its own thread pool.
    torch.set_num_threads(1)


def _simulate_one(task: tuple[np.ndarray, int, SurfaceGrid, MonteCarloConfig]):
    theta, seed, grid, mc = task
    with torch.no_grad():
        estimate = implied_vol_surface(RBergomiParams.from_vector(theta.tolist()), grid, mc, seed)
    return estimate.implied_vol.numpy(), estimate.implied_vol_stderr.numpy()


def simulate_surfaces(
    theta: np.ndarray,
    grid: SurfaceGrid,
    mc: MonteCarloConfig,
    base_seed: int,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one Monte Carlo surface per row of ``theta``, in parallel.

    Surface ``i`` uses the seed ``base_seed + i``, so the result does not depend
    on the number of workers and the Monte Carlo noise is independent across
    surfaces.
    """
    tasks = [(row, base_seed + i, grid, mc) for i, row in enumerate(theta)]
    with ProcessPoolExecutor(n_workers, initializer=_limit_worker_threads) as pool:
        results = list(
            tqdm(pool.map(_simulate_one, tasks, chunksize=4), total=len(tasks), desc="surfaces")
        )
    implied_vols, stderrs = zip(*results, strict=True)
    return np.stack(implied_vols), np.stack(stderrs)


def build_dataset(
    config: DatasetConfig,
    grid: SurfaceGrid,
    mc: MonteCarloConfig,
    n_workers: int,
) -> SurfaceDataset:
    theta, split = sample_parameters(config)
    implied_vol, stderr = simulate_surfaces(theta, grid, mc, config.seed, n_workers)
    return SurfaceDataset(theta, implied_vol, stderr, split, grid, mc)
