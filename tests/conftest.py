import numpy as np
import pytest

from rbergomi_nn.dataset import TRAIN_BOX, SurfaceDataset
from rbergomi_nn.surface import MonteCarloConfig, SurfaceGrid


def smooth_surface(theta: np.ndarray, grid: SurfaceGrid) -> np.ndarray:
    """A cheap, smooth stand-in for the model: lets network tests run without Monte Carlo.

    Each parameter moves the surface in a different way (level, slope, curvature,
    term structure), so the parameters are identifiable from the surface.
    """
    hurst, eta, rho, xi0 = (theta[:, i, None, None] for i in range(4))
    log_moneyness = grid.log_moneyness.numpy()[None]
    maturity = np.array(grid.maturities)[None, :, None]
    level = np.sqrt(xi0)
    slope = 0.3 * rho * eta * maturity ** (hurst - 0.5)
    return level * (1.0 + slope * log_moneyness + 0.5 * eta * log_moneyness**2)


@pytest.fixture(scope="session")
def synthetic_dataset() -> SurfaceDataset:
    grid = SurfaceGrid()
    theta = TRAIN_BOX.sobol(1024, seed=0)
    implied_vol = smooth_surface(theta, grid)
    split = np.array(["train"] * 768 + ["val"] * 128 + ["test"] * 128)
    return SurfaceDataset(
        theta=theta,
        implied_vol=implied_vol,
        implied_vol_stderr=np.full_like(implied_vol, 1e-4),
        split=split,
        grid=grid,
        mc=MonteCarloConfig(),
    )
