"""From model parameters to an implied volatility surface on a fixed grid (S0 = 1)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from rbergomi_nn.black_scholes import black_vega, implied_vol
from rbergomi_nn.pricing import mixed_estimator
from rbergomi_nn.simulator import DTYPE, RBergomiParams, simulate_variance


@dataclass
class SurfaceGrid:
    """Fixed grid of maturities and strikes on which surfaces are computed.

    Strikes are set in log-moneyness scaled by ``sqrt(T)``: the grid point
    ``(T, m)`` sits at ``log(K / S0) = m * sqrt(T)``. A grid fixed in strike
    would put short-dated points many standard deviations out of the money,
    where Monte Carlo implied volatilities are pure noise. The range leans
    towards puts because that is where equity smiles carry information.
    """

    maturities: tuple = (0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0)
    scaled_moneyness: tuple = (-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.maturities), len(self.scaled_moneyness)

    @property
    def size(self) -> int:
        return len(self.maturities) * len(self.scaled_moneyness)

    @property
    def log_moneyness(self) -> Tensor:
        """``log(K / S0)`` at each grid point, shape ``(n_maturities, n_strikes)``."""
        sqrt_t = torch.tensor(self.maturities, dtype=DTYPE).sqrt()
        return sqrt_t[:, None] * torch.tensor(self.scaled_moneyness, dtype=DTYPE)

    @property
    def strikes(self) -> Tensor:
        return torch.exp(self.log_moneyness)


@dataclass
class MonteCarloConfig:
    """Discretisation used to compute one surface."""

    n_paths: int = 16_384
    steps_per_year: int = 200
    antithetic: bool = True


@dataclass
class SurfaceEstimate:
    """Monte Carlo implied volatility surface with its statistical error."""

    implied_vol: Tensor  # (n_maturities, n_strikes)
    implied_vol_stderr: Tensor  # (n_maturities, n_strikes), same units as implied_vol


def implied_vol_surface(
    params: RBergomiParams, grid: SurfaceGrid, mc: MonteCarloConfig, seed: int
) -> SurfaceEstimate:
    """Simulate rough Bergomi once and read the whole surface off the same paths.

    The standard error of the implied volatility is obtained from the standard
    error of the price by the delta method: ``stderr_vol = stderr_price / vega``.
    """
    maturities, strikes = torch.tensor(grid.maturities, dtype=DTYPE), grid.strikes
    n_steps = round(max(grid.maturities) * mc.steps_per_year)
    generator = torch.Generator().manual_seed(seed)
    paths = simulate_variance(
        params, n_steps, 1 / mc.steps_per_year, mc.n_paths, generator, mc.antithetic
    )
    estimate = mixed_estimator(paths, params.rho, maturities, strikes)

    T = maturities[:, None]
    vol = implied_vol(estimate.price, 1.0, strikes, T, is_call=strikes >= 1.0)
    vega = black_vega(1.0, strikes, T, vol)
    return SurfaceEstimate(vol, estimate.stderr / vega)
