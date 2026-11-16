"""From model parameters to an implied volatility surface on a fixed grid."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from rbergomi_nn.black_scholes import black_vega, implied_vol
from rbergomi_nn.pricing import mixed_estimator
from rbergomi_nn.simulator import DTYPE, RBergomiParams, simulate_variance

SPOT = 1.0  # prices are homogeneous in the spot, so the surface is computed for S0 = 1


@dataclass(frozen=True)
class SurfaceGrid:
    """Fixed grid of maturities and strikes on which surfaces are computed.

    Strikes are set in log-moneyness scaled by ``sqrt(T)``: the grid point
    ``(T, m)`` sits at ``log(K / S0) = m * sqrt(T)``. A grid fixed in strike
    would put short-dated points many standard deviations out of the money,
    where Monte Carlo implied volatilities are pure noise. The range leans
    towards puts because that is where equity smiles carry information.
    """

    maturities: tuple[float, ...] = (0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0)
    scaled_moneyness: tuple[float, ...] = (
        -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15,
    )  # fmt: skip

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.maturities), len(self.scaled_moneyness)

    @property
    def size(self) -> int:
        return len(self.maturities) * len(self.scaled_moneyness)

    @property
    def maturity_tensor(self) -> Tensor:
        return torch.tensor(self.maturities, dtype=DTYPE)

    @property
    def log_moneyness(self) -> Tensor:
        """``log(K / S0)`` at each grid point, shape ``(n_maturities, n_strikes)``."""
        scaled = torch.tensor(self.scaled_moneyness, dtype=DTYPE)
        return torch.sqrt(self.maturity_tensor)[:, None] * scaled[None, :]

    @property
    def strikes(self) -> Tensor:
        return SPOT * torch.exp(self.log_moneyness)


@dataclass(frozen=True)
class MonteCarloConfig:
    """Discretisation used to compute one surface."""

    n_paths: int = 16_384
    steps_per_year: int = 200
    antithetic: bool = True

    @property
    def dt(self) -> float:
        return 1.0 / self.steps_per_year

    def n_steps(self, horizon: float) -> int:
        return math.ceil(horizon * self.steps_per_year - 1e-9)


@dataclass(frozen=True)
class SurfaceEstimate:
    """Monte Carlo implied volatility surface with its statistical error."""

    implied_vol: Tensor  # (n_maturities, n_strikes)
    implied_vol_stderr: Tensor  # (n_maturities, n_strikes), same units as implied_vol
    price: Tensor
    price_stderr: Tensor


def implied_vol_surface(
    params: RBergomiParams,
    grid: SurfaceGrid,
    mc: MonteCarloConfig,
    seed: int,
) -> SurfaceEstimate:
    """Simulate rough Bergomi once and read the whole surface off the same paths.

    The standard error of the implied volatility is obtained from the standard
    error of the price by the delta method: ``stderr_vol = stderr_price / vega``.
    """
    generator = torch.Generator().manual_seed(seed)
    maturities = grid.maturity_tensor
    strikes = grid.strikes

    paths = simulate_variance(
        params,
        n_steps=mc.n_steps(max(grid.maturities)),
        dt=mc.dt,
        n_paths=mc.n_paths,
        generator=generator,
        antithetic=mc.antithetic,
    )
    estimate = mixed_estimator(paths, params.rho, maturities, strikes, spot=SPOT)

    forward = torch.tensor(SPOT, dtype=DTYPE)
    maturity_column = maturities[:, None]
    vol = implied_vol(estimate.price, forward, strikes, maturity_column, is_call=strikes >= SPOT)
    vega = black_vega(forward, strikes, maturity_column, vol)
    return SurfaceEstimate(
        implied_vol=vol,
        implied_vol_stderr=estimate.stderr / vega,
        price=estimate.price,
        price_stderr=estimate.stderr,
    )
