"""Monte Carlo simulation of the rough Bergomi model (Bayer, Friz, Gatheral 2016).

Model, with zero rates and a flat forward variance curve ``xi0``::

    dS_t / S_t = sqrt(v_t) dZ_t
    v_t        = xi0 * exp(eta * sqrt(2H) * Y_t - eta^2 / 2 * t^(2H))
    Y_t        = int_0^t (t - s)^(H - 1/2) dW_s           (Volterra process)
    dZ_t       = rho dW_t + sqrt(1 - rho^2) dW_perp_t

``sqrt(2H) * Y_t`` has variance ``t^(2H)``, so ``E[v_t] = xi0`` for every ``t``.

``Y`` is simulated with the hybrid scheme of Bennedsen, Lunde and Pakkanen
(2017) with ``kappa = 1``. On a grid of step ``dt`` the kernel is treated

* exactly on the last step, where it is singular: the pair
  ``(dW_i, int_{t_{i-1}}^{t_i} (t_i - s)^alpha dW_s)`` is bivariate Gaussian with
  known covariance, and
* as a step function further back, evaluated at the optimal points ``b_k``::

    Y_{t_i} ~ int_{t_{i-1}}^{t_i} (t_i - s)^alpha dW_s
              + sum_{k=2}^{i} (b_k * dt)^alpha * (W_{t_{i-k+1}} - W_{t_{i-k}})

  with ``alpha = H - 1/2`` and ``b_k^alpha = (k^(alpha+1) - (k-1)^(alpha+1)) / (alpha+1)``.

For ``H = 1/2`` the kernel is constant and the scheme reproduces ``Y = W`` exactly.

Every operation is differentiable with respect to the model parameters, which
is what the differential-ML phase of the project will rely on.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

DTYPE = torch.float64

PARAMETER_NAMES = ("H", "eta", "rho", "xi0")


@dataclass(frozen=True)
class RBergomiParams:
    """Rough Bergomi parameters with a flat forward variance curve."""

    hurst: float | Tensor  # H in (0, 1/2]: roughness of the variance paths
    eta: float | Tensor  # volatility of variance
    rho: float | Tensor  # spot-variance correlation
    xi0: float | Tensor  # forward variance level (so sqrt(xi0) is a volatility)

    @classmethod
    def from_vector(cls, theta) -> "RBergomiParams":
        """Build from a vector ordered as ``PARAMETER_NAMES``."""
        hurst, eta, rho, xi0 = theta
        return cls(hurst, eta, rho, xi0)

    def as_tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return tuple(
            torch.as_tensor(value, dtype=DTYPE)
            for value in (self.hurst, self.eta, self.rho, self.xi0)
        )


@dataclass(frozen=True)
class VariancePaths:
    """Simulated variance paths together with the Brownian motion driving them."""

    dW: Tensor  # (n_paths, n_steps)      increments of W over each time step
    variance: Tensor  # (n_paths, n_steps + 1)  v at the grid times 0, dt, ..., T
    dt: float
    antithetic: bool  # True if the second half of the paths mirrors the first half

    @property
    def n_paths(self) -> int:
        return self.dW.shape[0]

    @property
    def n_steps(self) -> int:
        return self.dW.shape[1]

    @property
    def times(self) -> Tensor:
        return self.dt * torch.arange(self.n_steps + 1, dtype=DTYPE)

    def step_index(self, time: float) -> int:
        """Index of ``time`` on the simulation grid; fails if it is not a grid point."""
        index = round(time / self.dt)
        if index < 1 or index > self.n_steps or abs(index * self.dt - time) > 1e-9:
            raise ValueError(f"time {time} is not on the simulation grid (dt = {self.dt})")
        return index


def standard_normal(shape: tuple[int, int], generator: torch.Generator, antithetic: bool) -> Tensor:
    """Standard normal draws of shape ``(n_paths, n_steps)``.

    With ``antithetic=True`` only half of the paths are drawn and the other half
    is their mirror image, so ``n_paths`` must be even.
    """
    n_paths, n_steps = shape
    if not antithetic:
        return torch.randn(n_paths, n_steps, generator=generator, dtype=DTYPE)
    if n_paths % 2 != 0:
        raise ValueError("antithetic sampling needs an even number of paths")
    half = torch.randn(n_paths // 2, n_steps, generator=generator, dtype=DTYPE)
    return torch.cat([half, -half])


def hybrid_scheme_weights(alpha: Tensor, n_steps: int, dt: float) -> Tensor:
    """Matrix ``M`` such that ``dW @ M`` is the step-function part of the scheme.

    ``M[j, i]`` is the weight of the increment over ``[t_j, t_{j+1}]`` in
    ``Y_{t_{i+1}}``: it is ``(b_k * dt)^alpha`` with lag ``k = i + 1 - j`` when
    ``k >= 2`` and zero otherwise (the lag ``k = 1`` is the exact local term).
    """
    steps = torch.arange(n_steps, dtype=DTYPE)
    lag = steps[None, :] + 1.0 - steps[:, None]
    k = lag.clamp_min(2.0)  # keeps the powers well defined where the weight is zero anyway
    weights = (k ** (alpha + 1.0) - (k - 1.0) ** (alpha + 1.0)) / (alpha + 1.0) * dt**alpha
    return torch.where(lag >= 2.0, weights, torch.zeros_like(weights))


def simulate_volterra(
    hurst: Tensor,
    n_steps: int,
    dt: float,
    n_paths: int,
    generator: torch.Generator,
    antithetic: bool = True,
) -> tuple[Tensor, Tensor]:
    """Simulate ``W`` and the Volterra process ``Y`` with the hybrid scheme.

    Returns ``(dW, Y)`` of shapes ``(n_paths, n_steps)``: Brownian increments and
    ``Y`` at the grid times ``dt, 2 dt, ..., n_steps * dt``.
    """
    alpha = hurst - 0.5
    z_brownian = standard_normal((n_paths, n_steps), generator, antithetic)
    z_local = standard_normal((n_paths, n_steps), generator, antithetic)

    # Cholesky factorisation of the covariance of (dW_i, local_i), where
    # local_i = int_{t_{i-1}}^{t_i} (t_i - s)^alpha dW_s:
    #   Var(dW) = dt,  Cov = dt^(alpha+1) / (alpha+1),  Var(local) = dt^(2 alpha+1) / (2 alpha+1)
    local_std = dt ** (alpha + 0.5)
    loading_on_brownian = local_std / (alpha + 1.0)
    loading_on_residual = local_std * alpha.abs() / ((alpha + 1.0) * torch.sqrt(2.0 * alpha + 1.0))

    dW = dt**0.5 * z_brownian
    local_part = loading_on_brownian * z_brownian + loading_on_residual * z_local
    history_part = dW @ hybrid_scheme_weights(alpha, n_steps, dt)
    return dW, local_part + history_part


def simulate_variance(
    params: RBergomiParams,
    n_steps: int,
    dt: float,
    n_paths: int,
    generator: torch.Generator,
    antithetic: bool = True,
) -> VariancePaths:
    """Simulate rough Bergomi variance paths on a uniform grid of ``n_steps`` steps."""
    hurst, eta, _, xi0 = params.as_tensors()
    dW, volterra = simulate_volterra(hurst, n_steps, dt, n_paths, generator, antithetic)

    times = dt * torch.arange(1, n_steps + 1, dtype=DTYPE)
    exponent = eta * torch.sqrt(2.0 * hurst) * volterra - 0.5 * eta**2 * times ** (2.0 * hurst)
    initial_variance = xi0.expand(n_paths, 1)
    variance = torch.cat([initial_variance, xi0 * torch.exp(exponent)], dim=1)
    return VariancePaths(dW=dW, variance=variance, dt=dt, antithetic=antithetic)


def simulate_spot(
    paths: VariancePaths,
    rho: float | Tensor,
    generator: torch.Generator,
    spot: float = 1.0,
) -> Tensor:
    """Simulate the spot along given variance paths with a log-Euler scheme.

    Returns a tensor of shape ``(n_paths, n_steps + 1)`` starting at ``spot``.
    """
    rho = torch.as_tensor(rho, dtype=DTYPE)
    z_perp = standard_normal((paths.n_paths, paths.n_steps), generator, paths.antithetic)
    dZ = rho * paths.dW + torch.sqrt(1.0 - rho**2) * paths.dt**0.5 * z_perp

    v_left = paths.variance[:, :-1]  # the scheme is explicit: variance at the start of each step
    log_returns = torch.sqrt(v_left) * dZ - 0.5 * v_left * paths.dt
    log_spot = torch.cumsum(log_returns, dim=1)
    start = torch.zeros(paths.n_paths, 1, dtype=DTYPE)
    return spot * torch.exp(torch.cat([start, log_spot], dim=1))
