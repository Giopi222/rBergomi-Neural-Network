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


@dataclass
class RBergomiParams:
    """Rough Bergomi parameters with a flat forward variance curve."""

    hurst: float | Tensor  # H in (0, 1/2]: roughness of the variance paths
    eta: float | Tensor  # volatility of variance
    rho: float | Tensor  # spot-variance correlation
    xi0: float | Tensor  # forward variance level (so sqrt(xi0) is a volatility)


@dataclass
class VariancePaths:
    """Simulated variance paths together with the Brownian motion driving them."""

    dW: Tensor  # (n_paths, n_steps)      increments of W over each time step
    variance: Tensor  # (n_paths, n_steps + 1)  v at the grid times 0, dt, ..., T
    dt: float
    antithetic: bool  # True if the second half of the paths mirrors the first half

    @property
    def times(self) -> Tensor:
        return self.dt * torch.arange(self.dW.shape[1] + 1, dtype=DTYPE)


def standard_normal(n_paths: int, n_steps: int, generator, antithetic: bool) -> Tensor:
    """Standard normal draws of shape ``(n_paths, n_steps)``.

    With ``antithetic=True`` only half of the paths are drawn and the other half
    is their mirror image (``n_paths`` must be even).
    """
    if not antithetic:
        return torch.randn(n_paths, n_steps, generator=generator, dtype=DTYPE)
    half = torch.randn(n_paths // 2, n_steps, generator=generator, dtype=DTYPE)
    return torch.cat([half, -half])


def hybrid_scheme_weights(alpha: Tensor, n_steps: int, dt: float) -> Tensor:
    """Matrix ``M`` such that ``dW @ M`` is the step-function part of the scheme.

    ``M[j, i]`` is the weight of the increment over ``[t_j, t_{j+1}]`` in
    ``Y_{t_{i+1}}``: it is ``(b_k * dt)^alpha`` with lag ``k = i + 1 - j`` when
    ``k >= 2`` and zero otherwise (the lag ``k = 1`` is the exact local term).
    """
    steps = torch.arange(n_steps, dtype=DTYPE)
    lag = steps + 1.0 - steps[:, None]
    k = lag.clamp_min(2.0)  # keeps the powers well defined where the weight is zero anyway
    weights = (k ** (alpha + 1.0) - (k - 1.0) ** (alpha + 1.0)) / (alpha + 1.0) * dt**alpha
    return torch.where(lag >= 2.0, weights, 0.0)


def simulate_volterra(hurst: Tensor, n_steps, dt, n_paths, generator, antithetic=True):
    """Simulate ``W`` and the Volterra process ``Y`` with the hybrid scheme.

    Returns ``(dW, Y)`` of shapes ``(n_paths, n_steps)``: Brownian increments and
    ``Y`` at the grid times ``dt, 2 dt, ..., n_steps * dt``.
    """
    alpha = hurst - 0.5
    z = standard_normal(n_paths, n_steps, generator, antithetic)
    z_perp = standard_normal(n_paths, n_steps, generator, antithetic)

    # Cholesky factorisation of the covariance of (dW_i, local_i), where
    # local_i = int_{t_{i-1}}^{t_i} (t_i - s)^alpha dW_s:
    #   Var(dW) = dt,  Cov = dt^(alpha+1) / (alpha+1),  Var(local) = dt^(2 alpha+1) / (2 alpha+1)
    local_std = dt ** (alpha + 0.5)
    loading = local_std / (alpha + 1.0)
    loading_perp = local_std * alpha.abs() / ((alpha + 1.0) * torch.sqrt(2.0 * alpha + 1.0))

    dW = dt**0.5 * z
    local = loading * z + loading_perp * z_perp
    return dW, local + dW @ hybrid_scheme_weights(alpha, n_steps, dt)


def simulate_variance(params: RBergomiParams, n_steps, dt, n_paths, generator, antithetic=True):
    """Simulate rough Bergomi variance paths on a uniform grid of ``n_steps`` steps."""
    hurst = torch.as_tensor(params.hurst, dtype=DTYPE)
    eta = torch.as_tensor(params.eta, dtype=DTYPE)
    xi0 = torch.as_tensor(params.xi0, dtype=DTYPE)
    dW, Y = simulate_volterra(hurst, n_steps, dt, n_paths, generator, antithetic)

    t = dt * torch.arange(1, n_steps + 1, dtype=DTYPE)
    v = xi0 * torch.exp(eta * torch.sqrt(2.0 * hurst) * Y - 0.5 * eta**2 * t ** (2.0 * hurst))
    return VariancePaths(dW, torch.cat([xi0.expand(n_paths, 1), v], dim=1), dt, antithetic)


def simulate_spot(paths: VariancePaths, rho, generator, spot=1.0) -> Tensor:
    """Simulate the spot along given variance paths with a log-Euler scheme.

    Returns a tensor of shape ``(n_paths, n_steps + 1)`` starting at ``spot``.
    """
    rho = torch.as_tensor(rho, dtype=DTYPE)
    z_perp = standard_normal(*paths.dW.shape, generator, paths.antithetic)
    dZ = rho * paths.dW + torch.sqrt(1.0 - rho**2) * paths.dt**0.5 * z_perp

    v = paths.variance[:, :-1]  # the scheme is explicit: variance at the start of each step
    log_spot = torch.cumsum(torch.sqrt(v) * dZ - 0.5 * v * paths.dt, dim=1)
    start = torch.zeros(len(v), 1, dtype=DTYPE)
    return spot * torch.exp(torch.cat([start, log_spot], dim=1))
