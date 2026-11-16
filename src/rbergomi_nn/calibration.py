"""Calibration: find the parameters whose surface best matches a target surface.

The optimiser is the same whatever produces the surface, which keeps the
comparison between the network and Monte Carlo fair: both are just a
``Pricer``, a function ``theta -> flattened surface`` with an optional Jacobian.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import least_squares

from rbergomi_nn.dataset import ParameterBox
from rbergomi_nn.network import VolSurfaceMLP
from rbergomi_nn.simulator import DTYPE, RBergomiParams
from rbergomi_nn.surface import MonteCarloConfig, SurfaceGrid, implied_vol_surface

SurfaceFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Pricer:
    surface: SurfaceFunction  # theta (4,) -> implied vols (n_points,)
    jacobian: SurfaceFunction | None = (
        None  # theta (4,) -> (n_points, 4); None = finite differences
    )


@dataclass(frozen=True)
class CalibrationResult:
    theta: np.ndarray  # calibrated parameters (H, eta, rho, xi0)
    rmse: float  # root mean squared fit error, in volatility units
    n_evaluations: int  # number of surface evaluations used by the optimiser
    at_bound: np.ndarray  # per parameter: did the optimiser stop on the edge of the box?


def network_pricer(model: VolSurfaceMLP) -> Pricer:
    """Wrap a trained network; its Jacobian is exact and cheap."""

    def surface(theta: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.as_tensor(theta, dtype=DTYPE)[None])[0].numpy()

    def jacobian(theta: np.ndarray) -> np.ndarray:
        return model.jacobian(torch.as_tensor(theta, dtype=DTYPE)).numpy()

    return Pricer(surface, jacobian)


def monte_carlo_pricer(grid: SurfaceGrid, mc: MonteCarloConfig, seed: int) -> Pricer:
    """Wrap the simulator.

    The seed is fixed so that every evaluation reuses the same random numbers:
    the surface is then a smooth function of the parameters and finite
    differences are meaningful.
    """

    def surface(theta: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            estimate = implied_vol_surface(RBergomiParams.from_vector(theta), grid, mc, seed)
        return estimate.implied_vol.flatten().numpy()

    return Pricer(surface)


def calibrate(
    pricer: Pricer,
    target: np.ndarray,
    box: ParameterBox,
    initial_guess: np.ndarray | None = None,
) -> CalibrationResult:
    """Least-squares fit of ``pricer.surface(theta)`` to ``target`` within ``box``.

    Uses SciPy's trust-region reflective algorithm, a bounded variant of
    Levenberg-Marquardt, started from the centre of the box unless told otherwise.
    """
    lower, upper = box.lower_array, box.upper_array
    width = upper - lower
    if initial_guess is None:
        initial_guess = box.centre

    solution = least_squares(
        fun=lambda theta: pricer.surface(theta) - target,
        x0=initial_guess,
        jac=pricer.jacobian if pricer.jacobian is not None else "2-point",
        bounds=(lower, upper),
        x_scale=width,  # the parameters have very different magnitudes
        method="trf",
    )
    theta = solution.x
    tolerance = 1e-6 * width
    at_bound = (theta - lower < tolerance) | (upper - theta < tolerance)
    return CalibrationResult(
        theta=theta,
        rmse=float(np.sqrt(np.mean(solution.fun**2))),
        n_evaluations=int(solution.nfev),
        at_bound=at_bound,
    )
