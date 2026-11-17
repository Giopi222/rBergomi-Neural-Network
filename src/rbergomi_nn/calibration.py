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


@dataclass
class Pricer:
    surface: Callable  # theta (4,) -> implied vols (n_points,)
    jacobian: Callable | None = None  # theta (4,) -> (n_points, 4); None = finite differences


@dataclass
class CalibrationResult:
    theta: np.ndarray  # calibrated parameters (H, eta, rho, xi0)
    rmse: float  # root mean squared fit error, in volatility units
    n_evaluations: int  # number of surface evaluations used by the optimiser
    at_bound: np.ndarray  # per parameter: did the optimiser stop on the edge of the box?


def network_pricer(model: VolSurfaceMLP) -> Pricer:
    """Wrap a trained network; its Jacobian is exact and cheap."""

    def surface(theta):
        with torch.no_grad():
            return model(torch.as_tensor(theta, dtype=DTYPE)[None])[0].numpy()

    def jacobian(theta):
        return model.jacobian(torch.as_tensor(theta, dtype=DTYPE)).numpy()

    return Pricer(surface, jacobian)


def monte_carlo_pricer(grid: SurfaceGrid, mc: MonteCarloConfig, seed: int) -> Pricer:
    """Wrap the simulator.

    The seed is fixed so that every evaluation reuses the same random numbers:
    the surface is then a smooth function of the parameters and finite
    differences are meaningful.
    """

    def surface(theta):
        with torch.no_grad():
            estimate = implied_vol_surface(RBergomiParams(*theta), grid, mc, seed)
        return estimate.implied_vol.flatten().numpy()

    return Pricer(surface)


def calibrate(pricer: Pricer, target: np.ndarray, box: ParameterBox) -> CalibrationResult:
    """Least-squares fit of ``pricer.surface(theta)`` to ``target`` within ``box``.

    Trust-region reflective algorithm (a bounded Levenberg-Marquardt), started
    from the centre of the box.
    """
    solution = least_squares(
        lambda theta: pricer.surface(theta) - target,
        box.centre,
        jac=pricer.jacobian or "2-point",
        bounds=(box.lower, box.upper),
        x_scale=box.width,  # the parameters have very different magnitudes
        method="trf",
    )
    theta, tolerance = solution.x, 1e-6 * box.width
    at_bound = (theta - box.lower < tolerance) | (box.upper - theta < tolerance)
    rmse = float(np.sqrt(np.mean(solution.fun**2)))
    return CalibrationResult(theta, rmse, solution.nfev, at_bound)
