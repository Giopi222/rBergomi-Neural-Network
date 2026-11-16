"""Monte Carlo estimators of European option prices under rough Bergomi.

Two estimators are provided. Both price the out-of-the-money option at each
strike (calls above the spot, puts below), which is what the implied
volatility inversion needs.

``standard_estimator``
    Averages the payoff of simulated terminal spots. Used for validation.

``mixed_estimator``
    Conditions on the Brownian motion ``W`` that drives the variance
    (Romano-Touzi; "turbocharging" in McCrickerd-Pakkanen 2018). Given ``W``
    the spot is lognormal, so the orthogonal noise is integrated analytically::

        price = E[ Black( S0 * exp(rho * I_T - rho^2 / 2 * V_T),  K,  sqrt((1 - rho^2) * V_T) ) ]
        I_T = int_0^T sqrt(v) dW,     V_T = int_0^T v dt

    It needs no spot simulation, has much lower variance, and returns strictly
    positive prices, so implied volatilities always exist.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from rbergomi_nn.black_scholes import black_price
from rbergomi_nn.simulator import DTYPE, VariancePaths


@dataclass(frozen=True)
class PriceEstimate:
    """Monte Carlo price and its standard error, one entry per (maturity, strike)."""

    price: Tensor  # (n_maturities, n_strikes)
    stderr: Tensor  # (n_maturities, n_strikes)


def mean_and_stderr(samples: Tensor, antithetic: bool) -> tuple[Tensor, Tensor]:
    """Sample mean and standard error along the first (path) dimension.

    Antithetic pairs are not independent, so they are averaged first and the
    standard error is computed across pairs.
    """
    if antithetic:
        half = samples.shape[0] // 2
        samples = 0.5 * (samples[:half] + samples[half:])
    stderr = samples.std(dim=0) / math.sqrt(samples.shape[0])
    return samples.mean(dim=0), stderr


def with_control_variate(samples: Tensor, control: Tensor, control_mean: float) -> Tensor:
    """Subtract the best linear predictor of ``samples`` given a control of known mean.

    ``samples`` has shape ``(n_paths, n_strikes)`` and ``control`` ``(n_paths, 1)``.
    The slope is estimated per strike from the same paths, which adds a bias of
    order ``1 / n_paths``, negligible next to the statistical error.
    """
    centred_control = control - control_mean
    centred_samples = samples - samples.mean(dim=0)
    covariance = (centred_samples * centred_control).mean(dim=0)
    # A constant control (rho = 0) carries no information: the clamp turns 0 / 0 into slope 0.
    slope = covariance / centred_control.square().mean(dim=0).clamp_min(1e-300)
    return samples - slope * centred_control


def _as_strike_matrix(maturities: Tensor, strikes: Tensor) -> Tensor:
    strikes = torch.as_tensor(strikes, dtype=DTYPE)
    if strikes.ndim == 1:
        strikes = strikes.expand(len(maturities), -1)
    return strikes


def mixed_estimator(
    paths: VariancePaths,
    rho: float | Tensor,
    maturities: Tensor,
    strikes: Tensor,
    spot: float = 1.0,
    control_variate: bool = True,
) -> PriceEstimate:
    """Conditional Monte Carlo price of out-of-the-money options.

    ``strikes`` has shape ``(n_maturities, n_strikes)``, or ``(n_strikes,)`` if
    every maturity uses the same strikes. With ``control_variate=True`` the
    conditional forward, whose mean is known, is used as a control variate.
    """
    rho = torch.as_tensor(rho, dtype=DTYPE)
    strikes = _as_strike_matrix(maturities, strikes)

    v_left = paths.variance[:, :-1]
    integral_sqrt_v_dW = torch.cumsum(torch.sqrt(v_left) * paths.dW, dim=1)
    integral_v_dt = torch.cumsum(v_left, dim=1) * paths.dt

    prices, stderrs = [], []
    for maturity, strike_row in zip(maturities.tolist(), strikes, strict=True):
        last_step = paths.step_index(maturity) - 1
        I_T = integral_sqrt_v_dW[:, last_step, None]
        V_T = integral_v_dt[:, last_step, None]

        conditional_forward = spot * torch.exp(rho * I_T - 0.5 * rho**2 * V_T)
        conditional_std = torch.sqrt((1.0 - rho**2) * V_T)
        conditional_price = black_price(
            conditional_forward, strike_row, conditional_std, is_call=strike_row >= spot
        )
        if control_variate:
            # The conditional forward is a martingale: its mean is exactly the spot.
            conditional_price = with_control_variate(conditional_price, conditional_forward, spot)
        price, stderr = mean_and_stderr(conditional_price, paths.antithetic)
        prices.append(price)
        stderrs.append(stderr)
    return PriceEstimate(torch.stack(prices), torch.stack(stderrs))


def standard_estimator(
    spot_paths: Tensor,
    paths: VariancePaths,
    maturities: Tensor,
    strikes: Tensor,
    spot: float = 1.0,
) -> PriceEstimate:
    """Plain Monte Carlo price of out-of-the-money options from simulated spots."""
    strikes = _as_strike_matrix(maturities, strikes)

    prices, stderrs = [], []
    for maturity, strike_row in zip(maturities.tolist(), strikes, strict=True):
        terminal_spot = spot_paths[:, paths.step_index(maturity), None]
        call_payoff = (terminal_spot - strike_row).clamp_min(0.0)
        put_payoff = (strike_row - terminal_spot).clamp_min(0.0)
        payoff = torch.where(strike_row >= spot, call_payoff, put_payoff)
        price, stderr = mean_and_stderr(payoff, paths.antithetic)
        prices.append(price)
        stderrs.append(stderr)
    return PriceEstimate(torch.stack(prices), torch.stack(stderrs))
