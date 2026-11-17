"""Black-Scholes formulas with zero rates and dividends.

Everything in this project is quoted on the forward, so prices are undiscounted
and the only inputs are forward, strike and total standard deviation
``total_std = vol * sqrt(maturity)``. All functions broadcast their arguments.
"""

import math

import torch
from torch import Tensor
from torch.special import ndtr


def black_price(forward, strike: Tensor, total_std: Tensor, is_call: Tensor) -> Tensor:
    """Undiscounted Black price of a European call (``is_call``) or put."""
    std = total_std.clamp_min(1e-12)
    d1 = torch.log(forward / strike) / std + 0.5 * std
    d2 = d1 - std
    call = forward * ndtr(d1) - strike * ndtr(d2)
    put = strike * ndtr(-d2) - forward * ndtr(-d1)
    return torch.where(is_call, call, put)


def black_vega(forward, strike: Tensor, maturity: Tensor, vol: Tensor) -> Tensor:
    """Derivative of the Black price with respect to the volatility."""
    sqrt_t = torch.sqrt(maturity)
    std = (vol * sqrt_t).clamp_min(1e-12)
    d1 = torch.log(forward / strike) / std + 0.5 * std
    density = torch.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    return forward * density * sqrt_t


def implied_vol(price, forward, strike, maturity, is_call, max_vol=5.0, n_iterations=60):
    """Invert the Black formula by bisection on ``[0, max_vol]``.

    Slower than Newton, but it cannot diverge on noisy Monte Carlo prices, and 60
    halvings reach machine precision. Returns NaN where no volatility reproduces
    the price (outside the no-arbitrage bounds).
    """
    sqrt_t = torch.sqrt(maturity)
    low, high = torch.zeros_like(price), torch.full_like(price, max_vol)
    for _ in range(n_iterations):
        mid = 0.5 * (low + high)
        too_low = black_price(forward, strike, mid * sqrt_t, is_call) < price
        low, high = torch.where(too_low, mid, low), torch.where(too_low, high, mid)

    intrinsic = torch.where(is_call, forward - strike, strike - forward).clamp_min(0.0)
    upper_bound = black_price(forward, strike, max_vol * sqrt_t, is_call)
    attainable = (price > intrinsic) & (price < upper_bound)
    return torch.where(attainable, 0.5 * (low + high), torch.nan)
