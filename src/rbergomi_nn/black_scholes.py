"""Black-Scholes formulas with zero rates and dividends.

Everything in this project is quoted on the forward, so prices are undiscounted
and the only inputs are forward, strike and total standard deviation
``total_std = vol * sqrt(maturity)``.

All functions broadcast their tensor arguments.
"""

import math

import torch
from torch import Tensor
from torch.special import ndtr

_MIN_TOTAL_STD = 1e-12


def black_price(forward: Tensor, strike: Tensor, total_std: Tensor, is_call: Tensor) -> Tensor:
    """Undiscounted Black price of a European call (``is_call``) or put."""
    std = total_std.clamp_min(_MIN_TOTAL_STD)
    d1 = torch.log(forward / strike) / std + 0.5 * std
    d2 = d1 - std
    call = forward * ndtr(d1) - strike * ndtr(d2)
    put = strike * ndtr(-d2) - forward * ndtr(-d1)
    return torch.where(is_call, call, put)


def black_vega(forward: Tensor, strike: Tensor, maturity: Tensor, vol: Tensor) -> Tensor:
    """Derivative of the Black price with respect to the volatility."""
    sqrt_t = torch.sqrt(maturity)
    std = (vol * sqrt_t).clamp_min(_MIN_TOTAL_STD)
    d1 = torch.log(forward / strike) / std + 0.5 * std
    density = torch.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    return forward * density * sqrt_t


def implied_vol(
    price: Tensor,
    forward: Tensor,
    strike: Tensor,
    maturity: Tensor,
    is_call: Tensor,
    max_vol: float = 5.0,
    n_iterations: int = 60,
) -> Tensor:
    """Invert the Black formula by bisection.

    Bisection is slower than Newton but cannot diverge, which matters when the
    input prices carry Monte Carlo noise. Sixty halvings of ``[0, max_vol]``
    resolve the volatility to machine precision.

    Returns NaN where the price violates the no-arbitrage bounds, i.e. where no
    volatility in ``(0, max_vol)`` reproduces it.
    """
    price, forward, strike, maturity, is_call = torch.broadcast_tensors(
        price, forward, strike, maturity, is_call
    )
    sqrt_t = torch.sqrt(maturity)

    low = torch.zeros_like(price)
    high = torch.full_like(price, max_vol)
    for _ in range(n_iterations):
        mid = 0.5 * (low + high)
        model_price_too_low = black_price(forward, strike, mid * sqrt_t, is_call) < price
        low = torch.where(model_price_too_low, mid, low)
        high = torch.where(model_price_too_low, high, mid)
    vol = 0.5 * (low + high)

    intrinsic = torch.where(is_call, forward - strike, strike - forward).clamp_min(0.0)
    upper_bound = black_price(forward, strike, max_vol * sqrt_t, is_call)
    attainable = (price > intrinsic) & (price < upper_bound)
    return torch.where(attainable, vol, torch.full_like(vol, float("nan")))
