import pytest
import torch

from rbergomi_nn.black_scholes import black_price
from rbergomi_nn.pricing import mixed_estimator, standard_estimator
from rbergomi_nn.simulator import DTYPE, RBergomiParams, simulate_spot, simulate_variance
from rbergomi_nn.surface import MonteCarloConfig, SurfaceGrid, implied_vol_surface

MATURITIES = torch.tensor([0.5, 1.0], dtype=DTYPE)
STRIKES = torch.tensor([0.85, 1.0, 1.1], dtype=DTYPE)


def simulate(params: RBergomiParams, n_paths: int = 40_000, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    paths = simulate_variance(params, n_steps=100, dt=0.01, n_paths=n_paths, generator=generator)
    return paths, generator


def test_mixed_estimator_is_exact_without_vol_of_vol():
    """With eta = 0 the conditional price is the Black-Scholes price on every path."""
    params = RBergomiParams(hurst=0.1, eta=0.0, rho=0.0, xi0=0.04)
    paths, _ = simulate(params, n_paths=100)
    estimate = mixed_estimator(paths, params.rho, MATURITIES, STRIKES)

    forward = torch.tensor(1.0, dtype=DTYPE)
    exact = black_price(forward, STRIKES, 0.2 * MATURITIES.sqrt()[:, None], STRIKES >= 1.0)
    assert torch.allclose(estimate.price, exact, atol=1e-12)
    assert estimate.stderr.max() < 1e-12


def test_mixed_and_standard_estimators_agree():
    params = RBergomiParams(hurst=0.1, eta=1.9, rho=-0.7, xi0=0.04)
    paths, generator = simulate(params)
    mixed = mixed_estimator(paths, params.rho, MATURITIES, STRIKES)
    spot_paths = simulate_spot(paths, params.rho, generator)
    standard = standard_estimator(spot_paths, paths, MATURITIES, STRIKES)

    combined_stderr = (mixed.stderr**2 + standard.stderr**2).sqrt()
    assert ((mixed.price - standard.price).abs() < 4 * combined_stderr).all()


def test_mixed_estimator_reduces_variance():
    params = RBergomiParams(hurst=0.1, eta=1.9, rho=-0.7, xi0=0.04)
    paths, generator = simulate(params)
    mixed = mixed_estimator(paths, params.rho, MATURITIES, STRIKES)
    spot_paths = simulate_spot(paths, params.rho, generator)
    standard = standard_estimator(spot_paths, paths, MATURITIES, STRIKES)
    assert (mixed.stderr < standard.stderr).all()


def test_control_variate_does_not_move_the_price():
    params = RBergomiParams(hurst=0.1, eta=1.9, rho=-0.7, xi0=0.04)
    paths, _ = simulate(params)
    plain = mixed_estimator(paths, params.rho, MATURITIES, STRIKES, control_variate=False)
    controlled = mixed_estimator(paths, params.rho, MATURITIES, STRIKES, control_variate=True)
    assert ((plain.price - controlled.price).abs() < 4 * plain.stderr).all()
    assert (controlled.stderr <= plain.stderr * 1.001).all()


# Reference call prices from Bayer, Ben Hammouda, Tempone (2020), "Hierarchical adaptive
# sparse grids and quasi-Monte Carlo for option pricing under the rough Bergomi model",
# Table 5.1 (S0 = 1, T = 1; Monte Carlo with 500 time steps and 8e6 paths).
REFERENCE_PRICES = [
    # (H, eta, rho, xi0), strike, call price
    ((0.07, 1.9, -0.9, 0.235**2), 1.0, 0.0791),
    ((0.02, 0.4, -0.7, 0.1), 1.0, 0.1246),
    ((0.02, 0.4, -0.7, 0.1), 0.8, 0.2412),
    ((0.02, 0.4, -0.7, 0.1), 1.2, 0.0570),
]


@pytest.mark.parametrize("theta, strike, reference", REFERENCE_PRICES)
def test_prices_match_literature(theta, strike, reference):
    params = RBergomiParams(*theta)
    generator = torch.Generator().manual_seed(0)
    paths = simulate_variance(params, n_steps=500, dt=1 / 500, n_paths=40_000, generator=generator)
    estimate = mixed_estimator(
        paths, params.rho, torch.tensor([1.0], dtype=DTYPE), torch.tensor([strike], dtype=DTYPE)
    )
    otm_price, stderr = estimate.price.item(), estimate.stderr.item()
    call_price = otm_price if strike >= 1.0 else otm_price + 1.0 - strike  # put-call parity

    # 4 standard errors plus the rounding of the published figure (4 decimals).
    assert abs(call_price - reference) < 4 * stderr + 1e-4


def test_surface_has_expected_shape_and_equity_like_skew():
    grid = SurfaceGrid()
    params = RBergomiParams(hurst=0.1, eta=1.9, rho=-0.7, xi0=0.04)
    surface = implied_vol_surface(params, grid, MonteCarloConfig(n_paths=4_000), seed=0)

    assert surface.implied_vol.shape == grid.shape
    assert surface.implied_vol.isfinite().all()
    assert (surface.implied_vol_stderr > 0).all()
    # Negative correlation: implied volatility decreases with the strike.
    assert (surface.implied_vol[:, 0] > surface.implied_vol[:, -1]).all()


def test_surface_is_reproducible_given_the_seed():
    grid, mc = SurfaceGrid(), MonteCarloConfig(n_paths=1_000)
    params = RBergomiParams(hurst=0.2, eta=1.5, rho=-0.5, xi0=0.04)
    first = implied_vol_surface(params, grid, mc, seed=7).implied_vol
    second = implied_vol_surface(params, grid, mc, seed=7).implied_vol
    other = implied_vol_surface(params, grid, mc, seed=8).implied_vol
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_zero_correlation_surface_is_finite():
    """rho = 0 makes the control variate degenerate; the estimator must cope."""
    params = RBergomiParams(hurst=0.1, eta=1.5, rho=0.0, xi0=0.04)
    surface = implied_vol_surface(params, SurfaceGrid(), MonteCarloConfig(n_paths=2_000), seed=0)
    assert surface.implied_vol.isfinite().all()
