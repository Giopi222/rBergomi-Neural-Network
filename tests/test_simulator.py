"""Statistical tests of the simulator.

Seeds are fixed, so the tests are deterministic. Tolerances on Monte Carlo
means are expressed in standard errors: a correct simulator fails a
4-standard-error test with probability 6e-5.
"""

import math

import pytest
import torch

from rbergomi_nn.pricing import mean_and_stderr
from rbergomi_nn.simulator import (
    DTYPE,
    RBergomiParams,
    simulate_spot,
    simulate_variance,
    simulate_volterra,
)

N_PATHS = 40_000
N_STEPS = 100
DT = 0.01
ROUGH = RBergomiParams(hurst=0.1, eta=1.9, rho=-0.9, xi0=0.04)


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_antithetic_paths_are_mirror_images():
    dW, volterra = simulate_volterra(torch.tensor(0.1, dtype=DTYPE), 10, DT, 8, generator())
    assert torch.equal(dW[:4], -dW[4:])
    assert torch.allclose(volterra[:4], -volterra[4:])


def test_hurst_one_half_gives_brownian_motion_path_by_path():
    """With H = 1/2 the kernel is constant, so Y must equal W exactly."""
    dW, volterra = simulate_volterra(torch.tensor(0.5, dtype=DTYPE), N_STEPS, DT, 100, generator())
    assert torch.allclose(volterra, torch.cumsum(dW, dim=1), atol=1e-12)


@pytest.mark.parametrize("hurst", [0.05, 0.1, 0.3])
def test_volterra_variance_matches_theory(hurst):
    """Var(Y_t) = t^(2H) / (2H); the hybrid scheme reproduces it up to a small bias."""
    _, volterra = simulate_volterra(
        torch.tensor(hurst, dtype=DTYPE), N_STEPS, DT, N_PATHS, generator()
    )
    times = DT * torch.arange(1, N_STEPS + 1, dtype=DTYPE)
    theoretical = times ** (2 * hurst) / (2 * hurst)
    relative_error = volterra.var(dim=0) / theoretical - 1.0
    assert relative_error.abs().max() < 0.05


def test_brownian_volterra_covariance_matches_theory():
    """Cov(W_t, Y_t) = t^(H + 1/2) / (H + 1/2): this is what drives the skew."""
    hurst = 0.1
    dW, volterra = simulate_volterra(
        torch.tensor(hurst, dtype=DTYPE), N_STEPS, DT, N_PATHS, generator()
    )
    brownian_at_maturity = dW.sum(dim=1)
    covariance = (brownian_at_maturity * volterra[:, -1]).mean().item()
    assert covariance == pytest.approx(1.0 / (hurst + 0.5), rel=0.03)  # t = 1


def test_forward_variance_is_flat():
    """E[v_t] = xi0 at every time."""
    paths = simulate_variance(ROUGH, N_STEPS, DT, N_PATHS, generator())
    mean, stderr = mean_and_stderr(paths.variance[:, 1:], paths.antithetic)
    # The scheme slightly underestimates Var(Y), hence a small known bias on top of the noise.
    assert ((mean - ROUGH.xi0).abs() < 4 * stderr + 0.02 * ROUGH.xi0).all()


@pytest.mark.parametrize("hurst", [0.07, 0.3, 0.5])
def test_spot_is_a_martingale(hurst):
    """E[S_t] = S_0 at every time."""
    params = RBergomiParams(hurst=hurst, eta=1.9, rho=-0.9, xi0=0.04)
    random = generator()
    paths = simulate_variance(params, N_STEPS, DT, N_PATHS, random)
    spot = simulate_spot(paths, params.rho, random, spot=100.0)
    mean, stderr = mean_and_stderr(spot[:, 1:], paths.antithetic)
    assert ((mean - 100.0).abs() < 4 * stderr).all()


def test_zero_vol_of_vol_gives_black_scholes_spot():
    """With eta = 0 the variance is constant and S_T is lognormal with variance xi0 * T."""
    params = RBergomiParams(hurst=0.1, eta=0.0, rho=-0.5, xi0=0.09)
    random = generator()
    paths = simulate_variance(params, N_STEPS, DT, N_PATHS, random)
    assert torch.allclose(paths.variance, torch.full_like(paths.variance, 0.09))

    log_spot = simulate_spot(paths, params.rho, random)[:, -1].log()
    assert log_spot.var().item() == pytest.approx(0.09, rel=0.03)
    assert log_spot.mean().item() == pytest.approx(-0.045, abs=4 * 0.3 / math.sqrt(N_PATHS))


def test_simulation_is_differentiable_in_the_parameters():
    """Gradients must flow to all parameters: required for differential ML later on."""
    theta = torch.tensor([0.1, 1.9, -0.9, 0.04], dtype=DTYPE, requires_grad=True)
    params = RBergomiParams(*theta)
    random = generator()
    paths = simulate_variance(params, 20, DT, 1000, random)
    spot = simulate_spot(paths, params.rho, random)
    payoff = (spot[:, -1] - 1.0).clamp_min(0.0).mean()
    (gradient,) = torch.autograd.grad(payoff, theta)
    assert gradient.isfinite().all() and (gradient != 0).all()
