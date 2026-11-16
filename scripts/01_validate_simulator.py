"""Steps 2-3: validate the rough Bergomi simulator before trusting anything built on it.

Checks, each with a figure in reports/figures and numbers in reports/validation.json:

1. Martingale property  E[S_t] = S_0  and flat forward variance  E[v_t] = xi0.
2. Prices against published reference values.
3. At-the-money skew against its short-maturity asymptotics, including H = 1/2.
4. Convergence in the number of paths, for three estimators.
5. Convergence in the number of time steps (discretisation bias).

Usage:
    python scripts/01_validate_simulator.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from rbergomi_nn.black_scholes import implied_vol
from rbergomi_nn.pricing import PriceEstimate, mean_and_stderr, mixed_estimator, standard_estimator
from rbergomi_nn.project import save_figure, save_report
from rbergomi_nn.simulator import DTYPE, RBergomiParams, simulate_spot, simulate_variance

# Parameters fitted to SPX by Bayer, Friz and Gatheral (2016).
BFG_PARAMS = RBergomiParams(hurst=0.07, eta=1.9, rho=-0.9, xi0=0.235**2)

# Call prices from Bayer, Ben Hammouda, Tempone (2020), Table 5.1: S0 = 1, T = 1.
LITERATURE_PRICES = [
    ("Set 1", RBergomiParams(0.07, 1.9, -0.9, 0.235**2), 1.0, 0.0791),
    ("Set 2", RBergomiParams(0.02, 0.4, -0.7, 0.1), 1.0, 0.1246),
    ("Set 3", RBergomiParams(0.02, 0.4, -0.7, 0.1), 0.8, 0.2412),
    ("Set 4", RBergomiParams(0.02, 0.4, -0.7, 0.1), 1.2, 0.0570),
]

BATCH_PATHS = 2**14  # paths simulated at once; larger runs are split in batches to bound memory


def tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=DTYPE)


def price_in_batches(
    params: RBergomiParams,
    maturities: list[float],
    strikes: torch.Tensor,
    steps_per_year: int,
    n_batches: int,
    seed: int,
) -> PriceEstimate:
    """Mixed-estimator prices averaged over independent batches of ``BATCH_PATHS`` paths."""
    n_steps = round(max(maturities) * steps_per_year)
    batches = []
    for batch in range(n_batches):
        generator = torch.Generator().manual_seed(seed + batch)
        paths = simulate_variance(params, n_steps, 1.0 / steps_per_year, BATCH_PATHS, generator)
        batches.append(mixed_estimator(paths, params.rho, tensor(maturities), strikes))
    price = torch.stack([b.price for b in batches]).mean(dim=0)
    stderr = torch.stack([b.stderr**2 for b in batches]).sum(dim=0).sqrt() / n_batches
    return PriceEstimate(price, stderr)


def check_martingale() -> dict:
    """E[S_t] and E[v_t] along the path, for rough and non-rough dynamics."""
    n_steps, dt, n_paths = 400, 1 / 200, 2**16
    times = dt * np.arange(1, n_steps + 1)
    figure, (spot_axis, variance_axis) = plt.subplots(1, 2, figsize=(11, 3.8))
    report = {}

    for hurst, colour in [(0.07, "C3"), (0.25, "C0"), (0.5, "C2")]:
        params = RBergomiParams(hurst, BFG_PARAMS.eta, BFG_PARAMS.rho, BFG_PARAMS.xi0)
        generator = torch.Generator().manual_seed(0)
        paths = simulate_variance(params, n_steps, dt, n_paths, generator)
        spot = simulate_spot(paths, params.rho, generator)

        spot_mean, spot_stderr = mean_and_stderr(spot[:, 1:], antithetic=True)
        variance_mean, variance_stderr = mean_and_stderr(paths.variance[:, 1:], antithetic=True)
        z_scores = ((spot_mean - 1.0) / spot_stderr).abs()
        report[f"H={hurst}"] = {
            "E[S_T] at T=2": spot_mean[-1].item(),
            "stderr": spot_stderr[-1].item(),
            "max |z-score| of E[S_t] - S_0 over the path": z_scores.max().item(),
            "E[v_T] / xi0 at T=2": (variance_mean[-1] / params.xi0).item(),
            "stderr of E[v_T] / xi0": (variance_stderr[-1] / params.xi0).item(),
        }

        for axis, mean, stderr, target in [
            (spot_axis, spot_mean, spot_stderr, 1.0),
            (variance_axis, variance_mean / params.xi0, variance_stderr / params.xi0, 1.0),
        ]:
            axis.plot(times, mean, color=colour, lw=1.2, label=f"H = {hurst}")
            axis.fill_between(
                times, target - 2 * stderr, target + 2 * stderr, color=colour, alpha=0.12
            )
            axis.axhline(target, color="k", lw=0.8)

    spot_axis.set(title="Martingale check: $E[S_t]$", xlabel="t (years)", ylabel="$E[S_t] / S_0$")
    variance_axis.set(
        title="Forward variance: $E[v_t]$", xlabel="t (years)", ylabel=r"$E[v_t] / \xi_0$"
    )
    spot_axis.legend(title="bands: ±2 standard errors")
    save_figure(figure, "validation_martingale")
    return report


def check_literature_prices() -> list[dict]:
    rows = []
    for name, params, strike, reference in LITERATURE_PRICES:
        estimate = price_in_batches(params, [1.0], tensor([strike]), 500, n_batches=16, seed=100)
        otm_price, stderr = estimate.price.item(), estimate.stderr.item()
        call_price = otm_price if strike >= 1.0 else otm_price + 1.0 - strike  # put-call parity
        rows.append(
            {
                "set": name,
                "strike": strike,
                "reference": reference,
                "this code": round(call_price, 5),
                "stderr": round(stderr, 6),
                "difference in stderr": round((call_price - reference) / stderr, 2),
            }
        )
        print(f"  {name}  K={strike}:  {call_price:.5f} ± {stderr:.5f}   reference {reference}")
    return rows


def atm_skew(params: RBergomiParams, maturities: list[float], steps_per_year: int) -> np.ndarray:
    """At-the-money skew d(implied vol)/d(log-strike) by central finite differences.

    Both strikes are priced on the same paths, so the difference is not swamped by noise.
    """
    skews = []
    for maturity in maturities:
        half_width = 0.1 * math.sqrt(params.xi0 * maturity)
        log_strikes = tensor([-half_width, half_width])
        strikes = torch.exp(log_strikes)
        estimate = price_in_batches(params, [maturity], strikes, steps_per_year, 4, seed=200)
        vols = implied_vol(
            estimate.price[0], tensor(1.0), strikes, tensor(maturity), is_call=strikes >= 1.0
        )
        skews.append(((vols[1] - vols[0]) / (2 * half_width)).item())
    return np.array(skews)


def check_skew_term_structure() -> dict:
    r"""Compare the ATM skew with its small-maturity limit (Alòs-León-Vives, Fukasawa)::

        skew(T) ~ rho * eta * sqrt(2H) / (2 (H + 1/2) (H + 3/2)) * T^(H - 1/2)

    For H = 1/2 this is the constant rho * eta / 4 of a lognormal stochastic volatility
    model; for H < 1/2 the skew explodes like a power law, the signature of rough volatility.
    """
    eta, rho, xi0 = 1.0, -0.7, 0.04
    maturities = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    report = {}

    for hurst, colour in [(0.1, "C3"), (0.3, "C0"), (0.5, "C2")]:
        simulated = atm_skew(RBergomiParams(hurst, eta, rho, xi0), maturities, steps_per_year=1600)
        constant = rho * eta * math.sqrt(2 * hurst) / (2 * (hurst + 0.5) * (hurst + 1.5))
        asymptotic = constant * np.array(maturities) ** (hurst - 0.5)
        fitted_slope = np.polyfit(np.log(maturities), np.log(-simulated), 1)[0]
        report[f"H={hurst}"] = {
            "fitted power-law exponent": round(float(fitted_slope), 3),
            "theoretical exponent H - 1/2": round(hurst - 0.5, 3),
            "simulated skew at shortest maturity": round(float(simulated[0]), 4),
            "asymptotic skew at shortest maturity": round(float(asymptotic[0]), 4),
        }
        axis.loglog(maturities, -simulated, "o", color=colour, label=f"H = {hurst}")
        axis.loglog(maturities, -asymptotic, "-", color=colour, lw=1)

    axis.set_xticks(maturities, [str(maturity) for maturity in maturities])
    axis.minorticks_off()
    axis.plot([], [], "ko", label="Monte Carlo")
    axis.plot([], [], "k-", lw=1, label="small-T asymptotics")

    axis.set(
        title="At-the-money skew term structure",
        xlabel="maturity T (years)",
        ylabel=r"$-\partial_k \sigma_{BS}(k=0, T)$",
    )
    axis.legend(fontsize=8)
    save_figure(figure, "validation_skew")
    return report


def check_convergence_in_paths() -> dict:
    """RMSE of the at-the-money price against the number of paths, for three estimators."""
    params, maturity, steps_per_year, n_repeats = BFG_PARAMS, 1.0, 100, 40
    maturities, strikes = tensor([maturity]), tensor([1.0])
    path_counts = [2**k for k in range(8, 16)]

    # Same time grid as the estimators under test, so the reference shares their
    # discretisation bias and the error measured here is purely statistical.
    reference = price_in_batches(params, [maturity], strikes, steps_per_year, 64, seed=300)
    reference_price = reference.price.item()

    estimators = ("standard", "mixed", "mixed + control variate")
    rmse = {name: [] for name in estimators}
    for n_paths in path_counts:
        errors = {name: [] for name in estimators}
        for repeat in range(n_repeats):
            generator = torch.Generator().manual_seed(10_000 + repeat)
            paths = simulate_variance(
                params, steps_per_year, 1 / steps_per_year, n_paths, generator
            )
            spot = simulate_spot(paths, params.rho, generator)
            prices = {
                "standard": standard_estimator(spot, paths, maturities, strikes),
                "mixed": mixed_estimator(
                    paths, params.rho, maturities, strikes, control_variate=False
                ),
                "mixed + control variate": mixed_estimator(paths, params.rho, maturities, strikes),
            }
            for name in estimators:
                errors[name].append(prices[name].price.item() - reference_price)
        for name in estimators:
            rmse[name].append(float(np.sqrt(np.mean(np.square(errors[name])))))

    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    for name, colour in zip(estimators, ("C7", "C0", "C3"), strict=True):
        axis.loglog(path_counts, rmse[name], "o-", color=colour, label=name)
    guide = rmse["standard"][0] * np.sqrt(path_counts[0] / np.array(path_counts))
    axis.loglog(path_counts, guide, "k--", lw=0.8, label=r"$\propto N^{-1/2}$")
    axis.set(
        title="Monte Carlo error vs number of paths (ATM, T = 1)",
        xlabel="number of paths N",
        ylabel="RMSE of the price",
    )
    axis.legend()
    save_figure(figure, "validation_convergence_paths")

    slopes = {
        name: round(float(np.polyfit(np.log(path_counts), np.log(rmse[name]), 1)[0]), 3)
        for name in estimators
    }
    variance_ratio = (np.array(rmse["standard"]) / np.array(rmse["mixed + control variate"])) ** 2
    return {
        "reference price": reference_price,
        "reference stderr": reference.stderr.item(),
        "fitted convergence rate (theory: -0.5)": slopes,
        "variance reduction, standard -> mixed + control variate": round(
            float(np.exp(np.mean(np.log(variance_ratio)))), 1
        ),
        "rmse": {
            name: dict(zip(map(str, path_counts), rmse[name], strict=True)) for name in estimators
        },
    }


def check_convergence_in_steps() -> dict:
    """At-the-money implied volatility against the number of time steps per year."""
    params = BFG_PARAMS
    maturities = [0.1, 1.0]
    step_counts = [50, 100, 200, 400, 800, 1600]
    vols = {maturity: [] for maturity in maturities}
    errors = {maturity: [] for maturity in maturities}

    for steps_per_year in step_counts:
        estimate = price_in_batches(params, maturities, tensor([1.0]), steps_per_year, 16, seed=400)
        for row, maturity in enumerate(maturities):
            # ATM: price ~ vol * sqrt(T / 2 pi), so the vol error is the price error rescaled.
            vol = implied_vol(
                estimate.price[row, 0],
                tensor(1.0),
                tensor(1.0),
                tensor(maturity),
                torch.tensor(True),
            ).item()
            vols[maturity].append(100 * vol)
            errors[maturity].append(
                100 * estimate.stderr[row, 0].item() * math.sqrt(2 * math.pi / maturity)
            )

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for axis, maturity in zip(axes, maturities, strict=True):
        axis.errorbar(
            step_counts, vols[maturity], yerr=2 * np.array(errors[maturity]), fmt="o-", capsize=3
        )
        axis.axvline(200, color="C3", ls=":", label="used for the dataset")
        axis.set_xscale("log")
        axis.set(
            title=f"ATM implied vol, T = {maturity}",
            xlabel="time steps per year",
            ylabel="implied vol (%)",
        )
        axis.legend()
    save_figure(figure, "validation_convergence_steps")

    return {
        f"T={maturity}": {
            "steps per year": step_counts,
            "ATM implied vol (%)": [round(v, 3) for v in vols[maturity]],
            "stderr (vol pts)": [round(e, 3) for e in errors[maturity]],
        }
        for maturity in maturities
    }


def main() -> None:
    torch.set_grad_enabled(False)
    checks = {
        "martingale": check_martingale,
        "literature_prices": check_literature_prices,
        "skew_term_structure": check_skew_term_structure,
        "convergence_in_paths": check_convergence_in_paths,
        "convergence_in_steps": check_convergence_in_steps,
    }
    report = {}
    for name, check in checks.items():
        print(f"== {name}")
        report[name] = check()
    save_report(report, "validation")


if __name__ == "__main__":
    main()
