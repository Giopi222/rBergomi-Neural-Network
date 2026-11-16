"""Step 7: synthetic calibration. Can the network recover parameters it should know?

Experiments (the "true" parameters are always known):

A. Targets produced by the network itself. Tests the optimiser and the
   identifiability of the parameters in isolation: the fit can be perfect.
B. Targets produced by Monte Carlo (test split). The realistic noise-free case:
   the network's approximation error now leaks into the parameters.
C. As B, with Gaussian noise added to the targets, mimicking bid-ask noise.
D. Targets the network was never trained for (gap and out-of-domain splits).
E. One calibration done directly with Monte Carlo, to time the alternative.

Usage:
    python scripts/05_calibrate.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from rbergomi_nn.calibration import (
    CalibrationResult,
    Pricer,
    calibrate,
    monte_carlo_pricer,
    network_pricer,
)
from rbergomi_nn.dataset import TRAIN_BOX, SurfaceDataset
from rbergomi_nn.network import VolSurfaceMLP
from rbergomi_nn.project import DATASET_PATH, MODEL_PATH, save_figure, save_report
from rbergomi_nn.simulator import PARAMETER_NAMES

PARAMETER_LABELS = ("H", r"$\eta$", r"$\rho$", r"$\xi_0$")
NOISE_LEVELS_VOL_POINTS = (0.0, 0.1, 0.25, 0.5, 1.0)
N_NOISE_SURFACES = 300
BOX_WIDTH = TRAIN_BOX.upper_array - TRAIN_BOX.lower_array


def calibrate_all(pricer: Pricer, targets: np.ndarray) -> tuple[list[CalibrationResult], float]:
    """Calibrate to every row of ``targets``; also return the mean time per calibration."""
    start = time.perf_counter()
    results = [calibrate(pricer, target, TRAIN_BOX) for target in targets]
    return results, (time.perf_counter() - start) / len(targets)


def recovery_summary(results: list[CalibrationResult], true_theta: np.ndarray) -> dict:
    """How far the calibrated parameters are from the true ones."""
    calibrated = np.array([result.theta for result in results])
    absolute_error = np.abs(calibrated - true_theta)
    summary = {
        "n_surfaces": len(results),
        "median_fit_rmse_vol_points": round(100 * float(np.median([r.rmse for r in results])), 4),
        "fraction_with_a_parameter_at_bound": round(
            float(np.mean([r.at_bound.any() for r in results])), 4
        ),
        "mean_surface_evaluations": round(float(np.mean([r.n_evaluations for r in results])), 1),
    }
    for index, name in enumerate(PARAMETER_NAMES):
        summary[name] = {
            "median_abs_error": float(f"{np.median(absolute_error[:, index]):.3g}"),
            "p95_abs_error": float(f"{np.quantile(absolute_error[:, index], 0.95):.3g}"),
            # Error as a share of the training range, comparable across parameters.
            "median_error_pct_of_range": round(
                100 * float(np.median(absolute_error[:, index]) / BOX_WIDTH[index]), 3
            ),
            "p95_error_pct_of_range": round(
                100 * float(np.quantile(absolute_error[:, index], 0.95) / BOX_WIDTH[index]), 3
            ),
        }
    return summary


def print_summary(title: str, summary: dict) -> None:
    print(
        f"\n{title}  (n = {summary['n_surfaces']}, "
        f"fit RMSE {summary['median_fit_rmse_vol_points']} vol pts, "
        f"at bound {100 * summary['fraction_with_a_parameter_at_bound']:.1f}%)"
    )
    for name in PARAMETER_NAMES:
        row = summary[name]
        print(
            f"  {name:4s} median |error| {row['median_abs_error']:<9.3g} "
            f"({row['median_error_pct_of_range']:.2f}% of range)   "
            f"95%: {row['p95_error_pct_of_range']:.2f}% of range"
        )


def plot_recovery(
    results: list[CalibrationResult], true_theta: np.ndarray, name: str, title: str
) -> None:
    calibrated = np.array([result.theta for result in results])
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for index, (axis, label) in enumerate(zip(axes, PARAMETER_LABELS, strict=True)):
        low, high = TRAIN_BOX.lower[index], TRAIN_BOX.upper[index]
        axis.scatter(true_theta[:, index], calibrated[:, index], s=4, alpha=0.4)
        axis.plot([low, high], [low, high], "k-", lw=0.8)
        axis.set(title=label, xlabel="true", ylabel="calibrated" if index == 0 else None)
    figure.suptitle(title, y=1.03)
    save_figure(figure, name)


def plot_noise_sweep(sweep: dict[float, dict]) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    levels = list(sweep)
    for name, label in zip(PARAMETER_NAMES, PARAMETER_LABELS, strict=True):
        axis.plot(
            levels,
            [sweep[level][name]["median_error_pct_of_range"] for level in levels],
            "o-",
            label=label,
        )
    axis.set(
        title="Parameter recovery under noisy quotes",
        xlabel="noise added to each implied vol (vol points, std dev)",
        ylabel="median |error| (% of training range)",
    )
    axis.legend()
    save_figure(figure, "calibration_noise")


def time_monte_carlo_calibration(
    dataset: SurfaceDataset, target: np.ndarray, true_theta: np.ndarray
) -> dict:
    """One calibration with the simulator in the loop (finite-difference Jacobian)."""
    simulator = monte_carlo_pricer(dataset.grid, dataset.mc, seed=12_345)
    n_simulations = 0

    def counted_surface(theta: np.ndarray) -> np.ndarray:
        nonlocal n_simulations
        n_simulations += 1
        return simulator.surface(theta)

    start = time.perf_counter()
    result = calibrate(Pricer(counted_surface), target, TRAIN_BOX)
    elapsed = time.perf_counter() - start
    return {
        "seconds": round(elapsed, 1),
        "surfaces_simulated": n_simulations,
        "true_theta": true_theta.tolist(),
        "calibrated_theta": [float(f"{value:.5g}") for value in result.theta],
        "fit_rmse_vol_points": round(100 * result.rmse, 4),
    }


def main() -> None:
    dataset = SurfaceDataset.load(DATASET_PATH)
    model = VolSurfaceMLP.load(MODEL_PATH)
    pricer = network_pricer(model)
    rng = np.random.default_rng(0)
    report = {}

    test, gap, ood = dataset.subset("test"), dataset.subset("gap"), dataset.subset("ood")

    # A. Network-generated targets: the true parameters reproduce the target exactly.
    network_targets = np.array([pricer.surface(theta) for theta in test.theta])
    results, _ = calibrate_all(pricer, network_targets)
    report["A_network_targets"] = recovery_summary(results, test.theta)
    print_summary("A. network-generated targets", report["A_network_targets"])

    # B. Monte Carlo targets.
    results, seconds_per_calibration = calibrate_all(pricer, test.flat_implied_vol)
    report["B_monte_carlo_targets"] = recovery_summary(results, test.theta)
    print_summary("B. Monte Carlo targets (test split)", report["B_monte_carlo_targets"])
    plot_recovery(
        results,
        test.theta,
        "calibration_recovery",
        "Calibration to Monte Carlo surfaces of the test split",
    )

    # C. Monte Carlo targets plus quote noise.
    subset = slice(0, N_NOISE_SURFACES)
    sweep = {}
    for level in NOISE_LEVELS_VOL_POINTS:
        targets = test.flat_implied_vol[subset]
        noisy_targets = targets + 0.01 * level * rng.standard_normal(targets.shape)
        results, _ = calibrate_all(pricer, noisy_targets)
        sweep[level] = recovery_summary(results, test.theta[subset])
        print_summary(f"C. noise {level} vol pts", sweep[level])
    report["C_noisy_targets"] = {str(level): summary for level, summary in sweep.items()}
    plot_noise_sweep(sweep)

    # D. Surfaces the network was not trained for.
    results, _ = calibrate_all(pricer, gap.flat_implied_vol)
    report["D_gap_targets"] = recovery_summary(results, gap.theta)
    print_summary("D. gap split (H never seen, but inside the box)", report["D_gap_targets"])
    plot_recovery(
        results, gap.theta, "calibration_recovery_gap", "Calibration to surfaces from the Hurst gap"
    )

    results, _ = calibrate_all(pricer, ood.flat_implied_vol)
    report["D_out_of_domain_targets"] = recovery_summary(results, ood.theta)
    print_summary(
        "D. out-of-domain split (true parameters outside the box)",
        report["D_out_of_domain_targets"],
    )

    # E. Timing against a Monte Carlo calibration of the first test surface.
    print("\nE. timing one Monte Carlo calibration ...")
    monte_carlo = time_monte_carlo_calibration(dataset, test.flat_implied_vol[0], test.theta[0])
    network_first = calibrate(pricer, test.flat_implied_vol[0], TRAIN_BOX)
    report["E_timing"] = {
        "network_milliseconds_per_calibration": round(1e3 * seconds_per_calibration, 2),
        "monte_carlo_calibration": monte_carlo,
        "network_calibrated_theta_same_surface": [
            float(f"{value:.5g}") for value in network_first.theta
        ],
        "speedup": round(monte_carlo["seconds"] / seconds_per_calibration),
    }
    print(f"  network: {1e3 * seconds_per_calibration:.1f} ms per calibration")
    print(
        f"  Monte Carlo: {monte_carlo['seconds']} s  -> speed-up x{report['E_timing']['speedup']}"
    )
    print(f"  true {test.theta[0]}")
    print(f"  MC   {monte_carlo['calibrated_theta']}")
    print(f"  NN   {report['E_timing']['network_calibrated_theta_same_surface']}")

    save_report(report, "calibration")


if __name__ == "__main__":
    main()
