"""Step 6b: where is the network accurate, where does it fail, and how fast is it?

Compares the network with the Monte Carlo surfaces of every split:

* ``test``  same distribution as the training data (interpolation),
* ``gap``   a slab of Hurst exponents removed from training but surrounded by data,
* ``ood``   parameters outside the training box (extrapolation).

All errors are in volatility points (1 vol point = 0.01 of implied volatility).

Usage:
    python scripts/04_evaluate_network.py
"""

import time
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import torch

from rbergomi_nn.dataset import HURST_GAP, SPLITS, TRAIN_BOX, SurfaceDataset
from rbergomi_nn.network import VolSurfaceMLP
from rbergomi_nn.project import DATASET_PATH, MODEL_PATH, PARAMETER_LABELS, save_figure, save_report
from rbergomi_nn.simulator import DTYPE, PARAMETER_NAMES, RBergomiParams
from rbergomi_nn.surface import implied_vol_surface


def predict(model, dataset):
    """Network surfaces for every sample, shape ``(n, n_maturities, n_strikes)``."""
    with torch.no_grad():
        flat = model(torch.as_tensor(dataset.theta, dtype=DTYPE)).numpy()
    return flat.reshape(dataset.implied_vol.shape)


def surface_rmse(error):
    """Root mean squared error of each surface over its grid points."""
    return np.sqrt(np.mean(error**2, axis=(1, 2)))


def error_summary(error, mc_stderr):
    """Error statistics of one split, in vol points, next to the Monte Carlo noise.

    The Monte Carlo surfaces are themselves noisy, so the measured error can
    never fall below ``mc_noise``: the last entry removes it in quadrature to
    estimate the error against the true model surface.
    """
    absolute = 100 * np.abs(error)
    rmse = float(np.sqrt(np.mean(absolute**2)))
    mc_noise = 100 * float(np.sqrt(np.mean(mc_stderr**2)))
    stats = {
        "rmse": rmse,
        "mean_abs": absolute.mean(),
        "p95_abs": np.quantile(absolute, 0.95),
        "p99_abs": np.quantile(absolute, 0.99),
        "max_abs": absolute.max(),
        "mc_noise": mc_noise,
        "rmse_net_of_mc_noise": np.sqrt(max(rmse**2 - mc_noise**2, 0.0)),
    }
    return {"n_surfaces": len(error)} | {key: round(float(v), 4) for key, v in stats.items()}


def plot_error_by_split(errors):
    figure, axis = plt.subplots(figsize=(6.2, 4))
    axis.boxplot(
        [100 * surface_rmse(error) for error in errors.values()],
        tick_labels=[f"{name}\n(n = {len(error)})" for name, error in errors.items()],
        whis=(1, 99),
        showfliers=False,
    )
    axis.set_yscale("log")
    axis.set(title="Network error per surface, by split", ylabel="RMSE over the grid (vol points)")
    note = "boxes: quartiles, whiskers: 1%–99%"
    axis.text(0.02, 0.97, note, transform=axis.transAxes, va="top", fontsize=8)
    save_figure(figure, "error_by_split")


def plot_error_over_grid(test, test_error):
    """Mean absolute network error next to the Monte Carlo standard error, per grid point."""
    network = 100 * np.abs(test_error).mean(axis=0)
    monte_carlo = 100 * test.implied_vol_stderr.mean(axis=0)
    upper = max(network.max(), monte_carlo.max())
    moneyness, maturities = test.grid.scaled_moneyness, test.grid.maturities
    columns = range(0, len(moneyness), 2)  # every other strike, for legibility

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    panels = [
        (network, "Network: mean |error| on test set"),
        (monte_carlo, "Monte Carlo: mean standard error"),
    ]
    for axis, (values, title) in zip(axes, panels):
        image = axis.imshow(
            values, aspect="auto", cmap="viridis", vmin=0, vmax=upper, origin="lower"
        )
        axis.set_xticks(columns, [f"{moneyness[c]:+.2f}" for c in columns])
        axis.set_yticks(range(len(maturities)), maturities)
        axis.set(title=title, xlabel=r"scaled log-moneyness  $\log(K/S_0) / \sqrt{T}$")
        axis.grid(False)
    axes[0].set_ylabel("maturity (years)")
    figure.colorbar(image, ax=axes, label="vol points")
    save_figure(figure, "error_over_grid")


def plot_extrapolation(ood, ood_error, test_rmse):
    """Error against the distance from the training box."""
    rmse = 100 * surface_rmse(ood_error)
    distance = TRAIN_BOX.distance_outside(ood.theta)

    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.scatter(distance, rmse, s=5, alpha=0.4, color="C3", label="out-of-domain surfaces")
    axis.axhline(100 * np.median(test_rmse), color="C0", label="in-domain median")
    axis.axhline(
        100 * np.quantile(test_rmse, 0.99), color="C0", ls="--", label="in-domain 99% quantile"
    )
    axis.set_yscale("log")
    axis.set(
        title="Extrapolation error grows with distance from the training box",
        xlabel="distance outside the box (1 = one half-width in one parameter)",
        ylabel="RMSE over the grid (vol points)",
    )
    axis.legend(fontsize=8)
    save_figure(figure, "extrapolation_distance")

    binned = {}
    for low, high in pairwise([0.0, 0.25, 0.5, 1.0, np.inf]):
        in_bin = (distance > low) & (distance <= high)
        binned[f"distance in ({low}, {high}]"] = {
            "n_surfaces": int(in_bin.sum()),
            "median_rmse": round(float(np.median(rmse[in_bin])), 4),
            "p95_rmse": round(float(np.quantile(rmse[in_bin], 0.95)), 4),
        }
    return binned


def plot_extrapolation_by_parameter(ood, ood_error, test_rmse):
    """Which parameter hurts most when extrapolated?

    Uses the surfaces with exactly one parameter outside the training box.
    """
    rmse = 100 * surface_rmse(ood_error)
    normalised = TRAIN_BOX.normalise(ood.theta)
    outside = np.abs(normalised) > 1
    exactly_one_outside = outside.sum(axis=1) == 1

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    report = {}
    for index, (axis, label, name) in enumerate(zip(axes, PARAMETER_LABELS, PARAMETER_NAMES)):
        selected = exactly_one_outside & outside[:, index]
        position, selected_rmse = normalised[selected, index], rmse[selected]
        axis.scatter(position, selected_rmse, s=8, alpha=0.6, color="C3")
        axis.axvspan(-1, 1, color="C0", alpha=0.08, label="training range")
        axis.axhline(100 * np.median(test_rmse), color="C0", lw=1, label="in-domain median")
        axis.set_yscale("log")
        axis.set(title=f"only {label} outside", xlabel=f"{label}, normalised")
        for side, mask in [("below", position < -1), ("above", position > 1)]:
            if mask.any():
                report[f"{name} {side} range"] = {
                    "n_surfaces": int(mask.sum()),
                    "median_rmse": round(float(np.median(selected_rmse[mask])), 4),
                    "max_rmse": round(float(selected_rmse[mask].max()), 4),
                }
    axes[0].set_ylabel("RMSE over the grid (vol points)")
    axes[0].legend(fontsize=8)
    save_figure(figure, "extrapolation_by_parameter")
    return report


def plot_gap(gap, gap_error, test, test_error):
    """Error against H across the slab that was removed from the training data."""
    test_rmse, gap_rmse = 100 * surface_rmse(test_error), 100 * surface_rmse(gap_error)
    figure, axis = plt.subplots(figsize=(6.2, 4))
    axis.scatter(test.theta[:, 0], test_rmse, s=5, alpha=0.4, label="test (H seen in training)")
    axis.scatter(gap.theta[:, 0], gap_rmse, s=5, alpha=0.4, color="C1", label="gap (H never seen)")
    axis.axvspan(*HURST_GAP, color="C1", alpha=0.08)
    axis.set_yscale("log")
    axis.set(
        title="Interpolating across a hole in the training data",
        xlabel="Hurst exponent H",
        ylabel="RMSE over the grid (vol points)",
    )
    axis.legend(fontsize=8)
    save_figure(figure, "interpolation_gap")


def plot_example_smiles(model, datasets, errors):
    """Network against Monte Carlo smiles: a typical test surface and the worst OOD one."""
    test_rmse, ood_rmse = surface_rmse(errors["test"]), surface_rmse(errors["ood"])
    examples = [
        ("test, median error", datasets["test"], int(np.argsort(test_rmse)[len(test_rmse) // 2])),
        ("out-of-domain, worst error", datasets["ood"], int(np.argmax(ood_rmse))),
    ]
    maturity_rows = [0, 3, 7]

    figure, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for axis_row, (title, dataset, sample) in zip(axes, examples):
        network = predict(model, dataset)[sample]
        theta = ", ".join(
            f"{label} = {value:.3g}"
            for label, value in zip(PARAMETER_LABELS, dataset.theta[sample])
        )
        for axis, row in zip(axis_row, maturity_rows):
            log_moneyness = dataset.grid.log_moneyness[row].numpy()
            axis.errorbar(
                log_moneyness,
                100 * dataset.implied_vol[sample, row],
                yerr=200 * dataset.implied_vol_stderr[sample, row],
                fmt="o",
                ms=3,
                capsize=2,
                color="k",
                label="Monte Carlo ± 2 s.e.",
            )
            axis.plot(log_moneyness, 100 * network[row], color="C3", label="network")
            axis.set(title=f"T = {dataset.grid.maturities[row]}", xlabel=r"$\log(K/S_0)$")
        axis_row[0].set_ylabel("implied vol (%)")
        axis_row[0].legend(fontsize=8)
        axis_row[1].set_title(
            f"{title}\n{theta}\nT = {dataset.grid.maturities[maturity_rows[1]]}", fontsize=9
        )
    figure.tight_layout()
    save_figure(figure, "example_smiles")


def measure_speed(model, dataset):
    """Wall-clock time of one surface: network (single and batched) against Monte Carlo."""
    centre = torch.as_tensor(TRAIN_BOX.centre, dtype=DTYPE)
    single = centre[None]
    batch = torch.as_tensor(dataset.theta[:10_000], dtype=DTYPE)

    def seconds_per_call(function, repeats):
        function()  # warm-up
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        return (time.perf_counter() - start) / repeats

    params = RBergomiParams(*TRAIN_BOX.centre)
    with torch.no_grad():
        network_single = seconds_per_call(lambda: model(single), repeats=2_000)
        network_batch = seconds_per_call(lambda: model(batch), repeats=20) / len(batch)
        network_jacobian = seconds_per_call(lambda: model.jacobian(centre), repeats=2_000)
        monte_carlo = seconds_per_call(
            lambda: implied_vol_surface(params, dataset.grid, dataset.mc, seed=0), repeats=3
        )

    return {
        "torch_threads": torch.get_num_threads(),
        "monte_carlo_seconds_per_surface": round(monte_carlo, 3),
        "monte_carlo_paths": dataset.mc.n_paths,
        "network_microseconds_per_surface_single_call": round(1e6 * network_single, 1),
        "network_microseconds_per_surface_batched": round(1e6 * network_batch, 2),
        "network_microseconds_per_jacobian": round(1e6 * network_jacobian, 1),
        "speedup_single_call": round(monte_carlo / network_single),
        "speedup_batched": round(monte_carlo / network_batch),
    }


def main():
    dataset = SurfaceDataset.load(DATASET_PATH)
    model = VolSurfaceMLP.load(MODEL_PATH)

    datasets = {name: dataset.subset(name) for name in SPLITS}
    errors = {
        name: predict(model, subset) - subset.implied_vol for name, subset in datasets.items()
    }
    summary = {
        name: error_summary(errors[name], subset.implied_vol_stderr)
        for name, subset in datasets.items()
    }
    print(
        f"{'split':6s} {'n':>6s} {'rmse':>8s} {'mean':>8s} {'p99':>8s} {'max':>8s} "
        f"{'MC noise':>9s}   (vol points)"
    )
    for name, row in summary.items():
        print(
            f"{name:6s} {row['n_surfaces']:6d} {row['rmse']:8.3f} {row['mean_abs']:8.3f} "
            f"{row['p99_abs']:8.3f} {row['max_abs']:8.3f} {row['mc_noise']:9.3f}"
        )

    test_rmse = surface_rmse(errors["test"])
    plot_error_by_split({name: errors[name] for name in ("test", "gap", "ood")})
    plot_error_over_grid(datasets["test"], errors["test"])
    plot_gap(datasets["gap"], errors["gap"], datasets["test"], errors["test"])
    by_distance = plot_extrapolation(datasets["ood"], errors["ood"], test_rmse)
    by_parameter = plot_extrapolation_by_parameter(datasets["ood"], errors["ood"], test_rmse)
    plot_example_smiles(model, datasets, errors)

    speed = measure_speed(model, dataset)
    print("\nspeed:")
    for key, value in speed.items():
        print(f"  {key}: {value}")

    save_report(
        {
            "errors_in_vol_points": summary,
            "extrapolation_by_distance": by_distance,
            "extrapolation_by_parameter": by_parameter,
            "speed": speed,
        },
        "evaluation",
    )


if __name__ == "__main__":
    main()
