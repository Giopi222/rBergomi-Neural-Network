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

import matplotlib.pyplot as plt
import numpy as np
import torch

from rbergomi_nn.dataset import HURST_GAP, TRAIN_BOX, SurfaceDataset
from rbergomi_nn.network import VolSurfaceMLP
from rbergomi_nn.project import DATASET_PATH, MODEL_PATH, save_figure, save_report
from rbergomi_nn.simulator import DTYPE, PARAMETER_NAMES, RBergomiParams
from rbergomi_nn.surface import implied_vol_surface

PARAMETER_LABELS = ("H", r"$\eta$", r"$\rho$", r"$\xi_0$")


def predict(model: VolSurfaceMLP, dataset: SurfaceDataset) -> np.ndarray:
    """Network surfaces for every sample, shape ``(n, n_maturities, n_strikes)``."""
    with torch.no_grad():
        flat = model(torch.as_tensor(dataset.theta, dtype=DTYPE)).numpy()
    return flat.reshape(dataset.implied_vol.shape)


def surface_rmse(error: np.ndarray) -> np.ndarray:
    """Root mean squared error of each surface over its grid points."""
    return np.sqrt(np.mean(error**2, axis=(1, 2)))


def error_summary(error: np.ndarray, mc_stderr: np.ndarray) -> dict:
    """Error statistics of one split, in vol points, next to the Monte Carlo noise.

    The Monte Carlo surfaces are themselves noisy, so the measured error can
    never fall below ``mc_noise``: the last entry removes it in quadrature to
    estimate the error against the true model surface.
    """
    absolute = 100 * np.abs(error)
    rmse = float(np.sqrt(np.mean(absolute**2)))
    mc_noise = 100 * float(np.sqrt(np.mean(mc_stderr**2)))
    return {
        "n_surfaces": len(error),
        "rmse": round(rmse, 4),
        "mean_abs": round(float(absolute.mean()), 4),
        "p95_abs": round(float(np.quantile(absolute, 0.95)), 4),
        "p99_abs": round(float(np.quantile(absolute, 0.99)), 4),
        "max_abs": round(float(absolute.max()), 4),
        "mc_noise": round(mc_noise, 4),
        "rmse_net_of_mc_noise": round(float(np.sqrt(max(rmse**2 - mc_noise**2, 0.0))), 4),
    }


def plot_error_by_split(errors: dict[str, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4))
    names = list(errors)
    axis.boxplot(
        [100 * surface_rmse(errors[name]) for name in names],
        tick_labels=[f"{name}\n(n = {len(errors[name])})" for name in names],
        whis=(1, 99),
        showfliers=False,
    )
    axis.set_yscale("log")
    axis.set(
        title="Network error per surface, by split",
        ylabel="RMSE over the grid (vol points)",
    )
    axis.text(
        0.02,
        0.97,
        "boxes: quartiles, whiskers: 1%–99%",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    save_figure(figure, "error_by_split")


def plot_error_over_grid(test: SurfaceDataset, test_error: np.ndarray) -> None:
    """Mean absolute network error next to the Monte Carlo standard error, per grid point."""
    network = 100 * np.abs(test_error).mean(axis=0)
    monte_carlo = 100 * test.implied_vol_stderr.mean(axis=0)
    upper = max(network.max(), monte_carlo.max())

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    panels = [
        (network, "Network: mean |error| on test set"),
        (monte_carlo, "Monte Carlo: mean standard error"),
    ]
    for axis, (values, title) in zip(axes, panels, strict=True):
        image = axis.imshow(
            values, aspect="auto", cmap="viridis", vmin=0, vmax=upper, origin="lower"
        )
        columns = range(0, len(test.grid.scaled_moneyness), 2)  # every other strike, for legibility
        axis.set_xticks(columns, [f"{test.grid.scaled_moneyness[c]:+.2f}" for c in columns])
        axis.set_yticks(range(len(test.grid.maturities)), test.grid.maturities)
        axis.set(title=title, xlabel=r"scaled log-moneyness  $\log(K/S_0) / \sqrt{T}$")
        axis.grid(False)
    axes[0].set_ylabel("maturity (years)")
    figure.colorbar(image, ax=axes, label="vol points")
    save_figure(figure, "error_over_grid")


def plot_extrapolation(ood: SurfaceDataset, ood_error: np.ndarray, test_rmse: np.ndarray) -> dict:
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

    edges = [0.0, 0.25, 0.5, 1.0, np.inf]
    binned = {}
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (distance > low) & (distance <= high)
        binned[f"distance in ({low}, {high}]"] = {
            "n_surfaces": int(in_bin.sum()),
            "median_rmse": round(float(np.median(rmse[in_bin])), 4),
            "p95_rmse": round(float(np.quantile(rmse[in_bin], 0.95)), 4),
        }
    return binned


def plot_extrapolation_by_parameter(
    ood: SurfaceDataset, ood_error: np.ndarray, test_rmse: np.ndarray
) -> dict:
    """Which parameter hurts most when extrapolated?

    Uses the surfaces with exactly one parameter outside the training box.
    """
    rmse = 100 * surface_rmse(ood_error)
    normalised = TRAIN_BOX.normalise(ood.theta)
    outside = TRAIN_BOX.excess(ood.theta) > 0
    exactly_one_outside = outside.sum(axis=1) == 1

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    report = {}
    for index, (axis, label, name) in enumerate(
        zip(axes, PARAMETER_LABELS, PARAMETER_NAMES, strict=True)
    ):
        selected = exactly_one_outside & outside[:, index]
        position = normalised[selected, index]
        axis.scatter(position, rmse[selected], s=8, alpha=0.6, color="C3")
        axis.axvspan(-1, 1, color="C0", alpha=0.08, label="training range")
        axis.axhline(100 * np.median(test_rmse), color="C0", lw=1, label="in-domain median")
        axis.set_yscale("log")
        axis.set(title=f"only {label} outside", xlabel=f"{label}, normalised")
        for side, mask in [("below", position < -1), ("above", position > 1)]:
            if mask.any():
                report[f"{name} {side} range"] = {
                    "n_surfaces": int(mask.sum()),
                    "median_rmse": round(float(np.median(rmse[selected][mask])), 4),
                    "max_rmse": round(float(rmse[selected][mask].max()), 4),
                }
    axes[0].set_ylabel("RMSE over the grid (vol points)")
    axes[0].legend(fontsize=8)
    save_figure(figure, "extrapolation_by_parameter")
    return report


def plot_gap(
    gap: SurfaceDataset, gap_error: np.ndarray, test: SurfaceDataset, test_error: np.ndarray
) -> None:
    """Error against H across the slab that was removed from the training data."""
    figure, axis = plt.subplots(figsize=(6.2, 4))
    axis.scatter(
        test.theta[:, 0],
        100 * surface_rmse(test_error),
        s=5,
        alpha=0.4,
        label="test (H seen in training)",
    )
    axis.scatter(
        gap.theta[:, 0],
        100 * surface_rmse(gap_error),
        s=5,
        alpha=0.4,
        color="C1",
        label="gap (H never seen)",
    )
    axis.axvspan(*HURST_GAP, color="C1", alpha=0.08)
    axis.set_yscale("log")
    axis.set(
        title="Interpolating across a hole in the training data",
        xlabel="Hurst exponent H",
        ylabel="RMSE over the grid (vol points)",
    )
    axis.legend(fontsize=8)
    save_figure(figure, "interpolation_gap")


def plot_example_smiles(
    model: VolSurfaceMLP, datasets: dict[str, SurfaceDataset], errors: dict[str, np.ndarray]
) -> None:
    """Network against Monte Carlo smiles: a typical test surface and the worst OOD one."""
    test_rmse, ood_rmse = surface_rmse(errors["test"]), surface_rmse(errors["ood"])
    examples = [
        ("test, median error", datasets["test"], int(np.argsort(test_rmse)[len(test_rmse) // 2])),
        ("out-of-domain, worst error", datasets["ood"], int(np.argmax(ood_rmse))),
    ]
    maturity_rows = [0, 3, 7]

    figure, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for axis_row, (title, dataset, sample) in zip(axes, examples, strict=True):
        network = predict(model, dataset)[sample]
        theta = ", ".join(
            f"{label} = {value:.3g}"
            for label, value in zip(PARAMETER_LABELS, dataset.theta[sample], strict=True)
        )
        for axis, row in zip(axis_row, maturity_rows, strict=True):
            log_moneyness = dataset.grid.log_moneyness[row].numpy()
            monte_carlo = 100 * dataset.implied_vol[sample, row]
            band = 200 * dataset.implied_vol_stderr[sample, row]
            axis.errorbar(
                log_moneyness,
                monte_carlo,
                yerr=band,
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


def measure_speed(model: VolSurfaceMLP, dataset: SurfaceDataset) -> dict:
    """Wall-clock time of one surface: network (single and batched) against Monte Carlo."""
    centre = torch.as_tensor(TRAIN_BOX.centre, dtype=DTYPE)
    single = centre[None]
    batch = torch.as_tensor(dataset.theta[:10_000], dtype=DTYPE)

    def seconds_per_call(function, repeats: int) -> float:
        function()  # warm-up
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        return (time.perf_counter() - start) / repeats

    with torch.no_grad():
        network_single = seconds_per_call(lambda: model(single), repeats=2_000)
        network_batch = seconds_per_call(lambda: model(batch), repeats=20) / len(batch)
    network_jacobian = seconds_per_call(lambda: model.jacobian(centre), repeats=2_000)

    params = RBergomiParams.from_vector(TRAIN_BOX.centre.tolist())
    with torch.no_grad():
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


def main() -> None:
    dataset = SurfaceDataset.load(DATASET_PATH)
    model = VolSurfaceMLP.load(MODEL_PATH)

    datasets = {name: dataset.subset(name) for name in ("train", "val", "test", "gap", "ood")}
    errors = {
        name: predict(model, subset) - subset.implied_vol for name, subset in datasets.items()
    }

    summary = {
        name: error_summary(errors[name], datasets[name].implied_vol_stderr) for name in datasets
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
    evaluation_splits = {name: errors[name] for name in ("test", "gap", "ood")}
    plot_error_by_split(evaluation_splits)
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
