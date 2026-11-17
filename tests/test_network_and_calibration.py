"""Network, training and calibration tests on a synthetic, noise-free dataset."""

import numpy as np
import pytest
import torch

from conftest import smooth_surface
from rbergomi_nn.calibration import Pricer, calibrate, network_pricer
from rbergomi_nn.dataset import TRAIN_BOX
from rbergomi_nn.network import VolSurfaceMLP
from rbergomi_nn.simulator import DTYPE
from rbergomi_nn.training import TrainingConfig, rmse_vol_points, train_network


@pytest.fixture(scope="module")
def trained(synthetic_dataset):
    config = TrainingConfig(hidden_sizes=(32, 32), max_epochs=150, batch_size=64)
    model, history = train_network(
        synthetic_dataset.subset("train"), synthetic_dataset.subset("val"), config, verbose=False
    )
    return model, history


def test_output_shape_and_units(trained, synthetic_dataset):
    model, _ = trained
    theta = torch.as_tensor(synthetic_dataset.theta[:5], dtype=DTYPE)
    surfaces = model(theta)
    assert surfaces.shape == (5, synthetic_dataset.grid.size)
    assert (surfaces > 0.05).all() and (surfaces < 1.0).all()  # volatilities, not standardised


def test_training_learns_the_surface(trained, synthetic_dataset):
    model, history = trained
    assert history[-1]["val_rmse"] < 0.1 * history[0]["val_rmse"]

    test = synthetic_dataset.subset("test")
    test_rmse = rmse_vol_points(
        model,
        torch.as_tensor(test.theta, dtype=DTYPE),
        torch.as_tensor(test.flat_implied_vol, dtype=DTYPE),
    )
    assert test_rmse < 0.15  # vol points; the surfaces span roughly 10 to 40 vol points


def test_training_is_reproducible(synthetic_dataset):
    config = TrainingConfig(hidden_sizes=(8,), max_epochs=3)
    train, val = synthetic_dataset.subset("train"), synthetic_dataset.subset("val")
    _, first = train_network(train, val, config, verbose=False)
    _, second = train_network(train, val, config, verbose=False)
    assert first == second


def test_save_and_load_round_trip(trained, synthetic_dataset, tmp_path):
    model, _ = trained
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = VolSurfaceMLP.load(path)

    theta = torch.as_tensor(synthetic_dataset.theta[:10], dtype=DTYPE)
    with torch.no_grad():
        assert torch.equal(model(theta), loaded(theta))


def test_network_jacobian_matches_finite_differences(trained):
    model, _ = trained
    pricer = network_pricer(model)
    theta = TRAIN_BOX.centre
    jacobian = pricer.jacobian(theta)

    for index in range(4):
        bump = np.zeros(4)
        bump[index] = 1e-6
        finite_difference = (pricer.surface(theta + bump) - pricer.surface(theta - bump)) / 2e-6
        assert np.allclose(jacobian[:, index], finite_difference, atol=1e-6)


def test_analytic_jacobian_matches_automatic_differentiation(trained):
    model, _ = trained
    theta = torch.as_tensor(TRAIN_BOX.sobol(1, seed=5)[0], dtype=DTYPE)
    automatic = torch.func.jacrev(lambda t: model(t[None])[0])(theta)
    assert torch.allclose(model.jacobian(theta), automatic, atol=1e-12)


def test_calibration_recovers_parameters_of_an_exact_pricer(synthetic_dataset):
    """With an exact, noise-free pricer the optimiser must find the true parameters."""
    grid = synthetic_dataset.grid
    pricer = Pricer(lambda theta: smooth_surface(theta[None], grid).ravel())
    true_theta = np.array([0.12, 2.4, -0.55, 0.05])

    result = calibrate(pricer, pricer.surface(true_theta), TRAIN_BOX)
    assert np.allclose(result.theta, true_theta, rtol=1e-4)
    assert result.rmse < 1e-8
    assert not result.at_bound.any()


def test_calibration_through_the_network_recovers_parameters(trained, synthetic_dataset):
    model, _ = trained
    test = synthetic_dataset.subset("test")
    pricer = network_pricer(model)

    errors = []
    for theta, target in zip(test.theta[:20], test.flat_implied_vol[:20]):
        result = calibrate(pricer, target, TRAIN_BOX)
        errors.append(np.abs(TRAIN_BOX.normalise(result.theta) - TRAIN_BOX.normalise(theta)))
    # Normalised units: 2 = the full training range, so this is a median error below 5% of
    # the range. Loose on purpose: the fixture is a tiny network trained for a few seconds.
    assert np.median(errors) < 0.1


def test_calibration_flags_targets_outside_the_box(synthetic_dataset):
    """A target generated outside the box drives the optimiser onto the boundary."""
    grid = synthetic_dataset.grid
    pricer = Pricer(lambda theta: smooth_surface(theta[None], grid).ravel())
    outside = np.array([0.12, 2.4, -0.55, 0.20])  # xi0 well above the upper bound of 0.12

    result = calibrate(pricer, pricer.surface(outside), TRAIN_BOX)
    assert result.at_bound[3]
    assert result.rmse > 1e-3
