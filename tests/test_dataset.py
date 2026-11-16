import numpy as np

from rbergomi_nn.dataset import (
    HURST_GAP,
    OUTER_BOX,
    SPLITS,
    TRAIN_BOX,
    DatasetConfig,
    SurfaceDataset,
    sample_parameters,
    simulate_surfaces,
)
from rbergomi_nn.surface import MonteCarloConfig, SurfaceGrid

CONFIG = DatasetConfig(n_train_box=2048, n_outer_box=512)


def test_sobol_points_fill_the_box():
    points = TRAIN_BOX.sobol(1024, seed=0)
    assert TRAIN_BOX.contains(points).all()
    # Low discrepancy: every parameter's mean is close to the centre of its range.
    assert np.allclose(TRAIN_BOX.normalise(points).mean(axis=0), 0.0, atol=0.01)


def test_distance_outside_is_zero_inside_and_positive_outside():
    inside = TRAIN_BOX.centre
    outside = TRAIN_BOX.centre.copy()
    half_width = 0.5 * (TRAIN_BOX.upper[0] - TRAIN_BOX.lower[0])
    outside[0] = TRAIN_BOX.upper[0] + half_width
    assert TRAIN_BOX.distance_outside(inside) == 0.0
    assert np.isclose(TRAIN_BOX.distance_outside(outside), 1.0)


def test_training_box_is_strictly_inside_outer_box():
    assert (OUTER_BOX.lower_array < TRAIN_BOX.lower_array).all()
    assert (OUTER_BOX.upper_array > TRAIN_BOX.upper_array).all()


def test_every_sample_has_exactly_one_valid_split():
    theta, split = sample_parameters(CONFIG)
    assert len(theta) == len(split)
    assert set(split) == set(SPLITS)


def test_splits_respect_their_regions():
    theta, split = sample_parameters(CONFIG)
    hurst = theta[:, 0]
    in_gap = (hurst >= HURST_GAP[0]) & (hurst <= HURST_GAP[1])
    in_box = TRAIN_BOX.contains(theta)

    assert not in_box[split == "ood"].any()
    assert in_box[split != "ood"].all()
    assert in_gap[split == "gap"].all()
    # The network must never see the gap: not in training, not in model selection.
    for seen in ("train", "val", "test"):
        assert not in_gap[split == seen].any()


def test_sampling_is_reproducible():
    first_theta, first_split = sample_parameters(CONFIG)
    second_theta, second_split = sample_parameters(CONFIG)
    assert np.array_equal(first_theta, second_theta)
    assert np.array_equal(first_split, second_split)


def test_simulated_surfaces_do_not_depend_on_worker_count():
    grid, mc = SurfaceGrid(), MonteCarloConfig(n_paths=256)
    theta = TRAIN_BOX.sobol(4, seed=1)
    serial, _ = simulate_surfaces(theta, grid, mc, base_seed=3, n_workers=1)
    parallel, _ = simulate_surfaces(theta, grid, mc, base_seed=3, n_workers=2)
    assert serial.shape == (4, *grid.shape)
    assert np.array_equal(serial, parallel)


def test_save_and_load_round_trip(synthetic_dataset, tmp_path):
    path = tmp_path / "dataset.npz"
    synthetic_dataset.save(path)
    loaded = SurfaceDataset.load(path)

    assert np.array_equal(loaded.theta, synthetic_dataset.theta)
    assert np.array_equal(loaded.implied_vol, synthetic_dataset.implied_vol)
    assert np.array_equal(loaded.split, synthetic_dataset.split)
    assert loaded.grid == synthetic_dataset.grid
    assert loaded.mc == synthetic_dataset.mc
    assert len(loaded.subset("val")) == 128
