"""Training loop for the volatility surface network."""

import copy
from dataclasses import dataclass

import torch

from rbergomi_nn.dataset import TRAIN_BOX, SurfaceDataset
from rbergomi_nn.network import VolSurfaceMLP
from rbergomi_nn.simulator import DTYPE


@dataclass
class TrainingConfig:
    hidden_sizes: tuple = (64, 64, 64, 64)
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 600
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 15  # epochs without validation improvement before the rate is cut
    early_stopping_patience: int = 50
    seed: int = 0


def rmse_vol_points(model: VolSurfaceMLP, theta, implied_vol) -> float:
    """Root mean squared error over all surfaces and grid points, in vol points (1 = 1%)."""
    with torch.no_grad():
        return 100.0 * (model(theta) - implied_vol).square().mean().sqrt().item()


def train_network(train: SurfaceDataset, val: SurfaceDataset, config: TrainingConfig, verbose=True):
    """Fit the network on ``train`` and keep the weights with the best ``val`` error.

    The loss is the mean squared error on standardised implied volatilities, so
    every grid point carries the same weight regardless of its volatility level.

    Returns the trained model and the per-epoch history.
    """
    torch.manual_seed(config.seed)
    train_theta = torch.as_tensor(train.theta, dtype=DTYPE)
    train_vol = torch.as_tensor(train.flat_implied_vol, dtype=DTYPE)
    val_theta = torch.as_tensor(val.theta, dtype=DTYPE)
    val_vol = torch.as_tensor(val.flat_implied_vol, dtype=DTYPE)

    lower, upper = torch.tensor(TRAIN_BOX.lower), torch.tensor(TRAIN_BOX.upper)
    model = VolSurfaceMLP(lower, upper, train_vol.mean(0), train_vol.std(0), config.hidden_sizes)
    train_target = model.standardise(train_vol)

    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=config.lr_decay_factor, patience=config.lr_decay_patience
    )

    history, best_val_rmse, best_epoch = [], float("inf"), 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch in torch.randperm(len(train_theta)).split(config.batch_size):
            output = model.forward_standardised(train_theta[batch])
            loss = (output - train_target[batch]).square().mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        model.eval()
        train_rmse = rmse_vol_points(model, train_theta, train_vol)
        val_rmse = rmse_vol_points(model, val_theta, val_vol)
        scheduler.step(val_rmse)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})

        if val_rmse < best_val_rmse:
            best_val_rmse, best_epoch = val_rmse, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch == 1 or epoch % 10 == 0):
            rate = optimiser.param_groups[0]["lr"]
            print(f"epoch {epoch:4d}  train {train_rmse:.4f}  val {val_rmse:.4f}  lr {rate:.0e}")
        if epoch - best_epoch >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    return model.eval(), history
