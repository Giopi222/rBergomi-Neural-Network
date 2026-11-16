"""Neural network mapping model parameters to an implied volatility surface.

This is the "image-based" approach of Horvath, Muguruza and Tomas (2021): the
network outputs the whole surface on a fixed grid at once, one output neuron
per grid point.
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from rbergomi_nn.simulator import DTYPE


class VolSurfaceMLP(nn.Module):
    """Fully connected network ``theta -> implied volatilities on the grid``.

    The input and output scalings are stored inside the module, so the network
    consumes raw parameters and returns volatilities in natural units, and a
    saved checkpoint is self-contained.

    The activation is SiLU, which is infinitely differentiable: calibration
    uses the Jacobian of the network, and a smooth activation gives a smooth
    Jacobian (ReLU would make it piecewise constant).
    """

    def __init__(
        self,
        theta_lower: Tensor,
        theta_upper: Tensor,
        output_mean: Tensor,
        output_std: Tensor,
        hidden_sizes: tuple[int, ...] = (64, 64, 64, 64),
    ):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.register_buffer("theta_lower", theta_lower.to(DTYPE))
        self.register_buffer("theta_upper", theta_upper.to(DTYPE))
        self.register_buffer("output_mean", output_mean.to(DTYPE))
        self.register_buffer("output_std", output_std.to(DTYPE))

        layers: list[nn.Module] = []
        width = len(theta_lower)
        for hidden in self.hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, len(output_mean)))
        self.layers = nn.Sequential(*layers).to(DTYPE)

    def forward_standardised(self, theta: Tensor) -> Tensor:
        """Network output in standardised units (the space the loss lives in)."""
        scaled = 2.0 * (theta - self.theta_lower) / (self.theta_upper - self.theta_lower) - 1.0
        return self.layers(scaled)

    def standardise(self, implied_vol: Tensor) -> Tensor:
        return (implied_vol - self.output_mean) / self.output_std

    def forward(self, theta: Tensor) -> Tensor:
        """Implied volatilities, shape ``(batch, n_maturities * n_strikes)``."""
        return self.forward_standardised(theta) * self.output_std + self.output_mean

    @torch.no_grad()
    def jacobian(self, theta: Tensor) -> Tensor:
        """Derivative of the surface with respect to one parameter vector.

        ``theta`` has shape ``(4,)``; the result has shape ``(n_outputs, 4)``.

        The chain rule is applied layer by layer alongside the forward pass. This
        gives the same numbers as automatic differentiation at a fraction of the
        cost, which matters because calibration calls it at every iteration.
        """
        input_scale = 2.0 / (self.theta_upper - self.theta_lower)
        activation = input_scale * (theta - self.theta_lower) - 1.0
        jacobian = torch.diag(input_scale)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                activation = layer(activation)
                jacobian = layer.weight @ jacobian
            else:
                # SiLU: f(z) = z * s(z) with s the sigmoid, so f'(z) = s * (1 + z * (1 - s)).
                sigmoid = torch.sigmoid(activation)
                derivative = sigmoid * (1.0 + activation * (1.0 - sigmoid))
                activation = activation * sigmoid
                jacobian = derivative[:, None] * jacobian
        return self.output_std[:, None] * jacobian

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hidden_sizes": self.hidden_sizes, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path) -> "VolSurfaceMLP":
        checkpoint = torch.load(path, weights_only=True)
        state = checkpoint["state_dict"]
        model = cls(
            state["theta_lower"],
            state["theta_upper"],
            state["output_mean"],
            state["output_std"],
            hidden_sizes=tuple(checkpoint["hidden_sizes"]),
        )
        model.load_state_dict(state)
        return model.eval()
