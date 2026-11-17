"""Neural network mapping model parameters to an implied volatility surface.

This is the "image-based" approach of Horvath, Muguruza and Tomas (2021): the
network outputs the whole surface on a fixed grid at once, one output neuron
per grid point.
"""

import torch
from torch import Tensor, nn

from rbergomi_nn.simulator import DTYPE


class VolSurfaceMLP(nn.Module):
    """Fully connected network ``theta -> implied volatilities on the grid``.

    The input and output scalings are stored inside the module, so the network
    takes raw parameters, returns volatilities, and a checkpoint is self-contained.

    SiLU rather than ReLU because calibration uses the Jacobian of the network,
    and SiLU makes it smooth (with ReLU it would be piecewise constant).
    """

    def __init__(self, theta_lower, theta_upper, output_mean, output_std, hidden_sizes):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.register_buffer("theta_lower", theta_lower)
        self.register_buffer("theta_upper", theta_upper)
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("output_std", output_std)

        layers, width = [], len(theta_lower)
        for hidden in hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.layers = nn.Sequential(*layers, nn.Linear(width, len(output_mean))).to(DTYPE)

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
        """Derivative of the surface at one parameter vector: ``(4,) -> (n_outputs, 4)``.

        The chain rule is applied layer by layer alongside the forward pass: same
        numbers as autograd at a fraction of the cost, which matters because
        calibration calls it at every iteration.
        """
        scale = 2.0 / (self.theta_upper - self.theta_lower)
        z = scale * (theta - self.theta_lower) - 1.0
        jacobian = torch.diag(scale)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                z, jacobian = layer(z), layer.weight @ jacobian
            else:  # SiLU: f(z) = z * s(z), s = sigmoid, so f'(z) = s * (1 + z * (1 - s))
                s = torch.sigmoid(z)
                z, jacobian = z * s, (s * (1.0 + z * (1.0 - s)))[:, None] * jacobian
        return self.output_std[:, None] * jacobian

    def save(self, path):
        torch.save({"hidden_sizes": self.hidden_sizes, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path) -> "VolSurfaceMLP":
        checkpoint = torch.load(path, weights_only=True)
        state = checkpoint["state_dict"]
        names = ("theta_lower", "theta_upper", "output_mean", "output_std")
        model = cls(*(state[name] for name in names), checkpoint["hidden_sizes"])
        model.load_state_dict(state)
        return model.eval()
