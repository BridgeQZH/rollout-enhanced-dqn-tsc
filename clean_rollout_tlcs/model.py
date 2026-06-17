"""PyTorch Q-network for the rollout-enhanced TLCS agent.

A plain fully-connected MLP maps the 12-dimensional queue state to one Q-value per
action. Serialization uses ``state_dict`` checkpoints (never raw pickled modules),
with the architecture metadata stored alongside the weights so a checkpoint can be
reconstructed and validated on load. Loading uses ``weights_only=True`` for safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

CHECKPOINT_FORMAT_VERSION = 1


class MLP(nn.Module):
    """Fully-connected ReLU network with configurable depth and width."""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int, width: int) -> None:
        """Initialize the MLP.

        The network has one input projection, ``num_layers`` hidden blocks of
        ``width`` units, and a linear output head.

        Args:
            input_dim: Input feature dimension (state size).
            output_dim: Output dimension (number of actions).
            num_layers: Number of hidden ``width x width`` blocks.
            width: Units per hidden layer.
        """
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch_size, input_dim)``.

        Returns:
            Output tensor of shape ``(batch_size, output_dim)``.
        """
        return self.net(x)


class Model:
    """Q-network wrapper providing inference, a training step, and checkpointing."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        width: int,
        learning_rate: float,
        device: str | None = None,
    ) -> None:
        """Initialize the model, optimizer and loss.

        Args:
            input_dim: Input feature dimension (state size).
            output_dim: Output dimension (number of actions).
            num_layers: Number of hidden blocks.
            width: Units per hidden layer.
            learning_rate: Adam learning rate.
            device: Torch device string; defaults to CUDA when available, else CPU.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.width = width
        self.learning_rate = learning_rate
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.net = MLP(input_dim, output_dim, num_layers, width).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _forward_numpy(self, states: NDArray) -> NDArray:
        """Run a no-grad forward pass on a batch of states.

        Args:
            states: Array of shape ``(batch_size, input_dim)``.

        Returns:
            Array of shape ``(batch_size, output_dim)``.
        """
        self.net.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
            return self.net(tensor).cpu().numpy()

    def predict_one(self, state: NDArray) -> NDArray:
        """Predict Q-values for a single state.

        Args:
            state: 1-D array of shape ``(input_dim,)``.

        Returns:
            Array of shape ``(1, output_dim)``.
        """
        state_2d = np.asarray(state, dtype=np.float32).reshape(1, self.input_dim)
        return self._forward_numpy(state_2d)

    def predict_batch(self, states: NDArray) -> NDArray:
        """Predict Q-values for a batch of states.

        Args:
            states: Array of shape ``(batch_size, input_dim)``.

        Returns:
            Array of shape ``(batch_size, output_dim)``.
        """
        return self._forward_numpy(states)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train_batch(self, states: NDArray, targets: NDArray) -> float:
        """Perform one gradient step toward the target Q-values.

        Args:
            states: Input states of shape ``(batch_size, input_dim)``.
            targets: Target Q-values of shape ``(batch_size, output_dim)``.

        Returns:
            The scalar MSE loss for this batch.
        """
        self.net.train()

        states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        targets_t = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=self.device)

        self.optimizer.zero_grad()
        predictions = self.net(states_t)
        loss = self.loss_fn(predictions, targets_t)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Serialization (state_dict only — never raw class pickling)
    # ------------------------------------------------------------------ #
    def _architecture(self) -> dict[str, int]:
        """Return the architecture metadata stored in checkpoints."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "width": self.width,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save a self-describing checkpoint (weights + architecture metadata).

        Args:
            path: Destination file path (its parent is created if needed).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "state_dict": self.net.state_dict(),
            **self._architecture(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        learning_rate: float = 1e-3,
        device: str | None = None,
    ) -> Model:
        """Load a model from a ``state_dict`` checkpoint with safe deserialization.

        Args:
            path: Path to a checkpoint produced by :meth:`save_checkpoint`.
            learning_rate: Learning rate for the reconstructed optimizer.
            device: Torch device string; defaults to CUDA when available, else CPU.

        Returns:
            A :class:`Model` with the checkpoint weights loaded.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            KeyError: If the checkpoint is missing required fields.
            ValueError: If the checkpoint format version is unsupported.
        """
        if not path.exists():
            msg = f"Checkpoint not found: {path}"
            raise FileNotFoundError(msg)

        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)

        version = checkpoint.get("format_version")
        if version != CHECKPOINT_FORMAT_VERSION:
            msg = (
                f"Unsupported checkpoint format version {version!r}; "
                f"expected {CHECKPOINT_FORMAT_VERSION}"
            )
            raise ValueError(msg)

        required = ("input_dim", "output_dim", "num_layers", "width", "state_dict")
        missing = [key for key in required if key not in checkpoint]
        if missing:
            msg = f"Checkpoint is missing required fields: {missing}"
            raise KeyError(msg)

        model = cls(
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint["output_dim"],
            num_layers=checkpoint["num_layers"],
            width=checkpoint["width"],
            learning_rate=learning_rate,
            device=str(resolved_device),
        )
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.to(resolved_device)
        return model
