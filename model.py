"""MLP-based DQN for Snake state representation experiments.

Architecture: input → 256 → 128 → 64 → 3  (ReLU activations).
Save/load uses state_dict + config dict (not whole-model pickle).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPDQN(nn.Module):
    """Feed-forward Q-network.

    Parameters
    ----------
    input_dim : int
        Size of the encoded state vector.
    output_dim : int
        Number of actions (default 3: left, straight, right).
    """

    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state_dict + config to *path*."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> MLPDQN:
        """Load a model from *path* (returns a new ``MLPDQN`` instance)."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ckpt["input_dim"], ckpt["output_dim"])
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
