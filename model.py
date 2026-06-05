"""DQN models for Snake state representation experiments.

MLPDQN: input → 256 → 128 → 64 → 3  (ReLU activations).
CNNDQN: 4-channel grid → conv layers → FC → 3  (for egocentric spatial state).
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


class CNNDQN(nn.Module):
    """CNN Q-network for egocentric spatial state.

    Input: flat vector of size (3*K*K + 3), reshaped internally to:
        grid:    (B, 3, K, K)  — obstacle, release_time, fruit channels
        scalars: (B, 3)        — fruit_ego_dx, fruit_ego_dy, body_length_norm

    Architecture:
        Conv blocks with stride-2 downsampling (no pooling)
        → flatten → concat scalars → FC layers → 3 actions

    Parameters
    ----------
    input_dim : int
        Total flat input size (3*K*K + 3).
    grid_channels : int
        Number of spatial input channels (default 3).
    grid_size : int
        Spatial grid dimension (default 29).
    output_dim : int
        Number of actions (default 3).
    """

    def __init__(
        self,
        input_dim: int,
        grid_channels: int = 3,
        grid_size: int = 29,
        output_dim: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.grid_channels = grid_channels
        self.grid_size = grid_size
        self.output_dim = output_dim
        self._scalar_dim = 3  # fruit_dx, fruit_dy, body_length

        self.conv = nn.Sequential(
            # Block 1: 3 → 8, stride-2 downsample  29→15
            nn.Conv2d(grid_channels, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            # Block 2: 8 → 16, stride-2 downsample  15→8
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            # Block 3: 16 → 16, spatial preserved  8→8
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        # Calculate conv output size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, grid_channels, grid_size, grid_size)
            conv_out = self.conv(dummy)
            self._conv_flat = conv_out.numel()  # 16 * 8 * 8 = 1024 for K=29

        self.fc = nn.Sequential(
            nn.Linear(self._conv_flat + self._scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        grid_flat_size = self.grid_channels * self.grid_size * self.grid_size

        grid = x[:, :grid_flat_size].view(
            B, self.grid_channels, self.grid_size, self.grid_size
        )
        scalars = x[:, grid_flat_size:]

        conv_out = self.conv(grid).view(B, -1)
        combined = torch.cat([conv_out, scalars], dim=1)
        return self.fc(combined)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state_dict + config to *path*."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_type": "cnn",
                "input_dim": self.input_dim,
                "grid_channels": self.grid_channels,
                "grid_size": self.grid_size,
                "output_dim": self.output_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> CNNDQN:
        """Load a CNN model from *path*."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            ckpt["input_dim"],
            ckpt.get("grid_channels", 3),
            ckpt.get("grid_size", 29),
            ckpt["output_dim"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HybridDQN(nn.Module):
    """MLP + CNN late-fusion Q-network with residual correction.

    MLP branch preserves proven performance; CNN branch provides spatial
    pattern features as a correction signal.  The residual head is zero-
    initialized so training starts from pure MLP behaviour.

    Input: flat vector = [MLP part (mlp_dim)] [CNN grid (3*K*K)]

    Parameters
    ----------
    input_dim : int
        Total flat input size (mlp_dim + 3*K*K).
    mlp_dim : int
        Size of the MLP branch input.
    grid_channels : int
        Number of CNN spatial channels (default 3).
    grid_size : int
        Spatial grid dimension (default 29).
    output_dim : int
        Number of actions (default 3).
    """

    def __init__(
        self,
        input_dim: int,
        mlp_dim: int,
        grid_channels: int = 3,
        grid_size: int = 29,
        output_dim: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.grid_channels = grid_channels
        self.grid_size = grid_size
        self.output_dim = output_dim

        # --- MLP branch ---
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mlp_head = nn.Linear(128, output_dim)  # q_base

        # --- CNN branch ---
        self.conv = nn.Sequential(
            nn.Conv2d(grid_channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, grid_channels, grid_size, grid_size)
            self._conv_flat = self.conv(dummy).numel()  # 16*15*15 = 3600
        self.cnn_fc = nn.Linear(self._conv_flat, 64)

        # --- Residual head: zero-initialized ---
        self.residual = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        # Zero-init so q = q_base at start
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        mlp_in = x[:, : self.mlp_dim]
        cnn_in = x[:, self.mlp_dim :]

        # MLP branch → q_base
        mlp_feat = self.mlp(mlp_in)  # (B, 128)
        q_base = self.mlp_head(mlp_feat)  # (B, 3)

        # CNN branch → 64d feature
        grid_flat_size = self.grid_channels * self.grid_size * self.grid_size
        grid = cnn_in[:, :grid_flat_size].view(
            B, self.grid_channels, self.grid_size, self.grid_size
        )
        conv_out = self.conv(grid).view(B, -1)
        cnn_feat = torch.relu(self.cnn_fc(conv_out))  # (B, 64)

        # Residual correction
        combined = torch.cat([mlp_feat, cnn_feat], dim=1)  # (B, 192)
        q_correction = self.residual(combined)  # (B, 3)

        return q_base + q_correction

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_type": "hybrid",
                "input_dim": self.input_dim,
                "mlp_dim": self.mlp_dim,
                "grid_channels": self.grid_channels,
                "grid_size": self.grid_size,
                "output_dim": self.output_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> HybridDQN:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            ckpt["input_dim"],
            ckpt["mlp_dim"],
            ckpt.get("grid_channels", 3),
            ckpt.get("grid_size", 29),
            ckpt["output_dim"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
