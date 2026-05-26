"""CURL contrastive module for Snake.

Implements the InfoNCE objective with a bilinear similarity W and a
momentum (EMA) key encoder, following Srinivas, Laskin & Abbeel (2020).
The query encoder is shared with the downstream RL head so its parameters
receive gradients from both losses.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CURLModule(nn.Module):
    def __init__(self, query_encoder: nn.Module, z_dim: int,
                 momentum: float = 0.05):
        """
        Args:
            query_encoder: SnakeEncoder instance. Owned externally — we keep a
                reference, and the key encoder is built as a deep copy.
            z_dim: latent dimension.
            momentum: EMA coefficient `m` such that
                theta_k <- (1 - m) * theta_k + m * theta_q.
                The paper uses tau in the same role (0.05 for DMControl);
                we follow that convention.
        """
        super().__init__()
        self.q_enc = query_encoder
        self.k_enc = copy.deepcopy(query_encoder)
        for p in self.k_enc.parameters():
            p.requires_grad = False
        self.W = nn.Parameter(torch.randn(z_dim, z_dim) * 0.01)
        self.momentum = float(momentum)
        self.z_dim = z_dim

    @torch.no_grad()
    def update_key_encoder(self) -> None:
        m = self.momentum
        for pq, pk in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            pk.data.mul_(1.0 - m).add_(pq.data, alpha=m)

    def encode_query(self, grid: torch.Tensor, dir_onehot: torch.Tensor) -> torch.Tensor:
        return self.q_enc(grid, dir_onehot)

    @torch.no_grad()
    def encode_key(self, grid: torch.Tensor, dir_onehot: torch.Tensor) -> torch.Tensor:
        return self.k_enc(grid, dir_onehot)

    def info_nce_loss(self, z_q: torch.Tensor, z_k: torch.Tensor) -> torch.Tensor:
        """Bilinear InfoNCE. Positive pair: the diagonal (z_q[i], z_k[i]).

        logits[i, j] = z_q[i]^T W z_k[j]
        labels[i]    = i
        """
        Wk = torch.matmul(self.W, z_k.T)            # (z_dim, B)
        logits = torch.matmul(z_q, Wk)              # (B, B)
        logits = logits - logits.max(dim=1, keepdim=True).values  # stability
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)


def curl_loss_on_batch(curl: CURLModule,
                       grids: torch.Tensor,
                       dirs: torch.Tensor,
                       aug_fn,
                       aug_fn_key: Optional[callable] = None) -> torch.Tensor:
    """Convenience: two augmentations -> query/key embeddings -> InfoNCE.

    Same direction one-hot is used for both views (random_pad_and_crop does
    not change the agent's heading, and direction has no spatial axis to
    augment).
    """
    if aug_fn_key is None:
        aug_fn_key = aug_fn
    x_q = aug_fn(grids)
    x_k = aug_fn_key(grids)
    z_q = curl.encode_query(x_q, dirs)
    z_k = curl.encode_key(x_k, dirs)
    return curl.info_nce_loss(z_q, z_k)
