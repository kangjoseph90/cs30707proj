"""Build (embedding, ground-truth-state) datasets from a trained encoder."""

import os
import random as pyrandom
from typing import List, Tuple

import numpy as np
import torch

from model import SnakeEncoder
from snake import BoardX, BoardY, Snake, getDir


def load_encoder(run_dir: str, device: torch.device = None) -> Tuple[SnakeEncoder, int, int]:
    """Loads the encoder saved by zdqn_loop. Returns (encoder, z_dim, frame_stack)."""
    device = device or torch.device("cpu")
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location=device,
                      weights_only=False)
    if not isinstance(ckpt, dict) or "encoder" not in ckpt:
        raise ValueError(f"{run_dir} has no encoder checkpoint (setup A doesn't qualify).")
    z_dim = ckpt["z_dim"]
    frame_stack = ckpt.get("frame_stack", 1)
    enc = SnakeEncoder(in_channels=3 * frame_stack, z_dim=z_dim).to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    return enc, z_dim, frame_stack


def load_agent_nets(run_dir: str, device: torch.device = None):
    """Load encoder + Q-head from a zdqn_loop checkpoint.

    Returns (encoder, head, z_dim, frame_stack).
    head: nn.Sequential matching ZDQN architecture (z_dim -> 128 -> 64 -> 3).
    """
    import torch.nn as nn
    device = device or torch.device("cpu")
    ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location=device,
                      weights_only=False)
    if not isinstance(ckpt, dict) or "encoder" not in ckpt or "head" not in ckpt:
        raise ValueError(f"{run_dir} has no encoder+head checkpoint.")
    z_dim = ckpt["z_dim"]
    frame_stack = ckpt.get("frame_stack", 1)
    enc = SnakeEncoder(in_channels=3 * frame_stack, z_dim=z_dim).to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    head = nn.Sequential(
        nn.Linear(z_dim, 128), nn.ReLU(),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, 3),
    ).to(device)
    head.load_state_dict(ckpt["head"])
    head.eval()
    return enc, head, z_dim, frame_stack


def gt_features(game: Snake) -> np.ndarray:
    """Ground-truth low-dimensional summary of the game state.

    [head_x, head_y, apple_x, apple_y, dir_onehot(4), body_length]
    All coordinates normalized to [0, 1].
    """
    head = game.body[0]
    apple = game.apple
    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[game.dir] = 1.0
    return np.concatenate([
        np.array([head[0] / BoardX, head[1] / BoardY,
                  apple[0] / BoardX, apple[1] / BoardY], dtype=np.float32),
        dir_oh,
        np.array([len(game.body) / (BoardX * BoardY)], dtype=np.float32),
    ])


def collect_dataset(num_steps: int, frame_stack: int = 1,
                    policy: str = "random") -> Tuple[List[torch.Tensor],
                                                      List[torch.Tensor],
                                                      np.ndarray]:
    """Roll out a policy; return per-step (grid, dir_onehot, gt-vector)."""
    Game = Snake(frame_stack=frame_stack)
    grids, dirs, gts = [], [], []
    for _ in range(num_steps):
        grid, dir_oh = Game.getFullState()
        grids.append(grid)
        dirs.append(dir_oh)
        gts.append(gt_features(Game))
        if policy == "random":
            a = pyrandom.randrange(3)
        else:
            raise ValueError(policy)
        Game.changeDir(getDir(Game.dir)[a])
        Game.MoveSnake()
        if Game.isDead():
            Game = Snake(frame_stack=frame_stack)
    return grids, dirs, np.stack(gts)


@torch.no_grad()
def embed_all(encoder: SnakeEncoder, grids, dirs,
              batch_size: int = 256, device: torch.device = None) -> np.ndarray:
    device = device or torch.device("cpu")
    encoder.eval().to(device)
    out = []
    for s in range(0, len(grids), batch_size):
        g = torch.stack(grids[s:s + batch_size]).to(device)
        d = torch.stack(dirs[s:s + batch_size]).to(device)
        z = encoder(g, d)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)
