"""Evaluate a saved DQN checkpoint on Snake (deterministic, epsilon=0).

Usage::

    python evaluate_snake.py results/structural_seed42/checkpoint.pt --episodes 30
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from model import MLPDQN
from snake_env import SnakeEnv
from train_snake import make_encoder, seed_all


def evaluate(
    checkpoint_path: str,
    num_episodes: int = 100,
    seed: int = 0,
    render: bool = False,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    encoder = make_encoder(
        config["representation"],
        config["board_size"],
        config["max_length"],
        config.get("local_window_size"),
    )

    model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = SnakeEnv(
        board_size=config["board_size"],
        max_length=config["max_length"],
        max_steps=config.get("max_steps", 2000),
        headless=not render,
        seed=seed,
    )

    scores = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                sv = torch.FloatTensor(encoder.encode(state)).unsqueeze(0)
                action = torch.argmax(model(sv)).item()
            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if render:
                env.render()
        scores.append(info["score"])
        if (ep + 1) % 10 == 0:
            print(
                f"  Episode {ep+1}/{num_episodes}  "
                f"score={info['score']}  mean={np.mean(scores):.2f}"
            )

    print(f"\nEvaluation complete ({num_episodes} episodes):")
    print(f"  Mean score : {np.mean(scores):.2f}")
    print(f"  Std score  : {np.std(scores):.2f}")
    print(f"  Min / Max  : {min(scores)} / {max(scores)}")
    print(f"  Median     : {np.median(scores):.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a Snake DQN checkpoint.")
    p.add_argument("checkpoint", type=str, help="Path to checkpoint.pt")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", default=False)
    args = p.parse_args()
    evaluate(args.checkpoint, args.episodes, args.seed, args.render)


if __name__ == "__main__":
    main()
