"""Deterministic (epsilon=0) evaluation of a saved checkpoint."""

import argparse
import json
import os
from statistics import mean, pstdev

import torch

from model import DQN, SnakeEncoder, ZDQN
from snake import Snake, getDir

# At epsilon=0 a learned policy can loop forever without dying.
# Cap each episode so eval never hangs.
DEFAULT_MAX_STEPS = 2000


def eval_window(model_path: str, episodes: int,
                max_steps: int = DEFAULT_MAX_STEPS) -> list:
    """Setup A model."""
    agent = DQN(1, 9 ** 2 + 4, 3)
    agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.epsilon_threshold = 0.0
    Game = Snake()
    scores = []
    for _ in range(episodes):
        steps = 0
        while True:
            dirs = getDir(Game.dir)
            state = Game.getState()
            action = agent.select_action(state)
            Game.changeDir(dirs[action])
            Game.MoveSnake()
            steps += 1
            if Game.isDead() or steps >= max_steps:
                scores.append(Game.score)
                Game = Snake()
                break
    return scores


def eval_zdqn(model_path: str, episodes: int,
              max_steps: int = DEFAULT_MAX_STEPS) -> list:
    """Setups B/C/D model."""
    ckpt = torch.load(model_path, map_location="cpu")
    z_dim = ckpt["z_dim"]
    frame_stack = ckpt.get("frame_stack", 1)
    encoder = SnakeEncoder(in_channels=3 * frame_stack, z_dim=z_dim)
    encoder.load_state_dict(ckpt["encoder"])
    agent = ZDQN(1, z_dim=z_dim, output_size=3, encoder=encoder,
                 finetune_encoder=False)
    agent.head.load_state_dict(ckpt["head"])
    agent.epsilon_threshold = 0.0
    Game = Snake(frame_stack=frame_stack)
    scores = []
    for _ in range(episodes):
        steps = 0
        while True:
            dirs = getDir(Game.dir)
            grid, dir_oh = Game.getFullState()
            action = agent.select_action(grid, dir_oh)
            Game.changeDir(dirs[action])
            Game.MoveSnake()
            steps += 1
            if Game.isDead() or steps >= max_steps:
                scores.append(Game.score)
                Game = Snake(frame_stack=frame_stack)
                break
    return scores


def main(run_dir: str, episodes: int = 30,
         max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    model_path = os.path.join(run_dir, "model.pt")
    ckpt = torch.load(model_path, map_location="cpu")
    is_zdqn = isinstance(ckpt, dict) and "encoder" in ckpt
    scores = (eval_zdqn(model_path, episodes, max_steps=max_steps)
              if is_zdqn else
              eval_window(model_path, episodes, max_steps=max_steps))
    summary = {
        "run": run_dir,
        "episodes": episodes,
        "mean": mean(scores),
        "std": pstdev(scores) if len(scores) > 1 else 0.0,
        "max": max(scores),
        "min": min(scores),
        "scores": scores,
    }
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{run_dir}  mean={summary['mean']:.2f} +/- {summary['std']:.2f}  "
          f"max={summary['max']}  min={summary['min']}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = p.parse_args()
    main(args.run_dir, episodes=args.episodes, max_steps=args.max_steps)
