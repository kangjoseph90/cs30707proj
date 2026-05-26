"""Setup A: baseline DQN on the 85-dim hand-crafted windowed state."""

import argparse
import os

import torch

from model import DQN
from snake import Snake, getDir
from train_utils import dump_scores, run_dir, seed_all


def train(episode: int, seed: int = 0, save_root: str = "runs",
          max_steps_per_episode: int = 2000):
    seed_all(seed)
    out = run_dir("A_window", seed, root=save_root)

    Game = Snake()
    agent = DQN(episode, 9 ** 2 + 4, 3)

    scores = []
    for i in range(episode):
        steps = 0
        while True:
            dirs = getDir(Game.dir)
            state = Game.getState()
            action = agent.select_action(state)
            Game.changeDir(dirs[action])
            Game.MoveSnake()
            next_state = Game.getState()
            reward = Game.getReward()
            agent.memorize(state, action, reward, next_state)
            agent.optimize_model(load_data=False)
            steps += 1
            if Game.isDead() or steps >= max_steps_per_episode:
                scores.append(Game.score)
                agent.decay_epsilon()
                Game = Snake()
                break
        if (i + 1) % 10 == 0 or i == episode - 1:
            print(f"[A_window seed={seed}] {i+1}/{episode}  "
                  f"score={scores[-1]}  max={max(scores)}  "
                  f"eps={agent.epsilon_threshold:.4f}")

    torch.save(agent.model.state_dict(), os.path.join(out, "model.pt"))
    dump_scores(out, scores, meta={"setup": "A_window", "seed": seed,
                                    "episodes": episode})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-root", type=str, default="runs")
    args = p.parse_args()
    train(args.episodes, seed=args.seed, save_root=args.save_root)
