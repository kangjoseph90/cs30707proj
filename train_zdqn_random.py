"""Setup B: ZDQN on (3,20,20)+dir with a randomly initialized encoder.

No contrastive loss; the encoder learns purely from reward gradients.
"""

import argparse

from train_utils import run_dir, seed_all
from zdqn_loop import run_zdqn


def train(episode: int, seed: int = 0, save_root: str = "runs"):
    seed_all(seed)
    out = run_dir("B_random", seed, root=save_root)
    run_zdqn(setup_name="B_random", episode=episode, out_dir=out,
             seed_label=seed, use_curl=False, finetune_encoder=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-root", type=str, default="runs")
    args = p.parse_args()
    train(args.episodes, seed=args.seed, save_root=args.save_root)
