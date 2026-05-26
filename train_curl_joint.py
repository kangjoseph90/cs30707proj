"""Setup C (main): Joint CURL + DQN on (3,20,20)+dir.

Reproduces the CURL paper's main recipe: contrastive aux loss and the Q-loss
are summed with equal weight on every gradient step, sharing the encoder.
"""

import argparse

from train_utils import run_dir, seed_all
from zdqn_loop import run_zdqn


def train(episode: int, seed: int = 0, save_root: str = "runs",
          curl_weight: float = 1.0, aug_pad: int = 2,
          checkpoint: str = None, eps_start: float = 0.95):
    seed_all(seed)
    out = run_dir("C_curl_joint", seed, root=save_root)
    run_zdqn(setup_name="C_curl_joint", episode=episode, out_dir=out,
             seed_label=seed, use_curl=True, finetune_encoder=True,
             curl_weight=curl_weight, aug_pad=aug_pad,
             checkpoint_path=checkpoint, epsilon_start=eps_start)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-root", type=str, default="runs")
    p.add_argument("--curl-weight", type=float, default=1.0)
    p.add_argument("--aug-pad", type=int, default=2)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to model.pt to resume encoder+head from")
    p.add_argument("--eps-start", type=float, default=0.95,
                   help="starting epsilon (set to last run's final eps when resuming)")
    args = p.parse_args()
    train(args.episodes, seed=args.seed, save_root=args.save_root,
          curl_weight=args.curl_weight, aug_pad=args.aug_pad,
          checkpoint=args.checkpoint, eps_start=args.eps_start)
