"""Setup E: CURL joint + Target Network + Double DQN + z_dim=128.

Architectural improvements over Setup C:
  - Target network: frozen copy synced every target_update_freq gradient steps.
  - Double DQN: online net selects action, target net evaluates value.
  - z_dim 50 -> 128: larger latent for complex states (long snake).
"""

import argparse

from train_utils import run_dir, seed_all
from zdqn_loop import run_zdqn2


def train(episode: int, seed: int = 0, save_root: str = "runs",
          curl_weight: float = 1.0, aug_pad: int = 2,
          z_dim: int = 128, target_update_freq: int = 1000,
          checkpoint: str = None, eps_start: float = 0.95):
    seed_all(seed)
    out = run_dir("E_curl_improved", seed, root=save_root)
    run_zdqn2(setup_name="E_curl_improved", episode=episode, out_dir=out,
              seed_label=seed, use_curl=True, finetune_encoder=True,
              z_dim=z_dim, curl_weight=curl_weight, aug_pad=aug_pad,
              target_update_freq=target_update_freq,
              checkpoint_path=checkpoint, epsilon_start=eps_start)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-root", type=str, default="runs")
    p.add_argument("--curl-weight", type=float, default=1.0)
    p.add_argument("--aug-pad", type=int, default=2)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--target-update-freq", type=int, default=1000,
                   help="gradient steps between target network syncs")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="model.pt to resume encoder+head from")
    p.add_argument("--eps-start", type=float, default=0.95)
    args = p.parse_args()
    train(args.episodes, seed=args.seed, save_root=args.save_root,
          curl_weight=args.curl_weight, aug_pad=args.aug_pad,
          z_dim=args.z_dim, target_update_freq=args.target_update_freq,
          checkpoint=args.checkpoint, eps_start=args.eps_start)
