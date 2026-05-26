"""Setup D: CURL pretrain -> freeze -> ZDQN (encoder fixed, Q-head only).

Phase 1 collects K random-policy transitions, trains the encoder with
InfoNCE only (no reward signal). Phase 2 hands the frozen encoder to ZDQN
and trains the Q-head.

This mirrors the paper's E.3 ablation (detached encoder) and gives the
cleanest read for the project's state-aliasing / Markov-probe analyses,
since the encoder never sees reward gradients.
"""

import argparse
import os
import random as pyrandom
from functools import partial

import torch
import torch.optim as optim

from augment import random_pad_and_crop
from curl import CURLModule
from model import SnakeEncoder
from snake import Snake, getDir
from train_utils import run_dir, seed_all
from zdqn_loop import run_zdqn


def collect_random_buffer(num_steps: int, frame_stack: int = 1):
    """Roll out a uniformly random policy and return a list of (grid, dir)."""
    Game = Snake(frame_stack=frame_stack)
    buf = []
    while len(buf) < num_steps:
        grid, dir_oh = Game.getFullState()
        buf.append((grid, dir_oh))
        dirs = getDir(Game.dir)
        a = pyrandom.randrange(3)
        Game.changeDir(dirs[a])
        Game.MoveSnake()
        if Game.isDead():
            Game = Snake(frame_stack=frame_stack)
    return buf


def pretrain_encoder(buf, z_dim: int = 50, frame_stack: int = 1,
                     epochs: int = 5, batch_size: int = 256,
                     lr: float = 1e-3, aug_pad: int = 2,
                     device: torch.device = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SnakeEncoder(in_channels=3 * frame_stack, z_dim=z_dim).to(device)
    curl = CURLModule(encoder, z_dim=z_dim, momentum=0.05).to(device)
    aug = partial(random_pad_and_crop, pad=aug_pad)
    opt = optim.Adam([
        {"params": encoder.parameters(), "lr": lr},
        {"params": [curl.W], "lr": lr},
    ])
    n = len(buf)
    for ep in range(epochs):
        idx = list(range(n))
        pyrandom.shuffle(idx)
        running = 0.0
        steps = 0
        for s in range(0, n - batch_size + 1, batch_size):
            chunk = idx[s:s + batch_size]
            grids = torch.stack([buf[i][0] for i in chunk]).to(device)
            dirs = torch.stack([buf[i][1] for i in chunk]).to(device)
            x_q = aug(grids)
            x_k = aug(grids)
            z_q = curl.encode_query(x_q, dirs)
            z_k = curl.encode_key(x_k, dirs)
            loss = curl.info_nce_loss(z_q, z_k)
            opt.zero_grad()
            loss.backward()
            opt.step()
            curl.update_key_encoder()
            running += float(loss.item())
            steps += 1
        avg = running / max(1, steps)
        print(f"  pretrain epoch {ep+1}/{epochs}  curl_loss={avg:.4f}")
    return encoder


def train(episode: int, seed: int = 0, save_root: str = "runs",
          pretrain_steps: int = 20000, pretrain_epochs: int = 5,
          aug_pad: int = 2):
    seed_all(seed)
    out = run_dir("D_curl_pretrain", seed, root=save_root)

    print(f"[D_curl_pretrain seed={seed}] collecting {pretrain_steps} random steps...")
    buf = collect_random_buffer(pretrain_steps)
    print(f"[D_curl_pretrain seed={seed}] pretraining encoder ({pretrain_epochs} epochs)...")
    enc = pretrain_encoder(buf, epochs=pretrain_epochs, aug_pad=aug_pad)
    enc_path = os.path.join(out, "pretrained_encoder.pt")
    torch.save(enc.state_dict(), enc_path)

    print(f"[D_curl_pretrain seed={seed}] training Q-head with frozen encoder...")
    run_zdqn(setup_name="D_curl_pretrain", episode=episode, out_dir=out,
             seed_label=seed, use_curl=False, finetune_encoder=False,
             encoder_init_path=enc_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-root", type=str, default="runs")
    p.add_argument("--pretrain-steps", type=int, default=20000)
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--aug-pad", type=int, default=2)
    args = p.parse_args()
    train(args.episodes, seed=args.seed, save_root=args.save_root,
          pretrain_steps=args.pretrain_steps,
          pretrain_epochs=args.pretrain_epochs, aug_pad=args.aug_pad)
