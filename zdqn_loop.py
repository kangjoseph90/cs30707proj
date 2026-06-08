"""Shared ZDQN training loop used by setups B, C, D."""

import os
from functools import partial
from typing import Optional

import torch

from augment import random_pad_and_crop
from curl import CURLModule
from model import SnakeEncoder, ZDQN, ZDQN2
from snake import Snake, getDir
from train_utils import dump_scores


def make_encoder(frame_stack: int = 1, z_dim: int = 50) -> SnakeEncoder:
    return SnakeEncoder(in_channels=3 * frame_stack, z_dim=z_dim)


def run_zdqn(setup_name: str,
             episode: int,
             out_dir: str,
             seed_label: int,
             frame_stack: int = 1,
             z_dim: int = 50,
             use_curl: bool = False,
             finetune_encoder: bool = True,
             encoder_init_path: Optional[str] = None,
             checkpoint_path: Optional[str] = None,
             epsilon_start: float = 0.95,
             curl_weight: float = 1.0,
             aug_pad: int = 2,
             episodes_per_log: int = 10,
             max_steps_per_episode: int = 2000):
    """One full training run, writing model.pt and scores.json into out_dir."""
    encoder = make_encoder(frame_stack=frame_stack, z_dim=z_dim)
    if encoder_init_path is not None and os.path.exists(encoder_init_path):
        encoder.load_state_dict(torch.load(encoder_init_path,
                                           map_location="cpu"))
        print(f"[{setup_name}] loaded encoder from {encoder_init_path}")

    agent = ZDQN(episode, z_dim=z_dim, output_size=3, encoder=encoder,
                 finetune_encoder=finetune_encoder,
                 epsilon_start=epsilon_start)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.head.load_state_dict(ckpt["head"])
        print(f"[{setup_name}] resumed encoder+head from {checkpoint_path}")

    curl = None
    if use_curl:
        curl = CURLModule(agent.encoder, z_dim=z_dim, momentum=0.05)
        aug_fn = partial(random_pad_and_crop, pad=aug_pad)
        agent.attach_curl(curl, aug_fn=aug_fn, curl_weight=curl_weight)

    Game = Snake(frame_stack=frame_stack)
    scores = []
    last_q, last_c = float("nan"), float("nan")

    def _save(n_episodes):
        torch.save({
            "encoder": agent.encoder.state_dict(),
            "head": agent.head.state_dict(),
            "z_dim": z_dim,
            "frame_stack": frame_stack,
        }, os.path.join(out_dir, "model.pt"))
        dump_scores(out_dir, scores, meta={
            "setup": setup_name, "seed": seed_label, "episodes": n_episodes,
            "use_curl": use_curl, "finetune_encoder": finetune_encoder,
            "frame_stack": frame_stack, "z_dim": z_dim,
            "curl_weight": curl_weight if use_curl else None,
        })

    try:
        for i in range(episode):
            steps = 0
            while True:
                dirs = getDir(Game.dir)
                grid, dir_oh = Game.getFullState()
                action = agent.select_action(grid, dir_oh)
                Game.changeDir(dirs[action])
                Game.MoveSnake()
                next_grid, next_dir = Game.getFullState()
                reward = Game.getReward()
                done = Game.isDead() or (steps + 1) >= max_steps_per_episode
                agent.memorize(grid, dir_oh, action, reward, next_grid, next_dir, done)
                info = agent.optimize_model()
                if info is not None:
                    last_q = info["q_loss"]
                    if "curl_loss" in info:
                        last_c = info["curl_loss"]
                steps += 1
                if done:
                    scores.append(Game.score)
                    agent.decay_epsilon()
                    Game = Snake(frame_stack=frame_stack)
                    break
            if (i + 1) % episodes_per_log == 0 or i == episode - 1:
                tag = f"curl={last_c:.3f}" if use_curl else ""
                print(f"[{setup_name} seed={seed_label}] {i+1}/{episode}  "
                      f"score={scores[-1]}  max={max(scores)}  "
                      f"eps={agent.epsilon_threshold:.4f}  q={last_q:.3f}  {tag}")
    except KeyboardInterrupt:
        print(f"\n[{setup_name}] interrupted at episode {len(scores)} — saving checkpoint...")

    _save(len(scores))
    return scores


def run_zdqn2(setup_name: str,
              episode: int,
              out_dir: str,
              seed_label: int,
              frame_stack: int = 1,
              z_dim: int = 128,
              use_curl: bool = False,
              finetune_encoder: bool = True,
              encoder_init_path: Optional[str] = None,
              checkpoint_path: Optional[str] = None,
              epsilon_start: float = 0.95,
              curl_weight: float = 1.0,
              aug_pad: int = 2,
              target_update_freq: int = 1000,
              episodes_per_log: int = 10,
              max_steps_per_episode: int = 2000):
    """ZDQN with target network + Double DQN. Writes model.pt and scores.json."""
    encoder = make_encoder(frame_stack=frame_stack, z_dim=z_dim)
    if encoder_init_path is not None and os.path.exists(encoder_init_path):
        encoder.load_state_dict(torch.load(encoder_init_path, map_location="cpu"))
        print(f"[{setup_name}] loaded encoder from {encoder_init_path}")

    agent = ZDQN2(episode, z_dim=z_dim, output_size=3, encoder=encoder,
                  finetune_encoder=finetune_encoder,
                  epsilon_start=epsilon_start,
                  target_update_freq=target_update_freq)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.head.load_state_dict(ckpt["head"])
        agent._sync_target()
        print(f"[{setup_name}] resumed encoder+head from {checkpoint_path}")

    if use_curl:
        curl = CURLModule(agent.encoder, z_dim=z_dim, momentum=0.05)
        aug_fn = partial(random_pad_and_crop, pad=aug_pad)
        agent.attach_curl(curl, aug_fn=aug_fn, curl_weight=curl_weight)

    Game = Snake(frame_stack=frame_stack)
    scores = []
    last_q, last_c = float("nan"), float("nan")

    def _save(n_episodes):
        torch.save({
            "encoder": agent.encoder.state_dict(),
            "head": agent.head.state_dict(),
            "z_dim": z_dim,
            "frame_stack": frame_stack,
        }, os.path.join(out_dir, "model.pt"))
        dump_scores(out_dir, scores, meta={
            "setup": setup_name, "seed": seed_label, "episodes": n_episodes,
            "use_curl": use_curl, "finetune_encoder": finetune_encoder,
            "frame_stack": frame_stack, "z_dim": z_dim,
            "curl_weight": curl_weight if use_curl else None,
            "target_update_freq": target_update_freq,
        })

    try:
        for i in range(episode):
            steps = 0
            while True:
                dirs = getDir(Game.dir)
                grid, dir_oh = Game.getFullState()
                action = agent.select_action(grid, dir_oh)
                Game.changeDir(dirs[action])
                Game.MoveSnake()
                next_grid, next_dir = Game.getFullState()
                reward = Game.getReward()
                done = Game.isDead() or (steps + 1) >= max_steps_per_episode
                agent.memorize(grid, dir_oh, action, reward, next_grid, next_dir, done)
                info = agent.optimize_model()
                if info is not None:
                    last_q = info["q_loss"]
                    if "curl_loss" in info:
                        last_c = info["curl_loss"]
                steps += 1
                if done:
                    scores.append(Game.score)
                    agent.decay_epsilon()
                    Game = Snake(frame_stack=frame_stack)
                    break
            if (i + 1) % episodes_per_log == 0 or i == episode - 1:
                tag = f"curl={last_c:.3f}" if use_curl else ""
                print(f"[{setup_name} seed={seed_label}] {i+1}/{episode}  "
                      f"score={scores[-1]}  max={max(scores)}  "
                      f"eps={agent.epsilon_threshold:.4f}  q={last_q:.3f}  {tag}")
    except KeyboardInterrupt:
        print(f"\n[{setup_name}] interrupted at episode {len(scores)} — saving checkpoint...")

    _save(len(scores))
    return scores
