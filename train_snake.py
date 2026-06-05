"""CLI training script for Snake state-representation experiments.

Supports both ``--total-env-steps`` (step-based) and ``--episodes``
(episode-based) termination.  Includes periodic greedy evaluation,
timing breakdowns, sparsity measurement, and analysis collection.

Usage examples::

    python train_snake.py --representation structural --total-env-steps 100000 --eval-interval 5000 --seed 0
    python train_snake.py --representation local --local-window-size 7 --episodes 300 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch

from analysis_collector import AnalysisCollector, SparsityCollector
from model import CNNDQN, HybridDQN, MLPDQN
from replay_buffer import EncodedReplayBuffer, NStepReplayBuffer, ReplayBuffer
from snake_env import SnakeEnv
from state_encoders import (
    BlindSniffEncoder,
    CNNEgocentricEncoder,
    DenseOrderedGlobalEncoder,
    DenseOrderedLocalEncoder,
    EgocentricMergedObstacleLocalEncoder,
    EgocentricMergedOrderedLocalEncoder,
    FullStateEncoder,
    HybridEgocentricEncoder,
    HybridMinimalEncoder,
    LocalStateEncoder,
    MergedObstacleLocalEncoder,
    MergedOrderedLocalEncoder,
    OccupancyGlobalEncoder,
    OccupancyLocalEncoder,
    StateEncoder,
    StructuralStateEncoder,
)


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------

_WINDOW_REPS = {
    "local", "local_sparse_ordered",
    "dense_ordered_local", "occupancy_local",
    "merged_obstacle_local", "egocentric_merged_obstacle_local",
    "merged_ordered_local", "egocentric_merged_ordered_local",
    "cnn_egocentric", "hybrid_egocentric", "hybrid_minimal",
}


def make_encoder(
    representation: str,
    board_size: int,
    max_length: int,
    window_size: Optional[int] = None,
) -> StateEncoder:
    if representation in _WINDOW_REPS and window_size is None:
        raise ValueError(
            f"window_size is required for '{representation}' representation"
        )
    if representation == "full":
        return FullStateEncoder(board_size, max_length)
    if representation == "structural":
        return StructuralStateEncoder(board_size, max_length)
    if representation in ("local", "local_sparse_ordered"):
        return LocalStateEncoder(board_size, max_length, window_size)
    if representation == "dense_ordered_global":
        return DenseOrderedGlobalEncoder(board_size, max_length)
    if representation == "dense_ordered_local":
        return DenseOrderedLocalEncoder(board_size, max_length, window_size)
    if representation == "occupancy_global":
        return OccupancyGlobalEncoder(board_size, max_length)
    if representation == "occupancy_local":
        return OccupancyLocalEncoder(board_size, max_length, window_size)
    if representation == "merged_obstacle_local":
        return MergedObstacleLocalEncoder(board_size, max_length, window_size)
    if representation == "egocentric_merged_obstacle_local":
        return EgocentricMergedObstacleLocalEncoder(board_size, max_length, window_size)
    if representation == "merged_ordered_local":
        return MergedOrderedLocalEncoder(board_size, max_length, window_size)
    if representation == "egocentric_merged_ordered_local":
        return EgocentricMergedOrderedLocalEncoder(board_size, max_length, window_size)
    if representation == "blind_sniff":
        return BlindSniffEncoder(board_size, max_length)
    if representation == "cnn_egocentric":
        return CNNEgocentricEncoder(board_size, max_length, window_size)
    if representation == "hybrid_egocentric":
        return HybridEgocentricEncoder(board_size, max_length, window_size)
    if representation == "hybrid_minimal":
        return HybridMinimalEncoder(board_size, max_length, window_size)
    raise ValueError(f"Unknown representation: {representation}")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Greedy evaluation (isolated from training)
# ---------------------------------------------------------------------------

EVAL_SEEDS = list(range(1000, 1030))


def run_greedy_evaluation(
    model: torch.nn.Module,
    encoder: StateEncoder,
    board_size: int,
    max_length: int,
    max_steps: int,
    eval_episodes: int = 5,
) -> dict:
    """Run greedy (epsilon=0) evaluation on isolated environments.

    Does **not** touch training RNG or replay buffer.
    """
    model.eval()
    results = []
    seeds_to_use = EVAL_SEEDS[:eval_episodes]

    for seed in seeds_to_use:
        eval_env = SnakeEnv(
            board_size=board_size,
            max_length=max_length,
            max_steps=max_steps,
            headless=True,
            seed=seed,
        )
        state = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                sv = torch.FloatTensor(encoder.encode(state)).unsqueeze(0)
                action = torch.argmax(model(sv)).item()
            state, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        results.append({
            "score": info["score"],
            "reward": total_reward,
            "survival_steps": info["survived_steps"],
            "death_reason": info["death_reason"],
        })

    model.train()

    scores = [r["score"] for r in results]
    rewards = [r["reward"] for r in results]
    survivals = [r["survival_steps"] for r in results]
    deaths = [r["death_reason"] for r in results]

    n = len(results)
    return {
        "eval_score_mean": float(np.mean(scores)),
        "eval_score_std": float(np.std(scores)),
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "eval_survival_steps_mean": float(np.mean(survivals)),
        "eval_survival_steps_std": float(np.std(survivals)),
        "eval_wall_death_rate": sum(1 for d in deaths if d == "wall") / n,
        "eval_body_death_rate": sum(1 for d in deaths if d == "body") / n,
        "eval_timeout_rate": sum(1 for d in deaths if d == "timeout") / n,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    seed_all(args.seed)

    encoder = make_encoder(
        args.representation,
        args.board_size,
        args.max_length,
        args.local_window_size,
    )
    full_dim = FullStateEncoder(args.board_size, args.max_length).output_dim

    env = SnakeEnv(
        board_size=args.board_size,
        max_length=args.max_length,
        max_steps=args.max_steps,
        headless=True,
        seed=args.seed,
    )

    # Model: Hybrid for hybrid_egocentric, CNN for cnn_egocentric, MLP for all others
    if args.representation == "hybrid_egocentric":
        assert isinstance(encoder, HybridEgocentricEncoder)
        model = HybridDQN(input_dim=encoder.output_dim, mlp_dim=encoder.mlp_dim)
        target_model = HybridDQN(input_dim=encoder.output_dim, mlp_dim=encoder.mlp_dim)
    elif args.representation == "hybrid_minimal":
        assert isinstance(encoder, HybridMinimalEncoder)
        model = HybridDQN(
            input_dim=encoder.output_dim, mlp_dim=encoder.mlp_dim,
            grid_channels=2,
        )
        target_model = HybridDQN(
            input_dim=encoder.output_dim, mlp_dim=encoder.mlp_dim,
            grid_channels=2,
        )
    elif args.representation == "cnn_egocentric":
        model = CNNDQN(input_dim=encoder.output_dim)
        target_model = CNNDQN(input_dim=encoder.output_dim)
    else:
        model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)
        target_model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.SmoothL1Loss()  # Huber loss
    cache_encoded_replay = getattr(args, "cache_encoded_replay", False)

    # Determine termination mode
    if args.total_env_steps is not None:
        total_env_steps = args.total_env_steps
        use_episodes = False
    elif args.episodes is not None:
        total_env_steps = args.episodes * args.max_steps  # estimate
        use_episodes = True
        max_episodes = args.episodes
    else:
        raise ValueError("Must specify --total-env-steps or --episodes")

    # Epsilon schedule
    eps_start = 0.95
    eps_end = args.eps_end
    eps_decay = float(args.eps_decay) if args.eps_decay else float(total_env_steps)

    # Checkpoint milestones
    checkpoint_steps = set(args.checkpoint_steps) if args.checkpoint_steps else set()

    gamma = args.gamma
    n_step = args.n_step
    batch_size = args.batch_size
    learn_every = args.learn_every
    warmup_steps = args.warmup_steps
    target_update_freq = 1000  # steps between target network syncs
    grad_clip_norm = 10.0

    # -- Buffer (after gamma/n_step are defined) --------------------------
    if cache_encoded_replay and n_step > 1:
        buffer = NStepReplayBuffer(capacity=100_000, n_step=n_step, gamma=gamma)
    elif cache_encoded_replay:
        buffer = EncodedReplayBuffer(capacity=100_000)
    else:
        buffer = ReplayBuffer(capacity=100_000)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # -- Config -----------------------------------------------------------
    config = {
        "representation": args.representation,
        "board_size": args.board_size,
        "max_length": args.max_length,
        "total_env_steps": total_env_steps,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "local_window_size": args.local_window_size,
        "input_dimension": encoder.output_dim,
        "gamma": gamma,
        "learning_rate": 1e-3,
        "batch_size": batch_size,
        "learn_every": learn_every,
        "buffer_capacity": 100_000,
        "cache_encoded_replay": cache_encoded_replay,
        "warmup_steps": warmup_steps,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay": eps_decay,
        "target_update_freq": target_update_freq,
        "grad_clip_norm": grad_clip_norm,
        "n_step": n_step,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # -- Collectors -------------------------------------------------------
    analysis_collector = AnalysisCollector(max_transitions=100_000, seed=42)
    sparsity_collector = SparsityCollector(max_samples=10_000, seed=42)

    # -- Timing accumulators ----------------------------------------------
    t_env_step = 0.0
    t_action_encoding = 0.0
    t_replay_cache_encoding = 0.0
    t_batch_encoding = 0.0
    t_optimization = 0.0

    # -- Training loop ----------------------------------------------------
    t_wall_start = time.perf_counter()
    env_steps = 0
    epsilon = eps_start
    episode = 0

    # -- Resume from checkpoint -------------------------------------------
    resume_path = getattr(args, "resume", None)
    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        target_model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        env_steps = ckpt.get("env_steps", 0)
        episode = ckpt.get("episode", 0)
        epsilon = ckpt.get("epsilon", eps_end)
        # Recompute epsilon from schedule based on resumed env_steps
        epsilon = eps_end + (eps_start - eps_end) * math.exp(-env_steps / eps_decay)
        print(f"  Resumed at env_steps={env_steps}, episode={episode}, epsilon={epsilon:.4f}", flush=True)

    csv_rows: list[dict] = []
    eval_csv_rows: list[dict] = []

    state = env.reset()
    episode_reward = 0.0
    episode_loss_sum = 0.0
    episode_loss_count = 0
    episode_fruit_reward = 0.0
    episode_distance_reward = 0.0
    episode_death_penalty = 0.0
    # Initial evaluation at step 0
    if args.eval_interval is not None:
        eval_stats = run_greedy_evaluation(
            model, encoder, args.board_size, args.max_length,
            args.max_steps, args.eval_episodes,
        )
        eval_row = {"env_steps": 0}
        eval_row.update(eval_stats)
        eval_csv_rows.append(eval_row)

    done_training = False
    while not done_training:
        # --- Action selection (epsilon-greedy) ---
        t0 = time.perf_counter()
        encoded = encoder.encode(state)
        t_action_encoding += time.perf_counter() - t0

        sparsity_collector.push(encoded)

        if random.random() > epsilon:
            with torch.no_grad():
                sv = torch.FloatTensor(encoded).unsqueeze(0)
                action = torch.argmax(model(sv)).item()
        else:
            action = random.randrange(3)

        # --- Env step ---
        t0 = time.perf_counter()
        next_state, reward, terminated, truncated, info = env.step(action)
        t_env_step += time.perf_counter() - t0

        ate_fruit = info["ate_fruit"]

        # --- Store ---
        if cache_encoded_replay:
            t0 = time.perf_counter()
            next_encoded = encoder.encode(next_state)
            t_replay_cache_encoding += time.perf_counter() - t0
            buffer.push(encoded, action, reward, next_encoded, terminated, truncated)
        else:
            buffer.push(state, action, reward, next_state, terminated, truncated)
        analysis_collector.push(
            state, action, reward, next_state, terminated, truncated, ate_fruit,
        )

        env_steps += 1

        # --- Optimise (every learn_every steps) ---
        if env_steps >= warmup_steps and len(buffer) >= batch_size and env_steps % learn_every == 0:
            t0 = time.perf_counter()
            t_enc_start = time.perf_counter()
            sample_out = buffer.sample(batch_size, encoder)
            t_batch_encoding += time.perf_counter() - t_enc_start

            b_s, b_a, b_r, b_ns, b_term, b_trunc = sample_out[:6]
            b_discount = sample_out[6] if len(sample_out) > 6 else torch.full_like(b_r, gamma)

            current_q = model(b_s).gather(1, b_a)
            with torch.no_grad():
                # Double DQN: online net selects action, target net evaluates
                next_actions = model(b_ns).argmax(dim=1, keepdim=True)
                max_next_q = target_model(b_ns).gather(1, next_actions).squeeze(1)
                expected_q = b_r + b_discount * max_next_q * (1.0 - b_term)

            optimizer.zero_grad()
            loss = criterion(current_q.squeeze(1), expected_q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            t_optimization += time.perf_counter() - t0
            episode_loss_sum += loss.item()
            episode_loss_count += 1

        # --- Target network sync ---
        if env_steps % target_update_freq == 0 and env_steps >= warmup_steps:
            target_model.load_state_dict(model.state_dict())

        # --- Epsilon decay ---
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-1.0 * env_steps / eps_decay)

        episode_reward += reward
        rc = info["reward_components"]
        episode_fruit_reward += rc["fruit"]
        episode_distance_reward += rc["distance"]
        episode_death_penalty += rc["death"]

        # --- Periodic evaluation ---
        if args.eval_interval is not None and env_steps % args.eval_interval == 0:
            eval_stats = run_greedy_evaluation(
                model, encoder, args.board_size, args.max_length,
                args.max_steps, args.eval_episodes,
            )
            eval_row = {"env_steps": env_steps}
            eval_row.update(eval_stats)
            eval_csv_rows.append(eval_row)
            print(
                f"  [eval @ {env_steps}] "
                f"score={eval_stats['eval_score_mean']:.1f} "
                f"± {eval_stats['eval_score_std']:.1f}",
                flush=True,
            )

        # --- Checkpoint milestone ---
        if checkpoint_steps and env_steps in checkpoint_steps:
            ckpt_name = f"checkpoint_{env_steps // 1000}k.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "env_steps": env_steps,
                    "episode": episode,
                    "epsilon": epsilon,
                },
                os.path.join(output_dir, ckpt_name),
            )
            print(f"  [checkpoint] saved {ckpt_name}", flush=True)

        # --- Episode boundary ---
        if terminated or truncated:
            csv_rows.append({
                "episode": episode,
                "env_steps_total": env_steps,
                "episode_reward": round(episode_reward, 2),
                "score": info["score"],
                "survival_steps": info["survived_steps"],
                "epsilon": round(epsilon, 6),
                "loss_mean": round(episode_loss_sum / max(episode_loss_count, 1), 4),
                "terminated": terminated,
                "truncated": truncated,
                "death_reason": info["death_reason"],
                "fruit_reward_sum": round(episode_fruit_reward, 1),
                "distance_shaping_reward_sum": round(episode_distance_reward, 1),
                "death_penalty_sum": round(episode_death_penalty, 1),
            })

            episode += 1
            if (episode % 10 == 0) or (use_episodes and episode >= max_episodes):
                best = max(r["score"] for r in csv_rows)
                print(
                    f"[{args.representation} seed={args.seed}] "
                    f"ep={episode}  score={info['score']}  "
                    f"best={best}  eps={epsilon:.4f}  steps={env_steps}",
                    flush=True,
                )

            # Check termination
            if use_episodes and episode >= max_episodes:
                done_training = True
            elif env_steps >= total_env_steps:
                done_training = True
            else:
                state = env.reset()
                episode_reward = 0.0
                episode_loss_sum = 0.0
                episode_loss_count = 0
                episode_fruit_reward = 0.0
                episode_distance_reward = 0.0
                episode_death_penalty = 0.0
        else:
            if env_steps >= total_env_steps:
                done_training = True
            else:
                state = next_state

    t_wall = time.perf_counter() - t_wall_start

    # Flush partial episode data if training ended mid-episode
    if not (terminated or truncated) and env_steps >= total_env_steps:
        csv_rows.append({
            "episode": episode,
            "env_steps_total": env_steps,
            "episode_reward": round(episode_reward, 2),
            "score": env._score,
            "survival_steps": env._step_count,
            "epsilon": round(epsilon, 6),
            "loss_mean": round(episode_loss_sum / max(episode_loss_count, 1), 4),
            "terminated": False,
            "truncated": False,
            "death_reason": "budget_cutoff",
            "fruit_reward_sum": round(episode_fruit_reward, 1),
            "distance_shaping_reward_sum": round(episode_distance_reward, 1),
            "death_penalty_sum": round(episode_death_penalty, 1),
        })

    # Final evaluation
    if args.eval_interval is not None:
        last_eval_steps = eval_csv_rows[-1]["env_steps"] if eval_csv_rows else -1
        if env_steps != last_eval_steps:
            eval_stats = run_greedy_evaluation(
                model, encoder, args.board_size, args.max_length,
                args.max_steps, args.eval_episodes,
            )
            eval_row = {"env_steps": env_steps}
            eval_row.update(eval_stats)
            eval_csv_rows.append(eval_row)

    # -- Sparsity ---------------------------------------------------------
    sparsity_mean, sparsity_std = sparsity_collector.compute()

    # -- Learning curve AUC -----------------------------------------------
    normalized_auc = float("nan")
    if len(eval_csv_rows) >= 2:
        steps_arr = np.array([r["env_steps"] for r in eval_csv_rows], dtype=float)
        scores_arr = np.array([r["eval_score_mean"] for r in eval_csv_rows], dtype=float)
        auc = float(np.trapezoid(scores_arr, steps_arr))
        normalized_auc = auc / max(total_env_steps, 1)

    # -- Save outputs -----------------------------------------------------

    # scores.csv
    if csv_rows:
        csv_path = os.path.join(output_dir, "scores.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    # eval_scores.csv
    if eval_csv_rows:
        eval_path = os.path.join(output_dir, "eval_scores.csv")
        with open(eval_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=eval_csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(eval_csv_rows)

    # checkpoint.pt
    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "env_steps": env_steps,
            "episode": episode,
            "epsilon": epsilon,
        },
        ckpt_path,
    )

    # analysis_transitions.pkl
    analysis_path = os.path.join(output_dir, "analysis_transitions.pkl")
    analysis_collector.save(analysis_path)

    # metrics.json
    best_eval = max((r["eval_score_mean"] for r in eval_csv_rows), default=0.0)
    final_eval = eval_csv_rows[-1] if eval_csv_rows else {}
    scores = [r["score"] for r in csv_rows] if csv_rows else []

    metrics = {
        "representation": args.representation,
        "input_dimension": encoder.output_dim,
        "compression_ratio_vs_full": round(encoder.output_dim / full_dim, 4),
        "input_sparsity_mean": round(sparsity_mean, 4),
        "input_sparsity_std": round(sparsity_std, 4),
        "parameter_count": model.parameter_count,
        "training_time_seconds": round(t_wall, 2),
        "env_step_time_seconds": round(t_env_step, 4),
        "action_encoding_time_seconds": round(t_action_encoding, 4),
        "replay_cache_encoding_time_seconds": round(t_replay_cache_encoding, 4),
        "batch_encoding_time_seconds": round(t_batch_encoding, 4),
        "optimization_time_seconds": round(t_optimization, 4),
        "best_eval_score_mean": round(best_eval, 2),
        "final_eval_score_mean": round(final_eval.get("eval_score_mean", 0.0), 2),
        "final_eval_score_std": round(final_eval.get("eval_score_std", 0.0), 2),
        "best_score": max(scores) if scores else 0,
        "learning_curve_auc": round(normalized_auc, 4),
        "total_env_steps": env_steps,
        "total_episodes": episode,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Results saved to {output_dir}/", flush=True)
    print(f"  Total env steps: {env_steps}", flush=True)
    print(f"  Total episodes: {episode}", flush=True)
    print(f"  Training time: {t_wall:.1f}s", flush=True)
    if scores:
        print(f"  Best score: {max(scores)}", flush=True)
    print(f"  Best eval score: {best_eval:.1f}", flush=True)

    # Reward exploitation warning
    if csv_rows:
        total_fruit = sum(r.get("fruit_reward_sum", 0) for r in csv_rows)
        total_dist = sum(abs(r.get("distance_shaping_reward_sum", 0)) for r in csv_rows)
        if total_fruit > 0 and total_dist > 2 * total_fruit:
            print(f"  WARNING: distance shaping reward ({total_dist:.0f}) > 2x fruit reward ({total_fruit:.0f}). "
                  "Possible reward exploitation.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Train DQN on Snake with different state representations."
    )
    p.add_argument(
        "--representation",
        required=True,
        choices=[
            "full", "structural", "local", "local_sparse_ordered",
            "dense_ordered_global", "dense_ordered_local",
            "occupancy_global", "occupancy_local",
            "merged_obstacle_local", "egocentric_merged_obstacle_local",
            "merged_ordered_local", "egocentric_merged_ordered_local",
            "blind_sniff", "cnn_egocentric", "hybrid_egocentric", "hybrid_minimal",
        ],
        help="State representation to use.",
    )
    p.add_argument(
        "--local-window-size",
        type=int,
        default=7,
        choices=[5, 7, 9, 29],
        help="Window size K for local representation (default: 7).",
    )
    p.add_argument("--board-size", type=int, default=15)
    p.add_argument("--max-length", type=int, default=100)
    p.add_argument("--total-env-steps", type=int, default=None,
                    help="Total environment steps budget (step-based termination).")
    p.add_argument("--episodes", type=int, default=None,
                    help="Number of episodes (episode-based termination).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=2000,
                    help="Max steps per episode (truncation).")
    p.add_argument("--gamma", type=float, default=0.95,
                    help="Discount factor (default: 0.95).")
    p.add_argument("--n-step", type=int, default=1,
                    help="N-step return (default: 1).")
    p.add_argument("--batch-size", type=int, default=256,
                    help="Replay batch size (default: 256).")
    p.add_argument("--learn-every", type=int, default=1,
                    help="Optimize every N env steps (default: 1).")
    p.add_argument("--warmup-steps", type=int, default=256,
                    help="Random exploration steps before training starts.")
    p.add_argument("--cache-encoded-replay", action="store_true",
                    help="Store encoded state vectors in replay to avoid sample-time encoding.")
    p.add_argument("--eval-interval", type=int, default=None,
                    help="Run greedy evaluation every N env steps.")
    p.add_argument("--eps-end", type=float, default=1e-20,
                    help="Minimum epsilon for exploration.")
    p.add_argument("--eps-decay", type=int, default=None,
                    help="Epsilon decay constant in env steps (default: total_env_steps).")
    p.add_argument("--checkpoint-steps", type=int, nargs="*", default=None,
                    help="Env-step milestones at which to save a checkpoint "
                         "(e.g., 100000 300000 500000). Final checkpoint always saved.")
    p.add_argument("--eval-episodes", type=int, default=5,
                    help="Number of episodes per greedy evaluation.")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint.pt to resume training from. "
                         "Loads model, optimizer state, and adjusts env_steps/epsilon.")
    args = p.parse_args()

    if args.total_env_steps is None and args.episodes is None:
        p.error("Must specify --total-env-steps or --episodes")

    if args.output_dir is None:
        ws = f"_w{args.local_window_size}" if args.representation in _WINDOW_REPS else ""
        steps_tag = f"{args.total_env_steps or args.episodes}"
        args.output_dir = f"results/{args.representation}{ws}_seed{args.seed}_s{steps_tag}"

    train(args)


if __name__ == "__main__":
    main()
