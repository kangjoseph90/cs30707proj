"""PPO training script for Snake state-representation experiments.

On-policy Actor-Critic with PPO-Clip objective.  Reuses the same
encoder factory and evaluation infrastructure as ``train_snake.py``.

Usage::

    python train_ppo.py --representation egocentric_merged_obstacle_local \
        --local-window-size 29 --total-env-steps 100000 --seed 0 \
        --eval-interval 5000 --eval-episodes 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader

from analysis_collector import AnalysisCollector, SparsityCollector
from model import MLPActorCritic
from snake_env import SnakeEnv
from state_encoders import (
    BlindSniffEncoder,
    DenseOrderedGlobalEncoder,
    DenseOrderedLocalEncoder,
    EgocentricMergedObstacleLocalEncoder,
    EgocentricMergedOrderedLocalEncoder,
    FullStateEncoder,
    LocalStateEncoder,
    MergedObstacleLocalEncoder,
    MergedOrderedLocalEncoder,
    OccupancyGlobalEncoder,
    OccupancyLocalEncoder,
    StateEncoder,
    StructuralStateEncoder,
)


# ---------------------------------------------------------------------------
# Encoder factory (same as train_snake.py)
# ---------------------------------------------------------------------------

_WINDOW_REPS = {
    "local", "local_sparse_ordered",
    "dense_ordered_local", "occupancy_local",
    "merged_obstacle_local", "egocentric_merged_obstacle_local",
    "merged_ordered_local", "egocentric_merged_ordered_local",
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
# Greedy evaluation
# ---------------------------------------------------------------------------

EVAL_SEEDS = list(range(1000, 1030))


def run_greedy_evaluation(
    model: MLPActorCritic,
    encoder: StateEncoder,
    board_size: int,
    max_length: int,
    max_steps: int,
    eval_episodes: int = 5,
) -> dict:
    """Run greedy (argmax policy) evaluation on isolated environments."""
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
                logits, _ = model(sv)
                action = torch.argmax(logits).item()
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
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    env: SnakeEnv,
    model: MLPActorCritic,
    encoder: StateEncoder,
    num_steps: int,
):
    """Collect *num_steps* environment transitions under the current policy.

    Returns a dict of numpy arrays plus the bootstrap next_value.
    """
    states_list = []
    actions_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    log_probs_list = []

    state = env.reset()
    episode_scores = []
    current_score = 0

    for _ in range(num_steps):
        encoded = encoder.encode(state)
        with torch.no_grad():
            sv = torch.FloatTensor(encoded).unsqueeze(0)
            action, log_prob, value, _ = model.get_action_and_value(sv)

        action_int = action.item()
        next_state, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated

        states_list.append(encoded)
        actions_list.append(action_int)
        rewards_list.append(reward)
        dones_list.append(float(done))
        values_list.append(value.item())
        log_probs_list.append(log_prob.item())

        current_score = info["score"]

        if done:
            episode_scores.append(current_score)
            current_score = 0
            state = env.reset()
        else:
            state = next_state

    # Bootstrap value for the last state
    with torch.no_grad():
        encoded = encoder.encode(state)
        sv = torch.FloatTensor(encoded).unsqueeze(0)
        _, _, next_value, _ = model.get_action_and_value(sv)
        bootstrap_value = next_value.item()

    return {
        "states": np.array(states_list, dtype=np.float32),
        "actions": np.array(actions_list, dtype=np.int64),
        "rewards": np.array(rewards_list, dtype=np.float32),
        "dones": np.array(dones_list, dtype=np.float32),
        "values": np.array(values_list, dtype=np.float32),
        "log_probs": np.array(log_probs_list, dtype=np.float32),
        "bootstrap_value": bootstrap_value,
        "episode_scores": episode_scores,
    }


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns (discounted value targets).

    Returns (advantages, returns) as float32 arrays.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n)):
        if t == n - 1:
            next_value = bootstrap_value
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model: MLPActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    num_epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> dict:
    """Run *num_epochs* of minibatch PPO updates.  Returns loss stats."""
    states = torch.FloatTensor(rollout["states"])
    actions = torch.LongTensor(rollout["actions"])
    old_log_probs = torch.FloatTensor(rollout["old_log_probs"])
    advantages = torch.FloatTensor(rollout["advantages"])
    returns = torch.FloatTensor(rollout["returns"])

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    total_pg_loss = 0.0
    total_v_loss = 0.0
    total_ent = 0.0
    total_approx_kl = 0.0
    num_batches = 0

    for _ in range(num_epochs):
        for batch in loader:
            b_s, b_a, b_old_lp, b_adv, b_ret = batch

            _, new_log_prob, new_value, entropy = model.get_action_and_value(
                b_s, b_a
            )

            # Policy loss (PPO-Clip)
            ratio = torch.exp(new_log_prob - b_old_lp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * b_adv
            pg_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            v_loss = 0.5 * ((new_value - b_ret) ** 2).mean()

            # Entropy bonus
            ent_loss = entropy.mean()

            # Total loss
            loss = pg_loss + value_coef * v_loss - entropy_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_ent += ent_loss.item()
            total_approx_kl += (b_old_lp - new_log_prob).mean().item()
            num_batches += 1

    return {
        "pg_loss": total_pg_loss / num_batches,
        "value_loss": total_v_loss / num_batches,
        "entropy": total_ent / num_batches,
        "approx_kl": total_approx_kl / num_batches,
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

    model = MLPActorCritic(input_dim=encoder.output_dim, output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    total_env_steps = args.total_env_steps
    rollout_steps = args.rollout_steps
    gamma = args.gamma
    gae_lambda = args.gae_lambda

    # PPO hyperparams
    num_epochs = args.num_epochs
    minibatch_size = args.minibatch_size
    clip_ratio = args.clip_ratio
    value_coef = args.value_coef
    entropy_coef = args.entropy_coef
    max_grad_norm = args.max_grad_norm

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "algorithm": "PPO",
        "representation": args.representation,
        "board_size": args.board_size,
        "max_length": args.max_length,
        "total_env_steps": total_env_steps,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "local_window_size": args.local_window_size,
        "input_dimension": encoder.output_dim,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": args.lr,
        "rollout_steps": rollout_steps,
        "num_epochs": num_epochs,
        "minibatch_size": minibatch_size,
        "clip_ratio": clip_ratio,
        "value_coef": value_coef,
        "entropy_coef": entropy_coef,
        "max_grad_norm": max_grad_norm,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Collectors
    sparsity_collector = SparsityCollector(max_samples=10_000, seed=42)

    # Timing
    t_wall_start = time.perf_counter()
    t_rollout = 0.0
    t_update = 0.0
    t_eval = 0.0

    # Training loop
    env_steps = 0
    rollout_count = 0
    eval_csv_rows: list[dict] = []
    csv_rows: list[dict] = []

    # Initial evaluation
    if args.eval_interval is not None:
        eval_stats = run_greedy_evaluation(
            model, encoder, args.board_size, args.max_length,
            args.max_steps, args.eval_episodes,
        )
        eval_row = {"env_steps": 0}
        eval_row.update(eval_stats)
        eval_csv_rows.append(eval_row)

    while env_steps < total_env_steps:
        # --- Collect rollout ---
        t0 = time.perf_counter()
        rollout = collect_rollout(env, model, encoder, rollout_steps)
        t_rollout += time.perf_counter() - t0

        # Sparsity
        for i in range(0, rollout_steps, max(1, rollout_steps // 100)):
            sparsity_collector.push(rollout["states"][i])

        # --- Compute GAE ---
        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            rollout["bootstrap_value"],
            gamma,
            gae_lambda,
        )
        rollout["advantages"] = advantages
        rollout["returns"] = returns
        rollout["old_log_probs"] = rollout["log_probs"]

        # --- PPO update ---
        t0 = time.perf_counter()
        update_stats = ppo_update(
            model, optimizer, rollout,
            num_epochs, minibatch_size,
            clip_ratio, value_coef, entropy_coef, max_grad_norm,
        )
        t_update += time.perf_counter() - t0

        env_steps += rollout_steps
        rollout_count += 1

        # --- Episode scores from rollout ---
        ep_scores = rollout["episode_scores"]
        for i, sc in enumerate(ep_scores):
            csv_rows.append({
                "episode": len(csv_rows),
                "env_steps_total": env_steps,
                "score": sc,
                "rollout": rollout_count,
            })

        # --- Logging ---
        if ep_scores:
            best = max(ep_scores)
            mean_sc = np.mean(ep_scores)
            print(
                f"[rollout {rollout_count}  steps={env_steps}]  "
                f"ep={len(ep_scores)}  mean={mean_sc:.1f}  best={best}  "
                f"pg_loss={update_stats['pg_loss']:.4f}  "
                f"v_loss={update_stats['value_loss']:.4f}  "
                f"ent={update_stats['entropy']:.3f}  "
                f"kl={update_stats['approx_kl']:.4f}",
                flush=True,
            )

        # --- Periodic evaluation ---
        if args.eval_interval is not None and env_steps % args.eval_interval < rollout_steps:
            t0 = time.perf_counter()
            eval_stats = run_greedy_evaluation(
                model, encoder, args.board_size, args.max_length,
                args.max_steps, args.eval_episodes,
            )
            t_eval += time.perf_counter() - t0
            eval_row = {"env_steps": env_steps}
            eval_row.update(eval_stats)
            eval_csv_rows.append(eval_row)
            print(
                f"  [eval @ {env_steps}] "
                f"score={eval_stats['eval_score_mean']:.1f} "
                f"± {eval_stats['eval_score_std']:.1f}",
                flush=True,
            )

    t_wall = time.perf_counter() - t_wall_start

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

    # -- Sparsity --
    sparsity_mean, sparsity_std = sparsity_collector.compute()

    # -- Learning curve AUC --
    normalized_auc = float("nan")
    if len(eval_csv_rows) >= 2:
        steps_arr = np.array([r["env_steps"] for r in eval_csv_rows], dtype=float)
        scores_arr = np.array([r["eval_score_mean"] for r in eval_csv_rows], dtype=float)
        auc = float(np.trapezoid(scores_arr, steps_arr))
        normalized_auc = auc / max(total_env_steps, 1)

    # -- Save outputs --
    if csv_rows:
        csv_path = os.path.join(output_dir, "scores.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    if eval_csv_rows:
        eval_path = os.path.join(output_dir, "eval_scores.csv")
        with open(eval_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=eval_csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(eval_csv_rows)

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "env_steps": env_steps,
            "rollout_count": rollout_count,
        },
        ckpt_path,
    )

    best_eval = max((r["eval_score_mean"] for r in eval_csv_rows), default=0.0)
    final_eval = eval_csv_rows[-1] if eval_csv_rows else {}
    scores = [r["score"] for r in csv_rows] if csv_rows else []

    metrics = {
        "representation": args.representation,
        "algorithm": "PPO",
        "input_dimension": encoder.output_dim,
        "compression_ratio_vs_full": round(encoder.output_dim / full_dim, 4),
        "input_sparsity_mean": round(sparsity_mean, 4),
        "input_sparsity_std": round(sparsity_std, 4),
        "parameter_count": model.parameter_count,
        "training_time_seconds": round(t_wall, 2),
        "rollout_time_seconds": round(t_rollout, 4),
        "update_time_seconds": round(t_update, 4),
        "eval_time_seconds": round(t_eval, 4),
        "best_eval_score_mean": round(best_eval, 2),
        "final_eval_score_mean": round(final_eval.get("eval_score_mean", 0.0), 2),
        "final_eval_score_std": round(final_eval.get("eval_score_std", 0.0), 2),
        "best_score": max(scores) if scores else 0,
        "learning_curve_auc": round(normalized_auc, 4),
        "total_env_steps": env_steps,
        "total_rollouts": rollout_count,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Results saved to {output_dir}/", flush=True)
    print(f"  Total env steps: {env_steps}", flush=True)
    print(f"  Total rollouts: {rollout_count}", flush=True)
    print(f"  Training time: {t_wall:.1f}s", flush=True)
    if scores:
        print(f"  Best score: {max(scores)}", flush=True)
    print(f"  Best eval score: {best_eval:.1f}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Train PPO on Snake with different state representations."
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
            "blind_sniff",
        ],
        help="State representation to use.",
    )
    p.add_argument("--local-window-size", type=int, default=7,
                    help="Window size K for local representation (default: 7).")
    p.add_argument("--board-size", type=int, default=15)
    p.add_argument("--max-length", type=int, default=100)
    p.add_argument("--total-env-steps", type=int, required=True,
                    help="Total environment steps budget.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=2000,
                    help="Max steps per episode (truncation).")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--gae-lambda", type=float, default=0.95,
                    help="GAE lambda (default: 0.95).")
    p.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate (default: 3e-4).")
    p.add_argument("--rollout-steps", type=int, default=2048,
                    help="Steps per rollout (default: 2048).")
    p.add_argument("--num-epochs", type=int, default=10,
                    help="PPO epochs per rollout (default: 10).")
    p.add_argument("--minibatch-size", type=int, default=64,
                    help="Minibatch size for PPO update (default: 64).")
    p.add_argument("--clip-ratio", type=float, default=0.2,
                    help="PPO clip ratio (default: 0.2).")
    p.add_argument("--value-coef", type=float, default=0.5,
                    help="Value loss coefficient (default: 0.5).")
    p.add_argument("--entropy-coef", type=float, default=0.01,
                    help="Entropy bonus coefficient (default: 0.01).")
    p.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="Max gradient norm for clipping (default: 0.5).")
    p.add_argument("--eval-interval", type=int, default=None,
                    help="Run greedy evaluation every N env steps.")
    p.add_argument("--eval-episodes", type=int, default=5,
                    help="Number of episodes per greedy evaluation.")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    if args.output_dir is None:
        ws = f"_w{args.local_window_size}" if args.representation in _WINDOW_REPS else ""
        args.output_dir = f"results/ppo_{args.representation}{ws}_seed{args.seed}_s{args.total_env_steps}"

    train(args)


if __name__ == "__main__":
    main()
