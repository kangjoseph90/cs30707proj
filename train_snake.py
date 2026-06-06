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
from model import MLPDQN
from replay_buffer import EncodedReplayBuffer, NStepReplayBuffer, ReplayBuffer
from snake_env import SnakeEnv
from state_encoders import (
    BlindSniffEncoder,
    DenseOrderedGlobalEncoder,
    DenseOrderedLocalEncoder,
    EgocentricMergedObstacleLocalEncoder,
    EgocentricMergedOrderedLocalEncoder,
    EgocentricMergedReleaseTimeLocalEncoder,
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
# Encoder factory
# ---------------------------------------------------------------------------

_WINDOW_REPS = {
    "local", "local_sparse_ordered",
    "dense_ordered_local", "occupancy_local",
    "merged_obstacle_local", "egocentric_merged_obstacle_local",
    "merged_ordered_local", "egocentric_merged_ordered_local",
    "egocentric_merged_release_time_local",
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
    if representation == "egocentric_merged_release_time_local":
        return EgocentricMergedReleaseTimeLocalEncoder(board_size, max_length, window_size)
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
# Greedy evaluation (isolated from training)
# ---------------------------------------------------------------------------

EVAL_SEEDS = list(range(1000, 1030))


def run_greedy_evaluation(
    model: MLPDQN,
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

    model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)
    target_model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.SmoothL1Loss()  # Huber loss
    cache_encoded_replay = getattr(args, "cache_encoded_replay", False)

    if args.train_with_planner:
        from planner import SafetyPlanner
        planner = SafetyPlanner(
            planner_weight=args.planner_weight,
            area_weight=args.planner_area_weight,
            tail_reach_weight=args.planner_tail_reach_weight,
        )
    else:
        planner = None

    if args.train_with_beam_veto:
        if planner is None:
            raise ValueError("--train-with-beam-veto requires --train-with-planner")
        from beam_veto_planner import BeamSearchVetoPlanner, BeamVetoConfig
        import collections as _collections
        beam_config = BeamVetoConfig(
            max_depth=args.beam_max_depth,
            beam_width=args.beam_width,
            use_conditional_trigger=not args.beam_always_on,
            trigger_on_bfs_override=not args.beam_disable_trigger_on_bfs_override,
            trigger_on_tail_unreachable=not args.beam_disable_trigger_on_tail_unreachable,
            trigger_on_low_legal_actions=not args.beam_disable_trigger_on_low_legal_actions,
        )
        beam_planner = BeamSearchVetoPlanner(planner, beam_config)
        beam_trigger_counts = _collections.Counter()
    else:
        beam_planner = None
        beam_trigger_counts = None

    # -- Planner tracking -------------------------------------------------
    total_interventions = 0
    total_all_unsafe = 0
    total_safe_explore = 0
    all_reachable_areas = []
    all_tail_reachables = []

    # -- Beam veto tracking -----------------------------------------------
    total_beam_search_runs = 0
    total_beam_vetoes = 0
    total_beam_all_forced = 0
    total_beam_runtime_ms = 0.0

    # Episode-level stats
    episode_interventions = 0
    episode_all_unsafe = 0
    episode_beam_search_runs = 0
    episode_beam_vetoes = 0
    episode_beam_all_forced = 0

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
    batch_size = 256
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
        "train_with_planner": args.train_with_planner,
        "planner_weight": args.planner_weight,
        "planner_area_weight": args.planner_area_weight,
        "planner_tail_reach_weight": args.planner_tail_reach_weight,
        "planner_safe_exploration": args.planner_safe_exploration,
        "planner_mask_target_actions": args.planner_mask_target_actions,
        "train_with_beam_veto": args.train_with_beam_veto,
        "beam_max_depth": args.beam_max_depth,
        "beam_width": args.beam_width,
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
        print(f"Resuming from {resume_path}")
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
        print(f"  Resumed at env_steps={env_steps}, episode={episode}, epsilon={epsilon:.4f}")

    csv_rows: list[dict] = []
    eval_csv_rows: list[dict] = []

    state = env.reset()
    episode_reward = 0.0
    episode_loss_sum = 0.0
    episode_loss_count = 0
    episode_fruit_reward = 0.0
    episode_distance_reward = 0.0
    episode_death_penalty = 0.0
    episode_interventions = 0
    episode_all_unsafe = 0
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

        with torch.no_grad():
            sv = torch.FloatTensor(encoded).unsqueeze(0)
            q_values = model(sv).squeeze(0).numpy()
            dqn_argmax = int(np.argmax(q_values))

        is_intervention = False
        all_unsafe = False
        is_safe_exploration = False

        if planner is not None:
            # Planner-guided action selection
            decision = planner.choose_action(env, q_values)
            is_intervention = (decision.chosen_action != dqn_argmax)
            all_unsafe = all(info.immediate_death for info in decision.actions)

            if random.random() > epsilon:
                # Exploitation: argmax(combined_score) among safe actions
                baseline_action = decision.chosen_action
            else:
                # Exploration
                if args.planner_safe_exploration:
                    safe_actions = [info.action for info in decision.actions if not info.immediate_death]
                    if safe_actions:
                        baseline_action = random.choice(safe_actions)
                    else:
                        baseline_action = dqn_argmax
                    is_safe_exploration = True
                else:
                    baseline_action = random.randrange(3)

            # Beam veto layer (only during exploitation)
            if beam_planner is not None and not is_safe_exploration:
                beam_decision = beam_planner.choose_action(env, q_values, decision)
                action = beam_decision.chosen_action
                if beam_trigger_counts is not None:
                    beam_trigger_counts[beam_decision.trigger_reason] += 1
                if beam_decision.planner_triggered:
                    total_beam_search_runs += 1
                    episode_beam_search_runs += 1
                    total_beam_runtime_ms += beam_decision.runtime_ms
                if beam_decision.veto_applied:
                    total_beam_vetoes += 1
                    episode_beam_vetoes += 1
                if beam_decision.planner_triggered and all(
                    r.likely_forced_death for r in beam_decision.root_results.values()
                ):
                    total_beam_all_forced += 1
                    episode_beam_all_forced += 1
            else:
                action = baseline_action

            # Log step-level metrics
            chosen_info = next(info for info in decision.actions if info.action == action)
            all_reachable_areas.append(chosen_info.reachable_area)
            all_tail_reachables.append(1.0 if chosen_info.can_reach_tail else 0.0)
            if is_intervention:
                total_interventions += 1
                episode_interventions += 1
            if all_unsafe:
                total_all_unsafe += 1
                episode_all_unsafe += 1
            if is_safe_exploration:
                total_safe_explore += 1
        else:
            # Standard epsilon-greedy
            if random.random() > epsilon:
                action = dqn_argmax
            else:
                action = random.randrange(3)

        # --- Env step ---
        t0 = time.perf_counter()
        next_state, reward, terminated, truncated, info = env.step(action)
        t_env_step += time.perf_counter() - t0

        ate_fruit = info["ate_fruit"]

        # --- Compute next safe mask ---
        if planner is not None and not terminated:
            next_decision = planner.analyze_actions(env, [0.0, 0.0, 0.0])
            next_safe_mask = np.array([not info.immediate_death for info in next_decision], dtype=np.bool_)
        else:
            next_safe_mask = None

        # --- Store ---
        if cache_encoded_replay:
            t0 = time.perf_counter()
            next_encoded = encoder.encode(next_state)
            t_replay_cache_encoding += time.perf_counter() - t0
            buffer.push(encoded, action, reward, next_encoded, terminated, truncated, next_safe_mask)
        else:
            buffer.push(state, action, reward, next_state, terminated, truncated, next_safe_mask)
        analysis_collector.push(
            state, action, reward, next_state, terminated, truncated, ate_fruit,
        )

        env_steps += 1

        # --- Optimise ---
        if env_steps >= warmup_steps and len(buffer) >= batch_size:
            t0 = time.perf_counter()
            t_enc_start = time.perf_counter()
            sample_out = buffer.sample(batch_size, encoder)
            t_batch_encoding += time.perf_counter() - t_enc_start

            b_s, b_a, b_r, b_ns, b_term, b_trunc = sample_out[:6]
            if isinstance(buffer, NStepReplayBuffer):
                b_discount = sample_out[6]
                b_next_safe_mask = sample_out[7]
            else:
                b_discount = torch.full_like(b_r, gamma)
                b_next_safe_mask = sample_out[6]

            current_q = model(b_s).gather(1, b_a)
            with torch.no_grad():
                # Double DQN: online net selects action, target net evaluates
                if args.train_with_planner and args.planner_mask_target_actions:
                    online_next_q = model(b_ns)
                    masked_online_next_q = online_next_q.masked_fill(
                        ~b_next_safe_mask.to(online_next_q.device),
                        -torch.inf,
                    )
                    all_unsafe = ~b_next_safe_mask.to(online_next_q.device).any(dim=1)
                    masked_online_next_q[all_unsafe] = online_next_q[all_unsafe]
                    
                    next_actions = masked_online_next_q.argmax(dim=1, keepdim=True)
                else:
                    next_actions = model(b_ns).argmax(dim=1, keepdim=True)

                max_next_q = target_model(b_ns).gather(1, next_actions).squeeze(1)
                expected_q = b_r + b_discount.to(max_next_q.device) * max_next_q * (1.0 - b_term.to(max_next_q.device))

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
                f"± {eval_stats['eval_score_std']:.1f}"
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
            print(f"  [checkpoint] saved {ckpt_name}")

        # --- Episode boundary ---
        if terminated or truncated:
            row = {
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
            }
            if args.train_with_planner:
                row.update({
                    "planner_intervention_count": episode_interventions,
                    "all_actions_unsafe_count": episode_all_unsafe,
                })
            if args.train_with_beam_veto:
                row.update({
                    "beam_search_runs": episode_beam_search_runs,
                    "beam_vetoes": episode_beam_vetoes,
                    "beam_all_forced_death": episode_beam_all_forced,
                })
            csv_rows.append(row)

            episode += 1
            if (episode % 10 == 0) or (use_episodes and episode >= max_episodes):
                best = max(r["score"] for r in csv_rows)
                print(
                    f"[{args.representation} seed={args.seed}] "
                    f"ep={episode}  score={info['score']}  "
                    f"best={best}  eps={epsilon:.4f}  steps={env_steps}"
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
                episode_interventions = 0
                episode_all_unsafe = 0
                episode_beam_search_runs = 0
                episode_beam_vetoes = 0
                episode_beam_all_forced = 0
        else:
            if env_steps >= total_env_steps:
                done_training = True
            else:
                state = next_state

    t_wall = time.perf_counter() - t_wall_start

    # Flush partial episode data if training ended mid-episode
    if not (terminated or truncated) and env_steps >= total_env_steps:
        row = {
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
        }
        if args.train_with_planner:
            row.update({
                "planner_intervention_count": episode_interventions,
                "all_actions_unsafe_count": episode_all_unsafe,
            })
        csv_rows.append(row)

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
    if args.train_with_planner:
        metrics.update({
            "planner_interventions": total_interventions,
            "planner_intervention_rate": total_interventions / max(env_steps, 1),
            "planner_all_actions_unsafe": total_all_unsafe,
            "planner_all_actions_unsafe_rate": total_all_unsafe / max(env_steps, 1),
            "planner_safe_exploration_actions": total_safe_explore,
            "planner_mean_reachable_area": float(np.mean(all_reachable_areas)) if all_reachable_areas else 0.0,
            "planner_tail_reachable_rate": float(np.mean(all_tail_reachables)) if all_tail_reachables else 0.0,
        })
    if args.train_with_beam_veto:
        metrics.update({
            "beam_search_runs": total_beam_search_runs,
            "beam_search_run_rate": total_beam_search_runs / max(env_steps, 1),
            "beam_vetoes": total_beam_vetoes,
            "beam_veto_rate_per_env_step": total_beam_vetoes / max(env_steps, 1),
            "beam_veto_rate_per_search_run": total_beam_vetoes / max(total_beam_search_runs, 1),
            "beam_all_roots_forced_death": total_beam_all_forced,
            "beam_all_roots_forced_death_rate": total_beam_all_forced / max(env_steps, 1),
            "beam_runtime_ms_total": round(total_beam_runtime_ms, 2),
            "beam_runtime_ms_mean": round(total_beam_runtime_ms / max(total_beam_search_runs, 1), 2),
        })
        if beam_trigger_counts:
            for reason, count in beam_trigger_counts.most_common():
                metrics[f"beam_trigger_{reason}"] = count
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone. Results saved to {output_dir}/")
    print(f"  Total env steps: {env_steps}")
    print(f"  Total episodes: {episode}")
    print(f"  Training time: {t_wall:.1f}s")
    if scores:
        print(f"  Best score: {max(scores)}")
    print(f"  Best eval score: {best_eval:.1f}")

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
            "egocentric_merged_release_time_local",
            "blind_sniff",
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
    
    # Planner CLI args
    p.add_argument("--train-with-planner", action="store_true", default=False)
    p.add_argument("--planner-weight", type=float, default=2.0)
    p.add_argument("--planner-area-weight", type=float, default=1.0)
    p.add_argument("--planner-tail-reach-weight", type=float, default=1.0)
    p.add_argument("--planner-safe-exploration", action="store_true", default=True)
    p.add_argument("--no-planner-safe-exploration", action="store_false", dest="planner_safe_exploration")
    p.add_argument("--planner-mask-target-actions", action="store_true", default=True)
    p.add_argument("--no-planner-mask-target-actions", action="store_false", dest="planner_mask_target_actions")

    # Beam veto CLI args
    p.add_argument("--train-with-beam-veto", action="store_true", default=False)
    p.add_argument("--beam-max-depth", type=int, default=25)
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--beam-always-on", action="store_true", default=False)
    p.add_argument("--beam-disable-trigger-on-bfs-override", action="store_true", default=False)
    p.add_argument("--beam-disable-trigger-on-tail-unreachable", action="store_true", default=False)
    p.add_argument("--beam-disable-trigger-on-low-legal-actions", action="store_true", default=False)

    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint.pt to resume training from. "
                         "Loads model, optimizer state, and adjusts env_steps/epsilon.")
    args = p.parse_args()

    if args.total_env_steps is None and args.episodes is None:
        p.error("Must specify --total-env-steps or --episodes")

    if args.train_with_beam_veto and not args.train_with_planner:
        p.error("--train-with-beam-veto requires --train-with-planner")

    if args.output_dir is None:
        ws = f"_w{args.local_window_size}" if args.representation in _WINDOW_REPS else ""
        steps_tag = f"{args.total_env_steps or args.episodes}"
        args.output_dir = f"results/{args.representation}{ws}_seed{args.seed}_s{steps_tag}"

    train(args)


if __name__ == "__main__":
    main()
