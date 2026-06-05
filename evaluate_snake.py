"""Evaluate a saved DQN checkpoint on Snake (deterministic, epsilon=0).

Usage::

    python evaluate_snake.py results/structural_seed42/checkpoint.pt --episodes 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from model import MLPDQN
from snake_env import SnakeEnv
from train_snake import make_encoder, seed_all


def evaluate(
    checkpoint_path: str,
    num_episodes: int = 100,
    seed: int = 0,
    render: bool = False,
    use_planner: bool = False,
    planner_weight: float = 2.0,
    area_weight: float = 1.0,
    tail_reach_weight: float = 1.0,
    output_dir: str | None = None,
    representation_override: str | None = None,
    local_window_size_override: int | None = None,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    rep = representation_override if representation_override else config["representation"]
    ws = local_window_size_override if local_window_size_override is not None else config.get("local_window_size")

    encoder = make_encoder(
        rep,
        config["board_size"],
        config["max_length"],
        ws,
    )

    model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = SnakeEnv(
        board_size=config["board_size"],
        max_length=config["max_length"],
        max_steps=config.get("max_steps", 2000),
        headless=not render,
        seed=seed,
    )

    if use_planner:
        from planner import SafetyPlanner
        planner = SafetyPlanner(
            planner_weight=planner_weight,
            area_weight=area_weight,
            tail_reach_weight=tail_reach_weight,
        )
    else:
        planner = None

    scores = []
    survival_steps = []
    death_reasons = []
    episode_interventions = []
    episode_all_unsafe = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        interventions = 0
        all_unsafe_steps = 0

        while not done:
            with torch.no_grad():
                sv = torch.FloatTensor(encoder.encode(state)).unsqueeze(0)
                q_values = model(sv).squeeze(0).numpy()
                dqn_argmax = int(np.argmax(q_values))

            if planner is not None:
                decision = planner.choose_action(env, q_values)
                action = decision.chosen_action

                if action != dqn_argmax:
                    interventions += 1

                if all(info.immediate_death for info in decision.actions):
                    all_unsafe_steps += 1
            else:
                action = dqn_argmax

            state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if render:
                env.render()

        scores.append(info["score"])
        survival_steps.append(info["survived_steps"])
        death_reasons.append(info["death_reason"])
        episode_interventions.append(interventions)
        episode_all_unsafe.append(all_unsafe_steps)

        if (ep + 1) % 10 == 0:
            print(
                f"  Episode {ep+1}/{num_episodes}  "
                f"score={info['score']}  mean={np.mean(scores):.2f}"
            )

    scores = np.array(scores)
    survival_steps = np.array(survival_steps)

    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_score = float(np.std(scores))
    min_score = int(np.min(scores))
    max_score = int(np.max(scores))

    ge20_rate = float(np.mean(scores >= 20))
    ge30_rate = float(np.mean(scores >= 30))
    ge40_rate = float(np.mean(scores >= 40))

    total_episodes = len(scores)
    body_death_rate = sum(1 for r in death_reasons if r == "body") / total_episodes
    wall_death_rate = sum(1 for r in death_reasons if r == "wall") / total_episodes
    timeout_rate = sum(1 for r in death_reasons if r == "timeout") / total_episodes

    mean_survival_steps = float(np.mean(survival_steps))
    mean_interventions = float(np.mean(episode_interventions))
    total_interventions = int(sum(episode_interventions))
    total_all_unsafe = int(sum(episode_all_unsafe))

    print(f"\nEvaluation complete ({num_episodes} episodes):")
    print(f"  Mean score : {mean_score:.2f}")
    print(f"  Median     : {median_score:.1f}")
    print(f"  Std score  : {std_score:.2f}")
    print(f"  Min / Max  : {min_score} / {max_score}")
    print(f"  >=20       : {ge20_rate * 100:.1f}%")
    print(f"  >=30       : {ge30_rate * 100:.1f}%")
    print(f"  >=40       : {ge40_rate * 100:.1f}%")
    print(f"  Body death rate : {body_death_rate * 100:.1f}%")
    print(f"  Wall death rate : {wall_death_rate * 100:.1f}%")
    print(f"  Timeout rate    : {timeout_rate * 100:.1f}%")
    print(f"  Mean survival steps        : {mean_survival_steps:.1f}")
    print(f"  Planner intervention count : {total_interventions} (mean: {mean_interventions:.2f}/ep)")
    print(f"  All-actions-unsafe count   : {total_all_unsafe}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save per-episode results
        ep_file = os.path.join(output_dir, "episode_results.csv")
        with open(ep_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "score", "survival_steps", "death_reason", "interventions", "all_unsafe"])
            for idx in range(total_episodes):
                writer.writerow([
                    idx,
                    scores[idx],
                    survival_steps[idx],
                    death_reasons[idx],
                    episode_interventions[idx],
                    episode_all_unsafe[idx],
                ])
        print(f"Saved episode results to {ep_file}")

        # Save summary CSV
        sum_file = os.path.join(output_dir, "summary.csv")
        with open(sum_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["mean score", mean_score])
            writer.writerow(["median score", median_score])
            writer.writerow(["std", std_score])
            writer.writerow(["min", min_score])
            writer.writerow(["max", max_score])
            writer.writerow([">=20", ge20_rate])
            writer.writerow([">=30", ge30_rate])
            writer.writerow([">=40", ge40_rate])
            writer.writerow(["body death rate", body_death_rate])
            writer.writerow(["wall death rate", wall_death_rate])
            writer.writerow(["timeout rate", timeout_rate])
            writer.writerow(["mean survival steps", mean_survival_steps])
            writer.writerow(["planner intervention count", total_interventions])
            writer.writerow(["all-actions-unsafe count", total_all_unsafe])
        print(f"Saved summary to {sum_file}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a Snake DQN checkpoint.")
    p.add_argument("checkpoint", type=str, nargs="?", default=None, help="Path to checkpoint.pt")
    p.add_argument("--checkpoint", type=str, dest="checkpoint_opt", default=None, help="Path to checkpoint.pt")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true", default=False)
    
    # Planner CLI args
    p.add_argument("--use-planner", action="store_true", default=False)
    p.add_argument("--planner-weight", type=float, default=2.0)
    p.add_argument("--planner-area-weight", type=float, default=1.0)
    p.add_argument("--planner-tail-reach-weight", type=float, default=1.0)
    
    # Representation overrides
    p.add_argument("--representation", type=str, default=None)
    p.add_argument("--local-window-size", type=int, default=None)
    
    # Greedy flag compatibility
    p.add_argument("--greedy", action="store_true", default=True)
    p.add_argument("--output-dir", type=str, default=None)
    
    args = p.parse_args()
    
    checkpoint_path = args.checkpoint_opt if args.checkpoint_opt else args.checkpoint
    if not checkpoint_path:
        p.error("the following arguments are required: checkpoint")
        
    evaluate(
        checkpoint_path=checkpoint_path,
        num_episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        use_planner=args.use_planner,
        planner_weight=args.planner_weight,
        area_weight=args.planner_area_weight,
        tail_reach_weight=args.planner_tail_reach_weight,
        output_dir=args.output_dir,
        representation_override=args.representation,
        local_window_size_override=args.local_window_size,
    )


if __name__ == "__main__":
    main()
