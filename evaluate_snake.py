"""Evaluate a saved DQN checkpoint on Snake (deterministic, epsilon=0).

Usage::

    python evaluate_snake.py results/structural_seed42/checkpoint.pt --episodes 30
"""

from __future__ import annotations

import argparse
import collections
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
    use_beam_veto: bool = False,
    beam_max_depth: int = 6,
    beam_width: int = 8,
    beam_always_on: bool = False,
    beam_area_margin: int = 10,
    beam_trigger_on_bfs_override: bool = True,
    beam_trigger_on_tail_unreachable: bool = True,
    beam_trigger_on_low_legal_actions: bool = True,
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

    if use_beam_veto:
        if planner is None:
            raise ValueError("--use-beam-veto requires --use-planner")
        from beam_veto_planner import BeamSearchVetoPlanner, BeamVetoConfig
        beam_config = BeamVetoConfig(
            max_depth=beam_max_depth,
            beam_width=beam_width,
            use_conditional_trigger=not beam_always_on,
            trigger_area_margin=beam_area_margin,
            trigger_on_bfs_override=beam_trigger_on_bfs_override,
            trigger_on_tail_unreachable=beam_trigger_on_tail_unreachable,
            trigger_on_low_legal_actions=beam_trigger_on_low_legal_actions,
        )
        beam_planner = BeamSearchVetoPlanner(planner, beam_config)
    else:
        beam_planner = None

    scores = []
    survival_steps = []
    death_reasons = []
    episode_interventions = []
    episode_all_unsafe = []
    # Beam veto tracking
    episode_beam_calls = []
    episode_beam_vetoes = []
    episode_beam_all_forced = []
    episode_beam_expanded = []
    episode_beam_runtime_ms = []
    # Trigger reason counter
    trigger_reason_counts = collections.Counter()

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        interventions = 0
        all_unsafe_steps = 0
        beam_calls = 0
        beam_vetoes = 0
        beam_all_forced = 0
        beam_expanded = 0
        beam_runtime = 0.0

        while not done:
            with torch.no_grad():
                sv = torch.FloatTensor(encoder.encode(state)).unsqueeze(0)
                q_values = model(sv).squeeze(0).numpy()
                dqn_argmax = int(np.argmax(q_values))

            if beam_planner is not None:
                # SafetyPlanner baseline
                bfs_decision = planner.choose_action(env, q_values)
                # Beam veto layer
                beam_decision = beam_planner.choose_action(env, q_values, bfs_decision)
                action = beam_decision.chosen_action

                beam_calls += 1
                trigger_reason_counts[beam_decision.trigger_reason] += 1
                if beam_decision.planner_triggered:
                    beam_expanded += sum(
                        r.expanded_nodes for r in beam_decision.root_results.values()
                    )
                    beam_runtime += beam_decision.runtime_ms
                if beam_decision.veto_applied:
                    beam_vetoes += 1
                if beam_decision.planner_triggered and all(
                    r.likely_forced_death for r in beam_decision.root_results.values()
                ):
                    beam_all_forced += 1

                if action != dqn_argmax:
                    interventions += 1
                if all(info.immediate_death for info in bfs_decision.actions):
                    all_unsafe_steps += 1

            elif planner is not None:
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
        episode_beam_calls.append(beam_calls)
        episode_beam_vetoes.append(beam_vetoes)
        episode_beam_all_forced.append(beam_all_forced)
        episode_beam_expanded.append(beam_expanded)
        episode_beam_runtime_ms.append(beam_runtime)

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
    ge50_rate = float(np.mean(scores >= 50))
    ge60_rate = float(np.mean(scores >= 60))

    total_episodes = len(scores)
    body_death_rate = sum(1 for r in death_reasons if r == "body") / total_episodes
    wall_death_rate = sum(1 for r in death_reasons if r == "wall") / total_episodes
    timeout_rate = sum(1 for r in death_reasons if r == "timeout") / total_episodes

    mean_survival_steps = float(np.mean(survival_steps))
    mean_interventions = float(np.mean(episode_interventions))
    total_interventions = int(sum(episode_interventions))
    total_all_unsafe = int(sum(episode_all_unsafe))

    # Beam veto aggregates
    total_beam_calls = int(sum(episode_beam_calls))
    total_beam_vetoes = int(sum(episode_beam_vetoes))
    total_beam_all_forced = int(sum(episode_beam_all_forced))
    mean_beam_expanded = float(np.mean(episode_beam_expanded)) if episode_beam_expanded else 0.0
    mean_beam_runtime_ms = float(np.mean(episode_beam_runtime_ms)) if episode_beam_runtime_ms else 0.0
    beam_veto_rate = total_beam_vetoes / max(total_beam_calls, 1)

    print(f"\nEvaluation complete ({num_episodes} episodes):")
    print(f"  Mean score : {mean_score:.2f}")
    print(f"  Median     : {median_score:.1f}")
    print(f"  Std score  : {std_score:.2f}")
    print(f"  Min / Max  : {min_score} / {max_score}")
    print(f"  >=20       : {ge20_rate * 100:.1f}%")
    print(f"  >=30       : {ge30_rate * 100:.1f}%")
    print(f"  >=40       : {ge40_rate * 100:.1f}%")
    print(f"  >=50       : {ge50_rate * 100:.1f}%")
    print(f"  >=60       : {ge60_rate * 100:.1f}%")
    print(f"  Body death rate : {body_death_rate * 100:.1f}%")
    print(f"  Wall death rate : {wall_death_rate * 100:.1f}%")
    print(f"  Timeout rate    : {timeout_rate * 100:.1f}%")
    print(f"  Mean survival steps        : {mean_survival_steps:.1f}")
    print(f"  Planner intervention count : {total_interventions} (mean: {mean_interventions:.2f}/ep)")
    print(f"  All-actions-unsafe count   : {total_all_unsafe}")
    if use_beam_veto:
        total_triggered = sum(v for k, v in trigger_reason_counts.items() if k != "not_triggered")
        print(f"  Veto layer calls           : {total_beam_calls}")
        print(f"  Beam search triggered      : {total_triggered} ({total_triggered / max(total_beam_calls, 1) * 100:.1f}%)")
        print(f"  Beam veto count            : {total_beam_vetoes} (rate: {beam_veto_rate:.4f})")
        print(f"  All-root-forced-death count: {total_beam_all_forced}")
        print(f"  Mean expanded nodes/ep     : {mean_beam_expanded:.1f}")
        print(f"  Mean runtime ms/ep         : {mean_beam_runtime_ms:.2f}")
        print(f"  Trigger reasons:")
        for reason, count in trigger_reason_counts.most_common():
            pct = count / max(total_beam_calls, 1) * 100
            print(f"    {reason:25s}: {count:6d} ({pct:5.1f}%)")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save per-episode results
        ep_file = os.path.join(output_dir, "episode_results.csv")
        with open(ep_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["episode", "score", "survival_steps", "death_reason",
                       "interventions", "all_unsafe"]
            if use_beam_veto:
                header += ["beam_calls", "beam_vetoes", "beam_all_forced",
                           "beam_expanded_nodes", "beam_runtime_ms"]
            writer.writerow(header)
            for idx in range(total_episodes):
                row = [
                    idx,
                    scores[idx],
                    survival_steps[idx],
                    death_reasons[idx],
                    episode_interventions[idx],
                    episode_all_unsafe[idx],
                ]
                if use_beam_veto:
                    row += [
                        episode_beam_calls[idx],
                        episode_beam_vetoes[idx],
                        episode_beam_all_forced[idx],
                        episode_beam_expanded[idx],
                        episode_beam_runtime_ms[idx],
                    ]
                writer.writerow(row)
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
            writer.writerow([">=50", ge50_rate])
            writer.writerow([">=60", ge60_rate])
            writer.writerow(["body death rate", body_death_rate])
            writer.writerow(["wall death rate", wall_death_rate])
            writer.writerow(["timeout rate", timeout_rate])
            writer.writerow(["mean survival steps", mean_survival_steps])
            writer.writerow(["planner intervention count", total_interventions])
            writer.writerow(["all-actions-unsafe count", total_all_unsafe])
            if use_beam_veto:
                writer.writerow(["beam search calls", total_beam_calls])
                writer.writerow(["beam veto count", total_beam_vetoes])
                writer.writerow(["beam veto rate", beam_veto_rate])
                writer.writerow(["all-root-forced-death count", total_beam_all_forced])
                writer.writerow(["mean expanded nodes per call", mean_beam_expanded])
                writer.writerow(["mean runtime ms per call", mean_beam_runtime_ms])
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

    # Beam veto CLI args
    p.add_argument("--use-beam-veto", action="store_true", default=False)
    p.add_argument("--beam-max-depth", type=int, default=6)
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--beam-always-on", action="store_true", default=False)
    p.add_argument("--beam-area-margin", type=int, default=10)
    p.add_argument("--beam-disable-trigger-on-bfs-override", action="store_true", default=False)
    p.add_argument("--beam-disable-trigger-on-tail-unreachable", action="store_true", default=False)
    p.add_argument("--beam-disable-trigger-on-low-legal-actions", action="store_true", default=False)

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
        use_beam_veto=args.use_beam_veto,
        beam_max_depth=args.beam_max_depth,
        beam_width=args.beam_width,
        beam_always_on=args.beam_always_on,
        beam_area_margin=args.beam_area_margin,
        beam_trigger_on_bfs_override=not args.beam_disable_trigger_on_bfs_override,
        beam_trigger_on_tail_unreachable=not args.beam_disable_trigger_on_tail_unreachable,
        beam_trigger_on_low_legal_actions=not args.beam_disable_trigger_on_low_legal_actions,
        output_dir=args.output_dir,
        representation_override=args.representation,
        local_window_size_override=args.local_window_size,
    )


if __name__ == "__main__":
    main()
