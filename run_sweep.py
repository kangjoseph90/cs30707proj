"""Run a sweep of all representation × seed combinations.

Usage::

    python run_sweep.py --total-env-steps 100000 --seeds 0 1 2 3 4 --output-dir results/sweep_100k
    python run_sweep.py --total-env-steps 5000 --seeds 0 --output-dir results/pilot_5k
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd


REPRESENTATIONS = [
    ("full", None),
    ("structural", None),
    ("local", 9),
    ("local", 7),
    ("local", 5),
]

PHASE4_REPRESENTATIONS = [
    ("dense_ordered_global", None),
    ("dense_ordered_local", 9),
    ("dense_ordered_local", 7),
    ("dense_ordered_local", 5),
    ("occupancy_global", None),
    ("occupancy_local", 9),
    ("occupancy_local", 7),
    ("occupancy_local", 5),
]

SWEEP_500K_REPRESENTATIONS = [
    ("structural", None),
    ("dense_ordered_global", None),
    ("occupancy_global", None),
    ("dense_ordered_local", 5),
    ("occupancy_local", 5),
    ("occupancy_local", 7),
    ("occupancy_local", 9),
]

PHASE5_REPRESENTATIONS = [
    ("merged_obstacle_local", 5),
    ("merged_obstacle_local", 9),
    ("egocentric_merged_obstacle_local", 5),
    ("egocentric_merged_obstacle_local", 9),
]

FINAL_REPRESENTATIONS = [
    ("merged_obstacle_local", 5),
    ("merged_obstacle_local", 9),
    ("merged_obstacle_local", 29),
    ("merged_ordered_local", 5),
    ("merged_ordered_local", 9),
    ("merged_ordered_local", 29),
    ("egocentric_merged_obstacle_local", 5),
    ("egocentric_merged_obstacle_local", 9),
    ("egocentric_merged_obstacle_local", 29),
    ("egocentric_merged_ordered_local", 5),
    ("egocentric_merged_ordered_local", 9),
    ("egocentric_merged_ordered_local", 29),
    ("blind_sniff", None),
]


def generate_summary(sweep_dir: str) -> str:
    """Read metrics.json + config.json from each subdirectory, produce summary.csv."""
    rows = []
    for name in sorted(os.listdir(sweep_dir)):
        run_dir = os.path.join(sweep_dir, name)
        metrics_path = os.path.join(run_dir, "metrics.json")
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        with open(config_path) as f:
            c = json.load(f)

        rows.append({
            "representation": c.get("representation", ""),
            "local_window_size": c.get("local_window_size"),
            "seed": c.get("seed"),
            "input_dimension": m.get("input_dimension"),
            "compression_ratio_vs_full": m.get("compression_ratio_vs_full"),
            "input_sparsity_mean": m.get("input_sparsity_mean"),
            "input_sparsity_std": m.get("input_sparsity_std"),
            "parameter_count": m.get("parameter_count"),
            "training_time_seconds": m.get("training_time_seconds"),
            "best_eval_score_mean": m.get("best_eval_score_mean"),
            "final_eval_score_mean": m.get("final_eval_score_mean"),
            "final_eval_score_std": m.get("final_eval_score_std"),
            "best_score": m.get("best_score"),
            "learning_curve_auc": m.get("learning_curve_auc"),
            "total_env_steps": m.get("total_env_steps"),
            "total_episodes": m.get("total_episodes"),
        })

    if not rows:
        print("No completed runs found.")
        return ""

    df = pd.DataFrame(rows)
    summary_path = os.path.join(sweep_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(df.to_string(index=False))
    return summary_path


def main() -> None:
    p = argparse.ArgumentParser(description="Run representation comparison sweep.")
    p.add_argument("--total-env-steps", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--board-size", type=int, default=15)
    p.add_argument("--max-length", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=256)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--output-dir", type=str, default="results/sweep")
    p.add_argument("--force", action="store_true", help="Re-run existing results.")
    p.add_argument(
        "--suite",
        choices=["default", "phase4_spatial", "sweep_500k", "phase5_merged_egocentric", "final"],
        default="default",
        help="Which representation suite to run.",
    )
    p.add_argument(
        "--cache-encoded-replay", action="store_true",
        help="Use EncodedReplayBuffer to cache encoded states.",
    )
    args = p.parse_args()

    if args.eval_interval is None:
        args.eval_interval = args.total_env_steps // 10

    os.makedirs(args.output_dir, exist_ok=True)

    _SUITE_MAP = {
        "default": REPRESENTATIONS,
        "phase4_spatial": PHASE4_REPRESENTATIONS,
        "sweep_500k": SWEEP_500K_REPRESENTATIONS,
        "phase5_merged_egocentric": PHASE5_REPRESENTATIONS,
        "final": FINAL_REPRESENTATIONS,
    }
    reps = _SUITE_MAP[args.suite]

    # 500k-specific defaults (sweep_500k, phase5, and final)
    if args.suite in ("sweep_500k", "phase5_merged_egocentric", "final"):
        if args.warmup_steps == 256:
            args.warmup_steps = 10000
        if args.eval_episodes == 5:
            args.eval_episodes = 30
        if args.eval_interval is None:
            args.eval_interval = 25_000

    total_runs = len(reps) * len(args.seeds)
    completed = 0

    for rep, ws in reps:
        for seed in args.seeds:
            ws_suffix = f"_w{ws}" if ws else ""
            run_name = f"{rep}{ws_suffix}_seed{seed}"
            run_dir = os.path.join(args.output_dir, run_name)

            if not args.force and os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
                print(f"[SKIP] {run_name} (already exists)")
                completed += 1
                continue

            cmd = [
                sys.executable, "train_snake.py",
                "--representation", rep,
                "--total-env-steps", str(args.total_env_steps),
                "--seed", str(seed),
                "--board-size", str(args.board_size),
                "--max-length", str(args.max_length),
                "--max-steps", str(args.max_steps),
                "--warmup-steps", str(args.warmup_steps),
                "--eval-interval", str(args.eval_interval),
                "--eval-episodes", str(args.eval_episodes),
                "--output-dir", run_dir,
            ]
            if ws is not None:
                cmd.extend(["--local-window-size", str(ws)])
            if args.cache_encoded_replay:
                cmd.append("--cache-encoded-replay")

            # 500k-specific schedule
            if args.suite in ("sweep_500k", "phase5_merged_egocentric", "final"):
                cmd.extend([
                    "--eps-end", "0.05",
                    "--eps-decay", "80000",
                    "--checkpoint-steps", "100000", "300000", "500000",
                ])

            completed += 1
            print(f"\n[{completed}/{total_runs}] {run_name}")
            subprocess.run(cmd, check=True)

    generate_summary(args.output_dir)


if __name__ == "__main__":
    main()
