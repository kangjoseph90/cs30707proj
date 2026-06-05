"""Measure empirical Markov property violations across representations.

Uses a shared canonical corpus.  For each state in the corpus, all 3
actions are simulated via ``env.simulate_transition()``.  Transitions
where the snake ate fruit or the episode was truncated are excluded.

Key = (encoded_state_hash, action), Value = (reward, terminated, encoded_next_hash).
A "conflict" occurs when the same key maps to ≥2 distinct outcomes.

The violation rate denominator is **repeated** state-action pairs only;
singleton pairs are reported separately.

Usage::

    python analyze_markovianity.py --corpus analysis_corpus.pkl --board-size 15 --max-length 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from snake_env import SnakeEnv, SnakeState
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


def _encoded_hash(arr: np.ndarray) -> bytes:
    return hashlib.blake2b(arr.astype(np.float32).tobytes(), digest_size=16).digest()


def analyze_markov(
    states: list[SnakeState],
    encoder: StateEncoder,
    env: SnakeEnv,
) -> dict:
    """Compute Markov violation metrics for a single encoder."""
    # Build mapping: (encoded_hash, action) -> set of (reward, terminated, next_hash)
    sa_outcomes: dict[tuple[bytes, int], set[tuple[float, bool, bytes]]] = defaultdict(set)
    sa_obs_count: dict[tuple[bytes, int], int] = defaultdict(int)
    total_transitions = 0
    skipped_truncated = 0
    skipped_ate_fruit = 0

    for s in states:
        encoded_s = encoder.encode(s)
        h_s = _encoded_hash(encoded_s)

        for action in range(3):
            ns, reward, terminated, info = env.simulate_transition(s, action)
            ate_fruit = info.get("ate_fruit", False)

            # Skip non-deterministic transitions
            if ate_fruit:
                skipped_ate_fruit += 1
                continue
            if not terminated:
                pass

            total_transitions += 1
            encoded_ns = encoder.encode(ns)
            h_ns = _encoded_hash(encoded_ns)

            sa_outcomes[(h_s, action)].add(
                (round(reward, 4), terminated, h_ns)
            )
            sa_obs_count[(h_s, action)] += 1

    # Analyse
    unique_pairs = len(sa_outcomes)
    repeated_pairs = sum(1 for c in sa_obs_count.values() if c >= 2)
    repeat_coverage = repeated_pairs / unique_pairs if unique_pairs > 0 else 0.0

    conflicting_pairs = sum(
        1 for outcomes in sa_outcomes.values()
        if len(outcomes) >= 2
    )

    if repeated_pairs > 0:
        violation_rate = conflicting_pairs / repeated_pairs
    else:
        violation_rate = float("nan")

    outcomes_per_repeated = [
        len(sa_outcomes[k]) for k, c in sa_obs_count.items() if c >= 2
    ]
    mean_outcomes = float(np.mean(outcomes_per_repeated)) if outcomes_per_repeated else 0.0
    max_outcomes = max(outcomes_per_repeated, default=0)

    return {
        "analyzed_transition_count": total_transitions,
        "skipped_ate_fruit": skipped_ate_fruit,
        "unique_state_action_pairs": unique_pairs,
        "repeated_state_action_pairs": repeated_pairs,
        "repeat_coverage": round(repeat_coverage, 4),
        "conflicting_repeated_pairs": conflicting_pairs,
        "markov_violation_rate_among_repeated": (
            round(violation_rate, 6) if not np.isnan(violation_rate) else None
        ),
        "mean_outcomes_per_repeated_pair": round(mean_outcomes, 2),
        "max_outcomes_per_pair": max_outcomes,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze empirical Markov violations.")
    p.add_argument("--corpus", required=True)
    p.add_argument("--board-size", type=int, default=15)
    p.add_argument("--max-length", type=int, default=100)
    p.add_argument("--output-dir", type=str, default=".")
    args = p.parse_args()

    with open(args.corpus, "rb") as f:
        states: list[SnakeState] = pickle.load(f)

    print(f"Loaded {len(states)} canonical states")

    env = SnakeEnv(board_size=args.board_size, max_length=args.max_length, headless=True, seed=0)

    encoders = {
        "full": FullStateEncoder(args.board_size, args.max_length),
        "structural": StructuralStateEncoder(args.board_size, args.max_length),
        "local_sparse_ordered_9": LocalStateEncoder(args.board_size, args.max_length, 9),
        "local_sparse_ordered_7": LocalStateEncoder(args.board_size, args.max_length, 7),
        "local_sparse_ordered_5": LocalStateEncoder(args.board_size, args.max_length, 5),
        "dense_ordered_global": DenseOrderedGlobalEncoder(args.board_size, args.max_length),
        "dense_ordered_local_9": DenseOrderedLocalEncoder(args.board_size, args.max_length, 9),
        "dense_ordered_local_7": DenseOrderedLocalEncoder(args.board_size, args.max_length, 7),
        "dense_ordered_local_5": DenseOrderedLocalEncoder(args.board_size, args.max_length, 5),
        "occupancy_global": OccupancyGlobalEncoder(args.board_size, args.max_length),
        "occupancy_local_9": OccupancyLocalEncoder(args.board_size, args.max_length, 9),
        "occupancy_local_7": OccupancyLocalEncoder(args.board_size, args.max_length, 7),
        "occupancy_local_5": OccupancyLocalEncoder(args.board_size, args.max_length, 5),
        "merged_obstacle_local_5": MergedObstacleLocalEncoder(args.board_size, args.max_length, 5),
        "merged_obstacle_local_9": MergedObstacleLocalEncoder(args.board_size, args.max_length, 9),
        "merged_obstacle_local_29": MergedObstacleLocalEncoder(args.board_size, args.max_length, 29),
        "egocentric_merged_obstacle_local_5": EgocentricMergedObstacleLocalEncoder(args.board_size, args.max_length, 5),
        "egocentric_merged_obstacle_local_9": EgocentricMergedObstacleLocalEncoder(args.board_size, args.max_length, 9),
        "egocentric_merged_obstacle_local_29": EgocentricMergedObstacleLocalEncoder(args.board_size, args.max_length, 29),
        "merged_ordered_local_5": MergedOrderedLocalEncoder(args.board_size, args.max_length, 5),
        "merged_ordered_local_9": MergedOrderedLocalEncoder(args.board_size, args.max_length, 9),
        "merged_ordered_local_29": MergedOrderedLocalEncoder(args.board_size, args.max_length, 29),
        "egocentric_merged_ordered_local_5": EgocentricMergedOrderedLocalEncoder(args.board_size, args.max_length, 5),
        "egocentric_merged_ordered_local_9": EgocentricMergedOrderedLocalEncoder(args.board_size, args.max_length, 9),
        "egocentric_merged_ordered_local_29": EgocentricMergedOrderedLocalEncoder(args.board_size, args.max_length, 29),
        "blind_sniff": BlindSniffEncoder(args.board_size, args.max_length),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    for name, enc in encoders.items():
        print(f"  Analyzing {name}...", end=" ", flush=True)
        metrics = analyze_markov(states, enc, env)
        results[name] = metrics
        vr = metrics["markov_violation_rate_among_repeated"]
        print(
            f"unique_pairs={metrics['unique_state_action_pairs']}  "
            f"repeated={metrics['repeated_state_action_pairs']}  "
            f"conflicting={metrics['conflicting_repeated_pairs']}  "
            f"violation_rate={vr}"
        )

    out_path = os.path.join(args.output_dir, "markov_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
