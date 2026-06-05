"""Counterfactual Markov analysis using a shared canonical corpus.

For each encoder, canonical states are grouped into alias buckets
(states that encode to the same hash).  Inside each aliased bucket,
all 3 actions are simulated for every canonical state.  If the same
encoded state + same action leads to different outcomes, that is a
counterfactual Markov violation.

Transitions where the snake ate fruit are excluded (fruit respawn is
random and not encoded).

Usage::

    python analyze_markov_counterfactual.py --corpus analysis_corpus.pkl --board-size 15 --max-length 100
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


def analyze_counterfactual(
    states: list[SnakeState],
    encoder: StateEncoder,
    env: SnakeEnv,
) -> dict:
    """Compute counterfactual Markov violation metrics."""
    # Step 1: Group states by encoded hash
    hash_to_canonicals: dict[bytes, list[SnakeState]] = defaultdict(list)
    for s in states:
        h = _encoded_hash(encoder.encode(s))
        hash_to_canonicals[h].append(s)

    # Step 2: For each aliased bucket, simulate all actions
    aliased_buckets = {h: cs for h, cs in hash_to_canonicals.items() if len(cs) >= 2}
    total_aliased_buckets = len(aliased_buckets)
    total_states_in_aliased = sum(len(cs) for cs in aliased_buckets.values())

    # (encoded_hash, action) -> set of (reward, terminated, next_hash)
    sa_outcomes: dict[tuple[bytes, int], set[tuple[float, bool, bytes]]] = defaultdict(set)
    total_simulated = 0
    skipped_ate_fruit = 0

    for h, canonicals in aliased_buckets.items():
        for s in canonicals:
            for action in range(3):
                ns, reward, terminated, info = env.simulate_transition(s, action)
                if info.get("ate_fruit", False):
                    skipped_ate_fruit += 1
                    continue
                total_simulated += 1
                h_ns = _encoded_hash(encoder.encode(ns))
                sa_outcomes[(h, action)].add(
                    (round(reward, 4), terminated, h_ns)
                )

    # Step 3: Count violations
    unique_pairs = len(sa_outcomes)
    conflicting_pairs = sum(1 for outcomes in sa_outcomes.values() if len(outcomes) >= 2)

    outcomes_per_pair = [len(v) for v in sa_outcomes.values()]
    mean_outcomes = float(np.mean(outcomes_per_pair)) if outcomes_per_pair else 0.0
    max_outcomes = max(outcomes_per_pair, default=0)

    return {
        "aliased_buckets": total_aliased_buckets,
        "states_in_aliased_buckets": total_states_in_aliased,
        "total_simulated_transitions": total_simulated,
        "skipped_ate_fruit": skipped_ate_fruit,
        "unique_state_action_pairs": unique_pairs,
        "conflicting_pairs": conflicting_pairs,
        "violation_rate": round(conflicting_pairs / unique_pairs, 6) if unique_pairs > 0 else 0.0,
        "mean_outcomes_per_pair": round(mean_outcomes, 2),
        "max_outcomes_per_pair": max_outcomes,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Counterfactual Markov analysis.")
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
        metrics = analyze_counterfactual(states, enc, env)
        results[name] = metrics
        print(
            f"aliased_buckets={metrics['aliased_buckets']}  "
            f"conflicting={metrics['conflicting_pairs']}/{metrics['unique_state_action_pairs']}  "
            f"violation_rate={metrics['violation_rate']}"
        )

    out_path = os.path.join(args.output_dir, "counterfactual_markov_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
