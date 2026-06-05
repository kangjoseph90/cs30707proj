"""Measure state aliasing across representations using a shared canonical corpus.

For each encoder, every canonical state is encoded and hashed.  If two
different canonical states produce the same hash the representation has
*aliased* them.  Full and Structural encoders should show 0 aliasing.

Usage::

    python analyze_aliasing.py --corpus analysis_corpus.pkl --board-size 15 --max-length 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import defaultdict
from typing import Optional  # noqa: F401

import numpy as np

from snake_env import SnakeState
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
    """Stable hash of a float32 encoded vector (blake2b, 16-byte digest)."""
    return hashlib.blake2b(arr.astype(np.float32).tobytes(), digest_size=16).digest()


def analyze_representation(
    states: list[SnakeState],
    encoder: StateEncoder,
) -> dict:
    """Compute aliasing metrics for a single encoder."""
    hash_to_canonicals: dict[bytes, set[SnakeState]] = defaultdict(set)

    for s in states:
        encoded = encoder.encode(s)
        h = _encoded_hash(encoded)
        hash_to_canonicals[h].add(s)

    unique_canonical = len(states)
    unique_encoded = len(hash_to_canonicals)

    # Buckets with ≥2 distinct canonical states
    aliased_buckets = [
        bucket for bucket in hash_to_canonicals.values() if len(bucket) >= 2
    ]
    aliased_encoded = len(aliased_buckets)
    aliasing_rate = aliased_encoded / unique_encoded if unique_encoded > 0 else 0.0

    bucket_sizes = [len(b) for b in aliased_buckets]
    mean_alias_bucket_size = float(np.mean(bucket_sizes)) if bucket_sizes else 0.0
    max_alias_bucket_size = max(bucket_sizes, default=0)

    return {
        "unique_canonical_states": unique_canonical,
        "unique_encoded_states": unique_encoded,
        "aliased_encoded_states": aliased_encoded,
        "aliasing_rate": round(aliasing_rate, 6),
        "mean_alias_bucket_size": round(mean_alias_bucket_size, 2),
        "max_alias_bucket_size": max_alias_bucket_size,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze state aliasing.")
    p.add_argument("--corpus", required=True, help="Path to analysis_corpus.pkl")
    p.add_argument("--board-size", type=int, default=15)
    p.add_argument("--max-length", type=int, default=100)
    p.add_argument("--output-dir", type=str, default=".")
    args = p.parse_args()

    with open(args.corpus, "rb") as f:
        states: list[SnakeState] = pickle.load(f)

    print(f"Loaded {len(states)} canonical states")

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
        print(f"  Analyzing {name} (dim={enc.output_dim})...", end=" ", flush=True)
        metrics = analyze_representation(states, enc)
        results[name] = metrics
        print(
            f"aliasing_rate={metrics['aliasing_rate']}  "
            f"aliased={metrics['aliased_encoded_states']}/{metrics['unique_encoded_states']}"
        )

        # Warn if lossless encoder shows aliasing
        if name in ("full", "structural", "dense_ordered_global") and metrics["aliasing_rate"] > 0:
            print(f"  WARNING: {name} encoder has non-zero aliasing! Encoder bug?")

    out_path = os.path.join(args.output_dir, "aliasing_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
