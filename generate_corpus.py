"""Generate a shared canonical-state corpus for representation analysis.

Collects states from multiple sources with different exploration strategies:
  1. Random-policy rollouts (covers short snakes broadly)
  2. ε-greedy rollouts from trained checkpoints (covers longer snakes)
  3. Heuristic-policy rollouts across multiple seeds (covers diverse long-snake states)

States are deduplicated and then **stratified-sampled** by body-length bin so
every length range gets equal representation.  Source provenance is tracked
throughout.

Usage::

    python generate_corpus.py --num-states 10000 --output results/analysis/analysis_corpus.pkl

    # With explicit checkpoints:
    python generate_corpus.py --num-states 10000 \\
        --checkpoints results/sweep_100k/local_w5_seed0/checkpoint.pt \\
        --output results/analysis/analysis_corpus.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch

from model import MLPDQN
from snake_env import SnakeEnv, SnakeState
from train_snake import make_encoder


# ---------------------------------------------------------------------------
# Body-length bins
# ---------------------------------------------------------------------------

LENGTH_BINS = [
    (2, 5, "2-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 35, "21-35"),
    (36, 55, "36-55"),
    (56, 100, "56-100"),
]


def _bin_key(length: int) -> str:
    for lo, hi, label in LENGTH_BINS:
        if lo <= length <= hi:
            return label
    return "2-5"  # fallback


# ---------------------------------------------------------------------------
# Source 1: Random policy
# ---------------------------------------------------------------------------

def collect_random(
    num_episodes: int,
    board_size: int = 15,
    max_length: int = 100,
    max_steps: int = 2000,
    seed: int = 0,
) -> set[SnakeState]:
    """Collect states from random-policy rollouts."""
    env = SnakeEnv(board_size=board_size, max_length=max_length,
                   max_steps=max_steps, headless=True, seed=seed)
    rng = random.Random(seed)
    seen: set[SnakeState] = set()

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            seen.add(state)
            action = rng.randint(0, 2)
            state, _, term, trunc, _ = env.step(action)
            done = term or trunc

    return seen


# ---------------------------------------------------------------------------
# Source 2: ε-greedy rollout from a checkpoint
# ---------------------------------------------------------------------------

def collect_checkpoint(
    checkpoint_path: str,
    num_episodes: int = 100,
    epsilon: float = 0.1,
    seed_start: int = 2000,
    board_size: int = 15,
    max_length: int = 100,
    max_steps: int = 2000,
) -> set[SnakeState]:
    """Collect states from ε-greedy rollouts using a trained checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    enc = make_encoder(
        config["representation"],
        config["board_size"],
        config["max_length"],
        config.get("local_window_size"),
    )
    model = MLPDQN(input_dim=enc.output_dim, output_dim=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rng = random.Random(seed_start)
    seen: set[SnakeState] = set()

    for i in range(num_episodes):
        eval_env = SnakeEnv(
            board_size=board_size, max_length=max_length,
            max_steps=max_steps, headless=True, seed=seed_start + i,
        )
        state = eval_env.reset()
        done = False
        while not done:
            seen.add(state)
            if rng.random() < epsilon:
                action = rng.randint(0, 2)
            else:
                with torch.no_grad():
                    sv = torch.FloatTensor(enc.encode(state)).unsqueeze(0)
                    action = torch.argmax(model(sv)).item()
            state, _, term, trunc, _ = eval_env.step(action)
            done = term or trunc

    return seen


# ---------------------------------------------------------------------------
# Source 3: Heuristic policy (avoid death + approach fruit)
# ---------------------------------------------------------------------------

def collect_heuristic(
    num_episodes_per_seed: int,
    seeds: list[int],
    board_size: int = 15,
    max_length: int = 100,
    max_steps: int = 2000,
) -> set[SnakeState]:
    """Collect states using a heuristic policy across multiple seeds.

    Policy:
    1. For each of the 3 relative actions, check if it causes immediate death.
    2. Among surviving actions, pick the one that minimises Manhattan distance to fruit.
    3. If all actions cause death, pick a random one.
    """
    seen: set[SnakeState] = set()

    for seed in seeds:
        rng = random.Random(seed)
        env = SnakeEnv(board_size=board_size, max_length=max_length,
                       max_steps=max_steps, headless=True, seed=seed)

        for _ in range(num_episodes_per_seed):
            state = env.reset()
            done = False
            while not done:
                seen.add(state)
                action = _heuristic_action(env, state, rng)
                state, _, term, trunc, _ = env.step(action)
                done = term or trunc

    return seen


def _heuristic_action(
    env: SnakeEnv, state: SnakeState, rng: random.Random,
) -> int:
    """Pick action avoiding immediate death and approaching fruit."""
    from snake_env import DELTA, get_relative_dirs, manhattan

    bs = env.board_size
    hx, hy = state.head
    fx, fy = state.fruit
    body_set = set(state.body)

    abs_dirs = get_relative_dirs(state.direction)
    best_action = 0
    best_dist = float("inf")
    any_surviving = False

    # Shuffle to break ties randomly
    actions = list(range(3))
    rng.shuffle(actions)

    for action in actions:
        dx, dy = DELTA[abs_dirs[action]]
        nx, ny = hx + dx, hy + dy

        # Check wall
        if nx < 0 or nx >= bs or ny < 0 or ny >= bs:
            continue

        # Check self-collision
        if (nx, ny) in body_set:
            # Allow if it's the tail and we're not eating (tail will move)
            if (nx, ny) == state.body[-1] and (nx, ny) != (fx, fy):
                pass  # tail will vacate
            else:
                continue

        any_surviving = True
        dist = manhattan((nx, ny), (fx, fy))
        if dist < best_dist:
            best_dist = dist
            best_action = action

    if not any_surviving:
        return rng.randint(0, 2)

    return best_action


# ---------------------------------------------------------------------------
# Stratified corpus sampling
# ---------------------------------------------------------------------------

def _stratified_sample(
    state_sources: dict[SnakeState, set[str]],
    num_states: int,
    rng: random.Random,
) -> list[SnakeState]:
    """Stratified sampling by body-length bin with equal quota.

    1. Divide *num_states* equally across bins.
    2. If a bin has fewer states than its quota, take all and redistribute
       the surplus equally among bins that still have capacity.
    3. Within each bin, balance across sources as fairly as possible.
    """
    num_bins = len(LENGTH_BINS)
    base_quota = num_states // num_bins
    remainder = num_states % num_bins

    # Assign initial quotas
    bin_labels = [label for _, _, label in LENGTH_BINS]
    quotas: dict[str, int] = {}
    for i, label in enumerate(bin_labels):
        quotas[label] = base_quota + (1 if i < remainder else 0)

    # Group states by bin
    bin_states: dict[str, list[SnakeState]] = defaultdict(list)
    for state, sources in state_sources.items():
        bk = _bin_key(len(state.body))
        bin_states[bk].append((state, sources))

    # Shuffle within each bin for reproducibility
    for bk in bin_states:
        rng.shuffle(bin_states[bk])

    # Phase 1: fill quotas, collect surplus
    selected: list[SnakeState] = []
    surplus = 0
    underfilled: list[str] = []

    for label in bin_labels:
        available = bin_states.get(label, [])
        quota = quotas[label]
        if len(available) >= quota:
            taken = _balance_by_sources(available[:quota], quota)
            selected.extend(taken)
        else:
            taken = _balance_by_sources(available, len(available))
            selected.extend(taken)
            surplus += quota - len(available)
            underfilled.append(label)

    # Phase 2: redistribute surplus to bins with capacity
    if surplus > 0:
        bins_with_extra = []
        for label in bin_labels:
            available = bin_states.get(label, [])
            already_taken = min(quotas[label], len(available))
            remaining = len(available) - already_taken
            if remaining > 0:
                bins_with_extra.append((label, already_taken, remaining))

        # Sort by most remaining first
        bins_with_extra.sort(key=lambda x: x[2], reverse=True)

        while surplus > 0 and bins_with_extra:
            # Distribute evenly across bins with remaining states
            per_bin = max(1, surplus // len(bins_with_extra))
            new_extra = []
            for label, start, remaining in bins_with_extra:
                take = min(per_bin, remaining, surplus)
                if take > 0:
                    available = bin_states[label]
                    batch = available[start : start + take]
                    taken = _balance_by_sources(batch, take)
                    selected.extend(taken)
                    surplus -= take
                    new_start = start + take
                    new_remaining = remaining - take
                    if new_remaining > 0:
                        new_extra.append((label, new_start, new_remaining))
            bins_with_extra = new_extra
            per_bin = max(1, surplus // max(len(bins_with_extra), 1))

    return selected


def _balance_by_sources(
    candidates: list[tuple[SnakeState, set[str]]],
    quota: int,
) -> list[SnakeState]:
    """Select *quota* states from *candidates*, balancing across sources.

    Uses round-robin across source types (random, checkpoint, heuristic).
    States that belong to multiple sources are assigned to whichever source
    still needs filling during the round-robin.
    """
    SOURCE_ORDER = ["random", "checkpoint", "heuristic"]

    # Track which states can fill which source quota
    by_source: dict[str, list[SnakeState]] = defaultdict(list)
    for state, sources in candidates:
        for src in sources:
            by_source[src].append(state)

    # Remove duplicates within each source list (same state from multiple candidates)
    for src in by_source:
        seen = set()
        unique = []
        for s in by_source[src]:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        by_source[src] = unique

    result: list[SnakeState] = []
    used: set[SnakeState] = set()
    source_counts: dict[str, int] = {src: 0 for src in SOURCE_ORDER}

    # Round-robin across sources
    src_idx = 0
    while len(result) < quota:
        src = SOURCE_ORDER[src_idx % len(SOURCE_ORDER)]
        src_idx += 1

        # Find next unused state from this source
        for s in by_source.get(src, []):
            if s not in used:
                used.add(s)
                result.append(s)
                source_counts[src] += 1
                break
        else:
            # This source exhausted; try remaining sources
            filled = False
            for alt_src in SOURCE_ORDER:
                if alt_src == src:
                    continue
                for s in by_source.get(alt_src, []):
                    if s not in used:
                        used.add(s)
                        result.append(s)
                        source_counts[alt_src] += 1
                        filled = True
                        break
                if filled:
                    break
            if not filled:
                break  # all sources exhausted

    return result[:quota]


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------

def generate_corpus(
    num_states: int = 10_000,
    board_size: int = 15,
    max_length: int = 100,
    max_steps: int = 2000,
    checkpoint_paths: Optional[list[str]] = None,
    checkpoint_epsilon: float = 0.1,
    checkpoint_episodes: int = 200,
    heuristic_episodes_per_seed: int = 500,
    heuristic_seeds: Optional[list[int]] = None,
    random_episodes: int = 2000,
    seed: int = 999,
) -> tuple[list[SnakeState], dict]:
    """Generate a stratified mixed-source corpus of unique canonical states.

    Returns (corpus, metadata).
    """
    rng = random.Random(seed)

    # Default heuristic seeds
    if heuristic_seeds is None:
        heuristic_seeds = list(range(16000, 16010))

    # --- Collect from all sources, track provenance ---
    state_sources: dict[SnakeState, set[str]] = defaultdict(set)

    # Source 1: Random
    print(f"[1/3] Random rollouts ({random_episodes} episodes)...", flush=True)
    random_states = collect_random(
        num_episodes=random_episodes,
        board_size=board_size, max_length=max_length,
        max_steps=max_steps, seed=seed,
    )
    for s in random_states:
        state_sources[s].add("random")
    _print_length_stats("  Random", random_states)

    # Source 2: Checkpoint ε-greedy
    if checkpoint_paths:
        for ckpt_path in checkpoint_paths:
            name = os.path.basename(os.path.dirname(ckpt_path))
            print(f"[2/3] Checkpoint '{name}' ε={checkpoint_epsilon} "
                  f"({checkpoint_episodes} episodes)...", flush=True)
            ckpt_states = collect_checkpoint(
                checkpoint_path=ckpt_path,
                num_episodes=checkpoint_episodes,
                epsilon=checkpoint_epsilon,
                seed_start=seed,
                board_size=board_size, max_length=max_length,
                max_steps=max_steps,
            )
            for s in ckpt_states:
                state_sources[s].add("checkpoint")
            _print_length_stats(f"  {name}", ckpt_states)
    else:
        print("[2/3] No checkpoints provided, skipping.")

    # Source 3: Heuristic (multi-seed)
    total_heuristic_eps = heuristic_episodes_per_seed * len(heuristic_seeds)
    print(f"[3/3] Heuristic rollouts ({heuristic_episodes_per_seed} episodes × "
          f"{len(heuristic_seeds)} seeds = {total_heuristic_eps} total)...", flush=True)
    heuristic_states = collect_heuristic(
        num_episodes_per_seed=heuristic_episodes_per_seed,
        seeds=heuristic_seeds,
        board_size=board_size, max_length=max_length,
        max_steps=max_steps,
    )
    for s in heuristic_states:
        state_sources[s].add("heuristic")
    _print_length_stats("  Heuristic", heuristic_states)

    # --- Report total ---
    total_unique = len(state_sources)
    print(f"\nTotal unique states collected: {total_unique}")

    # --- Stratified sampling ---
    print(f"\nStratified sampling {num_states} states by body-length bin...")
    corpus = _stratified_sample(state_sources, num_states, rng)

    # --- Build metadata ---
    bin_counts: dict[str, int] = Counter()
    source_counts: dict[str, int] = Counter()
    for s in corpus:
        bin_counts[_bin_key(len(s.body))] += 1
        for src in state_sources[s]:
            source_counts[src] += 1

    lengths = [len(s.body) for s in corpus]
    metadata = {
        "total_collected": total_unique,
        "corpus_size": len(corpus),
        "seed": seed,
        "body_length_bins": {label: bin_counts.get(label, 0)
                             for _, _, label in LENGTH_BINS},
        "source_counts": dict(source_counts),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": round(float(np.mean(lengths)), 1),
        "heuristic_seeds_used": heuristic_seeds,
        "checkpoint_count": len(checkpoint_paths) if checkpoint_paths else 0,
    }

    # Print summary
    print(f"\nCorpus: {len(corpus)} states")
    print(f"  Body length: {metadata['length_min']}-{metadata['length_max']} "
          f"(mean {metadata['length_mean']})")
    for _, _, label in LENGTH_BINS:
        print(f"  {label}: {metadata['body_length_bins'][label]}")
    print(f"  Sources: {dict(source_counts)}")

    return corpus, metadata


def _print_length_stats(label: str, states: set[SnakeState] | list[SnakeState]) -> None:
    lengths = [len(s.body) for s in states]
    if not lengths:
        print(f"{label}: 0 states")
        return
    dist = Counter()
    for l in lengths:
        dist[_bin_key(l)] += 1
    bins_str = "  ".join(f"{lb}:{dist.get(lb, 0)}" for _, _, lb in LENGTH_BINS)
    print(f"{label}: {len(lengths)} states, "
          f"length {min(lengths)}-{max(lengths)} (mean {np.mean(lengths):.1f})  "
          f"[{bins_str}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate shared canonical-state corpus from mixed sources "
                    "with stratified body-length sampling."
    )
    p.add_argument("--num-states", type=int, default=10_000)
    p.add_argument("--board-size", type=int, default=15)
    p.add_argument("--max-length", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--output", type=str, default="analysis_corpus.pkl")
    p.add_argument(
        "--checkpoints", nargs="*", default=None,
        help="Paths to checkpoint.pt files for ε-greedy rollout."
    )
    p.add_argument("--checkpoint-epsilon", type=float, default=0.1)
    p.add_argument("--checkpoint-episodes", type=int, default=200)
    p.add_argument("--heuristic-episodes-per-seed", type=int, default=500)
    p.add_argument("--random-episodes", type=int, default=2000)
    args = p.parse_args()

    # Auto-discover checkpoints if not specified
    ckpt_paths = args.checkpoints
    if ckpt_paths is None:
        sweep_dir = "results/sweep_100k"
        if os.path.isdir(sweep_dir):
            ckpt_paths = sorted([
                os.path.join(sweep_dir, d, "checkpoint.pt")
                for d in os.listdir(sweep_dir)
                if os.path.isfile(os.path.join(sweep_dir, d, "checkpoint.pt"))
            ])
            print(f"Auto-discovered {len(ckpt_paths)} checkpoints in {sweep_dir}")

    corpus, metadata = generate_corpus(
        num_states=args.num_states,
        board_size=args.board_size,
        max_length=args.max_length,
        max_steps=args.max_steps,
        checkpoint_paths=ckpt_paths,
        checkpoint_epsilon=args.checkpoint_epsilon,
        checkpoint_episodes=args.checkpoint_episodes,
        random_episodes=args.random_episodes,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Save corpus
    with open(args.output, "wb") as f:
        pickle.dump(corpus, f)

    # Save metadata alongside
    import json
    meta_path = os.path.splitext(args.output)[0] + "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {len(corpus)} states to {args.output}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
