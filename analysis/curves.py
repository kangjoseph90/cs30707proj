"""Aggregate scores.json files into a median+IQR learning curve per setup."""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_scores(root: str = "runs") -> dict:
    """Return {setup: [[seed0 scores], [seed1 scores], ...]}."""
    pattern = os.path.join(root, "*_seed*", "scores.json")
    out = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        dir_name = os.path.basename(os.path.dirname(path))
        m = re.match(r"(?P<setup>.+)_seed(?P<seed>\d+)$", dir_name)
        if not m:
            continue
        setup = m.group("setup")
        with open(path) as f:
            data = json.load(f)
        out[setup].append(data["scores"])
    return out


def plot(setups: dict, out_path: str = "learning_curves.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for setup, runs in sorted(setups.items()):
        # truncate to the shortest run so all seeds align
        n = min(len(r) for r in runs)
        arr = np.stack([r[:n] for r in runs])  # (seeds, episodes)
        ep = np.arange(1, n + 1)
        median = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        line, = ax.plot(ep, median, label=f"{setup} (n={len(runs)})")
        ax.fill_between(ep, q25, q75, alpha=0.2, color=line.get_color())
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Snake DQN: representation comparison (median +/- IQR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=str, default="runs")
    p.add_argument("--out", type=str, default="learning_curves.png")
    args = p.parse_args()
    setups = load_scores(args.runs_root)
    if not setups:
        raise SystemExit(f"no runs found under {args.runs_root}")
    plot(setups, args.out)
