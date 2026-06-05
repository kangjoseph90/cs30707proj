"""Generate comparison plots from sweep results.

Usage::

    python plot_results.py --sweep-dir results/sweep_100k --output-dir plots/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent colours and labels per representation
REP_STYLE = {
    # Phase 3 (original)
    "full":                    {"color": "#e41a1c", "label": "Full (22,729d)"},
    "structural":              {"color": "#377eb8", "label": "Structural (850d)"},
    "local_w9":                {"color": "#4daf4a", "label": "Local Sparse K=9 (8,187d)"},
    "local_w7":                {"color": "#984ea3", "label": "Local Sparse K=7 (4,955d)"},
    "local_w5":                {"color": "#ff7f00", "label": "Local Sparse K=5 (2,531d)"},
    # Phase 4 (spatial ablation)
    "dense_ordered_global":    {"color": "#a65628", "label": "Dense Ord Global (904d)"},
    "dense_ordered_local_w9":  {"color": "#f781bf", "label": "Dense Ord Local K=9 (249d)"},
    "dense_ordered_local_w7":  {"color": "#999999", "label": "Dense Ord Local K=7 (153d)"},
    "dense_ordered_local_w5":  {"color": "#66c2a5", "label": "Dense Ord Local K=5 (81d)"},
    "occupancy_global":        {"color": "#fc8d62", "label": "Occ Global (679d)"},
    "occupancy_local_w9":      {"color": "#8da0cb", "label": "Occ Local K=9 (168d)"},
    "occupancy_local_w7":      {"color": "#e78ac3", "label": "Occ Local K=7 (104d)"},
    "occupancy_local_w5":      {"color": "#a6d854", "label": "Occ Local K=5 (56d)"},
    # Phase 5 (merged obstacle + egocentric)
    "merged_obstacle_local_w9":  {"color": "#ffd92f", "label": "Merged Obst K=9 (87d)"},
    "merged_obstacle_local_w5":  {"color": "#e5c494", "label": "Merged Obst K=5 (31d)"},
    "egocentric_merged_obstacle_local_w9": {"color": "#b3b3b3", "label": "Ego Merged K=9 (87d)"},
    "egocentric_merged_obstacle_local_w5": {"color": "#7fc97f", "label": "Ego Merged K=5 (31d)"},
    # Final sweep
    "merged_obstacle_local_w29":  {"color": "#d4a017", "label": "Merged Obst K=29 (847d)"},
    "merged_ordered_local_w5":    {"color": "#fb8072", "label": "Merged Ord K=5 (56d)"},
    "merged_ordered_local_w9":    {"color": "#80b1d3", "label": "Merged Ord K=9 (168d)"},
    "merged_ordered_local_w29":   {"color": "#fdb462", "label": "Merged Ord K=29 (1688d)"},
    "egocentric_merged_obstacle_local_w29": {"color": "#1b9e77", "label": "Ego Merged K=29 (847d)"},
    "egocentric_merged_ordered_local_w5":  {"color": "#b3de69", "label": "Ego Ord K=5 (56d)"},
    "egocentric_merged_ordered_local_w9":  {"color": "#fccde5", "label": "Ego Ord K=9 (168d)"},
    "egocentric_merged_ordered_local_w29": {"color": "#bc80bd", "label": "Ego Ord K=29 (1688d)"},
    "blind_sniff":  {"color": "#ffed6f", "label": "Blind Sniff (5d)"},
}


def _rep_key(row) -> str:
    """Canonical representation key from summary row or config dict."""
    rep = row["representation"] if isinstance(row, dict) else row.get("representation", "")
    ws = row.get("local_window_size") if isinstance(row, dict) else row.get("local_window_size")

    _LOCAL_REPS_WITH_WS = {"local", "dense_ordered_local", "occupancy_local",
                          "merged_obstacle_local", "egocentric_merged_obstacle_local",
                          "merged_ordered_local", "egocentric_merged_ordered_local"}
    if rep in _LOCAL_REPS_WITH_WS and ws is not None:
        return f"{rep}_w{int(ws)}"
    if rep == "local":
        # Backward compat: old runs used "local" with window_size
        if ws is not None:
            return f"local_w{int(ws)}"
    return rep


def _load_eval_curves(sweep_dir: str) -> dict[str, list[pd.DataFrame]]:
    """Load eval_scores.csv files, grouped by representation key."""
    groups: dict[str, list[pd.DataFrame]] = {}
    for name in sorted(os.listdir(sweep_dir)):
        run_dir = os.path.join(sweep_dir, name)
        csv_path = os.path.join(run_dir, "eval_scores.csv")
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(csv_path):
            continue
        with open(config_path) as f:
            config = json.load(f)
        key = _rep_key(config)
        df = pd.read_csv(csv_path)
        groups.setdefault(key, []).append(df)
    return groups


def _load_summary(sweep_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(sweep_dir, "summary.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


# -----------------------------------------------------------------------
# Plot 1: Steps vs eval score
# -----------------------------------------------------------------------

def plot_eval_score(groups: dict, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, style in REP_STYLE.items():
        dfs = groups.get(key, [])
        if not dfs:
            continue
        # Interpolate to common grid
        all_steps = sorted(set().union(*(set(df["env_steps"]) for df in dfs)))
        interp_scores = []
        for df in dfs:
            interp_scores.append(np.interp(all_steps, df["env_steps"], df["eval_score_mean"]))
        arr = np.array(interp_scores)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(all_steps, mean, color=style["color"], label=style["label"])
        ax.fill_between(all_steps, mean - std, mean + std, color=style["color"], alpha=0.15)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Score (mean)")
    ax.set_title("Learning Curve: Eval Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eval_score_curve.png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Plot 2: Steps vs eval reward
# -----------------------------------------------------------------------

def plot_eval_reward(groups: dict, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, style in REP_STYLE.items():
        dfs = groups.get(key, [])
        if not dfs:
            continue
        all_steps = sorted(set().union(*(set(df["env_steps"]) for df in dfs)))
        interp = []
        for df in dfs:
            interp.append(np.interp(all_steps, df["env_steps"], df["eval_reward_mean"]))
        arr = np.array(interp)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(all_steps, mean, color=style["color"], label=style["label"])
        ax.fill_between(all_steps, mean - std, mean + std, color=style["color"], alpha=0.15)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Reward (mean)")
    ax.set_title("Learning Curve: Eval Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eval_reward_curve.png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Plots 3-8: Bar charts from summary
# -----------------------------------------------------------------------

def _bar_plot(summary: pd.DataFrame, col: str, ylabel: str, title: str,
              out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = [k for k in REP_STYLE if k in summary["_key"].values]
    means, stds, colors, labels = [], [], [], []
    for k in keys:
        sub = summary[summary["_key"] == k]
        means.append(sub[col].mean())
        stds.append(sub[col].std() if len(sub) > 1 else 0.0)
        colors.append(REP_STYLE[k]["color"])
        labels.append(REP_STYLE[k]["label"])
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate comparison plots.")
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--output-dir", default="plots")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    groups = _load_eval_curves(args.sweep_dir)
    summary = _load_summary(args.sweep_dir)

    # Curves
    plot_eval_score(groups, args.output_dir)
    plot_eval_reward(groups, args.output_dir)

    # Bar charts from summary
    if summary is not None:
        summary["_key"] = summary.apply(_rep_key, axis=1)

        # Use first row per representation for deterministic metrics
        det = summary.groupby("_key").first().reset_index()

        _bar_plot(summary, "final_eval_score_mean", "Score",
                  "Final Eval Score", os.path.join(args.output_dir, "final_eval_score.png"))
        _bar_plot(summary, "learning_curve_auc", "AUC",
                  "Normalized Learning Curve AUC", os.path.join(args.output_dir, "auc.png"))

        # Deterministic (single-value) bar charts
        fig, ax = plt.subplots(figsize=(7, 4))
        keys = [k for k in REP_STYLE if k in det["_key"].values]
        vals = [det[det["_key"] == k]["input_dimension"].values[0] for k in keys]
        colors = [REP_STYLE[k]["color"] for k in keys]
        labels = [REP_STYLE[k]["label"] for k in keys]
        ax.bar(range(len(keys)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Dimensions")
        ax.set_title("Input Dimension")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "input_dimension.png"), dpi=150)
        plt.close(fig)

        _bar_plot(summary, "input_sparsity_mean", "Sparsity",
                  "Input Sparsity (mean)", os.path.join(args.output_dir, "sparsity.png"))

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        pcounts = [det[det["_key"] == k]["parameter_count"].values[0] for k in keys]
        ax2.bar(range(len(keys)), pcounts, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_xticks(range(len(keys)))
        ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax2.set_ylabel("Parameters")
        ax2.set_title("Parameter Count")
        ax2.grid(True, alpha=0.3, axis="y")
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.output_dir, "parameter_count.png"), dpi=150)
        plt.close(fig2)

        # --- Training time breakdown (stacked bar, read from metrics.json) ---
        metrics_per_key: dict[str, dict] = {}
        for name in sorted(os.listdir(args.sweep_dir)):
            run_dir = os.path.join(args.sweep_dir, name)
            mpath = os.path.join(run_dir, "metrics.json")
            cpath = os.path.join(run_dir, "config.json")
            if not os.path.isfile(mpath):
                continue
            with open(cpath) as f:
                cfg = json.load(f)
            with open(mpath) as f:
                m = json.load(f)
            k = _rep_key(cfg)
            if k not in metrics_per_key:
                metrics_per_key[k] = m

        fig_t, ax_t = plt.subplots(figsize=(9, 5))
        keys_t = [k for k in REP_STYLE if k in metrics_per_key]
        labels_t = [REP_STYLE[k]["label"] for k in keys_t]

        env_act = np.array([
            metrics_per_key[k].get("env_step_time_seconds", 0)
            + metrics_per_key[k].get("action_encoding_time_seconds", 0)
            for k in keys_t
        ])
        batch_enc = np.array([
            metrics_per_key[k].get("batch_encoding_time_seconds", 0) for k in keys_t
        ])
        optim = np.array([
            metrics_per_key[k].get("optimization_time_seconds", 0) for k in keys_t
        ])
        fwd_bwd = np.maximum(optim - batch_enc, 0)

        x_t = np.arange(len(keys_t))
        w = 0.6

        b1 = env_act
        b2 = b1 + fwd_bwd
        ax_t.bar(x_t, env_act, w, color="#66c2a5", label="Env step + action enc",
                 edgecolor="black", linewidth=0.3)
        ax_t.bar(x_t, fwd_bwd, w, bottom=b1, color="#8da0cb", label="Forward + Backward + Update",
                 edgecolor="black", linewidth=0.3)
        ax_t.bar(x_t, batch_enc, w, bottom=b2, color="#fc8d62", label="Batch encoding (replay)",
                 edgecolor="black", linewidth=0.3)

        ax_t.set_xticks(x_t)
        ax_t.set_xticklabels(labels_t, fontsize=7.5, rotation=20, ha="right")
        ax_t.set_ylabel("Seconds")
        ax_t.set_title("Training Time Breakdown")
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.3, axis="y")
        fig_t.tight_layout()
        fig_t.savefig(os.path.join(args.output_dir, "training_time.png"), dpi=150)
        plt.close(fig_t)

    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
