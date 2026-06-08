"""Latent-space visualisation: t-SNE projection + cluster board samples.

Workflow
--------
1. Collect game states via a random-policy rollout.
2. Pass every state through the trained encoder -> z vectors.
3. Project z to 2-D with t-SNE (or PCA).
4. Cluster the z vectors with k-means.
5. Plot A: scatter coloured by cluster.
6. Plot B: for each cluster, show N sample board images drawn from the
   states nearest to that cluster centre.

The point is to let the encoder decide what is "similar" — we then inspect
the actual boards to see whether the learned grouping is semantically
meaningful (e.g., do square-coiled states land in the same cluster?).
"""

import argparse
import os
import random as pyrandom
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from analysis.embed_utils import embed_all, load_agent_nets, load_encoder
from snake import Snake, getDir


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

def render_board(grid: np.ndarray) -> np.ndarray:
    """Convert a (3, H, W) float32 grid tensor to an (H, W, 3) RGB image.

    Background -> black
    Body       -> white
    Head       -> red
    Apple      -> bright green
    """
    H, W = grid.shape[1], grid.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)  # black background
    rgb[grid[0] > 0.5] = [1.00, 1.00, 1.00]   # body: white
    rgb[grid[1] > 0.5] = [1.00, 0.15, 0.15]   # head: red
    rgb[grid[2] > 0.5] = [0.10, 0.95, 0.30]   # apple: bright green
    return rgb


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect(num_steps: int, frame_stack: int = 1,
             policy: str = "random", enc=None, head=None,
             max_ep_steps: int = 500):
    Game = Snake(frame_stack=frame_stack)
    grids, dirs = [], []
    ep_steps = 0
    for _ in range(num_steps):
        grid, dir_oh = Game.getFullState()
        grids.append(grid)
        dirs.append(dir_oh)
        if policy == "greedy" and enc is not None and head is not None:
            g = grid.unsqueeze(0)
            d = dir_oh.unsqueeze(0)
            z = enc(g, d)
            action = int(head(z).argmax(dim=1).item())
        else:
            action = pyrandom.randrange(3)
        Game.changeDir(getDir(Game.dir)[action])
        Game.MoveSnake()
        ep_steps += 1
        if Game.isDead() or ep_steps >= max_ep_steps:
            Game = Snake(frame_stack=frame_stack)
            ep_steps = 0
    return grids, dirs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_dir: str, num_steps: int = 4000, n_clusters: int = 6,
         n_samples: int = 5, method: str = "tsne", out_prefix: str = None,
         policy: str = "random"):

    enc, z_dim, fs = load_encoder(run_dir)
    print(f"encoder loaded: z_dim={z_dim}, frame_stack={fs}")

    head = None
    if policy == "greedy":
        _, head, _, _ = load_agent_nets(run_dir)
        print("Q-head loaded for greedy rollout")

    grids, dirs = _collect(num_steps, frame_stack=fs, policy=policy,
                           enc=enc, head=head)
    z = embed_all(enc, grids, dirs)           # (N, z_dim)
    print(f"embedded {len(z)} states")

    # ---- 2-D projection -----------------------------------------------
    if method == "tsne":
        from sklearn.manifold import TSNE
        proj = TSNE(n_components=2, perplexity=30, random_state=0,
                    max_iter=1000).fit_transform(z)
        axis_label = "t-SNE"
    else:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(z)
        axis_label = "PC"

    # ---- k-means clustering on z (not on 2-D projection) --------------
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(z)
    centers = km.cluster_centers_           # (K, z_dim)

    # ---- Figure 1: scatter coloured by cluster -------------------------
    # Perceptually distinct palette; cycles if n_clusters > len(PALETTE).
    PALETTE = [
        "#e6194b",  # red
        "#3cb44b",  # green
        "#4363d8",  # blue
        "#f58231",  # orange
        "#911eb4",  # purple
        "#42d4f4",  # cyan
        "#f032e6",  # magenta
        "#bfef45",  # lime
        "#fabed4",  # pink
        "#469990",  # teal
    ]
    colors = [PALETTE[k % len(PALETTE)] for k in range(n_clusters)]

    fig1, ax1 = plt.subplots(figsize=(7, 6))
    for k in range(n_clusters):
        mask = labels == k
        ax1.scatter(proj[mask, 0], proj[mask, 1],
                    color=colors[k], s=6, alpha=0.6, label=f"C{k}",
                    edgecolors="none")
    ax1.set_xlabel(f"{axis_label} 1")
    ax1.set_ylabel(f"{axis_label} 2")
    ax1.set_title(f"Latent space — {os.path.basename(run_dir)}\n"
                  f"({method.upper()}, {num_steps} steps, k={n_clusters})")
    ax1.legend(markerscale=2, fontsize=8, loc="best")
    fig1.tight_layout()

    out1 = (f"{out_prefix}_scatter.png" if out_prefix else
            os.path.join(run_dir, f"embedding_{method}_scatter.png"))
    fig1.savefig(out1, dpi=150)
    print(f"wrote {out1}")

    # ---- Figure 2: sample boards per cluster ---------------------------
    CELL = 2.2          # inches per board thumbnail
    LABEL_W = 0.7       # width reserved for row label column
    fig2_w = LABEL_W + n_samples * CELL
    fig2_h = n_clusters * CELL + 0.5   # +0.5 for suptitle

    fig2 = plt.figure(figsize=(fig2_w, fig2_h), facecolor="black")

    grids_np = [g.numpy() if hasattr(g, "numpy") else np.array(g) for g in grids]

    for k in range(n_clusters):
        mask_idx = np.where(labels == k)[0]
        dists = np.linalg.norm(z[mask_idx] - centers[k], axis=1)
        closest = mask_idx[np.argsort(dists)[:n_samples]]

        for col, idx in enumerate(closest):
            # GridSpec-style manual axes placement
            left   = (LABEL_W + col * CELL) / fig2_w
            bottom = 1.0 - (k + 1) * CELL / fig2_h
            width  = CELL / fig2_w * 0.92
            height = CELL / fig2_h * 0.88

            ax = fig2.add_axes([left, bottom, width, height])
            board_img = render_board(grids_np[idx])
            ax.imshow(board_img, origin="upper", interpolation="nearest",
                      aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("white")
                spine.set_linewidth(1.5)
            ax.set_facecolor("black")

        # Row label (cluster index + colour swatch)
        lax = fig2.add_axes([0.0, 1.0 - (k + 1) * CELL / fig2_h,
                              LABEL_W / fig2_w, CELL / fig2_h])
        lax.set_facecolor("black")
        lax.axis("off")
        lax.text(0.85, 0.5, f"C{k}",
                 color=colors[k], fontsize=13, fontweight="bold",
                 ha="right", va="center", transform=lax.transAxes)

    fig2.suptitle(f"Nearest boards to each cluster centre — {os.path.basename(run_dir)}",
                  fontsize=9, color="white", y=0.995)
    fig2.patch.set_facecolor("black")

    out2 = (f"{out_prefix}_boards.png" if out_prefix else
            os.path.join(run_dir, f"embedding_{method}_boards.png"))
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")

    plt.close("all")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--clusters", type=int, default=6,
                   help="number of k-means clusters")
    p.add_argument("--samples", type=int, default=5,
                   help="sample boards shown per cluster")
    p.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    p.add_argument("--out-prefix", type=str, default=None,
                   help="output path prefix (two files: _scatter.png, _boards.png)")
    p.add_argument("--policy", choices=["random", "greedy"], default="random",
                   help="rollout policy: random or greedy (uses trained Q-head)")
    args = p.parse_args()
    main(args.run_dir,
         num_steps=args.steps,
         n_clusters=args.clusters,
         n_samples=args.samples,
         method=args.method,
         out_prefix=args.out_prefix,
         policy=args.policy)
