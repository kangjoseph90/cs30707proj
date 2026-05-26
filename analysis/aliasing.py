"""State-aliasing diagnostic for a trained encoder.

For each embedded state, look at its K nearest neighbors (in z-space) and
measure what fraction share a *coarse* ground-truth label. Low purity =>
distinct game states are mapping near each other in latent space = aliasing.

Coarse label = (head cell, direction, apple-relative quadrant). This bins
the state space finely enough that almost-identical board configurations
collapse to the same label, while different boards generally don't.
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.embed_utils import collect_dataset, embed_all, load_encoder


def coarse_labels(gts: np.ndarray) -> np.ndarray:
    """Build an integer label per state from the gt vector.
    gt = [head_x, head_y, apple_x, apple_y, dir(4), body_len]
    Label = (head_xy, dir, sign(apple - head) per axis).
    """
    head_x = (gts[:, 0] * 20).round().astype(int)
    head_y = (gts[:, 1] * 20).round().astype(int)
    dir_ix = gts[:, 4:8].argmax(axis=1)
    dx = np.sign((gts[:, 2] - gts[:, 0])).astype(int) + 1   # in {0,1,2}
    dy = np.sign((gts[:, 3] - gts[:, 1])).astype(int) + 1
    # Pack into a single integer.
    return ((head_x * 20 + head_y) * 4 + dir_ix) * 9 + (dx * 3 + dy)


def knn_purity(z: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    # Brute-force KNN; fine for a few thousand points.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    _, idx = nn.kneighbors(z)
    purities = []
    for i in range(len(z)):
        neighbors = idx[i, 1:]  # drop self
        same = (labels[neighbors] == labels[i]).mean()
        purities.append(same)
    return float(np.mean(purities))


def main(run_dir: str, num_steps: int = 4000, k: int = 5):
    enc, z_dim, fs = load_encoder(run_dir)
    grids, dirs, gts = collect_dataset(num_steps, frame_stack=fs)
    z = embed_all(enc, grids, dirs)
    labels = coarse_labels(gts)
    n_unique = len(np.unique(labels))
    purity = knn_purity(z, labels, k=k)
    summary = {
        "run": run_dir, "num_steps": num_steps, "k": k,
        "n_unique_coarse_labels": int(n_unique),
        "knn_purity": purity,
        "aliasing_rate": 1.0 - purity,
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(run_dir, "aliasing.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()
    main(args.run_dir, num_steps=args.steps, k=args.k)
