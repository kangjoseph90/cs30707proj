"""Markov-probe analysis (paper Appendix E.5 adapted to Snake).

Trains a small MLP to regress ground-truth state features from the
encoder's z. Lower test MSE => the latent contains the information you'd
need for a Markov policy.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.embed_utils import collect_dataset, embed_all, load_encoder


def train_probe(z: np.ndarray, y: np.ndarray, epochs: int = 50,
                hidden: int = 128, lr: float = 1e-3,
                test_frac: float = 0.2, device: str = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(z)
    perm = np.random.permutation(n)
    n_test = int(n * test_frac)
    test_ix = perm[:n_test]
    train_ix = perm[n_test:]

    z_t = torch.from_numpy(z).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    model = nn.Sequential(
        nn.Linear(z.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, y.shape[1]),
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    bs = 256
    for ep in range(epochs):
        model.train()
        np.random.shuffle(train_ix)
        for s in range(0, len(train_ix), bs):
            ix = train_ix[s:s + bs]
            opt.zero_grad()
            pred = model(z_t[ix])
            loss = loss_fn(pred, y_t[ix])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred_test = model(z_t[test_ix]).cpu().numpy()
        y_test = y[test_ix]
    mse_total = float(((pred_test - y_test) ** 2).mean())
    mse_per_dim = ((pred_test - y_test) ** 2).mean(axis=0).tolist()
    return {"test_mse": mse_total, "test_mse_per_dim": mse_per_dim,
            "n_train": int(len(train_ix)), "n_test": int(len(test_ix))}


def main(run_dir: str, num_steps: int = 8000, epochs: int = 50):
    enc, z_dim, fs = load_encoder(run_dir)
    grids, dirs, gts = collect_dataset(num_steps, frame_stack=fs)
    z = embed_all(enc, grids, dirs)
    result = train_probe(z, gts, epochs=epochs)
    result.update({"run": run_dir, "num_steps": num_steps,
                   "z_dim": z_dim, "gt_dim": gts.shape[1]})
    print(json.dumps(result, indent=2))
    with open(os.path.join(run_dir, "markov_probe.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()
    main(args.run_dir, num_steps=args.steps, epochs=args.epochs)
