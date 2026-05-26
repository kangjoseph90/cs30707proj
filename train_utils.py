"""Shared helpers for training scripts."""

import json
import os
import random
from typing import Dict, List

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(setup: str, seed: int, root: str = "runs") -> str:
    path = os.path.join(os.getcwd(), root, f"{setup}_seed{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def dump_scores(path: str, scores: List[int], meta: Dict) -> None:
    with open(os.path.join(path, "scores.json"), "w") as f:
        json.dump({"scores": scores, "meta": meta}, f, indent=2)
