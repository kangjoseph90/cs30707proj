"""Data collectors for analysis: transition reservoir and sparsity sampling."""

from __future__ import annotations

import pickle
import random
from typing import NamedTuple

import numpy as np

from snake_env import SnakeState


# ---------------------------------------------------------------------------
# AnalysisTransition
# ---------------------------------------------------------------------------

class AnalysisTransition(NamedTuple):
    state: SnakeState
    action: int
    reward: float
    next_state: SnakeState
    terminated: bool
    truncated: bool
    ate_fruit: bool


# ---------------------------------------------------------------------------
# AnalysisCollector – reservoir-sampled transition buffer
# ---------------------------------------------------------------------------

class AnalysisCollector:
    """Collect up to *max_transitions* transitions using reservoir sampling.

    Guarantees a uniform sample of the entire training trajectory regardless
    of when transitions arrive.
    """

    def __init__(self, max_transitions: int = 100_000, seed: int = 42):
        self._max = max_transitions
        self._transitions: list[AnalysisTransition] = []
        self._count = 0  # total seen (may exceed _max)
        self._rng = random.Random(seed)

    def push(
        self,
        state: SnakeState,
        action: int,
        reward: float,
        next_state: SnakeState,
        terminated: bool,
        truncated: bool,
        ate_fruit: bool,
    ) -> None:
        t = AnalysisTransition(state, action, reward, next_state, terminated, truncated, ate_fruit)
        self._count += 1
        if len(self._transitions) < self._max:
            self._transitions.append(t)
        else:
            j = self._rng.randint(0, self._count - 1)
            if j < self._max:
                self._transitions[j] = t

    def __len__(self) -> int:
        return len(self._transitions)

    @property
    def transitions(self) -> list[AnalysisTransition]:
        return self._transitions

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._transitions, f)

    @staticmethod
    def load(path: str) -> list[AnalysisTransition]:
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# SparsityCollector – reservoir-sampled encoded states
# ---------------------------------------------------------------------------

class SparsityCollector:
    """Collect up to *max_samples* encoded-state vectors via reservoir sampling.

    After collection call :meth:`compute` to get ``(mean, std)`` of
    ``1 - nonzero_count / input_dim``.
    """

    def __init__(self, max_samples: int = 10_000, seed: int = 42):
        self._max = max_samples
        self._samples: list[np.ndarray] = []
        self._count = 0
        self._rng = random.Random(seed)

    def push(self, encoded_state: np.ndarray) -> None:
        self._count += 1
        if len(self._samples) < self._max:
            self._samples.append(encoded_state)
        else:
            j = self._rng.randint(0, self._count - 1)
            if j < self._max:
                self._samples[j] = encoded_state

    def __len__(self) -> int:
        return len(self._samples)

    def compute(self) -> tuple[float, float]:
        """Return ``(sparsity_mean, sparsity_std)`` over collected samples."""
        if not self._samples:
            return (float("nan"), float("nan"))
        input_dim = self._samples[0].shape[0]
        values = np.array(
            [1.0 - np.count_nonzero(s) / input_dim for s in self._samples]
        )
        return float(values.mean()), float(values.std())
