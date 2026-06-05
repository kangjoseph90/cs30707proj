"""Replay buffers for Snake DQN training.

The default ``ReplayBuffer`` stores canonical ``SnakeState`` objects and
encodes at sample-time so the same buffer can be replayed with different
representations. ``EncodedReplayBuffer`` stores encoded vectors up front to
avoid repeated replay-time encoding when memory is available.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch

from snake_env import SnakeState
from state_encoders import StateEncoder


class Transition(NamedTuple):
    state: SnakeState
    action: int
    reward: float
    next_state: SnakeState
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """Fixed-size FIFO replay buffer storing canonical ``SnakeState`` snapshots.

    ``sample()`` applies the encoder at draw-time so that switching
    representations never requires re-collecting experience.
    """

    def __init__(self, capacity: int = 100_000):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(
        self,
        state: SnakeState,
        action: int,
        reward: float,
        next_state: SnakeState,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._buf.append(
            Transition(state, action, reward, next_state, terminated, truncated)
        )

    def sample(
        self, batch_size: int, encoder: StateEncoder
    ) -> tuple[
        torch.FloatTensor,   # states  (B, input_dim)
        torch.LongTensor,    # actions (B, 1)
        torch.FloatTensor,   # rewards (B,)
        torch.FloatTensor,   # next_states (B, input_dim)
        torch.FloatTensor,   # terminated  (B,)
        torch.FloatTensor,   # truncated   (B,)
    ]:
        transitions = random.sample(list(self._buf), batch_size)

        states = np.array([encoder.encode(t.state) for t in transitions])
        next_states = np.array([encoder.encode(t.next_state) for t in transitions])
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        terminated = np.array([t.terminated for t in transitions], dtype=np.float32)
        truncated = np.array([t.truncated for t in transitions], dtype=np.float32)

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions).unsqueeze(1),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(terminated),
            torch.from_numpy(truncated),
        )


class EncodedTransition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


class EncodedReplayBuffer:
    """Fixed-size FIFO replay buffer storing already-encoded state vectors.

    This trades memory for speed: each transition stores both encoded state
    and encoded next_state, so replay sampling does not call the encoder.
    """

    def __init__(self, capacity: int = 100_000):
        self._buf: deque[EncodedTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self._buf.append(
            EncodedTransition(
                state.copy(),
                action,
                reward,
                next_state.copy(),
                terminated,
                truncated,
            )
        )

    def sample(
        self, batch_size: int, encoder: StateEncoder | None = None
    ) -> tuple[
        torch.FloatTensor,   # states  (B, input_dim)
        torch.LongTensor,    # actions (B, 1)
        torch.FloatTensor,   # rewards (B,)
        torch.FloatTensor,   # next_states (B, input_dim)
        torch.FloatTensor,   # terminated  (B,)
        torch.FloatTensor,   # truncated   (B,)
    ]:
        transitions = random.sample(list(self._buf), batch_size)

        states = np.stack([t.state for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        terminated = np.array([t.terminated for t in transitions], dtype=np.float32)
        truncated = np.array([t.truncated for t in transitions], dtype=np.float32)

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions).unsqueeze(1),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(terminated),
            torch.from_numpy(truncated),
        )
