"""Replay buffers for Snake DQN training.

The default ``ReplayBuffer`` stores canonical ``SnakeState`` objects and
encodes at sample-time so the same buffer can be replayed with different
representations. ``EncodedReplayBuffer`` stores encoded vectors up front to
avoid repeated replay-time encoding when memory is available.
``NStepReplayBuffer`` accumulates n-step returns with per-transition bootstrap
discounts.
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
    next_safe_mask: np.ndarray | None = None


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
        next_safe_mask: np.ndarray | None = None,
    ) -> None:
        self._buf.append(
            Transition(state, action, reward, next_state, terminated, truncated, next_safe_mask)
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
        torch.BoolTensor,    # next_safe_masks (B, 3)
    ]:
        transitions = random.sample(list(self._buf), batch_size)

        states = np.array([encoder.encode(t.state) for t in transitions])
        next_states = np.array([encoder.encode(t.next_state) for t in transitions])
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        terminated = np.array([t.terminated for t in transitions], dtype=np.float32)
        truncated = np.array([t.truncated for t in transitions], dtype=np.float32)

        masks = []
        for t in transitions:
            if t.next_safe_mask is not None:
                masks.append(t.next_safe_mask)
            else:
                masks.append(np.array([True, True, True], dtype=np.bool_))
        next_safe_masks = np.stack(masks)

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions).unsqueeze(1),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(terminated),
            torch.from_numpy(truncated),
            torch.from_numpy(next_safe_masks),
        )


class EncodedTransition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    next_safe_mask: np.ndarray | None = None


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
        next_safe_mask: np.ndarray | None = None,
    ) -> None:
        self._buf.append(
            EncodedTransition(
                state.copy(),
                action,
                reward,
                next_state.copy(),
                terminated,
                truncated,
                next_safe_mask,
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
        torch.BoolTensor,    # next_safe_masks (B, 3)
    ]:
        transitions = random.sample(list(self._buf), batch_size)

        states = np.stack([t.state for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        terminated = np.array([t.terminated for t in transitions], dtype=np.float32)
        truncated = np.array([t.truncated for t in transitions], dtype=np.float32)

        masks = []
        for t in transitions:
            if t.next_safe_mask is not None:
                masks.append(t.next_safe_mask)
            else:
                masks.append(np.array([True, True, True], dtype=np.bool_))
        next_safe_masks = np.stack(masks)

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions).unsqueeze(1),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(terminated),
            torch.from_numpy(truncated),
            torch.from_numpy(next_safe_masks),
        )


# ---------------------------------------------------------------------------
# N-step replay buffer
# ---------------------------------------------------------------------------


class NStepEncodedTransition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    bootstrap_discount: float  # gamma ** actual_n
    next_safe_mask: np.ndarray | None = None


class NStepReplayBuffer:
    """Replay buffer that stores pre-computed n-step transitions.

    Accumulates consecutive transitions and emits n-step returns with
    per-transition bootstrap discounts (``gamma ** actual_n``).

    When ``n_step=1`` this behaves identically to ``EncodedReplayBuffer``
    but returns 7 tensors (with bootstrap_discount) instead of 6.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        n_step: int = 3,
        gamma: float = 0.95,
    ):
        self._buf: deque[NStepEncodedTransition] = deque(maxlen=capacity)
        self._n_step = n_step
        self._gamma = gamma
        self._pending: deque[tuple] = deque()

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
        next_safe_mask: np.ndarray | None = None,
    ) -> None:
        self._pending.append(
            (state.copy(), action, reward, next_state.copy(), terminated, truncated, next_safe_mask)
        )

        # Emit once we have n_step pending transitions
        if len(self._pending) >= self._n_step:
            self._emit_one()

        # Flush remaining pending on episode boundary
        if terminated or truncated:
            while self._pending:
                self._emit_one()

    def _emit_one(self) -> None:
        """Pop the oldest pending transition, emit an n-step (or shorter) return."""
        s0, a0 = self._pending[0][0], self._pending[0][1]

        cum_reward = 0.0
        actual_n = 0
        next_state = None
        terminated = False
        truncated = False
        next_safe_mask = None

        for i, (_, _, r, ns, term, trunc, mask) in enumerate(self._pending):
            if i >= self._n_step:
                break
            cum_reward += (self._gamma ** i) * r
            actual_n = i + 1
            next_state = ns
            terminated = term
            truncated = trunc
            next_safe_mask = mask
            if term or trunc:
                break

        self._buf.append(
            NStepEncodedTransition(
                s0,
                a0,
                cum_reward,
                next_state,
                terminated,
                truncated,
                self._gamma ** actual_n,
                next_safe_mask,
            )
        )
        self._pending.popleft()

    def sample(
        self,
        batch_size: int,
        encoder: StateEncoder | None = None,
    ) -> tuple[
        torch.FloatTensor,   # states  (B, input_dim)
        torch.LongTensor,    # actions (B, 1)
        torch.FloatTensor,   # rewards (B,)
        torch.FloatTensor,   # next_states (B, input_dim)
        torch.FloatTensor,   # terminated  (B,)
        torch.FloatTensor,   # truncated   (B,)
        torch.FloatTensor,   # bootstrap_discount (B,)
        torch.BoolTensor,    # next_safe_masks (B, 3)
    ]:
        transitions = random.sample(list(self._buf), batch_size)

        states = np.stack([t.state for t in transitions])
        next_states = np.stack([t.next_state for t in transitions])
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        terminated = np.array([t.terminated for t in transitions], dtype=np.float32)
        truncated = np.array([t.truncated for t in transitions], dtype=np.float32)
        discounts = np.array(
            [t.bootstrap_discount for t in transitions], dtype=np.float32
        )

        masks = []
        for t in transitions:
            if t.next_safe_mask is not None:
                masks.append(t.next_safe_mask)
            else:
                masks.append(np.array([True, True, True], dtype=np.bool_))
        next_safe_masks = np.stack(masks)

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions).unsqueeze(1),
            torch.from_numpy(rewards),
            torch.from_numpy(next_states),
            torch.from_numpy(terminated),
            torch.from_numpy(truncated),
            torch.from_numpy(discounts),
            torch.from_numpy(next_safe_masks),
        )
