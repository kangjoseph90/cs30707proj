"""Tests for state_encoders and replay_buffer."""

from __future__ import annotations

import random

import numpy as np
import pytest

from snake_env import SnakeEnv, SnakeState
from state_encoders import (
    FullStateEncoder,
    LocalStateEncoder,
    StateEncoder,
    StructuralStateEncoder,
    decode_structural_state,
)
from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(body, fruit=(5, 5), direction=0):
    """Convenience constructor for SnakeState."""
    return SnakeState(body=tuple(body), fruit=fruit, direction=direction)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestFullEncoderShape:
    def test_output_dim(self):
        enc = FullStateEncoder(board_size=15, max_length=100)
        assert enc.output_dim == 225 * 100 + 225 + 4  # 22,729

    def test_encode_shape(self):
        enc = FullStateEncoder(15, 100)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        assert vec.shape == (22729,)
        assert vec.dtype == np.float32

    def test_nonzero_count(self):
        enc = FullStateEncoder(15, 100)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        # 2 body segments + 1 fruit + 1 direction = 4 ones
        assert vec.sum() == 4.0


class TestStructuralEncoderShape:
    def test_output_dim(self):
        enc = StructuralStateEncoder(15, 100)
        assert enc.output_dim == 225 + 225 + 4 * 99 + 4  # 850

    def test_encode_shape(self):
        enc = StructuralStateEncoder(15, 100)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        assert vec.shape == (850,)
        assert vec.dtype == np.float32


class TestLocalEncoderShape:
    @pytest.mark.parametrize("k,expected", [
        (9, 101 * 81 + 6),   # 8,187
        (7, 101 * 49 + 6),   # 4,955
        (5, 101 * 25 + 6),   # 2,531
    ])
    def test_output_dim(self, k, expected):
        enc = LocalStateEncoder(15, 100, window_size=k)
        assert enc.output_dim == expected

    def test_encode_shape(self):
        enc = LocalStateEncoder(15, 100, 9)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        assert vec.shape == (8187,)
        assert vec.dtype == np.float32


# ---------------------------------------------------------------------------
# Zero-padding
# ---------------------------------------------------------------------------

class TestZeroPadding:
    def test_full_zero_padding(self):
        enc = FullStateEncoder(15, 100)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        # Body section: 225 * 100 values
        body_section = vec[: 225 * 100]
        # Should have exactly 2 ones (2 segments)
        assert body_section.sum() == 2.0

    def test_structural_zero_padding(self):
        enc = StructuralStateEncoder(15, 100)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        # Body path section: 4 * 99 values, starting at offset 450
        path_section = vec[450 : 450 + 4 * 99]
        # Should have exactly 1 one (1 transition)
        assert path_section.sum() == 1.0


# ---------------------------------------------------------------------------
# Structural roundtrip
# ---------------------------------------------------------------------------

class TestStructuralRoundtrip:
    def test_handcrafted_roundtrip(self):
        """Encode → decode a handcrafted state with 5 segments."""
        body = [(7, 7), (6, 7), (5, 7), (5, 6), (5, 5)]
        state = _make_state(body, fruit=(10, 10), direction=0)
        enc = StructuralStateEncoder(15, 100)
        vec = enc.encode(state)
        decoded = decode_structural_state(vec, 15, 100)
        assert decoded == state

    def test_rollout_roundtrip(self):
        """Encode → decode hundreds of states from random rollouts."""
        enc = StructuralStateEncoder(15, 100)
        env = SnakeEnv(board_size=15, max_length=100, max_steps=500, headless=True, seed=42)
        rng = random.Random(0)
        checked = 0

        for _ in range(200):
            state = env.reset()
            done = False
            while not done:
                vec = enc.encode(state)
                decoded = decode_structural_state(vec, 15, 100)
                assert decoded == state, f"Mismatch at step {checked}: {decoded} != {state}"
                checked += 1
                action = rng.randint(0, 2)
                state, _, term, trunc, _ = env.step(action)
                done = term or trunc

        assert checked >= 100, f"Only checked {checked} states"

    def test_two_segment_roundtrip(self):
        """Two-segment initial state should roundtrip (adjacent, not overlapping)."""
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        enc = StructuralStateEncoder(15, 100)
        vec = enc.encode(state)
        decoded = decode_structural_state(vec, 15, 100)
        assert decoded == state


# ---------------------------------------------------------------------------
# Local wall channel
# ---------------------------------------------------------------------------

class TestLocalWallChannel:
    def test_corner_head_wall(self):
        """Head at (0,0) — most of the window should be wall."""
        enc = LocalStateEncoder(15, 100, 9)
        state = _make_state([(0, 0), (1, 0)], fruit=(14, 14), direction=2)
        vec = enc.encode(state)
        k = 9
        wall_channel = vec[100 * k * k : 101 * k * k]
        # Head at (0,0), window origin = (-4, -4).
        # All cells with abs pos < 0 should be wall.
        # Count wall cells: at least those with x < 0 or y < 0.
        half = k // 2  # 4
        expected_wall = 0
        for gy in range(k):
            for gx in range(k):
                ax = gx - half
                ay = gy - half
                if ax < 0 or ay < 0 or ax >= 15 or ay >= 15:
                    expected_wall += 1
        assert wall_channel.sum() == expected_wall

    def test_center_head_no_wall(self):
        """Head at center — no wall cells."""
        enc = LocalStateEncoder(15, 100, 5)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        k = 5
        wall_channel = vec[100 * k * k : 101 * k * k]
        assert wall_channel.sum() == 0.0


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_stores_canonical_state(self):
        """Buffer stores SnakeState, not encoded vectors."""
        buf = ReplayBuffer(capacity=100)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        next_state = _make_state([(8, 7), (7, 7)], fruit=(3, 3), direction=0)
        buf.push(state, 1, 3.0, next_state, False, False)
        assert len(buf) == 1
        t = buf._buf[0]
        assert isinstance(t.state, SnakeState)
        assert isinstance(t.next_state, SnakeState)

    def test_sample_shape(self):
        """Sample returns tensors with correct shapes."""
        buf = ReplayBuffer(capacity=1000)
        # Fill with diverse transitions
        for i in range(300):
            s = _make_state([(7, 7), (6, 7)], fruit=(3 + i % 5, 3), direction=i % 4)
            ns = _make_state([(8, 7), (7, 7)], fruit=(3, 3), direction=0)
            buf.push(s, i % 3, float(i), ns, i == 0, False)

        enc = StructuralStateEncoder(15, 100)
        b_s, b_a, b_r, b_ns, b_term, b_trunc = buf.sample(32, enc)
        assert b_s.shape == (32, enc.output_dim)
        assert b_a.shape == (32, 1)
        assert b_r.shape == (32,)
        assert b_ns.shape == (32, enc.output_dim)
        assert b_term.shape == (32,)
        assert b_trunc.shape == (32,)
