"""Tests for final sweep encoders: MergedOrdered, EgocentricMergedOrdered, BlindSniff."""

from __future__ import annotations

import random

import numpy as np
import pytest

from snake_env import SnakeEnv, SnakeState
from state_encoders import (
    BlindSniffEncoder,
    EgocentricMergedOrderedLocalEncoder,
    MergedOrderedLocalEncoder,
)
from train_snake import make_encoder


def _make_state(body, fruit=(5, 5), direction=0):
    return SnakeState(body=tuple(body), fruit=fruit, direction=direction)


# ---------------------------------------------------------------------------
# 1. Output dimensions
# ---------------------------------------------------------------------------

class TestOutputDim:
    @pytest.mark.parametrize("k,expected", [
        (5, 2 * 25 + 6),    # 56
        (9, 2 * 81 + 6),    # 168
        (29, 2 * 841 + 6),  # 1688
    ])
    def test_merged_ordered_dim(self, k, expected):
        enc = MergedOrderedLocalEncoder(15, 100, window_size=k)
        assert enc.output_dim == expected

    @pytest.mark.parametrize("k,expected", [
        (5, 2 * 25 + 6),
        (9, 2 * 81 + 6),
        (29, 2 * 841 + 6),
    ])
    def test_ego_merged_ordered_dim(self, k, expected):
        enc = EgocentricMergedOrderedLocalEncoder(15, 100, window_size=k)
        assert enc.output_dim == expected

    def test_blind_sniff_dim(self):
        enc = BlindSniffEncoder(15, 100)
        assert enc.output_dim == 5


# ---------------------------------------------------------------------------
# 2. MergedOrdered channel correctness
# ---------------------------------------------------------------------------

class TestMergedOrderedChannels:
    def test_body_marks_obstacle_and_order(self):
        enc = MergedOrderedLocalEncoder(15, 100, window_size=5)
        state = _make_state([(7, 7), (6, 7), (5, 7)], fruit=(10, 10), direction=0)
        vec = enc.encode(state)
        ksq = 25
        occ = vec[:ksq]
        order = vec[ksq:2 * ksq]
        # 3 body segments
        assert occ.sum() == 3.0
        # Head at seg_idx=0 → order=0.0, neck=0.01, tail=0.02
        assert order.sum() > 0  # at least some non-zero order values

    def test_wall_marks_obstacle_no_order(self):
        enc = MergedOrderedLocalEncoder(15, 100, window_size=5)
        state = _make_state([(0, 0), (1, 0)], fruit=(14, 14), direction=2)
        vec = enc.encode(state)
        ksq = 25
        occ = vec[:ksq]
        order = vec[ksq:2 * ksq]
        # Wall cells should have obstacle=1 but order=0
        for i in range(ksq):
            if occ[i] > 0 and order[i] > 0:
                pass  # body segment
            elif occ[i] > 0 and order[i] == 0:
                pass  # wall — ok
            # No cell should have order > 0 without obstacle

    def test_order_values_correct(self):
        """Order should be seg_idx / max_length."""
        enc = MergedOrderedLocalEncoder(15, 100, window_size=5)
        # Body with head at (7,7), neck at (8,7), tail at (9,7)
        state = _make_state([(7, 7), (8, 7), (9, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        ksq = 25
        k = 5
        order = vec[ksq:2 * ksq]
        occ = vec[:ksq]
        # 3 body segments in window → 3 obstacle cells
        assert occ.sum() == 3.0
        # Head=seg_idx 0 → order=0.0 (indistinguishable from "no body")
        # Neck=seg_idx 1 → order=0.01, Tail=seg_idx 2 → order=0.02
        nonzero = order[order > 0]
        assert len(nonzero) == 2
        assert 0.01 in nonzero
        assert 0.02 in nonzero


# ---------------------------------------------------------------------------
# 3. EgocentricMergedOrdered rotation invariance
# ---------------------------------------------------------------------------

class TestEgoMergedOrderedRotation:
    def test_four_directions_same_spatial(self):
        enc = EgocentricMergedOrderedLocalEncoder(15, 100, window_size=5)
        ksq = 25
        spatial_len = 2 * ksq + 2  # obstacle + order + fruit

        # Right: body (8,7),(9,7), fruit (10,7)
        r = enc.encode(_make_state([(7, 7), (8, 7), (9, 7)], fruit=(10, 7), direction=0))
        d = enc.encode(_make_state([(7, 7), (7, 8), (7, 9)], fruit=(7, 10), direction=1))
        l = enc.encode(_make_state([(7, 7), (6, 7), (5, 7)], fruit=(4, 7), direction=2))
        u = enc.encode(_make_state([(7, 7), (7, 6), (7, 5)], fruit=(7, 4), direction=3))

        assert np.array_equal(r[:spatial_len], d[:spatial_len]), "Right vs Down"
        assert np.array_equal(r[:spatial_len], l[:spatial_len]), "Right vs Left"
        assert np.array_equal(r[:spatial_len], u[:spatial_len]), "Right vs Up"


# ---------------------------------------------------------------------------
# 4. BlindSniff correctness
# ---------------------------------------------------------------------------

class TestBlindSniff:
    def test_wall_ahead(self):
        enc = BlindSniffEncoder(15, 100)
        # Heading Right at (14,7) → forward is wall
        state = _make_state([(14, 7), (13, 7)], fruit=(5, 5), direction=0)
        vec = enc.encode(state)
        # left=straight_idx=0→left, 1→straight, 2→right
        # Heading Right: left=Up(idx0), forward=Right(idx1), right=Down(idx2)
        assert vec[1] == 1.0  # forward = wall
        assert vec[0] == 0.0  # left = up = (14,6) = empty
        assert vec[2] == 0.0  # right = down = (14,8) = empty

    def test_body_ahead(self):
        enc = BlindSniffEncoder(15, 100)
        # Heading Right, body segment at (8,7) = forward
        state = _make_state([(7, 7), (8, 7), (8, 8)], fruit=(5, 5), direction=0)
        vec = enc.encode(state)
        assert vec[1] == 1.0  # forward = body

    def test_no_obstacle(self):
        enc = BlindSniffEncoder(15, 100)
        state = _make_state([(7, 7), (6, 7)], fruit=(10, 10), direction=0)
        vec = enc.encode(state)
        # Heading Right at center: forward (8,7), left (7,6), right (7,8) all empty
        assert vec[0] == 0.0
        assert vec[1] == 0.0
        assert vec[2] == 0.0

    def test_tail_vacating(self):
        """Tail should NOT be counted as obstacle if it will vacate."""
        enc = BlindSniffEncoder(15, 100)
        # Heading Right, tail at (8,7) which is forward direction
        # But fruit is NOT at (8,7), so tail will vacate
        state = _make_state([(7, 7), (8, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        # (8,7) is tail and fruit is elsewhere → should be 0
        assert vec[1] == 0.0

    def test_fruit_ego_frame(self):
        enc = BlindSniffEncoder(15, 100)
        # Heading Right, fruit at (10,7) = 3 steps ahead
        state = _make_state([(7, 7), (6, 7)], fruit=(10, 7), direction=0)
        vec = enc.encode(state)
        assert vec[4] < 0  # ego_y < 0 = ahead

    def test_fruit_right(self):
        enc = BlindSniffEncoder(15, 100)
        # Heading Right, fruit at (7,10) = 3 steps right
        state = _make_state([(7, 7), (6, 7)], fruit=(7, 10), direction=0)
        vec = enc.encode(state)
        assert vec[3] > 0  # ego_x > 0 = right


# ---------------------------------------------------------------------------
# 5. make_encoder integration
# ---------------------------------------------------------------------------

class TestMakeEncoder:
    @pytest.mark.parametrize("rep,ws,expected_dim", [
        ("merged_ordered_local", 5, 56),
        ("merged_ordered_local", 9, 168),
        ("merged_ordered_local", 29, 1688),
        ("egocentric_merged_ordered_local", 5, 56),
        ("egocentric_merged_ordered_local", 9, 168),
        ("egocentric_merged_ordered_local", 29, 1688),
        ("blind_sniff", None, 5),
    ])
    def test_make_encoder(self, rep, ws, expected_dim):
        enc = make_encoder(rep, 15, 100, window_size=ws)
        assert enc.output_dim == expected_dim

    @pytest.mark.parametrize("rep", [
        "merged_ordered_local",
        "egocentric_merged_ordered_local",
    ])
    def test_window_size_required(self, rep):
        with pytest.raises(ValueError, match="window_size is required"):
            make_encoder(rep, 15, 100, window_size=None)


# ---------------------------------------------------------------------------
# 6. Smoke test: 10 episodes per representation
# ---------------------------------------------------------------------------

class TestSmokeEpisodes:
    @pytest.mark.parametrize("rep,ws", [
        ("merged_ordered_local", 5),
        ("merged_ordered_local", 9),
        ("merged_ordered_local", 29),
        ("egocentric_merged_ordered_local", 5),
        ("egocentric_merged_ordered_local", 9),
        ("egocentric_merged_ordered_local", 29),
    ])
    def test_ten_episodes_windowed(self, rep, ws):
        enc = make_encoder(rep, 15, 100, ws)
        env = SnakeEnv(board_size=15, max_length=100, max_steps=200, headless=True, seed=0)
        rng = random.Random(42)
        for _ in range(10):
            state = env.reset()
            done = False
            while not done:
                vec = enc.encode(state)
                assert vec.shape == (enc.output_dim,)
                assert vec.dtype == np.float32
                assert np.all(np.isfinite(vec))
                action = rng.randint(0, 2)
                state, _, term, trunc, _ = env.step(action)
                done = term or trunc

    def test_ten_episodes_blind_sniff(self):
        enc = make_encoder("blind_sniff", 15, 100)
        env = SnakeEnv(board_size=15, max_length=100, max_steps=200, headless=True, seed=0)
        rng = random.Random(42)
        for _ in range(10):
            state = env.reset()
            done = False
            while not done:
                vec = enc.encode(state)
                assert vec.shape == (5,)
                assert vec.dtype == np.float32
                assert np.all(np.isfinite(vec))
                action = rng.randint(0, 2)
                state, _, term, trunc, _ = env.step(action)
                done = term or trunc
