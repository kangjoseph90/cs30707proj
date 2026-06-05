"""Phase 5 tests: merged obstacle and egocentric encoders."""

from __future__ import annotations

import random

import numpy as np
import pytest

from snake_env import SnakeEnv, SnakeState
from state_encoders import (
    EgocentricMergedObstacleLocalEncoder,
    MergedObstacleLocalEncoder,
    _local_idx,
)
from train_snake import make_encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(body, fruit=(5, 5), direction=0):
    """Convenience constructor for SnakeState."""
    return SnakeState(body=tuple(body), fruit=fruit, direction=direction)


# ---------------------------------------------------------------------------
# 1-2. MergedObstacleLocalEncoder output_dim
# ---------------------------------------------------------------------------

class TestMergedOutputDim:
    @pytest.mark.parametrize("k,expected", [
        (5, 25 + 6),    # 31
        (9, 81 + 6),    # 87
    ])
    def test_output_dim(self, k, expected):
        enc = MergedObstacleLocalEncoder(15, 100, window_size=k)
        assert enc.output_dim == expected


# ---------------------------------------------------------------------------
# 3-4. EgocentricMergedObstacleLocalEncoder output_dim
# ---------------------------------------------------------------------------

class TestEgocentricMergedOutputDim:
    @pytest.mark.parametrize("k,expected", [
        (5, 25 + 6),    # 31
        (9, 81 + 6),    # 87
    ])
    def test_output_dim(self, k, expected):
        enc = EgocentricMergedObstacleLocalEncoder(15, 100, window_size=k)
        assert enc.output_dim == expected


# ---------------------------------------------------------------------------
# 5-7. Merged obstacle channel correctness
# ---------------------------------------------------------------------------

class TestMergedObstacleChannel:
    def test_body_marks_obstacle(self):
        """Body cells should be obstacle=1."""
        enc = MergedObstacleLocalEncoder(15, 100, window_size=5)
        body = [(7, 7), (6, 7), (5, 7)]
        state = _make_state(body, fruit=(10, 10), direction=0)
        vec = enc.encode(state)
        ksq = 25
        occ = vec[:ksq]
        # 3 body segments in window
        assert occ.sum() == 3.0

    def test_wall_marks_obstacle(self):
        """Out-of-board cells should be obstacle=1."""
        enc = MergedObstacleLocalEncoder(15, 100, window_size=5)
        state = _make_state([(0, 0), (1, 0)], fruit=(14, 14), direction=2)
        vec = enc.encode(state)
        ksq = 25
        # Count wall cells manually
        k = 5
        half = 2
        expected_wall = 0
        for wy in range(k):
            for wx in range(k):
                ax = 0 - half + wx
                ay = 0 - half + wy
                if ax < 0 or ax >= 15 or ay < 0 or ay >= 15:
                    expected_wall += 1
        # obstacle = wall + body_in_window (body at (0,0) and (1,0))
        assert vec[:ksq].sum() == expected_wall + 2.0

    def test_empty_interior_is_zero(self):
        """Empty interior cells should be obstacle=0."""
        enc = MergedObstacleLocalEncoder(15, 100, window_size=5)
        # Head at center, body segments nearby
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        ksq = 25
        occ = vec[:ksq]
        # Only 2 body cells, no wall → exactly 2.0
        assert occ.sum() == 2.0


# ---------------------------------------------------------------------------
# 8. Heading 4-directions: forward obstacle maps to ego-grid 위쪽
# ---------------------------------------------------------------------------

class TestEgoForwardMapping:
    """For each heading, world-forward obstacle should be at grid row < center."""

    @pytest.mark.parametrize("direction,body_ahead,world_label", [
        (0, (8, 7), "Right-forward=(8,7)"),
        (1, (7, 8), "Down-forward=(7,8)"),
        (2, (6, 7), "Left-forward=(6,7)"),
        (3, (7, 6), "Up-forward=(7,6)"),
    ])
    def test_forward_maps_to_grid_up(self, direction, body_ahead, world_label):
        enc = EgocentricMergedObstacleLocalEncoder(15, 100, window_size=5)
        state = _make_state([(7, 7), body_ahead], fruit=(3, 3), direction=direction)
        vec = enc.encode(state)
        k = 5
        center = 2
        ksq = k * k

        # Body should be exactly at (center, center-1) in ego grid
        pos = _local_idx(center, center - 1, k)
        assert vec[pos] == 1.0, (
            f"{world_label}: expected obstacle at ego grid ({center},{center-1})"
        )
        # Total obstacle count = 2 (head + forward segment)
        assert vec[:ksq].sum() == 2.0


# ---------------------------------------------------------------------------
# 9. Fruit dxdy heading 기준 회전
# ---------------------------------------------------------------------------

class TestEgoFruitDelta:
    def test_fruit_ahead_negative_ego_y(self):
        """Fruit directly ahead should have ego_y < 0."""
        enc = EgocentricMergedObstacleLocalEncoder(15, 100, window_size=5)
        # Heading Right, fruit at (10, 7) = 3 steps ahead
        state = _make_state([(7, 7), (6, 7)], fruit=(10, 7), direction=0)
        vec = enc.encode(state)
        ksq = 25
        assert vec[ksq + 1] < 0  # ego_y negative = ahead

    def test_fruit_right_positive_ego_x(self):
        """Fruit to the right should have ego_x > 0."""
        enc = EgocentricMergedObstacleLocalEncoder(15, 100, window_size=5)
        # Heading Right, fruit at (7, 10) = 3 steps right
        state = _make_state([(7, 7), (6, 7)], fruit=(7, 10), direction=0)
        vec = enc.encode(state)
        ksq = 25
        assert vec[ksq] > 0  # ego_x positive = right


# ---------------------------------------------------------------------------
# 10. ★ 90° 회전 동일 ego spatial encoding
# ---------------------------------------------------------------------------

class TestRotationInvariance:
    def test_four_directions_same_spatial(self):
        """Same relative layout at 4 headings → identical spatial encoding.

        Head at (7,7), body 2 segments straight ahead, fruit straight ahead.
        Only the spatial channels (obstacle grid + fruit = K²+2) should match.
        Direction one-hot differs by design.
        """
        enc = EgocentricMergedObstacleLocalEncoder(15, 100, window_size=5)
        ksq = 25

        # Right: body (8,7),(9,7), fruit (10,7)
        state_right = _make_state([(7, 7), (8, 7), (9, 7)], fruit=(10, 7), direction=0)
        # Down: body (7,8),(7,9), fruit (7,10)
        state_down = _make_state([(7, 7), (7, 8), (7, 9)], fruit=(7, 10), direction=1)
        # Left: body (6,7),(5,7), fruit (4,7)
        state_left = _make_state([(7, 7), (6, 7), (5, 7)], fruit=(4, 7), direction=2)
        # Up: body (7,6),(7,5), fruit (7,4)
        state_up = _make_state([(7, 7), (7, 6), (7, 5)], fruit=(7, 4), direction=3)

        vec_right = enc.encode(state_right)
        vec_down = enc.encode(state_down)
        vec_left = enc.encode(state_left)
        vec_up = enc.encode(state_up)

        # Compare spatial channels: obstacle (K²) + fruit (2) = K²+2
        spatial_len = ksq + 2
        spatial_right = vec_right[:spatial_len]
        spatial_down = vec_down[:spatial_len]
        spatial_left = vec_left[:spatial_len]
        spatial_up = vec_up[:spatial_len]

        assert np.array_equal(spatial_right, spatial_down), (
            f"Right vs Down spatial mismatch"
        )
        assert np.array_equal(spatial_right, spatial_left), (
            f"Right vs Left spatial mismatch"
        )
        assert np.array_equal(spatial_right, spatial_up), (
            f"Right vs Up spatial mismatch"
        )


# ---------------------------------------------------------------------------
# 11. make_encoder integration
# ---------------------------------------------------------------------------

class TestMakeEncoder:
    @pytest.mark.parametrize("rep", [
        "merged_obstacle_local",
        "egocentric_merged_obstacle_local",
    ])
    def test_make_encoder(self, rep):
        enc = make_encoder(rep, 15, 100, window_size=5)
        assert enc.output_dim == 31

    @pytest.mark.parametrize("rep", [
        "merged_obstacle_local",
        "egocentric_merged_obstacle_local",
    ])
    def test_window_size_required(self, rep):
        with pytest.raises(ValueError, match="window_size is required"):
            make_encoder(rep, 15, 100, window_size=None)


# ---------------------------------------------------------------------------
# 12. Checkpoint save/load
# ---------------------------------------------------------------------------

class TestCheckpointConfig:
    @pytest.mark.parametrize("rep", [
        "merged_obstacle_local",
        "egocentric_merged_obstacle_local",
    ])
    def test_config_roundtrip(self, rep, tmp_path):
        """Config roundtrip should restore the same encoder."""
        import json
        config = {
            "representation": rep,
            "board_size": 15,
            "max_length": 100,
            "local_window_size": 5,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loaded = json.loads(config_path.read_text())
        enc = make_encoder(
            loaded["representation"],
            loaded["board_size"],
            loaded["max_length"],
            loaded.get("local_window_size"),
        )

        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        assert vec.dtype == np.float32
        assert vec.shape == (enc.output_dim,)


# ---------------------------------------------------------------------------
# 13. Smoke test: 10 episodes per representation
# ---------------------------------------------------------------------------

class TestSmokeEpisodes:
    @pytest.mark.parametrize("rep,ws", [
        ("merged_obstacle_local", 5),
        ("merged_obstacle_local", 9),
        ("egocentric_merged_obstacle_local", 5),
        ("egocentric_merged_obstacle_local", 9),
    ])
    def test_ten_episodes(self, rep, ws):
        """Run 10 episodes — no crashes."""
        enc = make_encoder(rep, 15, 100, ws)
        env = SnakeEnv(board_size=15, max_length=100, max_steps=200, headless=True, seed=0)
        rng = random.Random(42)

        for ep in range(10):
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
