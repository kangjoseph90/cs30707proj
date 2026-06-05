"""Phase 4 tests: dense spatial encoders and ablation verification."""

from __future__ import annotations

import random

import numpy as np
import pytest

from snake_env import SnakeEnv, SnakeState
from state_encoders import (
    DenseOrderedGlobalEncoder,
    DenseOrderedLocalEncoder,
    OccupancyGlobalEncoder,
    OccupancyLocalEncoder,
    decode_dense_ordered_global,
    _global_idx,
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
# 1. Output dimension verification
# ---------------------------------------------------------------------------

class TestOutputDim:
    def test_dense_ordered_global(self):
        enc = DenseOrderedGlobalEncoder(15, 100)
        assert enc.output_dim == 4 * 225 + 4  # 904

    @pytest.mark.parametrize("k,expected", [
        (5, 3 * 25 + 6),    # 81
        (7, 3 * 49 + 6),    # 153
        (9, 3 * 81 + 6),    # 249
    ])
    def test_dense_ordered_local(self, k, expected):
        enc = DenseOrderedLocalEncoder(15, 100, window_size=k)
        assert enc.output_dim == expected

    def test_occupancy_global(self):
        enc = OccupancyGlobalEncoder(15, 100)
        assert enc.output_dim == 3 * 225 + 4  # 679

    @pytest.mark.parametrize("k,expected", [
        (5, 2 * 25 + 6),    # 56
        (7, 2 * 49 + 6),    # 104
        (9, 2 * 81 + 6),    # 168
    ])
    def test_occupancy_local(self, k, expected):
        enc = OccupancyLocalEncoder(15, 100, window_size=k)
        assert enc.output_dim == expected


# ---------------------------------------------------------------------------
# 2. dtype verification
# ---------------------------------------------------------------------------

class TestDtype:
    @pytest.mark.parametrize("enc_factory", [
        lambda: DenseOrderedGlobalEncoder(15, 100),
        lambda: DenseOrderedLocalEncoder(15, 100, 7),
        lambda: OccupancyGlobalEncoder(15, 100),
        lambda: OccupancyLocalEncoder(15, 100, 7),
    ])
    def test_encode_returns_float32(self, enc_factory):
        enc = enc_factory()
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        assert vec.dtype == np.float32
        assert vec.shape == (enc.output_dim,)


# ---------------------------------------------------------------------------
# 3. DenseOrderedGlobal body_occupancy
# ---------------------------------------------------------------------------

class TestDenseOrderedGlobalOccupancy:
    def test_body_positions_marked(self):
        """Body segment positions should be 1.0 in occupancy channel."""
        enc = DenseOrderedGlobalEncoder(15, 100)
        body = [(7, 7), (6, 7), (5, 7)]
        state = _make_state(body, fruit=(3, 3), direction=0)
        vec = enc.encode(state)

        bsq = 225
        occ = vec[:bsq]
        # All body positions should be 1.0
        for seg in body:
            idx = _global_idx(seg[0], seg[1], 15)
            assert occ[idx] == 1.0, f"Segment {seg} not marked in occupancy"

        # Exactly 3 cells should be occupied
        assert occ.sum() == 3.0

    def test_non_body_positions_zero(self):
        """Non-body positions should be 0.0 in occupancy."""
        enc = DenseOrderedGlobalEncoder(15, 100)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        bsq = 225
        occ = vec[:bsq]
        assert occ.sum() == 2.0


# ---------------------------------------------------------------------------
# 4. DenseOrderedGlobal body_order values
# ---------------------------------------------------------------------------

class TestDenseOrderedGlobalOrder:
    def test_order_values_tail_to_head(self):
        """tail=1/len(body), head=1.0."""
        enc = DenseOrderedGlobalEncoder(15, 100)
        body = [(7, 7), (6, 7), (5, 7)]  # head=idx0, tail=idx2
        state = _make_state(body, fruit=(3, 3), direction=0)
        vec = enc.encode(state)

        bsq = 225
        order = vec[bsq : 2 * bsq]
        blen = 3

        # head (idx=0): rank = 3/3 = 1.0
        assert order[_global_idx(7, 7, 15)] == pytest.approx(1.0)
        # middle (idx=1): rank = 2/3
        assert order[_global_idx(6, 7, 15)] == pytest.approx(2.0 / 3.0)
        # tail (idx=2): rank = 1/3
        assert order[_global_idx(5, 7, 15)] == pytest.approx(1.0 / 3.0)

    def test_order_empty_cells_zero(self):
        """Empty cells should have order 0.0."""
        enc = DenseOrderedGlobalEncoder(15, 100)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)
        bsq = 225
        order = vec[bsq : 2 * bsq]
        # A cell that isn't body should have order 0
        assert order[_global_idx(0, 0, 15)] == 0.0


# ---------------------------------------------------------------------------
# 5. DenseOrderedGlobal round-trip (★★★)
# ---------------------------------------------------------------------------

class TestDenseOrderedGlobalRoundtrip:
    def test_handcrafted_roundtrip(self):
        """Encode → decode a handcrafted state with 5 segments."""
        body = [(7, 7), (6, 7), (5, 7), (5, 6), (5, 5)]
        state = _make_state(body, fruit=(10, 10), direction=0)
        enc = DenseOrderedGlobalEncoder(15, 100)
        vec = enc.encode(state)
        decoded = decode_dense_ordered_global(vec, 15, 100)
        assert decoded == state

    def test_rollout_roundtrip(self):
        """Encode → decode hundreds of states from random rollouts."""
        enc = DenseOrderedGlobalEncoder(15, 100)
        env = SnakeEnv(board_size=15, max_length=100, max_steps=500, headless=True, seed=42)
        rng = random.Random(0)
        checked = 0

        for _ in range(200):
            state = env.reset()
            done = False
            while not done:
                vec = enc.encode(state)
                decoded = decode_dense_ordered_global(vec, 15, 100)
                assert decoded == state, (
                    f"Mismatch at step {checked}: {decoded} != {state}"
                )
                checked += 1
                action = rng.randint(0, 2)
                state, _, term, trunc, _ = env.step(action)
                done = term or trunc

        assert checked >= 100, f"Only checked {checked} states"

    def test_two_segment_roundtrip(self):
        """Two-segment initial state should roundtrip."""
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        enc = DenseOrderedGlobalEncoder(15, 100)
        vec = enc.encode(state)
        decoded = decode_dense_ordered_global(vec, 15, 100)
        assert decoded == state


# ---------------------------------------------------------------------------
# 6. DenseOrderedLocal crop verification
# ---------------------------------------------------------------------------

class TestDenseOrderedLocalCrop:
    def test_only_visible_segments_encoded(self):
        """Body segments outside window should not appear in occupancy."""
        enc = DenseOrderedLocalEncoder(15, 100, window_size=5)
        # head at (7,7), window covers (5,5) to (9,9)
        # segment at (3,7) is outside window → should not appear
        body = [(7, 7), (6, 7), (3, 7)]  # tail at (3,7) out of window
        state = _make_state(body, fruit=(10, 10), direction=0)
        vec = enc.encode(state)

        ksq = 25
        occ = vec[:ksq]
        # Only 2 segments visible (head and middle)
        assert occ.sum() == 2.0

    def test_order_channel_visible_only(self):
        """Order values only for visible segments (head=0 is still zero)."""
        enc = DenseOrderedLocalEncoder(15, 100, window_size=5)
        body = [(7, 7), (6, 7), (3, 7)]  # 3 segments, tail out of window
        state = _make_state(body, fruit=(10, 10), direction=0)
        vec = enc.encode(state)

        ksq = 25
        order = vec[ksq : 2 * ksq]
        # head (idx=0): order = 0/100 = 0.0 (same as empty!)
        # middle (idx=1): order = 1/100 = 0.01
        # Only 1 non-zero order value (middle segment; head is 0, tail is outside)
        nonzero = (order > 0).sum()
        assert nonzero == 1
        # Also verify the non-zero value is at the correct position
        # middle segment at (6,7), window origin (5,5), wx=1, wy=2
        mid_pos = _local_idx(1, 2, 5)
        assert order[mid_pos] == pytest.approx(1.0 / 100.0)


# ---------------------------------------------------------------------------
# 7. DenseOrderedLocal wall channel
# ---------------------------------------------------------------------------

class TestDenseOrderedLocalWall:
    def test_corner_head_wall(self):
        """Head at corner — many wall cells."""
        enc = DenseOrderedLocalEncoder(15, 100, window_size=5)
        state = _make_state([(0, 0), (1, 0)], fruit=(14, 14), direction=2)
        vec = enc.encode(state)

        k = 5
        ksq = k * k
        wall = vec[2 * ksq : 3 * ksq]

        half = k // 2  # 2
        expected_wall = 0
        for gy in range(k):
            for gx in range(k):
                ax = gx - half
                ay = gy - half
                if ax < 0 or ay < 0 or ax >= 15 or ay >= 15:
                    expected_wall += 1
        assert wall.sum() == expected_wall

    def test_center_head_no_wall(self):
        """Head at center — no wall cells."""
        enc = DenseOrderedLocalEncoder(15, 100, window_size=5)
        state = _make_state([(7, 7), (6, 7)], fruit=(3, 3), direction=0)
        vec = enc.encode(state)

        ksq = 25
        wall = vec[2 * ksq : 3 * ksq]
        assert wall.sum() == 0.0


# ---------------------------------------------------------------------------
# 8. OccupancyGlobal same encoding for different body order
# ---------------------------------------------------------------------------

class TestOccupancyGlobalAliasing:
    def test_different_order_same_encoding(self):
        """Different body orders with same occupancy + same head → identical encoding.

        Two valid snake bodies forming a 2×2 square, same head position,
        same cells, different body order:
          A: (5,5)→(5,6)→(4,6)→(4,5)   B: (5,5)→(4,5)→(4,6)→(5,6)
        """
        enc = OccupancyGlobalEncoder(15, 100)
        state_a = _make_state([(5, 5), (5, 6), (4, 6), (4, 5)], fruit=(10, 10), direction=0)
        state_b = _make_state([(5, 5), (4, 5), (4, 6), (5, 6)], fruit=(10, 10), direction=0)
        # Same occupancy set {(5,5),(5,6),(4,6),(4,5)}, same head (5,5)
        enc_a = enc.encode(state_a)
        enc_b = enc.encode(state_b)
        assert np.array_equal(enc_a, enc_b), "Same occupancy+head should produce same encoding"


# ---------------------------------------------------------------------------
# 9. OccupancyLocal window-outside ignored
# ---------------------------------------------------------------------------

class TestOccupancyLocalWindowIgnore:
    def test_outside_body_ignored(self):
        """Body changes outside window should not affect encoding."""
        enc = OccupancyLocalEncoder(15, 100, window_size=5)
        # head at (7,7), window covers (5,5)-(9,9)
        # Tail at (6,7) inside vs (3,7) outside → same encoding in window
        state_a = _make_state([(7, 7), (6, 7)], fruit=(10, 10), direction=0)
        state_b = _make_state([(7, 7), (3, 7)], fruit=(10, 10), direction=0)
        enc_a = enc.encode(state_a)
        enc_b = enc.encode(state_b)

        # The tail at (6,7) IS in the window for state_a (origin 5,5 → 6-5=1, 7-5=2)
        # The tail at (3,7) is NOT in the window for state_b
        # So encodings should differ because (6,7) is visible in a but not b
        # Let me fix: use a body where the extra segment is truly outside window
        # Window origin = (7-2, 7-2) = (5, 5)
        # (6,7) → (1,2) → IN window
        # Need tail completely outside: (2,7) → (-3,2) → OUT
        state_c = _make_state([(7, 7), (2, 7)], fruit=(10, 10), direction=0)
        enc_c = enc.encode(state_c)

        # For state_b and state_c, only (7,7) is in window (3,7) and (2,7) both out
        # So enc_b and enc_c should be identical
        assert np.array_equal(enc_b, enc_c), (
            "Body changes outside window should not affect encoding"
        )


# ---------------------------------------------------------------------------
# 10. Aliasing verification
# ---------------------------------------------------------------------------

class TestAliasingVerification:
    def test_dense_ordered_global_no_aliasing(self):
        """DenseOrderedGlobal should have zero aliasing (lossless)."""
        enc = DenseOrderedGlobalEncoder(15, 100)
        env = SnakeEnv(board_size=15, max_length=100, max_steps=500, headless=True, seed=99)
        rng = random.Random(42)
        seen_hashes: set[bytes] = {}
        collisions = 0

        import hashlib
        for _ in range(100):
            state = env.reset()
            done = False
            while not done:
                vec = enc.encode(state)
                h = hashlib.blake2b(vec.tobytes(), digest_size=16).digest()
                if h in seen_hashes and seen_hashes[h] != state:
                    collisions += 1
                seen_hashes[h] = state
                action = rng.randint(0, 2)
                state, _, term, trunc, _ = env.step(action)
                done = term or trunc

        assert collisions == 0, f"DenseOrderedGlobal has {collisions} aliasing collisions"

    def test_occupancy_global_has_aliasing(self):
        """OccupancyGlobal should produce aliasing (non-Markov)."""
        enc = OccupancyGlobalEncoder(15, 100)
        state_a = _make_state([(5, 5), (5, 6), (4, 6), (4, 5)], fruit=(10, 10), direction=0)
        state_b = _make_state([(5, 5), (4, 5), (4, 6), (5, 6)], fruit=(10, 10), direction=0)
        enc_a = enc.encode(state_a)
        enc_b = enc.encode(state_b)
        assert np.array_equal(enc_a, enc_b), "OccupancyGlobal should alias different body orders"
        assert state_a != state_b, "States themselves should differ"


# ---------------------------------------------------------------------------
# 11. Occupancy non-Markov targeted test (★★★)
# ---------------------------------------------------------------------------

class TestOccupancyNonMarkovTargeted:
    def test_occupancy_global_different_next_states(self):
        """Same occupancy encoding, same action → different next states.

        2×2 square bodies, same head (5,5), same cells, different body order:
          A: (5,5)→(5,6)→(4,6)→(4,5)   B: (5,5)→(4,5)→(4,6)→(5,6)

        Stepping straight (action=1) with direction=RIGHT:
          A: new head (6,5), tail (4,5) removed → body {(6,5),(5,5),(5,6),(4,6)}
          B: new head (6,5), tail (5,6) removed → body {(6,5),(5,5),(4,5),(4,6)}
        Different next states → Markov violation.
        """
        state_a = _make_state([(5, 5), (5, 6), (4, 6), (4, 5)], fruit=(10, 10), direction=0)
        state_b = _make_state([(5, 5), (4, 5), (4, 6), (5, 6)], fruit=(10, 10), direction=0)

        enc = OccupancyGlobalEncoder(15, 100)
        enc_a = enc.encode(state_a)
        enc_b = enc.encode(state_b)
        assert np.array_equal(enc_a, enc_b), "Should be aliased"

        env = SnakeEnv(board_size=15, max_length=100, max_steps=2000, headless=True, seed=0)
        env.set_state(state_a)
        ns_a, _, _, _, info_a = env.step(1)  # straight
        env.set_state(state_b)
        ns_b, _, _, _, info_b = env.step(1)  # straight

        assert ns_a != ns_b, (
            "Aliased states should produce different next states for same action "
            "(Markov violation)"
        )


# ---------------------------------------------------------------------------
# 12. make_encoder window_size validation
# ---------------------------------------------------------------------------

class TestMakeEncoderWindowSize:
    @pytest.mark.parametrize("rep", [
        "local", "local_sparse_ordered",
        "dense_ordered_local", "occupancy_local",
    ])
    def test_window_size_required(self, rep):
        """Local representations must have window_size."""
        with pytest.raises(ValueError, match="window_size is required"):
            make_encoder(rep, 15, 100, window_size=None)

    def test_global_reps_no_window_needed(self):
        """Global representations should work without window_size."""
        enc = make_encoder("dense_ordered_global", 15, 100)
        assert isinstance(enc, DenseOrderedGlobalEncoder)

        enc = make_encoder("occupancy_global", 15, 100)
        assert isinstance(enc, OccupancyGlobalEncoder)

    @pytest.mark.parametrize("rep", [
        "dense_ordered_local", "occupancy_local",
    ])
    def test_local_reps_with_window(self, rep):
        """Local representations should work with window_size."""
        enc = make_encoder(rep, 15, 100, window_size=5)
        assert enc.output_dim > 0


# ---------------------------------------------------------------------------
# 13. Checkpoint save/load
# ---------------------------------------------------------------------------

class TestCheckpointConfig:
    @pytest.mark.parametrize("rep,ws", [
        ("dense_ordered_global", None),
        ("dense_ordered_local", 7),
        ("occupancy_global", None),
        ("occupancy_local", 5),
    ])
    def test_config_roundtrip(self, rep, ws, tmp_path):
        """Config with new representation should restore the same encoder."""
        import json
        config = {
            "representation": rep,
            "board_size": 15,
            "max_length": 100,
            "local_window_size": ws,
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
# 14. Smoke test: 10 episodes per representation
# ---------------------------------------------------------------------------

class TestSmokeEpisodes:
    @pytest.mark.parametrize("rep,ws", [
        ("full", None),
        ("structural", None),
        ("local", 7),
        ("dense_ordered_global", None),
        ("dense_ordered_local", 7),
        ("occupancy_global", None),
        ("occupancy_local", 7),
    ])
    def test_ten_episodes(self, rep, ws):
        """Run 10 episodes with each representation — no crashes."""
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
