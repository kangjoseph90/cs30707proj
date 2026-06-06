"""State representation encoders for Snake.

Encoders that transform a canonical ``SnakeState`` into a flat
float32 vector suitable for an MLP:

Phase 3 (original):
* ``FullStateEncoder``          – one-hot per body segment on the full grid
* ``StructuralStateEncoder``    – lossless compression using body-path directions
* ``LocalStateEncoder``         – ordered local window around the head

Phase 4 (spatial ablation):
* ``DenseOrderedGlobalEncoder`` – dense spatial grid with body order (lossless)
* ``DenseOrderedLocalEncoder``  – dense spatial local window with body order
* ``OccupancyGlobalEncoder``    – dense spatial grid without body order
* ``OccupancyLocalEncoder``     – dense spatial local window without body order
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from snake_env import DELTA, SnakeState, get_relative_dirs

# Direction-to-index map for fast lookup in structural encoder
_DELTA_TO_IDX: dict[tuple[int, int], int] = {
    (1, 0): 0,   # Right
    (0, 1): 1,   # Down
    (-1, 0): 2,  # Left
    (0, -1): 3,  # Up
}


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------

def _global_idx(x: int, y: int, board_size: int) -> int:
    """Flat index for position (x, y) on a board_size×board_size grid."""
    return x * board_size + y


def _local_idx(x: int, y: int, window_size: int) -> int:
    """Flat index for position (x, y) in a K×K window (row-major)."""
    return y * window_size + x


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class StateEncoder(ABC):
    """Encode a ``SnakeState`` into a flat float32 vector."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

    @abstractmethod
    def encode(self, state: SnakeState) -> np.ndarray:
        """Return a float32 vector of length ``self.output_dim``."""
        ...


# ---------------------------------------------------------------------------
# 1. Full State Representation (Phase 3)
# ---------------------------------------------------------------------------

class FullStateEncoder(StateEncoder):
    """One-hot per body-segment position on the full grid + fruit + direction.

    Layout (flat vector):
        body segments : board_size² × max_length   (one-hot, zero-padded)
        fruit         : board_size²                 (one-hot)
        direction     : 4                           (one-hot)
    """

    def __init__(self, board_size: int = 15, max_length: int = 100):
        self.board_size = board_size
        self.max_length = max_length
        self._bsq = board_size * board_size

    @property
    def output_dim(self) -> int:
        return self._bsq * self.max_length + self._bsq + 4

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        bsq = self._bsq
        out = np.zeros(self.output_dim, dtype=np.float32)

        # Body segments
        for i, seg in enumerate(state.body):
            if i >= self.max_length:
                break
            idx = _global_idx(seg[0], seg[1], bs)
            out[i * bsq + idx] = 1.0

        # Fruit
        offset_body = bsq * self.max_length
        out[offset_body + _global_idx(state.fruit[0], state.fruit[1], bs)] = 1.0

        # Direction
        offset_fruit = offset_body + bsq
        out[offset_fruit + state.direction] = 1.0

        return out


# ---------------------------------------------------------------------------
# 2. Structural State Representation (Phase 3)
# ---------------------------------------------------------------------------

class StructuralStateEncoder(StateEncoder):
    """Lossless compression using head/fruit one-hot + body-path directions.

    Layout (flat vector):
        head       : board_size²              (one-hot)
        fruit      : board_size²              (one-hot)
        body_path  : 4 × (max_length - 1)     (direction one-hot, zero-padded)
        direction  : 4                        (one-hot)
    """

    def __init__(self, board_size: int = 15, max_length: int = 100):
        self.board_size = board_size
        self.max_length = max_length
        self._bsq = board_size * board_size

    @property
    def output_dim(self) -> int:
        return self._bsq + self._bsq + 4 * (self.max_length - 1) + 4

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        bsq = self._bsq
        out = np.zeros(self.output_dim, dtype=np.float32)

        # Head
        out[_global_idx(state.body[0][0], state.body[0][1], bs)] = 1.0

        # Fruit
        offset_head = bsq
        out[offset_head + _global_idx(state.fruit[0], state.fruit[1], bs)] = 1.0

        # Body path: direction from segment i to segment i+1
        offset_fruit = 2 * bsq
        max_transitions = self.max_length - 1
        for i in range(min(len(state.body) - 1, max_transitions)):
            dx = state.body[i + 1][0] - state.body[i][0]
            dy = state.body[i + 1][1] - state.body[i][1]
            dir_idx = _DELTA_TO_IDX.get((dx, dy))
            if dir_idx is not None:
                out[offset_fruit + i * 4 + dir_idx] = 1.0
            # If (dx, dy) is not a valid direction (shouldn't happen with
            # adjacent body), leave zeros for that transition.

        # Direction
        offset_path = offset_fruit + max_transitions * 4
        out[offset_path + state.direction] = 1.0

        return out


def decode_structural_state(
    vector: np.ndarray,
    board_size: int,
    max_length: int,
) -> SnakeState:
    """Decode a structural-state vector back to a ``SnakeState``.

    Returns a ``SnakeState`` with ``direction`` and full ``body``.
    ``fruit`` is reconstructed from the fruit one-hot section.
    """
    bs = board_size
    bsq = bs * bs

    # Head
    head_flat = int(np.argmax(vector[:bsq]))
    head = (head_flat // bs, head_flat % bs)

    # Fruit
    fruit_flat = int(np.argmax(vector[bsq : 2 * bsq]))
    fruit = (fruit_flat // bs, fruit_flat % bs)

    # Body path
    offset_path = 2 * bsq
    max_transitions = max_length - 1
    body = [head]
    for i in range(max_transitions):
        sl = vector[offset_path + i * 4 : offset_path + i * 4 + 4]
        if sl.max() == 0.0:
            # No more transitions
            break
        dir_idx = int(np.argmax(sl))
        dx, dy = DELTA[dir_idx]
        prev = body[-1]
        body.append((prev[0] + dx, prev[1] + dy))

    # Direction
    offset_dir = offset_path + max_transitions * 4
    direction = int(np.argmax(vector[offset_dir : offset_dir + 4]))

    return SnakeState(
        body=tuple(body),
        fruit=fruit,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# 3. Local (Ordered) State Representation (Phase 3)
# ---------------------------------------------------------------------------

class LocalStateEncoder(StateEncoder):
    """Ordered local window around the head.

    Layout (flat vector):
        body channels : max_length × K²  (one-hot per segment, zero if out of window)
        wall channel  : K²               (1 for out-of-board, 0 for valid)
        fruit delta   : 2                (normalized dx, dy in [-1, 1])
        direction     : 4                (one-hot)
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
        window_size: int = 9,
    ):
        self.board_size = board_size
        self.max_length = max_length
        self.window_size = window_size
        self._k = window_size
        self._half = window_size // 2

    @property
    def output_dim(self) -> int:
        channels = self.max_length + 1  # body segments + wall
        return channels * self._k * self._k + 2 + 4

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        k = self._k
        half = self._half
        hx, hy = state.head
        origin_x = hx - half
        origin_y = hy - half

        out = np.zeros(self.output_dim, dtype=np.float32)

        # Build a mapping: (x, y) -> segment index  (only if within window)
        seg_map: dict[tuple[int, int], int] = {}
        for i, seg in enumerate(state.body):
            if i >= self.max_length:
                break
            wx = seg[0] - origin_x
            wy = seg[1] - origin_y
            if 0 <= wx < k and 0 <= wy < k:
                seg_map[(wx, wy)] = i

        # Fill body segment channels and wall channel
        wall_channel = self.max_length  # last channel
        for gy in range(k):
            for gx in range(k):
                abs_x = origin_x + gx
                abs_y = origin_y + gy
                pos = _local_idx(gx, gy, k)

                # Wall channel
                if abs_x < 0 or abs_x >= bs or abs_y < 0 or abs_y >= bs:
                    out[wall_channel * k * k + pos] = 1.0
                else:
                    # Body segment at this grid position?
                    seg_idx = seg_map.get((gx, gy))
                    if seg_idx is not None:
                        out[seg_idx * k * k + pos] = 1.0

        # Fruit delta (normalized to [-1, 1])
        offset_grid = (self.max_length + 1) * k * k
        dx = (state.fruit[0] - hx) / (bs - 1)
        dy = (state.fruit[1] - hy) / (bs - 1)
        out[offset_grid] = dx
        out[offset_grid + 1] = dy

        # Direction
        offset_dir = offset_grid + 2
        out[offset_dir + state.direction] = 1.0

        return out


# ---------------------------------------------------------------------------
# 4. Dense Ordered Global Grid (Phase 4)
# ---------------------------------------------------------------------------

class DenseOrderedGlobalEncoder(StateEncoder):
    """Dense grid: body_occupancy + body_order + head + fruit + direction.

    Layout (flat vector):
        body_occupancy_grid : board_size²  (1.0 if body segment exists)
        body_order_grid     : board_size²  (rank_from_tail / len(body))
        head_grid           : board_size²  (1.0 at head)
        fruit_grid          : board_size²  (1.0 at fruit)
        direction           : 4            (one-hot)
    Total: 4 × board_size² + 4

    For a board_size=15: 4 × 225 + 4 = 904d

    Lossless: body occupancy + order → original body order recoverable.

    body_order convention:
        rank_from_tail = len(body) - idx   (tail=1, head=len(body))
        order_value    = rank_from_tail / len(body)
    So tail ≈ 1/length, head = 1.0.  Empty cells have order 0.
    """

    def __init__(self, board_size: int = 15, max_length: int = 100):
        self.board_size = board_size
        self.max_length = max_length
        self._bsq = board_size * board_size

    @property
    def output_dim(self) -> int:
        return 4 * self._bsq + 4

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        bsq = self._bsq
        out = np.zeros(self.output_dim, dtype=np.float32)
        blen = len(state.body)

        # Channel 0: body_occupancy_grid
        # Channel 1: body_order_grid
        for idx, seg in enumerate(state.body):
            gi = _global_idx(seg[0], seg[1], bs)
            out[gi] = 1.0  # occupancy
            rank = blen - idx  # tail=1, head=blen
            out[bsq + gi] = rank / blen  # order

        # Channel 2: head_grid
        out[2 * bsq + _global_idx(state.head[0], state.head[1], bs)] = 1.0

        # Channel 3: fruit_grid
        out[3 * bsq + _global_idx(state.fruit[0], state.fruit[1], bs)] = 1.0

        # Direction
        out[4 * bsq + state.direction] = 1.0

        return out


def decode_dense_ordered_global(
    vector: np.ndarray,
    board_size: int,
    max_length: int,
) -> SnakeState:
    """Decode a DenseOrderedGlobal vector back to a ``SnakeState``.

    Algorithm:
    1. occupancy grid에서 body cell 수집
    2. order grid 값 기준 내림차순 정렬 → head부터 tail 순서 복원
    3. head grid에서 head 위치 확정 (sanity check)
    4. fruit grid에서 fruit 위치 복원
    5. direction 복원
    """
    bs = board_size
    bsq = bs * bs

    # Collect body cells: (x, y, order_value)
    occupancy = vector[:bsq]
    order = vector[bsq : 2 * bsq]

    body_cells = []
    for flat_idx in range(bsq):
        if occupancy[flat_idx] > 0.5:
            x = flat_idx // bs
            y = flat_idx % bs
            body_cells.append((x, y, float(order[flat_idx])))

    # Sort by order descending → head first, tail last
    body_cells.sort(key=lambda c: c[2], reverse=True)
    body = tuple((c[0], c[1]) for c in body_cells)

    # Head grid (sanity check)
    head_flat = int(np.argmax(vector[2 * bsq : 3 * bsq]))
    head_from_grid = (head_flat // bs, head_flat % bs)

    # Fruit grid
    fruit_flat = int(np.argmax(vector[3 * bsq : 4 * bsq]))
    fruit = (fruit_flat // bs, fruit_flat % bs)

    # Direction
    direction = int(np.argmax(vector[4 * bsq : 4 * bsq + 4]))

    # Sanity: head matches
    if body and body[0] != head_from_grid:
        raise ValueError(
            f"Decoded head {body[0]} != head_grid {head_from_grid}"
        )

    return SnakeState(body=body, fruit=fruit, direction=direction)


# ---------------------------------------------------------------------------
# 5. Dense Ordered Local Grid (Phase 4)
# ---------------------------------------------------------------------------

class DenseOrderedLocalEncoder(StateEncoder):
    """Dense local grid: body_occupancy + body_order + wall + fruit_dxdy + direction.

    Layout (flat vector):
        local_body_occupancy : K²  (1.0 if body in window)
        local_body_order     : K²  (index_from_head / max_length, visible only)
        local_wall           : K²  (1.0 for out-of-board)
        fruit_dx_dy          : 2   (normalized)
        direction            : 4   (one-hot)
    Total: 3K² + 6

    K=5:  81d,  K=7: 153d,  K=9: 249d

    body_order convention:
        index_from_head = segment index (0 for head, 1 for neck, ...)
        order_value     = index_from_head / max_length
    So head = 0.0, neck = 0.01, ..., tail ≈ (len-1)/max_length.
    Empty cells have order 0.  Head_grid is a separate channel, so head=0
    does not conflict with empty cells (occupancy channel distinguishes).

    This convention uses a FIXED denominator (max_length) so that the order
    value does NOT leak total body length information for segments outside
    the window — ensuring a fair comparison with other local encoders.
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
        window_size: int = 9,
    ):
        self.board_size = board_size
        self.max_length = max_length
        self.window_size = window_size
        self._k = window_size
        self._half = window_size // 2

    @property
    def output_dim(self) -> int:
        return 3 * self._k * self._k + 6

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        k = self._k
        half = self._half
        hx, hy = state.head
        origin_x = hx - half
        origin_y = hy - half

        out = np.zeros(self.output_dim, dtype=np.float32)

        # Build mapping: window (gx, gy) -> segment index
        seg_map: dict[tuple[int, int], int] = {}
        for idx, seg in enumerate(state.body):
            if idx >= self.max_length:
                break
            gx = seg[0] - origin_x
            gy = seg[1] - origin_y
            if 0 <= gx < k and 0 <= gy < k:
                seg_map[(gx, gy)] = idx

        ksq = k * k

        # Channel 0: body_occupancy, Channel 1: body_order, Channel 2: wall
        for wy in range(k):
            for wx in range(k):
                abs_x = origin_x + wx
                abs_y = origin_y + wy
                pos = _local_idx(wx, wy, k)

                # Wall channel
                if abs_x < 0 or abs_x >= bs or abs_y < 0 or abs_y >= bs:
                    out[2 * ksq + pos] = 1.0
                else:
                    seg_idx = seg_map.get((wx, wy))
                    if seg_idx is not None:
                        # Occupancy
                        out[pos] = 1.0
                        # Order: index_from_head / max_length
                        out[ksq + pos] = seg_idx / self.max_length

        # Fruit dx, dy (normalized)
        offset_grid = 3 * ksq
        out[offset_grid] = (state.fruit[0] - hx) / (bs - 1)
        out[offset_grid + 1] = (state.fruit[1] - hy) / (bs - 1)

        # Direction
        offset_dir = offset_grid + 2
        out[offset_dir + state.direction] = 1.0

        return out


# ---------------------------------------------------------------------------
# 6. Occupancy Global Grid (Phase 4)
# ---------------------------------------------------------------------------

class OccupancyGlobalEncoder(StateEncoder):
    """Global occupancy only, no body order.

    Layout (flat vector):
        body_occupancy_grid : board_size²  (1.0 if body segment exists)
        head_grid           : board_size²  (1.0 at head)
        fruit_grid          : board_size²  (1.0 at fruit)
        direction           : 4            (one-hot)
    Total: 3 × board_size² + 4

    For a board_size=15: 3 × 225 + 4 = 679d

    Non-Markov: same occupancy + different body order → same encoding.
    """

    def __init__(self, board_size: int = 15, max_length: int = 100):
        self.board_size = board_size
        self.max_length = max_length
        self._bsq = board_size * board_size

    @property
    def output_dim(self) -> int:
        return 3 * self._bsq + 4

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        bsq = self._bsq
        out = np.zeros(self.output_dim, dtype=np.float32)

        # Channel 0: body_occupancy_grid
        for seg in state.body:
            out[_global_idx(seg[0], seg[1], bs)] = 1.0

        # Channel 1: head_grid
        out[bsq + _global_idx(state.head[0], state.head[1], bs)] = 1.0

        # Channel 2: fruit_grid
        out[2 * bsq + _global_idx(state.fruit[0], state.fruit[1], bs)] = 1.0

        # Direction
        out[3 * bsq + state.direction] = 1.0

        return out


# ---------------------------------------------------------------------------
# 7. Occupancy Local Grid (Phase 4)
# ---------------------------------------------------------------------------

class OccupancyLocalEncoder(StateEncoder):
    """Local occupancy only, no body order.

    Layout (flat vector):
        local_body_occupancy : K²  (1.0 if body in window)
        local_wall           : K²  (1.0 for out-of-board)
        fruit_dx_dy          : 2   (normalized)
        direction            : 4   (one-hot)
    Total: 2K² + 6

    K=5:  56d,  K=7: 104d,  K=9: 168d

    Non-Markov: window 밖 정보와 몸통 순서를 모두 제거.
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
        window_size: int = 9,
    ):
        self.board_size = board_size
        self.max_length = max_length
        self.window_size = window_size
        self._k = window_size
        self._half = window_size // 2

    @property
    def output_dim(self) -> int:
        return 2 * self._k * self._k + 6

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        k = self._k
        half = self._half
        hx, hy = state.head
        origin_x = hx - half
        origin_y = hy - half

        out = np.zeros(self.output_dim, dtype=np.float32)

        # Build set of body positions in window
        body_set: set[tuple[int, int]] = set()
        for seg in state.body:
            gx = seg[0] - origin_x
            gy = seg[1] - origin_y
            if 0 <= gx < k and 0 <= gy < k:
                body_set.add((gx, gy))

        ksq = k * k

        # Channel 0: body_occupancy, Channel 1: wall
        for wy in range(k):
            for wx in range(k):
                abs_x = origin_x + wx
                abs_y = origin_y + wy
                pos = _local_idx(wx, wy, k)

                # Wall channel
                if abs_x < 0 or abs_x >= bs or abs_y < 0 or abs_y >= bs:
                    out[ksq + pos] = 1.0
                else:
                    # Body at this position?
                    if (wx, wy) in body_set:
                        out[pos] = 1.0

        # Fruit dx, dy (normalized)
        offset_grid = 2 * ksq
        out[offset_grid] = (state.fruit[0] - hx) / (bs - 1)
        out[offset_grid + 1] = (state.fruit[1] - hy) / (bs - 1)

        # Direction
        offset_dir = offset_grid + 2
        out[offset_dir + state.direction] = 1.0

        return out


# ---------------------------------------------------------------------------
# Phase 5: Merged Obstacle + Egocentric Ablation
# ---------------------------------------------------------------------------


class MergedObstacleLocalEncoder(StateEncoder):
    """Merged obstacle (body + wall) in a head-centered local window.

    Layout (flat vector):
        obstacle_grid : K²  (1.0 if body OR wall, 0.0 otherwise)
        fruit_dx_dy   : 2   (normalized)
        direction     : 4   (one-hot)
    Total: K² + 6

    K=5: 31d,  K=9: 87d

    Identical to OccupancyLocalEncoder except body and wall share one channel.
    Uses absolute-coordinate head-centered crop (no heading rotation).
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
        window_size: int = 9,
    ):
        self.board_size = board_size
        self.max_length = max_length
        self.window_size = window_size
        self._k = window_size
        self._half = window_size // 2

    @property
    def output_dim(self) -> int:
        return self._k * self._k + 6

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        k = self._k
        half = self._half
        hx, hy = state.head
        origin_x = hx - half
        origin_y = hy - half

        out = np.zeros(self.output_dim, dtype=np.float32)

        # Build set of body positions in window
        body_set: set[tuple[int, int]] = set()
        for seg in state.body:
            gx = seg[0] - origin_x
            gy = seg[1] - origin_y
            if 0 <= gx < k and 0 <= gy < k:
                body_set.add((gx, gy))

        ksq = k * k

        # Single obstacle channel: body OR wall
        for wy in range(k):
            for wx in range(k):
                abs_x = origin_x + wx
                abs_y = origin_y + wy
                pos = _local_idx(wx, wy, k)

                if abs_x < 0 or abs_x >= bs or abs_y < 0 or abs_y >= bs:
                    out[pos] = 1.0  # wall
                elif (wx, wy) in body_set:
                    out[pos] = 1.0  # body

        # Fruit dx, dy (normalized)
        offset_grid = ksq
        out[offset_grid] = (state.fruit[0] - hx) / (bs - 1)
        out[offset_grid + 1] = (state.fruit[1] - hy) / (bs - 1)

        # Direction
        offset_dir = offset_grid + 2
        out[offset_dir + state.direction] = 1.0

        return out


class EgocentricMergedObstacleLocalEncoder(StateEncoder):
    """Egocentric merged obstacle: heading-normalized rotation.

    Layout (flat vector):
        obstacle_grid : K²  (1.0 if body OR wall, egocentric)
        fruit_dx_dy   : 2   (egocentric, normalized)
        direction     : 4   (one-hot)
    Total: K² + 6

    K=5: 31d,  K=9: 87d

    Same dimensionality as MergedObstacleLocalEncoder for controlled comparison.
    All spatial info is rotated so "forward" maps to grid-up (ego_y < 0).
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
        window_size: int = 9,
    ):
        self.board_size = board_size
        self.max_length = max_length
        self.window_size = window_size
        self._k = window_size
        self._center = window_size // 2

    @property
    def output_dim(self) -> int:
        return self._k * self._k + 6

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        k = self._k
        center = self._center
        hx, hy = state.head
        fx, fy = DELTA[state.direction]  # forward vector

        out = np.zeros(self.output_dim, dtype=np.float32)
        ksq = k * k

        # --- Channel 0: obstacle grid (body OR wall, egocentric) ---

        # Body segments → egocentric positions
        for seg in state.body:
            dx = seg[0] - hx
            dy = seg[1] - hy
            ego_x = -fy * dx + fx * dy
            ego_y = -fx * dx + -fy * dy
            gx = center + ego_x
            gy = center + ego_y
            if 0 <= gx < k and 0 <= gy < k:
                pos = _local_idx(gx, gy, k)
                out[pos] = 1.0

        # Wall cells: reverse-transform each ego grid cell to world
        for gy in range(k):
            for gx in range(k):
                if out[_local_idx(gx, gy, k)] == 1.0:
                    continue  # already body, skip wall check
                ego_x = gx - center
                ego_y = gy - center
                # Ego-to-world inverse transform
                wdx = -fy * ego_x + -fx * ego_y
                wdy = fx * ego_x + -fy * ego_y
                abs_x = hx + wdx
                abs_y = hy + wdy
                if abs_x < 0 or abs_x >= bs or abs_y < 0 or abs_y >= bs:
                    out[_local_idx(gx, gy, k)] = 1.0  # wall

        # --- Fruit dx, dy (egocentric, normalized) ---
        fdx = state.fruit[0] - hx
        fdy = state.fruit[1] - hy
        ego_fruit_x = -fy * fdx + fx * fdy
        ego_fruit_y = -fx * fdx + -fy * fdy

        offset_grid = ksq
        out[offset_grid] = ego_fruit_x / (bs - 1)
        out[offset_grid + 1] = ego_fruit_y / (bs - 1)

        # --- Direction (one-hot) ---
        offset_dir = offset_grid + 2
        out[offset_dir + state.direction] = 1.0

        return out


class MergedOrderedLocalEncoder(StateEncoder):
    """Merged obstacle + body order in a head-centered local window.

    Layout (flat vector):
        obstacle_grid : K²  (1.0 if body OR wall, 0.0 otherwise)
        order_grid    : K²  (seg_idx / max_length for body, 0.0 otherwise)
        fruit_dx_dy   : 2   (normalized)
        direction     : 4   (one-hot)
    Total: 2K² + 6

    K=5: 56d,  K=9: 168d,  K=29: 1688d

    Same as MergedObstacleLocalEncoder with an additional order channel.
    Uses absolute-coordinate head-centered crop (no heading rotation).
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
        window_size: int = 9,
    ):
        self.board_size = board_size
        self.max_length = max_length
        self.window_size = window_size
        self._k = window_size
        self._half = window_size // 2

    @property
    def output_dim(self) -> int:
        return 2 * self._k * self._k + 6

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        k = self._k
        half = self._half
        hx, hy = state.head
        origin_x = hx - half
        origin_y = hy - half
        ml = self.max_length

        out = np.zeros(self.output_dim, dtype=np.float32)
        ksq = k * k

        # --- Channel 0: obstacle grid (body OR wall) ---
        # --- Channel 1: order grid (seg_idx / max_length for body) ---

        # Body segments → obstacle + order
        for seg_idx, seg in enumerate(state.body):
            gx = seg[0] - origin_x
            gy = seg[1] - origin_y
            if 0 <= gx < k and 0 <= gy < k:
                pos = _local_idx(gx, gy, k)
                out[pos] = 1.0  # obstacle
                out[ksq + pos] = seg_idx / ml  # order

        # Wall cells → obstacle only
        for wy in range(k):
            for wx in range(k):
                abs_x = origin_x + wx
                abs_y = origin_y + wy
                pos = _local_idx(wx, wy, k)
                if abs_x < 0 or abs_x >= bs or abs_y < 0 or abs_y >= bs:
                    out[pos] = 1.0  # wall (no order value)

        # Fruit dx, dy (normalized)
        offset_grid = 2 * ksq
        out[offset_grid] = (state.fruit[0] - hx) / (bs - 1)
        out[offset_grid + 1] = (state.fruit[1] - hy) / (bs - 1)

        # Direction
        offset_dir = offset_grid + 2
        out[offset_dir + state.direction] = 1.0

        return out


class EgocentricMergedOrderedLocalEncoder(StateEncoder):
    """Egocentric merged obstacle + body order: heading-normalized rotation.

    Layout (flat vector):
        obstacle_grid : K²  (1.0 if body OR wall, egocentric)
        order_grid    : K²  (seg_idx / max_length for body, egocentric)
        fruit_dx_dy   : 2   (egocentric, normalized)
        direction     : 4   (one-hot)
    Total: 2K² + 6

    K=5: 56d,  K=9: 168d,  K=29: 1688d

    Egocentric version of MergedOrderedLocalEncoder.
    All spatial info is rotated so "forward" maps to grid-up (ego_y < 0).
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
        window_size: int = 9,
    ):
        self.board_size = board_size
        self.max_length = max_length
        self.window_size = window_size
        self._k = window_size
        self._center = window_size // 2

    @property
    def output_dim(self) -> int:
        return 2 * self._k * self._k + 6

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        k = self._k
        center = self._center
        hx, hy = state.head
        ml = self.max_length
        fx, fy = DELTA[state.direction]  # forward vector

        out = np.zeros(self.output_dim, dtype=np.float32)
        ksq = k * k

        # --- Channel 0: obstacle grid (body OR wall, egocentric) ---
        # --- Channel 1: order grid (seg_idx / max_length for body) ---

        # Body segments → egocentric positions
        for seg_idx, seg in enumerate(state.body):
            dx = seg[0] - hx
            dy = seg[1] - hy
            ego_x = -fy * dx + fx * dy
            ego_y = -fx * dx + -fy * dy
            gx = center + ego_x
            gy = center + ego_y
            if 0 <= gx < k and 0 <= gy < k:
                pos = _local_idx(gx, gy, k)
                out[pos] = 1.0  # obstacle
                out[ksq + pos] = seg_idx / ml  # order

        # Wall cells: reverse-transform each ego grid cell to world
        for gy in range(k):
            for gx in range(k):
                if out[_local_idx(gx, gy, k)] == 1.0:
                    continue  # already body
                ego_x = gx - center
                ego_y = gy - center
                wdx = -fy * ego_x + -fx * ego_y
                wdy = fx * ego_x + -fy * ego_y
                abs_x = hx + wdx
                abs_y = hy + wdy
                if abs_x < 0 or abs_x >= bs or abs_y < 0 or abs_y >= bs:
                    out[_local_idx(gx, gy, k)] = 1.0  # wall only

        # --- Fruit dx, dy (egocentric, normalized) ---
        fdx = state.fruit[0] - hx
        fdy = state.fruit[1] - hy
        ego_fruit_x = -fy * fdx + fx * fdy
        ego_fruit_y = -fx * fdx + -fy * fdy

        offset_grid = 2 * ksq
        out[offset_grid] = ego_fruit_x / (bs - 1)
        out[offset_grid + 1] = ego_fruit_y / (bs - 1)

        # --- Direction (one-hot) ---
        offset_dir = offset_grid + 2
        out[offset_dir + state.direction] = 1.0

        return out


class EgocentricMergedReleaseTimeLocalEncoder(StateEncoder):
    """Egocentric merged obstacle + body release-time: heading-normalized rotation.

    Layout (flat vector):
        obstacle_grid    : K²  (1.0 if body OR wall, egocentric)
        release_time_grid : K²  ((distance_from_tail + 1) / max_length for body)
        fruit_dx_dy      : 2   (egocentric, normalized)
        direction        : 4   (one-hot)
    Total: 2K² + 6

    K=5: 56d,  K=9: 168d,  K=29: 1688d

    Release-time semantics:
        tail segment  → distance_from_tail=0 → value = 1/max_length  (small, frees soon)
        head segment  → distance_from_tail=N → value = (N+1)/max_length (large, blocks long)
        wall / empty  → 0.0

    Note: this is the REVERSE ordering of EgocentricMergedOrderedLocalEncoder,
    which uses seg_idx / max_length (head→0, tail→large).
    Here: tail→small, head→large.
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
        window_size: int = 9,
    ):
        self.board_size = board_size
        self.max_length = max_length
        self.window_size = window_size
        self._k = window_size
        self._center = window_size // 2

    @property
    def output_dim(self) -> int:
        return 2 * self._k * self._k + 6

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        k = self._k
        center = self._center
        hx, hy = state.head
        ml = self.max_length
        fx, fy = DELTA[state.direction]  # forward vector
        body_len = len(state.body)

        out = np.zeros(self.output_dim, dtype=np.float32)
        ksq = k * k

        # --- Channel 0: obstacle grid (body OR wall, egocentric) ---
        # --- Channel 1: release-time grid (distance_from_tail + 1) / max_length ---

        # Body segments → egocentric positions
        for seg_idx, seg in enumerate(state.body):
            dx = seg[0] - hx
            dy = seg[1] - hy
            ego_x = -fy * dx + fx * dy
            ego_y = -fx * dx + -fy * dy
            gx = center + ego_x
            gy = center + ego_y
            if 0 <= gx < k and 0 <= gy < k:
                pos = _local_idx(gx, gy, k)
                out[pos] = 1.0  # obstacle
                dist_from_tail = body_len - 1 - seg_idx
                release_steps = dist_from_tail + 1
                out[ksq + pos] = release_steps / ml  # release-time

        # Wall cells: reverse-transform each ego grid cell to world
        for gy in range(k):
            for gx in range(k):
                if out[_local_idx(gx, gy, k)] == 1.0:
                    continue  # already body
                ego_x = gx - center
                ego_y = gy - center
                wdx = -fy * ego_x + -fx * ego_y
                wdy = fx * ego_x + -fy * ego_y
                abs_x = hx + wdx
                abs_y = hy + wdy
                if abs_x < 0 or abs_x >= bs or abs_y < 0 or abs_y >= bs:
                    out[_local_idx(gx, gy, k)] = 1.0  # wall only

        # --- Fruit dx, dy (egocentric, normalized) ---
        fdx = state.fruit[0] - hx
        fdy = state.fruit[1] - hy
        ego_fruit_x = -fy * fdx + fx * fdy
        ego_fruit_y = -fx * fdx + -fy * fdy

        offset_grid = 2 * ksq
        out[offset_grid] = ego_fruit_x / (bs - 1)
        out[offset_grid + 1] = ego_fruit_y / (bs - 1)

        # --- Direction (one-hot) ---
        offset_dir = offset_grid + 2
        out[offset_dir + state.direction] = 1.0

        return out


class BlindSniffEncoder(StateEncoder):
    """Minimal egocentric representation: 3-direction obstacle + fruit delta.

    Layout (flat vector):
        obstacle_left     : 1  (1.0 if wall or body at 1-step left)
        obstacle_straight : 1  (1.0 if wall or body at 1-step forward)
        obstacle_right    : 1  (1.0 if wall or body at 1-step right)
        fruit_dx          : 1  (egocentric, normalized)
        fruit_dy          : 1  (egocentric, normalized)
    Total: 5d

    Heading-based (egocentric) obstacle check with tail-vacating logic.
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
    ):
        self.board_size = board_size
        self.max_length = max_length

    @property
    def output_dim(self) -> int:
        return 5

    def encode(self, state: SnakeState) -> np.ndarray:
        bs = self.board_size
        hx, hy = state.head
        body_set = set(state.body)
        out = np.zeros(5, dtype=np.float32)

        # 3 directions: left, forward, right (relative to heading)
        abs_dirs = get_relative_dirs(state.direction)
        for i, abs_dir in enumerate(abs_dirs):
            dx, dy = DELTA[abs_dir]
            nx, ny = hx + dx, hy + dy
            if nx < 0 or nx >= bs or ny < 0 or ny >= bs:
                out[i] = 1.0  # wall
            elif (nx, ny) in body_set:
                # Tail vacating: if it's the tail and we're not eating fruit
                if (nx, ny) == state.body[-1] and (nx, ny) != state.fruit:
                    out[i] = 0.0
                else:
                    out[i] = 1.0
            else:
                out[i] = 0.0

        # Fruit delta in egocentric frame
        fwd_x, fwd_y = DELTA[state.direction]
        fdx = state.fruit[0] - hx
        fdy = state.fruit[1] - hy
        ego_fruit_x = -fwd_y * fdx + fwd_x * fdy
        ego_fruit_y = -fwd_x * fdx + -fwd_y * fdy
        out[3] = ego_fruit_x / (bs - 1)
        out[4] = ego_fruit_y / (bs - 1)

        return out
