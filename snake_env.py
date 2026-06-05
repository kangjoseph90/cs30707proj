"""Snake environment with gym-like API for state representation experiments.

Board size: 15x15, max snake length: 100.
Action space: 3 relative actions (left, straight, right).
Reward: +10 apple, -10 death, distance delta (symmetric +1/-1/0).
Episode truncation at max_steps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Direction indices: 0=Right, 1=Down, 2=Left, 3=Up
DELTA: list[tuple[int, int]] = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIRECTION_COUNT = 4
ACTION_COUNT = 3  # left, straight, right


def opposite(dir: int) -> int:
    """Return the opposite direction."""
    return (dir + 2) % 4


def get_relative_dirs(heading: int) -> list[int]:
    """Return [left_abs, forward_abs, right_abs] for the given heading."""
    return [(heading - 1) % 4, heading, (heading + 1) % 4]


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# SnakeState – immutable snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SnakeState:
    """Canonical, hashable game-state snapshot.

    ``score`` is not included; it is derivable as ``len(body) - 2``
    and is reported through ``info["score"]`` by the environment.
    """

    body: tuple[tuple[int, int], ...]  # head-first, tail-last
    fruit: tuple[int, int]
    direction: int  # 0-3

    @property
    def head(self) -> tuple[int, int]:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


# ---------------------------------------------------------------------------
# SnakeEnv
# ---------------------------------------------------------------------------

class SnakeEnv:
    """Snake game environment with gym-like step/reset API.

    Parameters
    ----------
    board_size : int
        Width and height of the square board (default 15).
    max_length : int
        Maximum snake length (default 100).  Episode ends when reached.
    max_steps : int
        Episode truncation limit (default 2000).
    headless : bool
        If True (default), pygame is not initialised and ``render()`` is no-op.
    seed : int | None
        RNG seed for reproducible fruit placement.
    """

    def __init__(
        self,
        board_size: int = 15,
        max_length: int = 100,
        max_steps: int = 2000,
        headless: bool = True,
        seed: Optional[int] = None,
    ):
        self.board_size = board_size
        self.max_length = max_length
        self.max_steps = max_steps
        self.headless = headless
        self._rng = random.Random(seed)

        # Pygame state (lazy init)
        self._pygame = None
        self._screen = None
        self._font = None
        self._clock = None

        # Mutable game state
        self._body: list[list[int]] = []       # list of [x, y]
        self._fruit: list[int] = [0, 0]
        self._direction: int = 0
        self._score: int = 0
        self._step_count: int = 0
        self._consumed: bool = False
        self._last_distance: int = 0
        self._board: Optional[np.ndarray] = None  # occupancy grid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> SnakeState:
        """Reset the environment and return the initial state."""
        bs = self.board_size
        cx, cy = bs // 2, bs // 2

        # Two adjacent, non-overlapping segments
        self._body = [[cx, cy], [cx - 1, cy]]
        self._direction = 0  # RIGHT
        self._score = 0
        self._step_count = 0
        self._consumed = False

        # Occupancy grid
        self._board = np.zeros((bs, bs), dtype=np.int32)
        self._board[cx][cy] += 1
        self._board[cx - 1][cy] += 1

        self._gen_fruit()
        self._last_distance = manhattan(
            (self._body[0][0], self._body[0][1]),
            (self._fruit[0], self._fruit[1]),
        )
        return self._get_state()

    def step(self, action: int) -> tuple[SnakeState, float, bool, bool, dict]:
        """Execute one step.

        Parameters
        ----------
        action : int
            0=left, 1=straight, 2=right (relative to current heading).

        Returns
        -------
        next_state : SnakeState
            On termination the pre-step state is returned as next_state
            (the snake head may be out of bounds).
        reward : float
        terminated : bool   – True on wall/body collision or max_length.
        truncated : bool    – True on max_steps timeout.
        info : dict
        """
        # Snapshot state before this step (used as next_state on termination)
        pre_step_state = self._get_state()

        if not (0 <= action <= 2):
            raise ValueError(f"action must be 0, 1, or 2, got {action}")

        bs = self.board_size
        self._step_count += 1
        ate_fruit = False

        # Map relative action to absolute direction
        abs_dirs = get_relative_dirs(self._direction)
        self._direction = abs_dirs[action]

        # Compute new head position
        dx, dy = DELTA[self._direction]
        head = self._body[0]
        new_head = [head[0] + dx, head[1] + dy]

        # Insert new head
        self._body.insert(0, new_head)

        # Check wall collision
        hit_wall = (
            new_head[0] < 0 or new_head[0] >= bs
            or new_head[1] < 0 or new_head[1] >= bs
        )

        if hit_wall:
            self._body.pop(0)  # Remove OOB head to keep internal state valid
            reward = self._compute_reward(dead=True)
            rc = {"fruit": 0.0, "death": reward, "distance": 0.0}
            terminated = True
            truncated = False
            death_reason = "wall"
            next_state = pre_step_state  # safe: head out of bounds not encoded
        else:
            self._board[new_head[0], new_head[1]] += 1

            # Check apple
            if new_head[0] == self._fruit[0] and new_head[1] == self._fruit[1]:
                ate_fruit = True
                self._consumed = True
                self._score += 1
                rc = {"fruit": 10.0, "death": 0.0, "distance": 0.0}
                # Check max_length reached
                if len(self._body) >= self.max_length:
                    if np.any(self._board == 0):
                        self._gen_fruit()  # place fruit for state consistency
                    self._consumed = False  # reset after using
                    self._last_distance = manhattan(
                        (self._body[0][0], self._body[0][1]),
                        (self._fruit[0], self._fruit[1]),
                    )
                    reward = 10.0
                    terminated = True
                    truncated = False
                    death_reason = "max_length"
                    next_state = self._get_state()
                else:
                    self._gen_fruit()
                    self._consumed = False  # reset after using
                    self._last_distance = manhattan(
                        (self._body[0][0], self._body[0][1]),
                        (self._fruit[0], self._fruit[1]),
                    )
                    reward = 10.0
                    terminated = False
                    truncated = False
                    death_reason = "none"
                    next_state = self._get_state()
                    # Check truncation
                    if self._step_count >= self.max_steps:
                        truncated = True
                        death_reason = "timeout"
            else:
                tail = self._body.pop()
                self._board[tail[0], tail[1]] -= 1

                # Check self-collision
                hit_body = self._board[new_head[0], new_head[1]] > 1

                if hit_body:
                    # Rollback: undo head insert + tail pop
                    self._body.pop(0)
                    self._board[new_head[0], new_head[1]] -= 1
                    self._body.append(tail)
                    self._board[tail[0], tail[1]] += 1
                    reward = self._compute_reward(dead=True)
                    rc = {"fruit": 0.0, "death": reward, "distance": 0.0}
                    terminated = True
                    truncated = False
                    death_reason = "body"
                    next_state = pre_step_state  # safe: pre-collision state
                else:
                    reward = self._compute_reward(dead=False)
                    rc = {"fruit": 0.0, "death": 0.0, "distance": reward}
                    terminated = False
                    truncated = False
                    death_reason = "none"
                    next_state = self._get_state()
                    # Check truncation
                    if self._step_count >= self.max_steps:
                        truncated = True
                        death_reason = "timeout"

        info = {
            "score": self._score,
            "survived_steps": self._step_count,
            "death_reason": death_reason,
            "ate_fruit": ate_fruit,
            "reward_components": rc,
        }
        return next_state, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # State restoration (for analysis)
    # ------------------------------------------------------------------

    def set_state(self, state: SnakeState) -> None:
        """Restore environment to a given canonical state.

        Used for counterfactual analysis.  After calling this the env is
        ready for ``step()`` calls starting from *state*.

        Body segments outside the board are filtered out for safety.
        """
        bs = self.board_size
        self._body = [
            [s[0], s[1]] for s in state.body
            if 0 <= s[0] < bs and 0 <= s[1] < bs
        ]
        if not self._body:
            # Fallback: shouldn't happen with valid states
            self._body = [[bs // 2, bs // 2]]
        self._fruit = [state.fruit[0], state.fruit[1]]
        self._direction = state.direction
        self._score = max(0, len(self._body) - 2)
        self._step_count = 0
        self._consumed = False

        # Rebuild board occupancy
        self._board = np.zeros((bs, bs), dtype=np.int32)
        for seg in self._body:
            self._board[seg[0], seg[1]] += 1

        self._last_distance = manhattan(
            (self._body[0][0], self._body[0][1]),
            (self._fruit[0], self._fruit[1]),
        )

    # ------------------------------------------------------------------
    # Pure transition helper (for counterfactual analysis)
    # ------------------------------------------------------------------

    def simulate_transition(
        self, state: SnakeState, action: int,
    ) -> tuple[SnakeState, float, bool, dict]:
        """Simulate a single step from *state* without modifying ``self``.

        Returns ``(next_state, reward, terminated, info)``.
        ``info`` includes ``ate_fruit`` and ``death_reason`` but not
        ``survived_steps`` (always 1).

        Uses a temporary env copy so the caller's env is untouched.
        """
        tmp = SnakeEnv.__new__(SnakeEnv)
        tmp.board_size = self.board_size
        tmp.max_length = self.max_length
        tmp.max_steps = self.max_steps
        tmp.headless = True
        tmp._rng = random.Random(0)  # deterministic fruit placement
        tmp._pygame = None
        tmp._screen = None
        tmp._font = None
        tmp._clock = None
        tmp.set_state(state)
        # Copy fruit position from state (don't let set_state regenerate)
        # set_state already sets _fruit from state.fruit, so it's fine.
        ns, r, term, trunc, info = tmp.step(action)
        return ns, r, term, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Render the current state (no-op in headless mode)."""
        if self.headless:
            return
        self._ensure_pygame()
        pg = self._pygame
        ppb = 40

        self._screen.fill((0, 0, 0))
        for seg in self._body:
            rect = pg.Rect(seg[0] * ppb + 1, seg[1] * ppb + 1, ppb - 2, ppb - 2)
            pg.draw.rect(self._screen, (255, 255, 255), rect)
        rect = pg.Rect(
            self._fruit[0] * ppb + 1, self._fruit[1] * ppb + 1, ppb - 2, ppb - 2
        )
        pg.draw.rect(self._screen, (255, 0, 0), rect)
        pg.display.flip()
        self._clock.tick(60)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> SnakeState:
        return SnakeState(
            body=tuple((s[0], s[1]) for s in self._body),
            fruit=(self._fruit[0], self._fruit[1]),
            direction=self._direction,
        )

    def _compute_reward(self, dead: bool) -> float:
        if self._consumed:
            self._consumed = False
            self._last_distance = manhattan(
                (self._body[0][0], self._body[0][1]),
                (self._fruit[0], self._fruit[1]),
            )
            return 10.0
        if dead:
            return -10.0
        now_dist = manhattan(
            (self._body[0][0], self._body[0][1]),
            (self._fruit[0], self._fruit[1]),
        )
        delta = self._last_distance - now_dist  # +1 closer, -1 farther, 0 same
        self._last_distance = now_dist
        return float(delta)

    def _gen_fruit(self) -> None:
        """Place a fruit on a random empty cell."""
        bs = self.board_size
        empty = []
        for x in range(bs):
            for y in range(bs):
                if self._board[x][y] == 0:
                    empty.append((x, y))
        if not empty:
            raise RuntimeError("No empty cell for fruit placement")
        fx, fy = self._rng.choice(empty)
        self._fruit = [fx, fy]

    def _ensure_pygame(self) -> None:
        if self._pygame is not None:
            return
        import os
        os.environ["SDL_VIDEODRIVER"] = ""
        import pygame
        pygame.init()
        ppb = 40
        bs = self.board_size
        self._pygame = pygame
        self._screen = pygame.display.set_mode((bs * ppb, bs * ppb), pygame.DOUBLEBUF)
        pygame.display.set_caption("Snake RL")
        self._font = pygame.font.SysFont("consolas", 30, True, False)
        self._clock = pygame.time.Clock()
