from dataclasses import dataclass
from typing import Sequence
import collections
import numpy as np

from snake_env import SnakeEnv, SnakeState


@dataclass
class PlannerActionInfo:
    action: int
    immediate_death: bool
    reachable_area: int
    can_reach_tail: bool
    planner_score: float
    q_value: float
    combined_score: float


@dataclass
class PlannerDecision:
    chosen_action: int
    actions: list[PlannerActionInfo]


def clone_env(env: SnakeEnv) -> SnakeEnv:
    """Safely clone the SnakeEnv without copying non-serializable pygame objects."""
    clone = SnakeEnv(
        board_size=env.board_size,
        max_length=env.max_length,
        max_steps=env.max_steps,
        headless=True,
    )
    clone.set_state(env._get_state())
    # Replicate internal mutable variables
    clone._step_count = env._step_count
    clone._score = env._score
    clone._consumed = env._consumed
    clone._last_distance = env._last_distance
    
    # Replicate RNG state so cloned operations are isolated from original sequence
    clone._rng.setstate(env._rng.getstate())
    return clone


class SafetyPlanner:
    def __init__(
        self,
        planner_weight: float = 2.0,
        area_weight: float = 1.0,
        tail_reach_weight: float = 1.0,
    ):
        self.planner_weight = planner_weight
        self.area_weight = area_weight
        self.tail_reach_weight = tail_reach_weight

    def analyze_actions(
        self,
        env: SnakeEnv,
        q_values: Sequence[float],
    ) -> list[PlannerActionInfo]:
        actions_info = []
        board_size = env.board_size

        for action in range(3):
            # 1. Clone environment
            clone = clone_env(env)

            # 2. Simulate step
            next_state, reward, terminated, truncated, info = clone.step(action)

            # Check immediate death
            if terminated and info.get("death_reason") in ("wall", "body"):
                actions_info.append(
                    PlannerActionInfo(
                        action=action,
                        immediate_death=True,
                        reachable_area=0,
                        can_reach_tail=False,
                        planner_score=0.0,
                        q_value=float(q_values[action]),
                        combined_score=-float("inf"),
                    )
                )
            else:
                # Calculate BFS flood fill from the new head
                head = next_state.body[0]
                tail = next_state.body[-1]

                # Blocked cells: snake body cells except tail
                blocked = set(next_state.body)
                if tail in blocked:
                    blocked.discard(tail)

                # BFS queue & visited
                queue = collections.deque([head])
                visited = {head}
                can_reach_tail = False

                while queue:
                    curr = queue.popleft()
                    if curr == tail:
                        can_reach_tail = True

                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = curr[0] + dx, curr[1] + dy
                        neighbor = (nx, ny)

                        if 0 <= nx < board_size and 0 <= ny < board_size:
                            if neighbor not in blocked and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

                reachable_area = len(visited)

                # Calculate planner score
                area_ratio = reachable_area / (board_size * board_size)
                tail_bonus = 1.0 if can_reach_tail else 0.0

                planner_score = (
                    self.area_weight * area_ratio
                    + self.tail_reach_weight * tail_bonus
                )

                combined_score = float(q_values[action]) + self.planner_weight * planner_score

                actions_info.append(
                    PlannerActionInfo(
                        action=action,
                        immediate_death=False,
                        reachable_area=reachable_area,
                        can_reach_tail=can_reach_tail,
                        planner_score=planner_score,
                        q_value=float(q_values[action]),
                        combined_score=combined_score,
                    )
                )
        return actions_info

    def choose_action(
        self,
        env: SnakeEnv,
        q_values: Sequence[float],
    ) -> PlannerDecision:
        actions_info = self.analyze_actions(env, q_values)

        # Decide action
        # If all actions are unsafe, fall back to argmax Q-value
        all_unsafe = all(info.immediate_death for info in actions_info)
        if all_unsafe:
            chosen_action = int(np.argmax(q_values))
        else:
            # Choose safe action with maximum combined score
            combined_scores = [info.combined_score for info in actions_info]
            chosen_action = int(np.argmax(combined_scores))

        return PlannerDecision(chosen_action=chosen_action, actions=actions_info)
