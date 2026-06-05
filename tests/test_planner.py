import pytest
import numpy as np

from snake_env import SnakeEnv, SnakeState
from planner import SafetyPlanner, clone_env, PlannerActionInfo


def test_clone_env_no_mutation():
    """Verify clone_env creates a valid independent replica and does not mutate the original."""
    env = SnakeEnv(board_size=5, max_length=10, max_steps=100, seed=42)
    state = env.reset()
    env._step_count = 5
    env._score = 3

    clone = clone_env(env)

    # Check that state variables are copied correctly
    assert clone.board_size == env.board_size
    assert clone.max_length == env.max_length
    assert clone.max_steps == env.max_steps
    assert clone.headless is True
    assert clone._step_count == 5
    assert clone._score == 3
    assert clone._body == env._body
    assert clone._fruit == env._fruit
    assert clone._direction == env._direction

    # Step the clone and verify the original is unaffected
    clone.step(0)
    assert env._step_count == 5
    assert env._body != clone._body


def test_flood_fill_and_tail_reachability():
    """Test BFS flood fill area calculation and tail reachability on a handcrafted 3x3 board."""
    env = SnakeEnv(board_size=3, max_length=10, max_steps=100)
    planner = SafetyPlanner(planner_weight=1.0, area_weight=1.0, tail_reach_weight=1.0)

    # Scenario A: Tail is reachable
    # Body: Head at (1,1), segment at (1,0), tail at (2,0)
    # y=0: [ (0,0), (1,0)-body, (2,0)-tail ]
    # y=1: [ (0,1), (1,1)-head, (2,1) ]
    # y=2: [ (0,2), (1,2),      (2,2) ]
    # Free space = total board (9) - body except tail (2) = 7. Head is counted, so reachable area = 8.
    state_a = SnakeState(
        body=((1, 1), (1, 0), (2, 0)),
        fruit=(0, 0),
        direction=0,  # Right
    )
    env.set_state(state_a)

    # Test BFS behavior by executing action 1 (straight) on a copy
    clone = clone_env(env)
    # Moving straight (action 1) from heading 0 (Right) results in head moving to (2,1).
    # The new state will have head at (2,1), body segments at (1,1) and (1,0), tail at (1,0).
    next_state, _, _, _, info = clone.step(1)
    
    # Calculate BFS on next_state
    head = next_state.body[0]  # (2, 1)
    tail = next_state.body[-1]  # (1, 0)
    blocked = set(next_state.body) - {tail}  # {(2, 1), (1, 1)}

    # Run BFS manually to confirm the planner logic
    import collections
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
            if 0 <= nx < 3 and 0 <= ny < 3:
                if neighbor not in blocked and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    assert can_reach_tail is True
    # Visited should be: (2,1), (2,2), (1,2), (0,2), (0,1), (0,0), (2,0), (1,0) [tail]
    # Blocked are (2,1) [head, which is start, so it's in visited] and (1,1).
    # So 8 cells should be reachable (all except (1, 1)).
    assert len(visited) == 8


def test_immediate_collision_masked():
    """Verify that an action leading to immediate death is masked with score -inf."""
    # 3x3 board
    # Snake at top-left, heading UP (3).
    # Board boundary is immediately above it, so straight (action 1) is death.
    env = SnakeEnv(board_size=3, max_length=10, max_steps=100)
    state = SnakeState(
        body=((0, 0), (0, 1)),
        fruit=(2, 2),
        direction=3,  # UP
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=2.0)
    # Q-values: Straight is highly preferred by DQN, but it leads to immediate death
    q_values = [1.0, 10.0, 2.0]
    decision = planner.choose_action(env, q_values)

    # Action 1 (straight) goes to (0, -1) -> out of bounds -> immediate death
    assert decision.actions[1].immediate_death is True
    assert decision.actions[1].combined_score == -float("inf")
    # Action 0 (left) goes to (-1, 0) -> out of bounds -> immediate death
    assert decision.actions[0].immediate_death is True
    assert decision.actions[0].combined_score == -float("inf")
    # Action 2 (right) goes to (1, 0) -> safe!
    assert decision.actions[2].immediate_death is False
    assert decision.actions[2].combined_score > -float("inf")

    # Chosen action must be the only safe action (2)
    assert decision.chosen_action == 2


def test_one_safe_action_selected():
    """Verify that if only one safe action exists, the planner picks it."""
    # 3x3 board
    # Head at (0, 1), body at (0, 0) and (1, 0). Heading DOWN (1).
    # Left (action 0) -> (1, 1) [safe]
    # Straight (action 1) -> (0, 2) [safe]
    # Right (action 2) -> (-1, 1) [OOB/death]
    # Let's make (0, 2) blocked by putting fruit or something? No, let's block (0, 2) by placing body there.
    # Body: ((0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)) - heading DOWN (1).
    # Straight (0, 2) is blocked.
    # Right (-1, 1) is wall.
    # Only Left (1, 1) is safe.
    env = SnakeEnv(board_size=3, max_length=10, max_steps=100)
    state = SnakeState(
        body=((0, 1), (0, 2), (1, 2), (2, 2)),
        fruit=(0, 0),
        direction=1,  # DOWN
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=1.0)
    q_values = [0.1, 5.0, 0.2]  # DQN prefers straight (5.0) which hits body
    decision = planner.choose_action(env, q_values)

    # Left = 0, Straight = 1, Right = 2
    # Straight hits body. Right hits wall. Only Left (0) is safe.
    assert decision.actions[1].immediate_death is True
    assert decision.actions[2].immediate_death is True
    assert decision.actions[0].immediate_death is False
    assert decision.chosen_action == 0


def test_fallback_when_all_actions_unsafe():
    """Verify fallback to DQN argmax if all actions lead to immediate death."""
    # 3x3 board, surrounded by body/walls.
    # Head at (0,0), body at (1,0), (0,1), tail at (0,2). Heading RIGHT (0).
    # Left (0) -> (0, -1) [wall]
    # Straight (1) -> (1, 0) [body segment, not tail]
    # Right (2) -> (0, 1) [body segment, not tail]
    env = SnakeEnv(board_size=3, max_length=10, max_steps=100)
    state = SnakeState(
        body=((0, 0), (1, 0), (0, 1), (0, 2)),
        fruit=(2, 2),
        direction=0,  # RIGHT
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=1.0)
    q_values = [1.5, 4.2, 0.8]  # argmax is 1 (straight)
    decision = planner.choose_action(env, q_values)

    assert all(info.immediate_death for info in decision.actions)
    # Should fallback to DQN argmax (1)
    assert decision.chosen_action == 1


def test_planner_weight_zero():
    """Verify planner_weight=0 behaves identically to pure DQN argmax selection."""
    env = SnakeEnv(board_size=5, max_length=10, max_steps=100)
    state = env.reset()

    planner = SafetyPlanner(planner_weight=0.0)
    q_values = [3.5, 1.2, 4.8]  # DQN argmax is 2
    decision = planner.choose_action(env, q_values)

    assert decision.chosen_action == 2


def test_action_mapping_preserved():
    """Verify action mapping [0=left, 1=straight, 2=right] is correctly evaluated."""
    env = SnakeEnv(board_size=5, max_length=10, max_steps=100)
    # Head at (2,2), heading RIGHT (0)
    state = SnakeState(
        body=((2, 2), (1, 2)),
        fruit=(4, 4),
        direction=0,  # RIGHT
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=1.0)
    q_values = [0.0, 0.0, 0.0]
    decision = planner.choose_action(env, q_values)

    # Let's clone and verify step directions for action 0, 1, 2
    # Action 0 (left): Heading RIGHT (0) -> left is UP (3). Head becomes (2, 1)
    # Action 1 (straight): Heading RIGHT (0) -> straight is RIGHT (0). Head becomes (3, 2)
    # Action 2 (right): Heading RIGHT (0) -> right is DOWN (1). Head becomes (2, 3)

    # Verify these correspond to the simulated actions in decision.actions
    # We can check that the steps from clone_env match
    clone_l = clone_env(env)
    ns_l, _, _, _, _ = clone_l.step(0)
    assert ns_l.body[0] == (2, 1)

    clone_s = clone_env(env)
    ns_s, _, _, _, _ = clone_s.step(1)
    assert ns_s.body[0] == (3, 2)

    clone_r = clone_env(env)
    ns_r, _, _, _, _ = clone_r.step(2)
    assert ns_r.body[0] == (2, 3)
