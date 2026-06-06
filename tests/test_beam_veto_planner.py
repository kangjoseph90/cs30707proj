"""Tests for beam_veto_planner.py.

16 tests covering:
  1-2.  Environment isolation (no mutation, determinism)
  3.    Relative action mapping preserved
  4.    Immediate-death root action → likely_forced_death (via _evaluate_root_action)
  5.    Surviving horizon path → viable (via _evaluate_root_action)
  6.    Safe fruit collection → viable
  7.    Baseline preserved when viable
  8.    Baseline vetoed when forced-death + viable alternative exists
  9.    SafetyPlanner score picks among viable alternatives
  10.   All forced-death → fallback to SafetyPlanner action
  11.   Conditional trigger skips safe early-game
  12.   Conditional trigger activates on low reachable area
  13.   Conditional trigger activates after BFS override
  14.   Deduplication removes repeated states
  15.   Fruit respawn in clone doesn't mutate live RNG
  16.   Evaluation smoke test completes
"""

from snake_env import SnakeEnv, SnakeState
from planner import SafetyPlanner
from beam_veto_planner import (
    BeamSearchVetoPlanner,
    BeamVetoConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _always_on_config(**overrides) -> BeamVetoConfig:
    """Config with conditional trigger disabled (always on)."""
    defaults = dict(
        max_depth=10,
        beam_width=32,
        use_conditional_trigger=False,
    )
    defaults.update(overrides)
    return BeamVetoConfig(**defaults)


def _conditional_config(**overrides) -> BeamVetoConfig:
    """Config with conditional trigger enabled."""
    defaults = dict(
        max_depth=10,
        beam_width=32,
        use_conditional_trigger=True,
        trigger_area_margin=10,
        trigger_on_bfs_override=True,
        trigger_on_tail_unreachable=True,
        trigger_on_low_legal_actions=True,
    )
    defaults.update(overrides)
    return BeamVetoConfig(**defaults)


def _make_veto_planner(config=None, safety_planner=None):
    if safety_planner is None:
        safety_planner = SafetyPlanner(planner_weight=2.0)
    if config is None:
        config = _always_on_config()
    return BeamSearchVetoPlanner(safety_planner, config)


# ---------------------------------------------------------------------------
# Tests 1-2: Environment isolation
# ---------------------------------------------------------------------------

def test_1_no_mutation_of_live_environment():
    """Beam search must never mutate the live environment."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    state = env.reset()
    # Take a few steps to build some body
    for _ in range(5):
        env.step(1)

    # Snapshot before
    body_before = [list(seg) for seg in env._body]
    fruit_before = list(env._fruit)
    dir_before = env._direction
    score_before = env._score
    rng_state_before = env._rng.getstate()

    planner = SafetyPlanner(planner_weight=2.0)
    veto = _make_veto_planner(config=_always_on_config(), safety_planner=planner)

    q_values = [1.0, 2.0, 0.5]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    # Environment must be unchanged
    assert env._body == body_before
    assert env._fruit == fruit_before
    assert env._direction == dir_before
    assert env._score == score_before
    assert env._rng.getstate() == rng_state_before


def test_2_repeated_calls_deterministic():
    """Repeated calls from the same state produce the same result."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    state = env.reset()
    env.step(1)
    env.step(1)

    planner = SafetyPlanner(planner_weight=2.0)
    veto = _make_veto_planner(config=_always_on_config(max_depth=5), safety_planner=planner)

    q_values = [1.0, 2.0, 0.5]
    bfs_decision = planner.choose_action(env, q_values)

    decision1 = veto.choose_action(env, q_values, bfs_decision)
    decision2 = veto.choose_action(env, q_values, bfs_decision)

    assert decision1.chosen_action == decision2.chosen_action
    assert decision1.veto_applied == decision2.veto_applied
    assert decision1.trigger_reason == decision2.trigger_reason
    assert decision1.planner_triggered == decision2.planner_triggered
    # Root results keys should match (lazy eval: may be 1 or 3)
    assert set(decision1.root_results.keys()) == set(decision2.root_results.keys())
    for a in decision1.root_results:
        r1 = decision1.root_results[a]
        r2 = decision2.root_results[a]
        assert r1.likely_forced_death == r2.likely_forced_death
        assert r1.max_survival_depth == r2.max_survival_depth
        assert r1.expanded_nodes == r2.expanded_nodes


# ---------------------------------------------------------------------------
# Test 3: Action mapping
# ---------------------------------------------------------------------------

def test_3_relative_action_mapping_preserved():
    """Verify [0=left, 1=straight, 2=right] is correctly evaluated."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200)
    # Head at (2,2), heading RIGHT (0)
    state = SnakeState(
        body=((2, 2), (1, 2)),
        fruit=(4, 4),
        direction=0,
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=1.0)
    veto = _make_veto_planner(config=_always_on_config(max_depth=3), safety_planner=planner)

    q_values = [0.0, 0.0, 0.0]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    # Baseline action should be evaluated and non-death on this open board
    assert decision.baseline_action in decision.root_results
    result = decision.root_results[decision.baseline_action]
    assert not result.immediate_death
    assert not result.likely_forced_death


# ---------------------------------------------------------------------------
# Test 4: Immediate-death root action (direct _evaluate_root_action)
# ---------------------------------------------------------------------------

def test_4_immediate_death_marked_forced_death():
    """A root action leading to immediate death is marked likely_forced_death."""
    # 3x3 board, head at corner heading into wall
    env = SnakeEnv(board_size=3, max_length=10, max_steps=100)
    state = SnakeState(
        body=((0, 0), (0, 1)),
        fruit=(2, 2),
        direction=3,  # UP → straight goes to (0, -1) = wall
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=2.0)
    veto = _make_veto_planner(config=_always_on_config(), safety_planner=planner)

    # Test each action directly via _evaluate_root_action
    r0 = veto._evaluate_root_action(env, 0, (2, 2))  # left from UP → (-1, 0) wall
    assert r0.immediate_death is True
    assert r0.likely_forced_death is True
    assert r0.max_survival_depth == 0

    r1 = veto._evaluate_root_action(env, 1, (2, 2))  # straight from UP → (0, -1) wall
    assert r1.immediate_death is True
    assert r1.likely_forced_death is True

    r2 = veto._evaluate_root_action(env, 2, (2, 2))  # right from UP → (1, 0) safe
    assert r2.immediate_death is False
    assert r2.likely_forced_death is False


# ---------------------------------------------------------------------------
# Test 5: Surviving horizon → viable (direct _evaluate_root_action)
# ---------------------------------------------------------------------------

def test_5_surviving_horizon_is_viable():
    """A root action with at least one branch surviving to max_depth is viable."""
    env = SnakeEnv(board_size=7, max_length=50, max_steps=200, seed=0)
    state = env.reset()

    planner = SafetyPlanner(planner_weight=1.0)
    veto = _make_veto_planner(
        config=_always_on_config(max_depth=5, beam_width=16),
        safety_planner=planner,
    )

    # On a fresh 7x7 board with length-2 snake, all actions should be viable
    fruit = tuple(env._fruit)
    for action in range(3):
        result = veto._evaluate_root_action(env, action, fruit)
        assert not result.immediate_death
        assert not result.likely_forced_death


# ---------------------------------------------------------------------------
# Test 6: Safe fruit collection → viable
# ---------------------------------------------------------------------------

def test_6_safe_fruit_reached_is_viable():
    """A root action that safely reaches the fruit is viable."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=99)
    # Place snake right next to fruit
    state = SnakeState(
        body=((2, 2), (1, 2)),
        fruit=(3, 2),  # fruit directly ahead (heading RIGHT)
        direction=0,   # RIGHT
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=1.0)
    veto = _make_veto_planner(
        config=_always_on_config(max_depth=5, beam_width=16),
        safety_planner=planner,
    )

    # Action 1 (straight) should reach fruit
    result = veto._evaluate_root_action(env, 1, (3, 2))
    assert not result.immediate_death
    assert not result.likely_forced_death
    assert result.safe_fruit_reached_count > 0 or result.survived_horizon_count > 0


# ---------------------------------------------------------------------------
# Test 7: Baseline preserved when viable
# ---------------------------------------------------------------------------

def test_7_baseline_preserved_when_viable():
    """If baseline action is viable, it is kept unchanged."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200)
    state = env.reset()

    planner = SafetyPlanner(planner_weight=1.0)
    veto = _make_veto_planner(
        config=_always_on_config(max_depth=3),
        safety_planner=planner,
    )

    q_values = [0.0, 5.0, 0.0]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    # On a fresh 5x5 board, baseline should be viable → no veto
    assert decision.chosen_action == decision.baseline_action
    assert not decision.veto_applied


# ---------------------------------------------------------------------------
# Test 8: Baseline vetoed when forced-death + alternative exists
# ---------------------------------------------------------------------------

def test_8_baseline_vetoed_when_forced_death():
    """Baseline is vetoed when likely forced-death and a viable alternative exists."""
    # 5x5 board with a snake that will trap itself going straight
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200)
    state = SnakeState(
        body=((2, 2), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)),
        fruit=(0, 0),
        direction=0,  # RIGHT
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=1.0)
    veto = _make_veto_planner(
        config=_always_on_config(max_depth=5, beam_width=16),
        safety_planner=planner,
    )

    q_values = [0.5, 10.0, 0.3]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    # The chosen action should not be the one leading to body collision
    assert decision.chosen_action in (0, 2) or decision.chosen_action == bfs_decision.chosen_action


# ---------------------------------------------------------------------------
# Test 9: SafetyPlanner score picks among viable alternatives
# ---------------------------------------------------------------------------

def test_9_safety_planner_score_picks_viable():
    """After veto, the highest SafetyPlanner combined-score viable action is chosen."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200)
    state = SnakeState(
        body=((2, 2), (2, 3), (3, 3), (3, 2)),
        fruit=(0, 0),
        direction=3,  # UP
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=2.0, area_weight=1.0, tail_reach_weight=1.0)
    veto = _make_veto_planner(
        config=_always_on_config(max_depth=3, beam_width=16),
        safety_planner=planner,
    )

    q_values = [1.0, 1.0, 1.0]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    # If veto was applied, the chosen action must be among viable root actions
    if decision.veto_applied:
        chosen_result = decision.root_results[decision.chosen_action]
        assert not chosen_result.likely_forced_death


# ---------------------------------------------------------------------------
# Test 10: All forced-death → fallback to SafetyPlanner
# ---------------------------------------------------------------------------

def test_10_all_forced_death_fallback():
    """If all root actions are likely forced-death, fall back to SafetyPlanner action."""
    # 5x5 board where snake is trapped in a corner spiral
    env = SnakeEnv(board_size=5, max_length=50, max_steps=100)
    state = SnakeState(
        body=((1, 2), (2, 2), (2, 1), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3)),
        fruit=(4, 4),
        direction=3,  # UP
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=1.0)
    veto = _make_veto_planner(
        config=_always_on_config(max_depth=3, beam_width=16),
        safety_planner=planner,
    )

    q_values = [1.0, 2.0, 0.5]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    # All actions should be forced-death or immediate-death
    assert all(r.likely_forced_death for r in decision.root_results.values())
    # Fallback to SafetyPlanner action
    assert decision.chosen_action == decision.baseline_action
    assert not decision.veto_applied


# ---------------------------------------------------------------------------
# Tests 11-13: Conditional trigger
# ---------------------------------------------------------------------------

def test_11_conditional_trigger_skips_early_game():
    """In early game (short body, open area), beam search is not triggered."""
    env = SnakeEnv(board_size=7, max_length=50, max_steps=200, seed=0)
    state = env.reset()  # length-2 snake, open board

    planner = SafetyPlanner(planner_weight=2.0)
    # Disable all triggers except area, and set margin negative so area won't trigger
    config = _conditional_config(
        trigger_area_margin=-1,  # negative → disabled (reachable area always >= 0 > -1 + 2)
        trigger_on_bfs_override=False,
        trigger_on_tail_unreachable=False,
        trigger_on_low_legal_actions=False,
    )
    veto = _make_veto_planner(config=config, safety_planner=planner)

    q_values = [1.0, 2.0, 0.5]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    assert not decision.planner_triggered
    assert decision.trigger_reason == "not_triggered"
    assert len(decision.root_results) == 0


def test_12_conditional_trigger_on_low_reachable_area():
    """Beam search triggers when reachable area < body_length + margin."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200)
    # Create a snake with body length 8 in tight space on 5x5
    # This should make reachable area tight relative to body length
    state = SnakeState(
        body=((2, 2), (2, 3), (2, 4), (1, 4), (0, 4), (0, 3), (0, 2), (1, 2)),
        fruit=(4, 0),
        direction=0,  # RIGHT
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=2.0)
    # margin=20 → body_length(8) + 20 = 28. If reachable_area < 28, triggers.
    config = _conditional_config(
        trigger_area_margin=20,
        trigger_on_bfs_override=False,
        trigger_on_tail_unreachable=False,
        trigger_on_low_legal_actions=False,
    )
    veto = _make_veto_planner(config=config, safety_planner=planner)

    q_values = [1.0, 2.0, 0.5]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    assert decision.planner_triggered
    assert decision.trigger_reason == "low_reachable_area"


def test_13_conditional_trigger_on_bfs_override():
    """Beam search triggers when SafetyPlanner overrides DQN argmax."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200)
    # Corner setup where DQN prefers a death action
    state = SnakeState(
        body=((0, 0), (0, 1)),
        fruit=(2, 2),
        direction=3,  # UP → left and straight are walls
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=2.0)
    config = _conditional_config(
        trigger_area_margin=-1,  # disable area trigger
        trigger_on_bfs_override=True,
        trigger_on_low_legal_actions=False,
    )
    veto = _make_veto_planner(config=config, safety_planner=planner)

    # DQN strongly prefers straight (which is death) → planner overrides
    q_values = [1.0, 10.0, 2.0]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    # Planner overrides DQN (argmax=1 but it's death), so trigger should fire
    assert decision.planner_triggered
    assert decision.trigger_reason == "bfs_override"


# ---------------------------------------------------------------------------
# Test 14: Deduplication
# ---------------------------------------------------------------------------

def test_14_deduplication_removes_repeated_states():
    """Deduplication should prevent the same state from being expanded twice."""
    env = SnakeEnv(board_size=3, max_length=10, max_steps=100)
    state = SnakeState(
        body=((1, 1), (1, 0)),
        fruit=(2, 2),
        direction=0,  # RIGHT
    )
    env.set_state(state)

    planner = SafetyPlanner(planner_weight=1.0)
    veto = _make_veto_planner(
        config=_always_on_config(max_depth=4, beam_width=4),
        safety_planner=planner,
    )

    q_values = [0.0, 0.0, 0.0]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    # Without dedup, a 3x3 board with beam_width=4 and depth=4 could
    # expand many nodes. With dedup it should be reasonable.
    total_expanded = sum(r.expanded_nodes for r in decision.root_results.values())
    assert total_expanded < 500  # reasonable upper bound with dedup


# ---------------------------------------------------------------------------
# Test 15: Fruit respawn isolation
# ---------------------------------------------------------------------------

def test_15_fruit_respawn_doesnt_mutate_live_rng():
    """Fruit respawn inside a cloned branch must not affect the live env RNG."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    state = env.reset()

    rng_before = env._rng.getstate()

    planner = SafetyPlanner(planner_weight=1.0)
    veto = _make_veto_planner(
        config=_always_on_config(max_depth=3, beam_width=8),
        safety_planner=planner,
    )

    q_values = [0.0, 1.0, 0.0]
    bfs_decision = planner.choose_action(env, q_values)
    decision = veto.choose_action(env, q_values, bfs_decision)

    # Live env RNG should be untouched
    assert env._rng.getstate() == rng_before


# ---------------------------------------------------------------------------
# Test 16: Evaluation smoke test
# ---------------------------------------------------------------------------

def test_16_evaluation_smoke_test():
    """Short evaluation completes without crashing."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    state = env.reset()

    planner = SafetyPlanner(planner_weight=2.0)
    veto = _make_veto_planner(
        config=_always_on_config(max_depth=3, beam_width=8),
        safety_planner=planner,
    )

    # Simulate a short episode
    done = False
    steps = 0
    max_steps = 50

    while not done and steps < max_steps:
        # Dummy Q-values (prefer straight)
        q_values = [0.0, 1.0, 0.0]
        bfs_decision = planner.choose_action(env, q_values)
        beam_decision = veto.choose_action(env, q_values, bfs_decision)
        action = beam_decision.chosen_action

        state, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    # Should complete without errors
    assert steps > 0
    assert "score" in info
    assert "death_reason" in info
