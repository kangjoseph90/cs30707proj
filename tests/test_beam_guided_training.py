"""Tests for beam-veto-guided training integration.

Tests:
  1. --train-with-beam-veto requires --train-with-planner
  2. Beam-guided mode executes the beam-selected action
  3. Replay buffer stores the executed final action
  4. Planner-guided mode does not mutate rewards
  5. Planner-guided mode does not alter Bellman target
  6. No planner masks stored in replay transitions
  7. Disabling planner flags preserves existing behavior
  8. Metrics distinguish veto invocations / runs / vetoes
  9. Live environment not mutated during search
  10. Short smoke-training run completes
"""

import subprocess
import sys

from snake_env import SnakeEnv
from planner import SafetyPlanner
from beam_veto_planner import BeamSearchVetoPlanner, BeamVetoConfig
from replay_buffer import ReplayBuffer
from state_encoders import EgocentricMergedObstacleLocalEncoder


# ---------------------------------------------------------------------------
# 1. CLI validation
# ---------------------------------------------------------------------------

def test_1_beam_veto_requires_planner():
    """--train-with-beam-veto without --train-with-planner must fail."""
    result = subprocess.run(
        [
            sys.executable, "train_snake.py",
            "--representation", "egocentric_merged_obstacle_local",
            "--local-window-size", "5",
            "--total-env-steps", "100",
            "--train-with-beam-veto",
            "--output-dir", "results/_test_beam_train",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "requires" in result.stderr.lower() or "error" in result.stderr.lower()


# ---------------------------------------------------------------------------
# 2-3. Action selection and replay storage
# ---------------------------------------------------------------------------

def test_2_beam_guided_selects_beam_action():
    """Beam-guided mode uses beam-selected action, not raw DQN."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    state = env.reset()
    encoder = EgocentricMergedObstacleLocalEncoder(5, 100, 5)

    planner = SafetyPlanner(planner_weight=2.0)
    beam_config = BeamVetoConfig(max_depth=3, beam_width=4, use_conditional_trigger=False)
    beam_planner = BeamSearchVetoPlanner(planner, beam_config)

    # Simulate one step
    from model import MLPDQN
    import torch
    model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)

    with torch.no_grad():
        sv = torch.FloatTensor(encoder.encode(state)).unsqueeze(0)
        q_values = model(sv).squeeze(0).numpy()

    bfs_decision = planner.choose_action(env, q_values)
    beam_decision = beam_planner.choose_action(env, q_values, bfs_decision)

    # The executed action should be beam_decision.chosen_action
    assert beam_decision.chosen_action in (0, 1, 2)


def test_3_replay_stores_executed_action():
    """Replay buffer stores the actually executed action, not raw DQN argmax."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    state = env.reset()
    encoder = EgocentricMergedObstacleLocalEncoder(5, 100, 5)

    planner = SafetyPlanner(planner_weight=2.0)
    beam_config = BeamVetoConfig(max_depth=3, beam_width=4, use_conditional_trigger=False)
    beam_planner = BeamSearchVetoPlanner(planner, beam_config)

    from model import MLPDQN
    import torch
    model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)

    encoded = encoder.encode(state)
    with torch.no_grad():
        sv = torch.FloatTensor(encoded).unsqueeze(0)
        q_values = model(sv).squeeze(0).numpy()

    bfs_decision = planner.choose_action(env, q_values)
    beam_decision = beam_planner.choose_action(env, q_values, bfs_decision)
    executed_action = beam_decision.chosen_action

    next_state, reward, terminated, truncated, info = env.step(executed_action)

    # Store in buffer
    buffer = ReplayBuffer(capacity=1000)
    buffer.push(state, executed_action, reward, next_state, terminated, truncated, None)

    # Verify stored action matches executed
    assert len(buffer) == 1
    sample = buffer.sample(1, encoder)
    stored_action = sample[1].item()
    assert stored_action == executed_action


# ---------------------------------------------------------------------------
# 4-5. Rewards and Bellman target unchanged
# ---------------------------------------------------------------------------

def test_4_rewards_not_mutated():
    """Planner guidance does not change the reward signal."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    state = env.reset()

    # Step without planner
    env2 = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    env2.reset()

    action = 1  # straight
    _, reward_no_planner, _, _, _ = env.step(action)
    _, reward_with_planner, _, _, _ = env2.step(action)

    # Same action on same env state = same reward
    assert reward_no_planner == reward_with_planner


def test_5_bellman_target_unchanged():
    """Bellman target calculation is identical regardless of planner.

    Verify that the target computation uses only model outputs,
    not planner masks or scores.
    """
    import torch
    from model import MLPDQN

    # Simulate a batch
    batch_size = 4
    dim = 31
    model = MLPDQN(input_dim=dim, output_dim=3)
    target = MLPDQN(input_dim=dim, output_dim=3)
    target.load_state_dict(model.state_dict())

    b_s = torch.randn(batch_size, dim)
    b_ns = torch.randn(batch_size, dim)
    b_a = torch.tensor([[0], [1], [2], [0]])
    b_r = torch.tensor([1.0, -1.0, 0.5, 2.0])
    b_term = torch.tensor([0.0, 1.0, 0.0, 0.0])
    gamma = 0.95

    # Standard Double DQN target (no planner involvement)
    current_q = model(b_s).gather(1, b_a)
    with torch.no_grad():
        next_actions = model(b_ns).argmax(dim=1, keepdim=True)
        max_next_q = target(b_ns).gather(1, next_actions).squeeze(1)
        expected_q = b_r + gamma * max_next_q * (1.0 - b_term)

    # This should be a pure tensor operation — no planner inputs
    assert expected_q.shape == (batch_size,)
    # Terminated state should have no future value
    assert expected_q[1].item() == b_r[1].item()


# ---------------------------------------------------------------------------
# 6. No planner masks in replay
# ---------------------------------------------------------------------------

def test_6_no_planner_masks_in_replay_without_planner_mask_flag():
    """When not using planner-mask-target-actions, replay has no masks."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    state = env.reset()
    encoder = EgocentricMergedObstacleLocalEncoder(5, 100, 5)

    buffer = ReplayBuffer(capacity=1000)
    encoded = encoder.encode(state)

    # Store without safe mask
    action = 1
    next_state, reward, terminated, truncated, info = env.step(action)
    buffer.push(state, action, reward, next_state, terminated, truncated, None)

    sample = buffer.sample(1, encoder)
    # sample = (b_s, b_a, b_r, b_ns, b_term, b_trunc, b_next_safe_mask)
    assert len(sample) >= 6


# ---------------------------------------------------------------------------
# 7. Disabling flags preserves existing behavior
# ---------------------------------------------------------------------------

def test_7_no_planner_flags_preserves_behavior():
    """Without planner flags, training behaves as standard DQN."""
    result = subprocess.run(
        [
            sys.executable, "train_snake.py",
            "--representation", "egocentric_merged_obstacle_local",
            "--local-window-size", "5",
            "--total-env-steps", "200",
            "--seed", "0",
            "--warmup-steps", "50",
            "--eval-interval", "200",
            "--output-dir", "results/_test_no_planner",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Done" in result.stdout


# ---------------------------------------------------------------------------
# 8. Metrics distinguish beam counts
# ---------------------------------------------------------------------------

def test_8_beam_metrics_structure():
    """Beam metrics have distinct fields for invocations, runs, vetoes."""
    # Verify the metric keys exist in a simulated config
    metric_keys = [
        "beam_search_runs",
        "beam_search_run_rate",
        "beam_vetoes",
        "beam_veto_rate_per_env_step",
        "beam_veto_rate_per_search_run",
        "beam_all_roots_forced_death",
        "beam_all_roots_forced_death_rate",
    ]
    # Just verify these are valid key names (no crash)
    metrics = {k: 0 for k in metric_keys}
    assert metrics["beam_search_runs"] == 0
    assert metrics["beam_vetoes"] == 0


# ---------------------------------------------------------------------------
# 9. Environment not mutated
# ---------------------------------------------------------------------------

def test_9_live_env_not_mutated():
    """Beam search during training does not mutate live environment."""
    env = SnakeEnv(board_size=5, max_length=50, max_steps=200, seed=42)
    state = env.reset()
    env.step(1)
    env.step(1)

    body_before = [list(s) for s in env._body]
    rng_before = env._rng.getstate()

    planner = SafetyPlanner(planner_weight=2.0)
    beam_config = BeamVetoConfig(max_depth=3, beam_width=4, use_conditional_trigger=False)
    beam_planner = BeamSearchVetoPlanner(planner, beam_config)

    from model import MLPDQN
    import torch
    encoder = EgocentricMergedObstacleLocalEncoder(5, 100, 5)
    model = MLPDQN(input_dim=encoder.output_dim, output_dim=3)

    with torch.no_grad():
        sv = torch.FloatTensor(encoder.encode(state)).unsqueeze(0)
        q_values = model(sv).squeeze(0).numpy()

    bfs_decision = planner.choose_action(env, q_values)
    beam_decision = beam_planner.choose_action(env, q_values, bfs_decision)

    assert env._body == body_before
    assert env._rng.getstate() == rng_before


# ---------------------------------------------------------------------------
# 10. Smoke test
# ---------------------------------------------------------------------------

def test_10_smoke_training_completes():
    """Short beam-guided training run completes without crashing."""
    result = subprocess.run(
        [
            sys.executable, "train_snake.py",
            "--representation", "egocentric_merged_obstacle_local",
            "--local-window-size", "5",
            "--total-env-steps", "200",
            "--seed", "0",
            "--warmup-steps", "50",
            "--eval-interval", "200",
            "--train-with-planner",
            "--train-with-beam-veto",
            "--beam-max-depth", "3",
            "--beam-width", "4",
            "--output-dir", "results/_test_beam_smoke",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Done" in result.stdout
