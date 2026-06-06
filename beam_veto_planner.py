"""Inference-time beam-search veto planner for Snake.

This module implements a lightweight veto layer that sits on top of the
existing DQN + SafetyPlanner controller.  Its sole role is to detect when
the baseline-selected action is likely to lead to unavoidable death within
a short lookahead horizon and, if so, veto it in favor of a viable
alternative.

It does NOT replace the SafetyPlanner.  It does NOT optimize for maximum
reachable area.  It ONLY vetoes when the baseline action has no viable
continuation.

Trigger strategy: only run beam search when danger signals appear (BFS
override, tail unreachable, low legal actions, tight space).  Baseline
action is evaluated first; remaining actions are only checked if baseline
is likely forced-death (lazy evaluation).
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from planner import SafetyPlanner, PlannerDecision, clone_env
from snake_env import SnakeEnv, DELTA, get_relative_dirs


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BeamVetoConfig:
    """Configuration for the beam-search veto planner."""

    max_depth: int = 6
    beam_width: int = 8

    # Conditional trigger settings
    use_conditional_trigger: bool = True
    trigger_on_bfs_override: bool = True
    trigger_on_tail_unreachable: bool = True
    trigger_on_low_legal_actions: bool = True
    trigger_area_margin: int = 10  # trigger when reachable_area < body_length + margin


@dataclass
class RootActionSearchResult:
    """Result of beam-search evaluation for one root action."""

    action: int
    immediate_death: bool
    likely_forced_death: bool

    max_survival_depth: int
    survived_horizon_count: int
    safe_fruit_reached_count: int

    expanded_nodes: int
    runtime_ms: float


@dataclass
class BeamVetoDecision:
    """Final decision produced by the beam-search veto layer."""

    chosen_action: int
    baseline_action: int
    planner_triggered: bool
    veto_applied: bool
    trigger_reason: str

    root_results: dict[int, RootActionSearchResult]
    runtime_ms: float


# ---------------------------------------------------------------------------
# Internal beam state
# ---------------------------------------------------------------------------


@dataclass
class _BeamState:
    """A single state in the beam."""

    env: SnakeEnv
    survival_depth: int
    safe_fruit_reached: bool
    current_fruit: tuple[int, int]
    terminated: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_state_key(env: SnakeEnv) -> tuple:
    """Create a deterministic hashable key for deduplication."""
    return (
        tuple(tuple(pos) for pos in env._body),
        int(env._direction),
        tuple(env._fruit),
    )


def _bfs_reachable_area(env: SnakeEnv) -> tuple[int, bool]:
    """Compute BFS reachable area and tail reachability from current head.

    Returns (reachable_area, can_reach_tail).
    """
    state = env._get_state()
    head = state.head
    tail = state.body[-1]
    board_size = env.board_size

    blocked = set(state.body)
    blocked.discard(tail)

    queue = collections.deque([head])
    visited = {head}
    _can_reach_tail = False

    while queue:
        curr = queue.popleft()
        if curr == tail:
            _can_reach_tail = True

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = curr[0] + dx, curr[1] + dy
            neighbor = (nx, ny)

            if 0 <= nx < board_size and 0 <= ny < board_size:
                if neighbor not in blocked and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return len(visited), _can_reach_tail


def _count_legal_actions(env: SnakeEnv) -> int:
    """Count how many relative actions are non-immediate-death from current state."""
    state = env._get_state()
    abs_dirs = get_relative_dirs(state.direction)
    head = state.head
    body_set = set(state.body)
    tail_pos = state.body[-1]
    count = 0
    for ad in abs_dirs:
        dx, dy = DELTA[ad]
        nx, ny = head[0] + dx, head[1] + dy
        in_bounds = 0 <= nx < env.board_size and 0 <= ny < env.board_size
        if in_bounds:
            is_tail = ((nx, ny) == tail_pos)
            if (nx, ny) not in body_set or is_tail:
                count += 1
    return count


def _beam_rank_key(
    bs: _BeamState,
    body_length: int,
    max_q_value: float,
) -> tuple:
    """Lexicographic ranking key for beam pruning.

    Order (higher is better):
      1. Safe fruit reached (bool → int)
      2. Longer survival depth
      3. At least one legal next action (has_legal_action)
      4. Tail reachable
      5. Sufficient reachable area (capped at body_length + 10)
      6. DQN max Q-value as tie-breaker
    """
    if bs.terminated:
        return (0, bs.survival_depth, 0, 0, 0, max_q_value)

    reachable_area, can_reach_tail = _bfs_reachable_area(bs.env)

    # Check if at least one next action is non-immediate-death
    has_legal_action = _count_legal_actions(bs.env) > 0

    sufficient_area = min(reachable_area, body_length + 10)

    return (
        int(bs.safe_fruit_reached),
        bs.survival_depth,
        int(has_legal_action),
        int(can_reach_tail),
        sufficient_area,
        max_q_value,
    )


# ---------------------------------------------------------------------------
# BeamSearchVetoPlanner
# ---------------------------------------------------------------------------


class BeamSearchVetoPlanner:
    """Inference-time beam-search veto layer.

    Parameters
    ----------
    safety_planner : SafetyPlanner
        The existing one-step safety planner whose decision is the baseline.
    config : BeamVetoConfig
        Configuration for beam search parameters and trigger conditions.
    """

    def __init__(
        self,
        safety_planner: SafetyPlanner,
        config: BeamVetoConfig | None = None,
    ):
        self.safety_planner = safety_planner
        self.config = config or BeamVetoConfig()

    # ------------------------------------------------------------------
    # Conditional trigger
    # ------------------------------------------------------------------

    def _should_trigger(
        self,
        env: SnakeEnv,
        bfs_decision: PlannerDecision,
        dqn_argmax: int,
    ) -> tuple[bool, str]:
        """Decide whether beam search should run and return the reason."""
        cfg = self.config

        if not cfg.use_conditional_trigger:
            return True, "always_on"

        # Condition 1: BFS overrode DQN argmax
        if cfg.trigger_on_bfs_override and bfs_decision.chosen_action != dqn_argmax:
            return True, "bfs_override"

        # Get baseline action info
        chosen_info = next(
            info for info in bfs_decision.actions
            if info.action == bfs_decision.chosen_action
        )

        if chosen_info.immediate_death:
            return True, "baseline_death"

        # Condition 2: Tail unreachable after baseline action
        if cfg.trigger_on_tail_unreachable and not chosen_info.can_reach_tail:
            return True, "tail_unreachable"

        # Condition 3: Legal action count ≤ 1 after baseline action
        if cfg.trigger_on_low_legal_actions:
            clone = clone_env(env)
            clone.step(bfs_decision.chosen_action)
            legal_count = _count_legal_actions(clone)
            if legal_count <= 1:
                return True, "low_legal_actions"

        # Condition 4: Reachable area < body_length + margin
        if cfg.trigger_area_margin >= 0:
            body_length = len(env._get_state().body)
            if chosen_info.reachable_area < body_length + cfg.trigger_area_margin:
                return True, "low_reachable_area"

        return False, "not_triggered"

    # ------------------------------------------------------------------
    # Single root-action beam search
    # ------------------------------------------------------------------

    def _evaluate_root_action(
        self,
        env: SnakeEnv,
        action: int,
        current_fruit: tuple[int, int],
    ) -> RootActionSearchResult:
        """Run beam search from one candidate root action."""
        t0 = time.perf_counter()
        cfg = self.config
        expanded_nodes = 0

        # 1. Clone and simulate root action
        clone = clone_env(env)
        next_state, reward, terminated, truncated, info = clone.step(action)

        # Immediate death check
        if terminated and info.get("death_reason") in ("wall", "body"):
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return RootActionSearchResult(
                action=action,
                immediate_death=True,
                likely_forced_death=True,
                max_survival_depth=0,
                survived_horizon_count=0,
                safe_fruit_reached_count=0,
                expanded_nodes=1,
                runtime_ms=elapsed_ms,
            )

        # 2. Initialise beam with the single resulting state
        ate_fruit = info.get("ate_fruit", False)
        safe_fruit = False
        if ate_fruit:
            # Check post-fruit viability: at least one non-immediate-death action
            if _count_legal_actions(clone) > 0:
                safe_fruit = True

        initial = _BeamState(
            env=clone,
            survival_depth=1 if not terminated else 0,
            safe_fruit_reached=safe_fruit,
            current_fruit=current_fruit,
            terminated=terminated or truncated,
        )
        beam: list[_BeamState] = [initial]
        expanded_nodes += 1

        max_survival = initial.survival_depth
        survived_horizon = 0
        safe_fruit_count = 1 if safe_fruit else 0

        # 3. Expand depth by depth
        for depth in range(1, cfg.max_depth):
            if not beam:
                break

            next_beam: list[_BeamState] = []
            seen_keys: set[tuple] = set()

            for bs_state in beam:
                if bs_state.terminated:
                    continue

                # Expand 3 relative actions
                for rel_action in range(3):
                    child_clone = clone_env(bs_state.env)
                    ns, r, term, trunc, child_info = child_clone.step(rel_action)
                    expanded_nodes += 1

                    child_terminated = term or trunc
                    child_ate_fruit = child_info.get("ate_fruit", False)
                    child_safe_fruit = False

                    if child_ate_fruit and not child_terminated:
                        if _count_legal_actions(child_clone) > 0:
                            child_safe_fruit = True

                    child_depth = bs_state.survival_depth + 1
                    child_sf = bs_state.safe_fruit_reached or child_safe_fruit

                    child = _BeamState(
                        env=child_clone,
                        survival_depth=child_depth,
                        safe_fruit_reached=child_sf,
                        current_fruit=bs_state.current_fruit,
                        terminated=child_terminated,
                    )

                    # Deduplication (only for non-terminated states)
                    if not child_terminated:
                        key = _canonical_state_key(child_clone)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                    next_beam.append(child)

                    # Track stats
                    if not child_terminated:
                        max_survival = max(max_survival, child_depth)
                    if child_depth >= cfg.max_depth and not child_terminated:
                        survived_horizon += 1
                    if child_safe_fruit:
                        safe_fruit_count += 1

            # Prune: keep only top beam_width states
            if len(next_beam) > cfg.beam_width:
                body_length = len(env._get_state().body)
                next_beam.sort(
                    key=lambda s: _beam_rank_key(s, body_length, 0.0),
                    reverse=True,
                )
                next_beam = next_beam[: cfg.beam_width]

            beam = next_beam

        # Also count survivors from final beam
        for bs_state in beam:
            if not bs_state.terminated and bs_state.survival_depth >= cfg.max_depth:
                survived_horizon += 1
            if bs_state.safe_fruit_reached:
                safe_fruit_count += 1

        # Determine viability (immediate death already returned above)
        likely_forced_death = (survived_horizon == 0 and safe_fruit_count == 0)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return RootActionSearchResult(
            action=action,
            immediate_death=False,
            likely_forced_death=likely_forced_death,
            max_survival_depth=max_survival,
            survived_horizon_count=survived_horizon,
            safe_fruit_reached_count=safe_fruit_count,
            expanded_nodes=expanded_nodes,
            runtime_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def choose_action(
        self,
        env: SnakeEnv,
        q_values: Sequence[float],
        bfs_decision: PlannerDecision,
    ) -> BeamVetoDecision:
        """Decide whether to veto the baseline SafetyPlanner action.

        Uses lazy evaluation: checks baseline action first. Only evaluates
        remaining actions if baseline is likely forced-death.

        Parameters
        ----------
        env : SnakeEnv
            The live environment (will NOT be mutated).
        q_values : Sequence[float]
            Raw DQN Q-values for the 3 actions.
        bfs_decision : PlannerDecision
            The SafetyPlanner's baseline decision.

        Returns
        -------
        BeamVetoDecision
        """
        t0 = time.perf_counter()
        q_values = list(q_values)
        dqn_argmax = int(np.argmax(q_values))
        baseline_action = bfs_decision.chosen_action

        # --- Check trigger conditions ---
        triggered, trigger_reason = self._should_trigger(
            env, bfs_decision, dqn_argmax,
        )

        if not triggered:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return BeamVetoDecision(
                chosen_action=baseline_action,
                baseline_action=baseline_action,
                planner_triggered=False,
                veto_applied=False,
                trigger_reason=trigger_reason,
                root_results={},
                runtime_ms=elapsed_ms,
            )

        # --- Lazy evaluation: check baseline first ---
        current_fruit = tuple(env._fruit)
        baseline_result = self._evaluate_root_action(
            env, baseline_action, current_fruit
        )
        root_results: dict[int, RootActionSearchResult] = {baseline_action: baseline_result}

        # If baseline action is viable, keep it unchanged — no need to check others
        if not baseline_result.likely_forced_death:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return BeamVetoDecision(
                chosen_action=baseline_action,
                baseline_action=baseline_action,
                planner_triggered=True,
                veto_applied=False,
                trigger_reason=trigger_reason,
                root_results=root_results,
                runtime_ms=elapsed_ms,
            )

        # --- Baseline is likely forced-death → evaluate remaining actions ---
        for action in range(3):
            if action == baseline_action:
                continue
            root_results[action] = self._evaluate_root_action(
                env, action, current_fruit
            )

        # Pick the highest SafetyPlanner combined-score among viable alternatives
        viable_actions = []
        for info in bfs_decision.actions:
            if not info.immediate_death:
                result = root_results[info.action]
                if not result.likely_forced_death:
                    viable_actions.append(info)

        if viable_actions:
            best = max(viable_actions, key=lambda a: a.combined_score)
            chosen = best.action
        else:
            # All likely forced-death → fall back to original SafetyPlanner action
            chosen = baseline_action

        veto_applied = (chosen != baseline_action)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return BeamVetoDecision(
            chosen_action=chosen,
            baseline_action=baseline_action,
            planner_triggered=True,
            veto_applied=veto_applied,
            trigger_reason=trigger_reason,
            root_results=root_results,
            runtime_ms=elapsed_ms,
        )
