"""A candidate policy must actually reach the planner.

``lsp-explore`` is parameterized by a frontier-statistics estimator, and
operators are built **once** per environment (``Environment.__init__`` resolves
them; ``get_actions`` re-grounds those same objects forever). So an operator
that closed over the estimator *object* would freeze whichever policy happened
to be installed at construction — the arena would report one policy's beliefs on
the dashboard while planning with another's. These tests pin the indirection
that prevents it (``railroad.lsp.env_mixin._LiveFrontierStatistics``), and the
truth-carrying oracle that a replay candidate needs.
"""

from __future__ import annotations

import numpy as np
import pytest

from railroad.lsp.frontier_statistics import (
    FixedPriorFrontierStatistics,
    OracleFrontierStatistics,
)
from railroad.navigation.constants import FREE_VAL, UNOBSERVED_VAL
from railroad.replay import build_replay_env

from .conftest import build_log_from_ascii, parse_ascii_grid

# One frontier (the gap at row 5) opens into unobserved space that really does
# contain the goal, so an oracle carrying the true map must call it feasible
# while the replay arena's own confinement grid calls it dead.
MAP = """
##########################
#........#???????????????#
#...S....#???????????????#
#........#???????????????#
#........####?????????????
#.............???????????#
#........####?????G??????#
#........#???????????????#
##########################
"""


def _explore_probs(env) -> dict[str, float]:
    """Grounded ``lsp-explore`` branch probabilities, by action name."""
    return {
        action.name: round(float(effect.prob_effects[0][0]), 6)
        for action in env.get_actions()
        if action.name.startswith("lsp-explore")
        for effect in action.effects
        if effect.prob_effects
    }


@pytest.fixture
def log():
    return build_log_from_ascii(MAP)


@pytest.fixture
def true_grid() -> np.ndarray:
    """The real world: the ``?`` region the deployment never saw is open space."""
    grid, _ = parse_ascii_grid(MAP)
    grid[grid == UNOBSERVED_VAL] = FREE_VAL
    return grid


@pytest.mark.parametrize("prob", [0.05, 0.5, 0.95])
def test_applied_policy_drives_explore_probability(log, prob: float) -> None:
    """The candidate's prob_feasible must reach the grounded explore action."""
    env = build_replay_env(log)
    env.apply_policy(FixedPriorFrontierStatistics(prob_feasible=prob))
    probs = _explore_probs(env)
    assert probs, "expected at least one grounded lsp-explore action"
    assert set(probs.values()) == {prob}


def test_policies_differ_on_one_arena(log) -> None:
    """Two candidates must not plan identically.

    If the explore operator read anything other than the currently-installed
    estimator, every candidate in a policy comparison would score the same by
    construction, and the comparison would be meaningless while looking fine.
    """
    env = build_replay_env(log)
    env.apply_policy(FixedPriorFrontierStatistics(0.05))
    cautious = _explore_probs(env)

    env.apply_policy(FixedPriorFrontierStatistics(0.95))
    optimistic = _explore_probs(env)

    assert cautious.keys() == optimistic.keys()
    assert cautious != optimistic


def test_policy_is_reswappable_on_a_built_arena(log) -> None:
    """Applying policies in sequence must keep taking effect, not latch."""
    env = build_replay_env(log)
    seen = []
    for prob in (0.1, 0.9, 0.2, 0.8):
        env.apply_policy(FixedPriorFrontierStatistics(prob))
        seen.append(set(_explore_probs(env).values()))
    assert seen == [{0.1}, {0.9}, {0.2}, {0.8}]


def test_replay_arena_reports_no_oracle(log) -> None:
    """A replay arena must report that it has no oracle.

    Its ``_true_grid`` is a confinement grid, so the inherited ``oracle_labels``
    would label against it and call this frontier infeasible — when in truth the
    goal lies beyond it (see
    ``test_oracle_carrying_true_grid_sees_the_real_world``). Reporting no oracle
    keeps that silently-wrong answer from being served.
    """
    env = build_replay_env(log)
    assert env.oracle_available is False
    assert dict(env.oracle_labels) == {}


def test_oracle_cannot_be_built_without_a_true_map() -> None:
    """The replay arena's own grid must never be mistaken for ground truth.

    Its ``_true_grid`` is a *confinement* grid (unobserved -> wall), so an
    env-sourced oracle would call every frontier dead while still looking like
    an oracle. Requiring the map at construction makes that unwritable.
    """
    with pytest.raises(TypeError):
        OracleFrontierStatistics()  # ty: ignore[missing-argument]


def test_oracle_carrying_true_grid_sees_the_real_world(log, true_grid) -> None:
    """With the scene's true map, the goal-bearing frontier is feasible.

    This is the whole point of the oracle candidate: it is a black box to the
    bound, so it may consult ground truth even though the arena cannot.
    """
    env = build_replay_env(log)
    env.apply_policy(OracleFrontierStatistics(true_grid))
    probs = _explore_probs(env)
    assert probs, "expected at least one grounded lsp-explore action"
    # The goal genuinely lies beyond the frontier, so the oracle backs it fully.
    assert max(probs.values()) == pytest.approx(1.0)
