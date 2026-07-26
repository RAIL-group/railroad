"""Policy builders — one per problem family, each taking only what it needs.

A policy IS an estimator here, and which kind depends on the problem class:
navigation consumes a ``FrontierStatisticsEstimator``, object search an
``ObjectFindEstimator``. The *registry* of which policies a study compares lives
with each experiment (``scripts/replay/*.py``); these tests cover the builders.
"""

from __future__ import annotations

import numpy as np
import pytest

from railroad.navigation.constants import FREE_VAL, UNOBSERVED_VAL
from railroad.experimental.unknown_search import FixedObjectFind
from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics
from railroad.replay import (
    OracleObjectFind,
    build_replay_env,
    constant_frontier_statistics,
    learned_container_find,
    learned_frontier_statistics,
    oracle_frontier_statistics,
    oracle_object_find,
    target_container_cells,
)

from .conftest import build_log_from_ascii, parse_ascii_grid

# Navigation policies that consult no ground truth. Mirrors the tuning the
# experiment scripts register, so drift there shows up here.
TRUTH_FREE_NAV = {
    "optimistic": constant_frontier_statistics(0.9, exploration_cost=8.0),
    "cautious": constant_frontier_statistics(0.3, exploration_cost=20.0),
    "uniform": constant_frontier_statistics(0.5),
    "fixed-prior": FixedPriorFrontierStatistics(prob_feasible=0.5),
}

# The gap at row 5 is the only way into the unobserved right-hand region, which
# really contains both the point goal (G) and, in the search tests, a container.
MAP = """
##########################
#........#???????????????#
#...S....#???????????????#
#........#???????????????#
#........####?????????????
#.............???????????#
#........#####????G??????#
#........#???????????????#
##########################
"""

HIDDEN_CELL = (6, 18)  # where G sits: beyond the frontier, unobserved
VISIBLE_CELL = (2, 4)  # inside the observed room, next to the start


class _FakeScene:
    """The `.grid` / `.locations` / `.object_locations` trio both scenes expose."""

    def __init__(self, grid, locations, object_locations):
        self.grid = grid
        self.locations = locations
        self.object_locations = object_locations


@pytest.fixture
def true_grid() -> np.ndarray:
    """The real world: the region the deployment never saw is open space."""
    grid, _ = parse_ascii_grid(MAP)
    grid[grid == UNOBSERVED_VAL] = FREE_VAL
    return grid


@pytest.fixture
def log():
    return build_log_from_ascii(MAP)


def _explore_probs(env) -> dict[str, float]:
    return {
        action.name: round(float(effect.prob_effects[0][0]), 6)
        for action in env.get_actions()
        if action.name.startswith("lsp-explore")
        for effect in action.effects
        if effect.prob_effects
    }


@pytest.mark.parametrize("name", sorted(TRUTH_FREE_NAV))
def test_constant_navigation_policy_needs_no_scene(name: str) -> None:
    """A navigation policy that consults no ground truth takes no scene."""
    estimator = TRUTH_FREE_NAV[name]
    statistics = estimator.get("robot1", "frontier_1")
    assert 0.0 <= statistics.prob_feasible <= 1.0


@pytest.mark.parametrize("probability", [0.1, 0.5, 0.9])
def test_constant_search_policy_needs_no_scene(probability: float) -> None:
    """Likewise for search: FixedObjectFind is already the whole policy."""
    estimator = FixedObjectFind(probability)
    assert estimator.container_probability("r", "l", "o") == probability
    assert estimator.frontier_probability("r", "f", "o") == probability


def test_learned_navigation_needs_weights() -> None:
    """No trained LSPFrontierNet ships, so navigation must be given one."""
    with pytest.raises(ValueError, match="network_file"):
        learned_frontier_statistics(None)


def test_oracle_reads_truth_straight_off_the_scene(log, true_grid) -> None:
    """The scene is the only argument: it holds every truth the oracle needs."""
    scene = _FakeScene(
        grid=true_grid,
        locations={"start_loc": (2, 4), "goal_loc": (6, 18), "shelf": HIDDEN_CELL},
        object_locations={"shelf": {"book"}},
    )
    estimator = oracle_object_find(scene, target_objects=("book",))

    # Container truth came from scene.object_locations.
    assert estimator.container_probability("robot1", "shelf", "book") == 1.0
    assert estimator.container_probability("robot1", "shelf", "sock") == 0.0

    # The hidden shelf came from scene.locations, so frontiers toward it score.
    env = build_replay_env(log)
    estimator.refresh(env)
    probs = [
        estimator.frontier_probability("robot1", fid, "book") for fid in env.frontiers
    ]
    assert max(probs) == pytest.approx(1.0)


def test_oracle_ignores_containers_without_a_target(true_grid) -> None:
    """A container holding nothing we want must not drive exploration."""
    scene = _FakeScene(
        grid=true_grid,
        locations={"shelf": HIDDEN_CELL, "bin": VISIBLE_CELL},
        object_locations={"shelf": {"sock"}, "bin": {"sock"}},
    )
    estimator = oracle_object_find(scene, target_objects=("book",))
    assert estimator.container_probability("robot1", "shelf", "book") == 0.0


def test_oracle_policy_sees_the_real_world_in_replay(log, true_grid) -> None:
    """The whole point: replay's arena is confined, the oracle candidate is not."""
    scene = _FakeScene(
        grid=true_grid, locations={"goal_loc": (6, 18)}, object_locations={}
    )
    env = build_replay_env(log)
    env.apply_policy(oracle_frontier_statistics(scene))

    probs = _explore_probs(env)
    assert probs, "expected a grounded lsp-explore action"
    # The goal really is beyond that frontier, so the oracle backs it fully —
    # whereas the arena's own confinement grid would call it dead (0.0).
    assert max(probs.values()) == pytest.approx(1.0)


def test_navigation_oracle_builds_no_search_half(true_grid) -> None:
    """A navigation study never constructs object-search machinery.

    A navigation scene has no containers, so there is nothing for a search
    oracle to know — building one would carry empty truth tables and a copy of
    the true grid that nothing ever reads.
    """
    scene = _FakeScene(
        grid=true_grid, locations={"goal_loc": (6, 18)}, object_locations={}
    )
    assert oracle_frontier_statistics(scene) is not None
    # No containers in a navigation scene, so nothing for a search oracle to know.
    assert target_container_cells(scene) == {}


def _oracle_object_find(true_grid, cells) -> OracleObjectFind:
    return OracleObjectFind(true_grid, {"shelf": {"book"}, "bin": {"book"}}, cells)


def test_search_frontier_oracle_backs_a_hidden_container(log, true_grid) -> None:
    """A target container behind the frontier makes exploring it feasible."""
    env = build_replay_env(log)
    estimator = _oracle_object_find(true_grid, {"shelf": HIDDEN_CELL})
    estimator.refresh(env)

    probs = [
        estimator.frontier_probability("robot1", fid, "book") for fid in env.frontiers
    ]
    assert probs, "expected at least one frontier"
    assert max(probs) == pytest.approx(1.0)


def test_oracle_container_belief_is_decisive(true_grid) -> None:
    """Ground truth: 1 where the object is, 0 everywhere else."""
    estimator = _oracle_object_find(true_grid, {"shelf": HIDDEN_CELL})
    assert estimator.container_probability("robot1", "shelf", "book") == 1.0
    assert estimator.container_probability("robot1", "shelf", "sock") == 0.0
    assert estimator.container_probability("robot1", "nowhere", "book") == 0.0


def test_search_frontier_oracle_is_zero_once_targets_are_visible(
    log, true_grid
) -> None:
    """Nothing left to explore toward: a visible container needs no frontier."""
    env = build_replay_env(log)
    estimator = _oracle_object_find(true_grid, {"bin": VISIBLE_CELL})
    estimator.refresh(env)

    probs = [
        estimator.frontier_probability("robot1", fid, "book") for fid in env.frontiers
    ]
    assert probs, "expected at least one frontier"
    assert max(probs) == pytest.approx(0.0)


def test_refresh_replaces_rather_than_accumulates(log, true_grid) -> None:
    """One policy object may serve both roles — this is why.

    A script builds each policy once and hands the same object to the deployment
    environment and then to a replay arena. That is only sound because
    ``refresh`` fully *replaces* an estimator's cache: whatever it learned about
    the previous environment must not survive into the next one. If an estimator
    ever starts accumulating across refreshes, this test fails and the registry
    must go back to building one policy per role.
    """
    estimator = _oracle_object_find(true_grid, {"shelf": HIDDEN_CELL})

    env = build_replay_env(log)
    estimator.refresh(env)
    scored = {
        fid: estimator.frontier_probability("robot1", fid, "book")
        for fid in env.frontiers
    }
    assert max(scored.values()) == pytest.approx(1.0), "expected a live prediction"

    # A different world, with no frontiers at all: every earlier answer must go.
    class _EmptyEnv:
        frontiers: dict = {}
        observed_grid = np.zeros((4, 4), dtype=float)
        goal_cell = (0, 0)

    estimator.refresh(_EmptyEnv())
    assert all(
        estimator.frontier_probability("robot1", fid, "book") == 0.0 for fid in scored
    ), "stale predictions survived a refresh"


@pytest.mark.slow
def test_learned_container_find_is_informative() -> None:
    """The known-map `learned` policy must actually consult its model.

    It is backed by ProcTHOR's trained `FCNNforObjectSearch`, which scores
    (room, container, object) triples. An estimator that ignored the model and
    returned a flat constant would load a checkpoint and then plan identically
    to `uniform`, silently. So assert the belief is genuinely object-conditioned:
    a spread across containers, with the true one above the median — and never
    0/1, which would mean the oracle is wired in by mistake.
    """
    from railroad.environment.procthor import ProcTHORScene

    scene = ProcTHORScene(seed=1089)
    estimator = learned_container_find(scene)

    obj = sorted({o for objs in scene.object_locations.values() for o in objs})[0]
    true_container = next(
        c for c, objs in scene.object_locations.items() if obj in objs
    )
    scored = {
        c: estimator.container_probability("robot1", c, obj)
        for c in scene.object_locations
    }

    assert all(0.0 <= p <= 1.0 for p in scored.values())
    # Informative, not a constant: the spread is what `uniform` lacks.
    assert max(scored.values()) - min(scored.values()) > 0.1
    # Object-conditioned: the true container beats the median container.
    ranked = sorted(scored.values())
    assert scored[true_container] > ranked[len(ranked) // 2]
