"""Integration: object-search replay over a real ProcTHOR deployment.

Drives the whole package pipeline — deploy an informed search policy, record it
with ``build_rollout_log`` (which captures revealed containers + their true
contents), rebuild the arena with ``SearchReplayEnvironment.from_log``, and
replay a candidate with ``run_search_replay`` — then asserts the load-bearing
properties: the truth is recorded, outcomes resolve from it, and the bounds are
admissible and in deployment units (seconds).

Slow + procthor-gated (the package ``conftest`` skips when the extra is absent).
"""

from __future__ import annotations

import numpy as np
import pytest

from railroad._bindings import Fluent, State
from railroad.core import Fluent as F, get_action_by_name
from railroad.environment.skill import NavigationMoveSkill
from railroad.environment.symbolic import LocationRegistry
from railroad.environment.types import Pose
from railroad.experimental.unknown_search import (
    NavigationConfig,
    UnknownSpaceEnvironment,
)
from railroad.experimental.unknown_search.operators import (
    construct_move_navigable_operator,
    construct_search_at_site_operator,
    construct_search_frontier_operator,
)
from railroad.operators import construct_no_op_operator
from railroad.planner import MCTSPlanner
from railroad.replay import (
    SearchReplayEnvironment,
    build_rollout_log,
    run_search_replay,
)

pytestmark = pytest.mark.slow

SEED = 1089


def _operators(env_ref, site_prob_fn):
    def move_time(robot, a, b):
        return 5.0 if env_ref[0] is None else env_ref[0].estimate_move_time_safe(robot, a, b)

    return [
        construct_move_navigable_operator(move_time),
        construct_search_frontier_operator(
            object_find_prob=lambda r, f, o: 0.5, search_time=20.0
        ),
        construct_search_at_site_operator(
            site_prob_fn, search_time=20.0, container_type="container"
        ),
        construct_no_op_operator(no_op_time=300.0, extra_cost=100.0),
    ]


def _env_kwargs(start, hidden_sites, target):
    pose = Pose(float(start[0]), float(start[1]), 0.0)
    fluents: set[Fluent] = {
        F("at robot1 start_loc"), F("free robot1"), F("revealed start_loc")
    }
    return dict(
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": {"robot1"}, "location": {"start_loc"},
            "container": set(), "frontier": set(), "object": {target},
        },
        robot_initial_poses={"robot1": pose},
        location_registry=LocationRegistry(
            {"start_loc": np.array(start, dtype=float)}
        ),
        hidden_sites=hidden_sites,
        config=NavigationConfig(
            sensor_range=60.0, max_move_action_time=10_000.0,
            interrupt_min_new_cells=30000, interrupt_min_dt=30000.0,
        ),
    )


def _drive(env, goal, prob_for_replay=None, max_iter=60):
    for _ in range(max_iter):
        if goal.evaluate(env.state.fluents):
            return True
        actions = env.get_actions()
        if not actions:
            return False
        name = MCTSPlanner(actions)(
            env.state, goal, max_iterations=2000, c=300, max_depth=20,
            heuristic_multiplier=2,
        )
        if name == "NONE":
            return False
        env.act(get_action_by_name(actions, name))
    return goal.evaluate(env.state.fluents)


def test_procthor_object_search_replay_end_to_end() -> None:
    from railroad.environment.procthor import ProcTHORScene

    scene = ProcTHORScene(seed=SEED)
    true_grid = np.asarray(scene.grid, dtype=float)
    hidden_sites = {
        n: (int(p[0]), int(p[1])) for n, p in scene.locations.items() if n != "start_loc"
    }
    true_obj_locs = scene.object_locations
    start = scene.locations["start_loc"]
    target = sorted({o for objs in true_obj_locs.values() for o in objs})[0]
    true_container = next(c for c, objs in true_obj_locs.items() if target in objs)
    goal = F(f"found {target}")

    # --- Deployment: informed (decisive) policy finds the target. ---
    env_ref: list = [None]
    dep_env = UnknownSpaceEnvironment(
        operators=_operators(
            env_ref, lambda r, loc, o: 1.0 if o in true_obj_locs.get(loc, set()) else 0.0
        ),
        skill_overrides={"move": NavigationMoveSkill},
        true_grid=true_grid,
        true_object_locations=true_obj_locs,
        **_env_kwargs(start, hidden_sites, target),
    )
    env_ref[0] = dep_env
    assert _drive(dep_env, goal), "informed deployment should find the target"

    # --- Record: the true container is captured WITH the target in contents. ---
    log = build_rollout_log(
        dep_env,
        goal_cell=(int(start[0]), int(start[1])),
        robot_starts={"robot1": Pose(float(start[0]), float(start[1]), 0.0)},
        problem_class="object-search",
    )
    assert log.actual_total_cost > 0  # deployment makespan recorded (seconds)
    tc = next((s for s in log.subgoals if s.signature == true_container), None)
    assert tc is not None, "true container not recorded"
    assert target in tc.contents, "recorded container is missing the target object"

    # --- Replay: an informed candidate beelines to the true container. ---
    arena = SearchReplayEnvironment.from_log(log, target_object=target)
    res = run_search_replay(
        arena,
        frontier_find_prob=lambda r, f, o: 0.5,
        container_find_prob=lambda r, loc, o: 1.0 if loc == true_container else 0.05,
        mcts_iterations=2000,
    )

    assert res.goal_reached, "replay should resolve found=True from recorded truth"
    # Outcome came from the recording: the true container resolved as found.
    assert any(loc == true_container and found for loc, _, found in res.search_log)
    # Bounds are admissible and in seconds (comparable to the deployment makespan).
    assert np.isfinite(res.bounds.optimistic_lb)
    assert res.bounds.optimistic_lb <= res.total_cost + 1e-6
    assert res.total_cost > 0
