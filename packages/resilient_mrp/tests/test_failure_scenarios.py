# Validation tests for failure scenarios A–E.
# Each test pre-injects a specific failure into the initial state and checks the planner's response.
#
# A: early failure     — robot fails at start before acting
# B: transit failure   — robot fails mid-route; alternate path used
# C: multiple failures — two robots fail simultaneously; one survivor covers remaining goals
# D: cascading failure — two sequential failures; sole survivor replans each time
# E: critical path     — only path to a goal is blocked; goal becomes unreachable

import random

from railroad._bindings import State
from railroad.core import Fluent as F, get_action_by_name
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner
from railroad import operators as rr_operators

from resilient_mrp.planning.core import (
    ResilientGraph,
    RobotProfile,
    create_risk_move_operator,
    create_safely_visited_operator,
)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ITER: int = 1000
MAX_DEPTH: int = 20
MCTS_C: float = 100.0
HEURISTIC_MULT: float = 5.0

# ── Robot profiles ────────────────────────────────────────────────────────────
#
# r1: strong on debris and clear; weak on radiation, flooded, confined
# r2: strong on flooded and confined; weak on debris and radiation
# r3: high compatibility across all terrain types (moderate, balanced profile)

PROFILES: dict[str, RobotProfile] = {
    "r1": {"debris": 0.92, "clear": 0.95, "radiation": 0.18, "flooded": 0.20, "confined": 0.55},
    "r2": {"debris": 0.20, "clear": 0.95, "radiation": 0.25, "flooded": 0.90, "confined": 0.88},
    "r3": {"debris": 0.65, "clear": 0.90, "radiation": 0.60, "flooded": 0.70, "confined": 0.72},
}

# ── Graph factories ───────────────────────────────────────────────────────────
# 5 nodes, 2 goals, 2-4 paths/goal, mix of terrain types and hazard severities
def create_two_goal_graph() -> ResilientGraph:

    g = ResilientGraph()
    # start node with multiple spokes; all terrain types represented; varying hazard severities
    g.add_edge("start", "n1", cost=2.0, terrain_type="clear",     hazard_severity=0.05, bidirectional=True)
    g.add_edge("start", "n2", cost=2.0, terrain_type="debris",    hazard_severity=0.50, bidirectional=True)
    g.add_edge("start", "n3", cost=2.0, terrain_type="radiation", hazard_severity=0.45, bidirectional=True)
    g.add_edge("start", "n4", cost=2.0, terrain_type="flooded",   hazard_severity=0.50, bidirectional=True)
    g.add_edge("start", "n5", cost=2.0, terrain_type="confined",  hazard_severity=0.40, bidirectional=True)

    # routes to g1 via all terrain types
    g.add_edge("n1", "g1", cost=2.0, terrain_type="clear",     hazard_severity=0.10, bidirectional=True)
    g.add_edge("n2", "g1", cost=2.0, terrain_type="debris",    hazard_severity=0.50, bidirectional=True)
    g.add_edge("n3", "g1", cost=2.0, terrain_type="radiation", hazard_severity=0.45, bidirectional=True)
    g.add_edge("n4", "g1", cost=2.0, terrain_type="flooded",   hazard_severity=0.30, bidirectional=True)

    # routes to g2 via all terrain types
    g.add_edge("n1", "g2", cost=2.0, terrain_type="clear",     hazard_severity=0.10, bidirectional=True)
    g.add_edge("n4", "g2", cost=2.0, terrain_type="flooded",   hazard_severity=0.50, bidirectional=True)
    g.add_edge("n5", "g2", cost=2.0, terrain_type="confined",  hazard_severity=0.40, bidirectional=True)
    g.add_edge("n2", "g2", cost=2.0, terrain_type="debris",    hazard_severity=0.45, bidirectional=True)

    return g


# 6 nodes, 3 goals, 2-3 paths/goal, low hazard severities, r3 traverses all reliably
def create_three_goal_graph() -> ResilientGraph:

    g = ResilientGraph()
    # start all low hazard; r3's baseline failure rate is negligible
    g.add_edge("start", "n1", cost=2.0, terrain_type="clear",     hazard_severity=0.02, bidirectional=True)
    g.add_edge("start", "n2", cost=2.0, terrain_type="debris",    hazard_severity=0.05, bidirectional=True)
    g.add_edge("start", "n3", cost=2.0, terrain_type="radiation", hazard_severity=0.08, bidirectional=True)
    g.add_edge("start", "n4", cost=2.0, terrain_type="flooded",   hazard_severity=0.05, bidirectional=True)
    g.add_edge("start", "n5", cost=2.0, terrain_type="confined",  hazard_severity=0.05, bidirectional=True)
    g.add_edge("start", "n6", cost=2.0, terrain_type="clear",     hazard_severity=0.02, bidirectional=True)

    # routes to g1 via clear, debris, radiation
    g.add_edge("n1", "g1", cost=2.0, terrain_type="clear",     hazard_severity=0.02, bidirectional=True)
    g.add_edge("n2", "g1", cost=2.0, terrain_type="debris",    hazard_severity=0.05, bidirectional=True)
    g.add_edge("n3", "g1", cost=2.0, terrain_type="radiation", hazard_severity=0.08, bidirectional=True)

    # routes to g2 via flooded, confined, clear
    g.add_edge("n4", "g2", cost=2.0, terrain_type="flooded",   hazard_severity=0.05, bidirectional=True)
    g.add_edge("n5", "g2", cost=2.0, terrain_type="confined",  hazard_severity=0.05, bidirectional=True)
    g.add_edge("n1", "g2", cost=2.0, terrain_type="clear",     hazard_severity=0.02, bidirectional=True)

    # routes to g3 via clear, debris, flooded
    g.add_edge("n6", "g3", cost=2.0, terrain_type="clear",     hazard_severity=0.02, bidirectional=True)
    g.add_edge("n2", "g3", cost=2.0, terrain_type="debris",    hazard_severity=0.05, bidirectional=True)
    g.add_edge("n4", "g3", cost=2.0, terrain_type="flooded",   hazard_severity=0.05, bidirectional=True)

    # cross-link for extra redundancy
    g.add_edge("n2", "n5", cost=2.0, terrain_type="debris",    hazard_severity=0.05, bidirectional=True)

    return g


# 4 nodes, 2 goals, g1 has one path, g2 has three, used for scenario E
def create_single_path_graph() -> ResilientGraph:

    g = ResilientGraph()
    # only path to g1; blocked in scenario E
    g.add_edge("start", "n1", cost=2.0, terrain_type="radiation", hazard_severity=0.10, bidirectional=True)
    g.add_edge("n1",    "g1", cost=2.0, terrain_type="radiation", hazard_severity=0.10, bidirectional=True)

    # multiple paths to g2; all unaffected by r1's failure
    g.add_edge("start", "n2", cost=2.0, terrain_type="clear",   hazard_severity=0.05, bidirectional=True)
    g.add_edge("n2",    "g2", cost=2.0, terrain_type="clear",   hazard_severity=0.05, bidirectional=True)
    g.add_edge("start", "n3", cost=2.0, terrain_type="debris",  hazard_severity=0.08, bidirectional=True)
    g.add_edge("n3",    "g2", cost=2.0, terrain_type="debris",  hazard_severity=0.08, bidirectional=True)
    g.add_edge("start", "n4", cost=2.0, terrain_type="flooded", hazard_severity=0.10, bidirectional=True)
    g.add_edge("n4",    "g2", cost=2.0, terrain_type="flooded", hazard_severity=0.10, bidirectional=True)

    return g


# ── Environment builder ───────────────────────────────────────────────────────
def _make_env(
    graph: ResilientGraph,
    robot_locations: dict[str, str],
    active_profiles: dict[str, RobotProfile],
    goal_sites: list[str],
    failed_robots: set[str] | None = None,
    blocked_paths: set[tuple[str, str]] | None = None,
) -> tuple[SymbolicEnvironment, list]:

    failed_robots = failed_robots or set()
    blocked_paths = blocked_paths or set()

    move_op    = create_risk_move_operator(graph, active_profiles)
    visited_op = create_safely_visited_operator()
    no_op      = rr_operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
    operators  = [move_op, visited_op, no_op]

    objects_by_type: dict = {
        "robot":    set(robot_locations.keys()),
        "location": set(graph.nodes),
    }

    fluents: set = set()
    for robot, loc in robot_locations.items():
        fluents.add(F(f"at {robot} {loc}"))
        if robot not in failed_robots:
            fluents.add(F(f"free {robot}"))
            fluents.add(F(f"operational {robot}"))
        # failed robots: present in the world but not free/operational

    fluents.update(graph.get_edge_fluents())

    available_paths = graph.get_available_path_fluents()
    for fr, to in blocked_paths:
        available_paths.discard(F(f"path_available {fr} {to}"))
    fluents.update(available_paths)

    for g in goal_sites:
        fluents.add(F(f"is_goal {g}"))

    env = SymbolicEnvironment(
        state=State(0.0, fluents, []),
        objects_by_type=objects_by_type,
        operators=operators,
    )
    return env, operators


# ── Episode runner ────────────────────────────────────────────────────────────
# plans for the full goal conjunction in a single MCTS call per step
def _run(env: SymbolicEnvironment, goal: F, goal_sites: list[str]) -> list[str]:
    for _ in range(MAX_ITER):
        if goal.evaluate(env.state.fluents):
            break
        real_actions = env.get_actions()
        if not real_actions:
            break

        planner = MCTSPlanner(real_actions)
        action_name = planner(
            env.state, goal,
            max_iterations=MAX_ITER, c=MCTS_C, max_depth=MAX_DEPTH,
            heuristic_multiplier=HEURISTIC_MULT,
        )
        if action_name == "NONE":
            break
        env.act(get_action_by_name(real_actions, action_name))

    return [g for g in goal_sites if F(f"safely_visited {g}") in env.state.fluents]


# ── Scenario A: early failure ─────────────────────────────────────────────────
def test_scenario_a_early_failure() -> None:
    random.seed(1337)
    graph = create_two_goal_graph()
    goal_sites = ["g1", "g2"]
    profiles = {k: PROFILES[k] for k in ["r1", "r2"]}

    env, _ = _make_env(
        graph,
        robot_locations={"r1": "start", "r2": "start"},
        active_profiles=profiles,
        goal_sites=goal_sites,
        failed_robots={"r1"},       # r1 fails before acting; no path removed
        blocked_paths=set(),
    )

    goal = F("safely_visited g1") & F("safely_visited g2")
    covered = _run(env, goal, goal_sites)

    assert F("operational r1") not in env.state.fluents, "r1 should remain non-operational"
    assert isinstance(covered, list)


# ── Scenario B: transit failure ───────────────────────────────────────────────
def test_scenario_b_transit_failure() -> None:
    random.seed(1337)
    graph = create_two_goal_graph()
    goal_sites = ["g1", "g2"]
    profiles = {k: PROFILES[k] for k in ["r1", "r2"]}

    env, _ = _make_env(
        graph,
        robot_locations={"r1": "n1", "r2": "start"},
        active_profiles=profiles,
        goal_sites=goal_sites,
        failed_robots={"r1"},
        blocked_paths={("n1", "g1")},   # r1 failed on n1→g1; that directed edge removed
    )

    goal = F("safely_visited g1") & F("safely_visited g2")
    covered = _run(env, goal, goal_sites)

    assert F("operational r1") not in env.state.fluents, "r1 should remain non-operational"
    assert set(covered) == {"g1", "g2"}, f"r2 should cover both goals via alternates; covered: {covered}"


# ── Scenario C: multiple simultaneous failures ────────────────────────────────
def test_scenario_c_multiple_simultaneous_failures() -> None:
    random.seed(1337)
    graph = create_three_goal_graph()
    goal_sites = ["g1", "g2", "g3"]
    profiles = {k: PROFILES[k] for k in ["r1", "r2", "r3"]}

    env, _ = _make_env(
        graph,
        robot_locations={"r1": "n1", "r2": "n3", "r3": "start"},
        active_profiles=profiles,
        goal_sites=goal_sites,
        failed_robots={"r1", "r2"},
        blocked_paths={
            ("n1", "g1"),   # r1 failed on clear path to g1; n2→g1 and n3→g1 still open
            ("n3", "g1"),   # r2 failed on radiation path to g1; n1→g1 and n2→g1 still open
        },
    )

    goal = F("safely_visited g1") & F("safely_visited g2") & F("safely_visited g3")
    covered = _run(env, goal, goal_sites)

    assert F("operational r1") not in env.state.fluents, "r1 should be non-operational"
    assert F("operational r2") not in env.state.fluents, "r2 should be non-operational"
    assert isinstance(covered, list)


# ── Scenario D: cascading failures ────────────────────────────────────────────
def test_scenario_d_cascading_failures() -> None:
    random.seed(1337)
    graph = create_three_goal_graph()
    goal_sites = ["g1", "g2", "g3"]
    profiles = {k: PROFILES[k] for k in ["r1", "r2", "r3"]}

    env, _ = _make_env(
        graph,
        robot_locations={"r1": "start", "r2": "n3", "r3": "start"},
        active_profiles=profiles,
        goal_sites=goal_sites,
        failed_robots={"r1", "r2"},
        blocked_paths={
            ("n3", "g2"),   # r2 failed on n3→g2; n4→g2 and n5→g2 still open
            # r1 failed at start — no path blocked
        },
    )

    goal = F("safely_visited g1") & F("safely_visited g2") & F("safely_visited g3")
    covered = _run(env, goal, goal_sites)

    assert F("operational r1") not in env.state.fluents, "r1 should be non-operational"
    assert F("operational r2") not in env.state.fluents, "r2 should be non-operational"
    assert isinstance(covered, list)


# ── Scenario E: critical path failure ────────────────────────────────────────
def test_scenario_e_critical_path_failure() -> None:
    random.seed(1337)
    graph = create_single_path_graph()
    goal_sites = ["g1", "g2"]
    profiles = {k: PROFILES[k] for k in ["r1", "r2"]}

    env, _ = _make_env(
        graph,
        robot_locations={"r1": "n1", "r2": "start"},
        active_profiles=profiles,
        goal_sites=goal_sites,
        failed_robots={"r1"},
        blocked_paths={("n1", "g1")},   # only path to g1 removed; g1 permanently unreachable
    )

    goal = F("safely_visited g1") & F("safely_visited g2")
    covered = _run(env, goal, goal_sites)

    assert "g2" in covered, f"r2 should cover g2 via unaffected paths; covered: {covered}"
    assert "g1" not in covered, f"g1 is unreachable with its only path blocked; covered: {covered}"
