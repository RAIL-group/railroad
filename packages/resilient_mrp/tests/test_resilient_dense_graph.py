# dense grid graph tests for the resilient planner.
#
# ── Graph A: single terminal Z (multi_goal=False, default) ───────────────────
#
#           S (start)
#         / | \
#        /  |  \
#       A───B───C        ← layer 1 (horizontal: bidirectional)
#       |\ /|\ /|
#       | X | X |        ← X pattern: downward + diagonal forward edges
#       |/ \|/ \|
#       D───E───F        ← layer 2 (horizontal: bidirectional)
#       |\ /|\ /|
#       | X | X |
#       |/ \|/ \|
#       G───H───I        ← layer 3 (horizontal: bidirectional)
#        \  |  /
#         \ | /
#          \|/
#           Z            ← single goal node
#
#   goal semantics: OR — mission complete when any robot reaches Z
#
# ── Graph B: three terminal goals (multi_goal=True) ──────────────────────────
#
#           S (start)
#         / | \
#        /  |  \
#       A───B───C        ← layer 1 (horizontal: bidirectional)
#       |\ /|\ /|
#       | X | X |        ← X pattern: same as Graph A
#       |/ \|/ \|
#       D───E───F        ← layer 2 (horizontal: bidirectional)
#       |\ /|\ /|
#       | X | X |
#       |/ \|/ \|
#       G───H───I        ← layer 3 (horizontal: bidirectional)
#       |   |   |
#      ZG  ZH  ZI        ← three distinct goal nodes (one per column)
#
#   goal semantics: AND — mission complete when all three goals are safely visited
#
# ─────────────────────────────────────────────────────────────────────────────
#
# shared edge structure (both variants, seed=42):
#   forward directed edges (20 + 3 terminal):
#     S → A, B, C
#     A → D, E        B → D, E, F        C → E, F
#     D → G, H        E → G, H, I        F → H, I
#     Graph A: G → Z,  H → Z,  I → Z
#     Graph B: G → ZG, H → ZH, I → ZI
#   horizontal bidirectional edges (6 pairs = 12 directed):
#     layer 1: A ↔ B, B ↔ C
#     layer 2: D ↔ E, E ↔ F
#     layer 3: G ↔ H, H ↔ I
#
# robot profiles (terrain compatibility, 0=incompatible, 1=fully suited):
#   r1 (compact)  — narrow:0.9, normal:0.7, rugged:0.1
#   r2 (balanced) — narrow:0.5, normal:0.9, rugged:0.4
#   r3 (robust)   — narrow:0.1, normal:0.8, rugged:0.95

import random
from typing import Dict, List
from resilient_mrp.planning.core import (
    ResilientGraph,
    RobotProfile,
    create_risk_move_operator,
    compute_p_success,
)
from railroad.core import Fluent, State
from railroad.planner import MCTSPlanner, get_usable_actions

F = Fluent

GRAPH_SEED = 42

ROBOT_PROFILES: Dict[str, RobotProfile] = {
    "r1": {"_type": "compact",  "narrow": 0.9, "normal": 0.7, "rugged": 0.1},
    "r2": {"_type": "balanced", "narrow": 0.5, "normal": 0.9, "rugged": 0.4},
    "r3": {"_type": "robust",   "narrow": 0.1, "normal": 0.8, "rugged": 0.95},
}

TERRAIN_TYPES = ["narrow", "normal", "rugged"]

def create_grid_graph(seed: int = GRAPH_SEED, multi_goal: bool = False) -> ResilientGraph:
    rng = random.Random(seed)
    graph = ResilientGraph()

    # add a directed forward edge with randomised terrain and risk.
    def fwd(fr: str, to: str) -> None:
        terrain = rng.choice(TERRAIN_TYPES)
        cost = round(rng.uniform(1.0, 5.0), 2)
        risk = round(rng.uniform(0.1, 0.8), 2)
        graph.add_edge(fr, to, cost=cost, terrain_type=terrain,
                       hazard_severity=risk, bidirectional=False)

    # add a horizontal bidirectional edge with randomised terrain and risk.
    def horiz(a: str, b: str) -> None:
        terrain = rng.choice(TERRAIN_TYPES)
        cost = round(rng.uniform(1.0, 5.0), 2)
        risk = round(rng.uniform(0.1, 0.8), 2)
        graph.add_edge(a, b, cost=cost, terrain_type=terrain,
                       hazard_severity=risk, bidirectional=True)

    # S → layer 1
    fwd("S", "A"); fwd("S", "B"); fwd("S", "C")

    # layer 1 → layer 2 (X pattern: straight + diagonals)
    fwd("A", "D"); fwd("A", "E")
    fwd("B", "D"); fwd("B", "E"); fwd("B", "F")
    fwd("C", "E"); fwd("C", "F")

    # layer 2 → layer 3 (X pattern)
    fwd("D", "G"); fwd("D", "H")
    fwd("E", "G"); fwd("E", "H"); fwd("E", "I")
    fwd("F", "H"); fwd("F", "I")

    # layer 3 → goal(s)
    if multi_goal:
        fwd("G", "ZG"); fwd("H", "ZH"); fwd("I", "ZI")
    else:
        fwd("G", "Z"); fwd("H", "Z"); fwd("I", "Z")

    # horizontal bidirectional edges within each layer
    horiz("A", "B"); horiz("B", "C")   # layer 1
    horiz("D", "E"); horiz("E", "F")   # layer 2
    horiz("G", "H"); horiz("H", "I")   # layer 3

    return graph


def build_initial_fluents(
    graph: ResilientGraph,
    robot_profiles: Dict[str, RobotProfile],
    robot_locations: Dict[str, str],
) -> set:
    fluents = set()
    for robot in robot_profiles:
        location = robot_locations.get(robot, "S")
        fluents.add(F(f"at {robot} {location}"))
        fluents.add(F(f"free {robot}"))
        fluents.add(F(f"operational {robot}"))
    for robot, profile in robot_profiles.items():
        robot_type = profile.get("_type", "")
        if robot_type:
            fluents.add(F(f"{robot_type} {robot}"))
    fluents.update(graph.get_edge_fluents())
    return fluents


# simulate deterministic successful arrival at destination: update fluents
# and accumulate edge cost into state.time. without this, all non-goal states
# look identical (time=0) to MCTS, giving no reward gradient for path selection.
def apply_success_arrival(state: State, action_name: str,
                          graph: ResilientGraph) -> State:
    parts = action_name.split()  # "risk_move robot from to"
    robot, from_, to_ = parts[1], parts[2], parts[3]
    cost = graph.edges.get((from_, to_), {}).get("cost", 1.0)
    new_fluents = set(state.fluents)
    new_fluents.discard(F(f"at {robot} {from_}"))
    new_fluents.add(F(f"at {robot} {to_}"))
    new_fluents.add(F(f"free {robot}"))
    new_fluents.add(F(f"safely_visited {to_}"))
    return State(time=state.time + cost, fluents=new_fluents)


# shared execution loop for dense graph tests. takes a pre-built graph (A or B),
# a goal expression (OR or AND), and the terminal node names for arrival logging.
# runs MCTS step-by-step using apply_success_arrival to simulate deterministic
# traversal. returns True if the mission completed within the step budget.
def _run_dense_graph_scenario(
    graph: ResilientGraph,
    goal,
    goal_nodes: List[str],
    max_steps: int,
    max_iterations: int,
) -> bool:
    robots = list(ROBOT_PROFILES.keys())

    objects_by_type = {
        "robot": robots,
        "location": sorted(graph.nodes),
    }

    move_op = create_risk_move_operator(graph, ROBOT_PROFILES)
    all_actions = move_op.instantiate(objects_by_type)
    initial_fluents = build_initial_fluents(
        graph, ROBOT_PROFILES, {r: "S" for r in robots})

    # print edge table with per-robot success probabilities
    print(f"\n{'=' * 68}")
    print(f"  graph: {len(graph.nodes)} nodes | "
          f"{len(graph.edges)} directed edges | "
          f"{len(all_actions)} grounded actions | "
          f"goals: {', '.join(goal_nodes)}")
    print(f"  {'edge':10}  {'terrain':6}  {'risk':4}  "
          f"{'p(r1)':6}  {'p(r2)':6}  {'p(r3)':6}  {'dir':5}")
    seen: set = set()
    for (fr, to), props in sorted(graph.edges.items()):
        key = tuple(sorted([fr, to]))
        bidir = "↔" if (to, fr) in graph.edges else "→"
        if key in seen and bidir == "↔":
            continue
        seen.add(key)
        terrain, risk = props['terrain_type'], props['hazard_severity']
        ps = {r: compute_p_success(ROBOT_PROFILES[r], terrain, risk) for r in robots}
        print(f"  {fr+' '+bidir+' '+to:10}  {terrain:6}  {risk:.2f}  "
              f"  {ps['r1']:.2f}    {ps['r2']:.2f}    {ps['r3']:.2f}  {bidir}")
    print(f"{'=' * 68}")

    robot_paths: Dict[str, List[str]] = {r: ["S"] for r in robots}
    robot_costs: Dict[str, float] = {r: 0.0 for r in robots}

    mcts = MCTSPlanner(all_actions)
    state = State(time=0, fluents=initial_fluents)
    mission_complete = False

    for step in range(max_steps):
        if goal.evaluate(state.fluents):
            mission_complete = True
            break

        usable = get_usable_actions(state, all_actions)
        action_name = mcts(state, goal, max_iterations=max_iterations,
                           c=100, heuristic_multiplier=0)

        if action_name == "NONE":
            break

        parts = action_name.split()
        robot, from_, to_ = parts[1], parts[2], parts[3]
        edge_cost = graph.edges.get((from_, to_), {}).get("cost", 1.0)
        robot_paths[robot].append(to_)
        robot_costs[robot] += edge_cost

        print(f"  step {step + 1:2d}  usable={len(usable):2d}  "
              f"t={state.time:.2f}  → {action_name}  (cost={edge_cost:.2f})")

        state = apply_success_arrival(state, action_name, graph)

    if not mission_complete:
        mission_complete = goal.evaluate(state.fluents)

    # log results
    print(f"\n  {'─' * 50}")
    print(f"  mission complete: {mission_complete}")
    for robot in robots:
        arrived_at = next(
            (gn for gn in goal_nodes if F(f"at {robot} {gn}").evaluate(state.fluents)),
            None,
        )
        path_str = " → ".join(robot_paths[robot])
        status = f"ARRIVED {arrived_at}" if arrived_at else f"at {robot_paths[robot][-1]}"
        print(f"  {robot} [{status}]  path: {path_str}  "
              f"total cost: {robot_costs[robot]:.2f}")
    print(f"  {'─' * 50}")
    print(f"\n  MCTS trace (last step):")
    print(mcts.get_trace_from_last_mcts_tree())

    return mission_complete


def test_dense_graph_first_arrival():
    graph = create_grid_graph(seed=GRAPH_SEED)
    goal = F("at r1 Z") | F("at r2 Z") | F("at r3 Z")
    mission_complete = _run_dense_graph_scenario(
        graph, goal, ["Z"], max_steps=40, max_iterations=10000)
    assert mission_complete, "at least one robot should reach Z within 40 steps"


def test_dense_graph_all_goals():
    graph = create_grid_graph(seed=GRAPH_SEED, multi_goal=True)
    goal = (F("safely_visited ZG") &
            F("safely_visited ZH") &
            F("safely_visited ZI"))
    mission_complete = _run_dense_graph_scenario(
        graph, goal, ["ZG", "ZH", "ZI"], max_steps=80, max_iterations=10000)
    assert mission_complete, "each robot should reach a distinct goal zone within 80 steps"
