# 3-robot 3-region resilient planning scenario.
#
# validates that encoding robot-terrain failure interaction as stochastic PDDL effects
# causes the MCTS planner to converge toward failure-resilient assignments on its own —
# without explicit assignment rules. compatible pairings are an emergent result of the
# failure model, not a programmed constraint.
#
# three robots visit different regions (narrow, normal, rugged) from a common start. 
# terrain types have different failure probabilities
# planner is not given any explicit rules about which robot is best suited for which terrain
#        Start
#       /  |  \
#      A   B   C
# narrow  normal  rugged
#
# robot profiles (terrain compatibility, 0=incompatible, 1=fully suited):
#   r1 (compact)  — narrow:0.9, normal:0.7, rugged:0.1
#   r2 (balanced) — narrow:0.5, normal:0.9, rugged:0.4
#   r3 (robust)   — narrow:0.1, normal:0.8, rugged:0.95

from typing import Dict
from resilient_mrp.planning.core import (
    ResilientGraph,
    RobotProfile,
    create_risk_move_operator,
    compute_p_success,
)
from railroad.core import Fluent, State
from railroad.planner import MCTSPlanner, get_usable_actions

F = Fluent

# test configuration

# robot profiles: terrain_type -> compatibility [0, 1]
ROBOT_PROFILES: Dict[str, RobotProfile] = {
    "r1": {"_type": "compact",  "narrow": 0.9, "normal": 0.7, "rugged": 0.1},
    "r2": {"_type": "balanced", "narrow": 0.5, "normal": 0.9, "rugged": 0.4},
    "r3": {"_type": "robust",   "narrow": 0.1, "normal": 0.8, "rugged": 0.95},
}


# build the 3-region graph from formulation
def create_3region_graph() -> ResilientGraph:
    graph = ResilientGraph()
    graph.add_edge("Start", "RegionA", cost=3.0, terrain_type="narrow", hazard_severity=0.7)
    graph.add_edge("Start", "RegionB", cost=2.0, terrain_type="normal", hazard_severity=0.2)
    graph.add_edge("Start", "RegionC", cost=4.0, terrain_type="rugged", hazard_severity=0.5)
    return graph


# build the initial state fluents for the scenario
def create_resilient_initial_fluents(
    graph: ResilientGraph,
    robot_profiles: Dict[str, RobotProfile],
    robot_locations: Dict[str, str],
) -> set:
    fluents = set()
    for robot in robot_profiles:
        location = robot_locations.get(robot, "Start")
        fluents.add(F(f"at {robot} {location}"))
        fluents.add(F(f"free {robot}"))
        fluents.add(F(f"operational {robot}"))
    for robot, profile in robot_profiles.items():
        robot_type = profile.get("_type", "")
        if robot_type:
            fluents.add(F(f"{robot_type} {robot}"))
    fluents.update(graph.get_edge_fluents())
    return fluents


# test that compute_p_success matches formulation
# formula: 1 - risk × (1 - compatibility), where (1 - compatibility) is robot's susceptibility
def test_p_success_computation():
    # r1 compact on narrow: compatibility=0.9, susceptibility=0.1, risk=0.7 → 1 - 0.7×0.1 = 0.93
    p = compute_p_success(ROBOT_PROFILES["r1"], "narrow", 0.7)
    assert abs(p - 0.93) < 0.001, f"Expected 0.93, got {p}"

    # r3 robust on narrow: compatibility=0.1, susceptibility=0.9, risk=0.7 → 1 - 0.7×0.9 = 0.37
    p = compute_p_success(ROBOT_PROFILES["r3"], "narrow", 0.7)
    assert abs(p - 0.37) < 0.001, f"Expected 0.37, got {p}"

    # r3 robust on rugged: compatibility=0.95, susceptibility=0.05, risk=0.5 → 1 - 0.5×0.05 = 0.975
    p = compute_p_success(ROBOT_PROFILES["r3"], "rugged", 0.5)
    assert abs(p - 0.975) < 0.001, f"Expected 0.975, got {p}"

    # r1 compact on rugged: compatibility=0.1, susceptibility=0.9, risk=0.5 → 1 - 0.5×0.9 = 0.55
    p = compute_p_success(ROBOT_PROFILES["r1"], "rugged", 0.5)
    assert abs(p - 0.55) < 0.001, f"Expected 0.55, got {p}"


# test that initial fluents are generated correctly
def test_initial_fluents():
    graph = create_3region_graph()
    robot_locations = {"r1": "Start", "r2": "Start", "r3": "Start"}

    fluents = create_resilient_initial_fluents(graph, ROBOT_PROFILES, robot_locations)

    # robot state
    assert F("at r1 Start") in fluents
    assert F("at r2 Start") in fluents
    assert F("at r3 Start") in fluents
    assert F("free r1") in fluents
    assert F("operational r1") in fluents
    assert F("operational r2") in fluents
    assert F("operational r3") in fluents

    # robot profile types
    assert F("compact r1") in fluents
    assert F("balanced r2") in fluents
    assert F("robust r3") in fluents

    # graph edges
    assert F("edge Start RegionA") in fluents
    assert F("edge Start RegionB") in fluents
    assert F("edge Start RegionC") in fluents


# test that operator instantiates and produces valid actions
def test_operator_instantiation():
    graph = create_3region_graph()
    objects_by_type = {
        "robot": ["r1", "r2", "r3"],
        "location": sorted(graph.nodes),
    }

    move_op = create_risk_move_operator(graph, ROBOT_PROFILES)
    all_actions = move_op.instantiate(objects_by_type)

    # should produce actions for each robot x edge combination
    assert len(all_actions) > 0

    # check action names contain risk_move
    action_names = [a.name for a in all_actions]
    assert any("risk_move" in name for name in action_names), \
        f"Expected risk_move actions, got: {action_names[:5]}"


# test that planner finds an action for 3-robot scenario
def test_basic_planning():
    graph = create_3region_graph()
    robot_locations = {"r1": "Start", "r2": "Start", "r3": "Start"}

    objects_by_type = {
        "robot": ["r1", "r2", "r3"],
        "location": sorted(graph.nodes),
    }

    move_op = create_risk_move_operator(graph, ROBOT_PROFILES)
    all_actions = move_op.instantiate(objects_by_type)

    initial_state = State(time=0, fluents=create_resilient_initial_fluents(
        graph, ROBOT_PROFILES, robot_locations))

    goal = F("safely_visited RegionA") & F("safely_visited RegionB") & F("safely_visited RegionC")

    mcts = MCTSPlanner(all_actions)
    action = mcts(initial_state, goal, max_iterations=5000, c=500)

    print(f"\nFirst action: {action}")
    assert action != "NONE", "Planner should find an action"
    assert "risk_move" in action, f"Should be risk_move, got: {action}"


# validates that the failure model, not explicit rules, causes the planner to prefer
# failure-resilient assignments. runs 3 planner steps per trial (one per robot dispatch)
# to capture the full assignment, not just the first action, which is always r2→RegionB
# (cheapest regardless of terrain awareness). heuristic_multiplier=0 keeps the planner
# in pure MCTS mode: reward signal from the 200-unit failure penalty drives all decisions.
def test_resilient_assignment_preference():
    graph = create_3region_graph()

    objects_by_type = {
        "robot": ["r1", "r2", "r3"],
        "location": sorted(graph.nodes),
    }

    move_op = create_risk_move_operator(graph, ROBOT_PROFILES)
    all_actions = move_op.instantiate(objects_by_type)
    initial_fluents = create_resilient_initial_fluents(
        graph, ROBOT_PROFILES, {"r1": "Start", "r2": "Start", "r3": "Start"})
    goal = F("safely_visited RegionA") & F("safely_visited RegionB") & F("safely_visited RegionC")

    # apply departure effects (t=0, non-probabilistic) to advance state after each dispatch
    def apply_departure(state: State, action_name: str) -> State:
        action = next(a for a in all_actions if a.name == action_name)
        new_fluents = set(state.fluents)
        for grounded_eff in action.effects:
            if not grounded_eff.is_probabilistic:
                for fluent in grounded_eff.resulting_fluents:
                    if fluent.negated:
                        new_fluents.discard(~fluent)  # ~negated → discard positive form
                    else:
                        new_fluents.add(fluent)
        return State(time=0, fluents=new_fluents)

    num_trials = 20
    r1_destinations: list = []
    r2_destinations: list = []
    r3_destinations: list = []

    mcts = MCTSPlanner(all_actions)

    for trial_idx in range(num_trials):
        state = State(time=0, fluents=initial_fluents)
        trial_assignment: dict = {}
        is_last = (trial_idx == num_trials - 1)

        # dispatch all 3 robots: each planner call assigns one free robot
        for step in range(3):
            if is_last:
                usable = get_usable_actions(state, all_actions)
                print(f"\n  [trial {trial_idx+1}, step {step+1}] usable actions ({len(usable)}):")
                for a in usable:
                    print(f"    {a.name}")

            action_name = mcts(state, goal, max_iterations=10000, c=1.414, heuristic_multiplier=0)

            if is_last:
                print(f"  → selected: {action_name}")
                print(mcts.get_trace_from_last_mcts_tree())

            # action name format: "risk_move <robot> <from> <to>"
            parts = action_name.split()
            if len(parts) >= 4:
                trial_assignment[parts[1]] = parts[3]
            state = apply_departure(state, action_name)

        r1_destinations.append(trial_assignment.get("r1", "unknown"))
        r2_destinations.append(trial_assignment.get("r2", "unknown"))
        r3_destinations.append(trial_assignment.get("r3", "unknown"))

    from collections import Counter
    r1_counts = Counter(r1_destinations)
    r2_counts = Counter(r2_destinations)
    r3_counts = Counter(r3_destinations)

    print(f"\nr1 (compact) destinations over {num_trials} trials:")
    for region, count in sorted(r1_counts.items()):
        print(f"  r1→{region}: {count} ({count/num_trials:.0%})")

    print(f"\nr2 (balanced) destinations over {num_trials} trials:")
    for region, count in sorted(r2_counts.items()):
        print(f"  r2→{region}: {count} ({count/num_trials:.0%})")

    print(f"\nr3 (robust) destinations over {num_trials} trials:")
    for region, count in sorted(r3_counts.items()):
        print(f"  r3→{region}: {count} ({count/num_trials:.0%})")

    # r1 (compact) should prefer narrow (RegionA) over rugged (RegionC)
    assert r1_counts.get("RegionA", 0) > r1_counts.get("RegionC", 0), (
        f"Compact robot should prefer narrow: "
        f"r1→A={r1_counts.get('RegionA', 0)}, r1→C={r1_counts.get('RegionC', 0)}"
    )

    # r2 (balanced) should prefer normal (RegionB) over both narrow and rugged
    assert r2_counts.get("RegionB", 0) > r2_counts.get("RegionA", 0), (
        f"Balanced robot should prefer normal over narrow: "
        f"r2→B={r2_counts.get('RegionB', 0)}, r2→A={r2_counts.get('RegionA', 0)}"
    )
    assert r2_counts.get("RegionB", 0) > r2_counts.get("RegionC", 0), (
        f"Balanced robot should prefer normal over rugged: "
        f"r2→B={r2_counts.get('RegionB', 0)}, r2→C={r2_counts.get('RegionC', 0)}"
    )

    # r3 (robust) should prefer rugged (RegionC) over narrow (RegionA)
    assert r3_counts.get("RegionC", 0) > r3_counts.get("RegionA", 0), (
        f"Robust robot should prefer rugged: "
        f"r3→C={r3_counts.get('RegionC', 0)}, r3→A={r3_counts.get('RegionA', 0)}"
    )
