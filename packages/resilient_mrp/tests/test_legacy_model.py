# Every test of the superseded homogeneous model in planning/legacy.py, in one place.
# That model has no terrain and no robot compatibility: each edge carries a flat p_success, and a
# failure costs failure_penalty time units. These tests are what keeps the module alive; nothing
# in src imports it. New work belongs against planning/core.py instead.

import pytest

from railroad.core import Fluent, State
from railroad.planner import MCTSPlanner, get_usable_actions

from resilient_mrp.planning.legacy import SimpleGraph, create_move_operator

F = Fluent

FAILURE_PENALTY = 200.0
SAFE_CHOICE_RATIO = 0.7


# ── Builders ─────────────────────────────────────────────────────────────────

# Two edges out of A: B is the safe arm, C the risky one.
def two_arm_graph(cost_b: float, p_b: float, cost_c: float, p_c: float) -> SimpleGraph:
    graph = SimpleGraph()
    graph.add_edge("A", "B", cost=cost_b, p_success=p_b)
    graph.add_edge("A", "C", cost=cost_c, p_success=p_c)
    return graph


# Three two-hop paths A->{B,C,D}->E, from a {'A_B': {'cost':…, 'p_success':…}} spec.
def multi_hop_graph(edges: dict) -> SimpleGraph:
    graph = SimpleGraph()
    for edge_key, props in edges.items():
        frm, to = edge_key.split("_")
        graph.add_edge(frm, to, cost=props["cost"], p_success=props["p_success"])
    return graph


def start_state(graph: SimpleGraph, robots=("r1",), at: str = "A") -> State:
    fluents = set()
    for r in robots:
        fluents.add(F(f"at {r} {at}"))
        fluents.add(F(f"free {r}"))
    fluents.update(graph.get_edge_fluents())
    return State(time=0, fluents=fluents)


def actions_for(graph: SimpleGraph, robots=("r1",), locations=("A", "B", "C")) -> list:
    move_op = create_move_operator(graph, failure_penalty=FAILURE_PENALTY)
    return move_op.instantiate({"robot": list(robots), "location": sorted(locations)})


SIMPLE_GRAPH = two_arm_graph(cost_b=2.0, p_b=0.95, cost_c=1.0, p_c=0.60)
EXTREME_GRAPH = two_arm_graph(cost_b=20.0, p_b=0.99, cost_c=1.0, p_c=0.30)

REACH_B_OR_C = F("at r1 B") | F("at r1 C")


@pytest.fixture
def objects_by_type():
    return {"robot": ["r1"], "location": ["A", "B", "C"]}


# ── Operator and basic planning ──────────────────────────────────────────────

def test_operator_creation():
    move_op = create_move_operator(SIMPLE_GRAPH, failure_penalty=FAILURE_PENALTY)
    assert move_op.name == "traverse_edge"
    assert len(move_op.parameters) == 3
    for param in (("?robot", "robot"), ("?from", "location"), ("?to", "location")):
        assert param in move_op.parameters


def test_basic_planning():
    mcts = MCTSPlanner(actions_for(SIMPLE_GRAPH))
    action = mcts(start_state(SIMPLE_GRAPH), REACH_B_OR_C, max_iterations=10000, c=1414)
    assert action != "NONE"
    assert isinstance(action, str) and action


def test_safe_vs_risky_path_choice():
    state = start_state(SIMPLE_GRAPH)
    usable = get_usable_actions(state, actions_for(SIMPLE_GRAPH))
    mcts = MCTSPlanner(usable)

    chosen = [mcts(state, REACH_B_OR_C, max_iterations=1000, c=500) for _ in range(25)]
    assert sum("A B" in a for a in chosen) >= SAFE_CHOICE_RATIO * len(chosen)


# ── Cost and probability sweeps ──────────────────────────────────────────────

# Runs the two-arm choice num_trials times and checks the planner picked the safer arm in most.
def _run_two_arm_trial(cost_b, cost_c, prob_b, prob_c,
                       expected_safer, max_iterations, num_trials=10):
    graph = two_arm_graph(cost_b, prob_b, cost_c, prob_c)
    state = start_state(graph)
    mcts = MCTSPlanner(actions_for(graph))

    expected = f"traverse_edge r1 A {'B' if expected_safer == 'B' else 'C'}"
    correct = 0

    for _ in range(num_trials):
        action_name = mcts(state, REACH_B_OR_C, max_iterations=max_iterations, c=500)
        correct += action_name == expected

    assert correct >= SAFE_CHOICE_RATIO * num_trials, \
        f"expected {expected_safer} in most trials, got {correct}/{num_trials}"


@pytest.mark.parametrize("cost_b, cost_c, expected_safer", [
    (5.0, 1.5, "B"),   # B expensive but safe
    (2.0, 2.0, "B"),   # equal cost, B safer
    (1.0, 5.0, "B"),   # B cheaper and safer
])
def test_planning_with_varying_costs(cost_b, cost_c, expected_safer):
    _run_two_arm_trial(cost_b, cost_c, prob_b=0.95, prob_c=0.60,
                       expected_safer=expected_safer, max_iterations=1000)


@pytest.mark.parametrize("prob_b, prob_c, expected_safer", [
    (0.50, 0.95, "C"),   # B risky and expensive
    (0.90, 0.90, "C"),   # equal risk, C cheaper
    (0.95, 0.50, "B"),   # B safe and expensive
])
def test_planning_with_varying_probabilities(prob_b, prob_c, expected_safer):
    _run_two_arm_trial(cost_b=3.0, cost_c=1.5, prob_b=prob_b, prob_c=prob_c,
                       expected_safer=expected_safer, max_iterations=10000)


# B costs 20x more but the failure penalty should still make the 30% arm look worse.
def test_extreme_tradeoff():
    state = start_state(EXTREME_GRAPH)
    mcts = MCTSPlanner(actions_for(EXTREME_GRAPH))
    chosen = [mcts(state, REACH_B_OR_C, max_iterations=1000, c=500) for _ in range(10)]
    assert sum("A B" in a for a in chosen) >= SAFE_CHOICE_RATIO * len(chosen)


# ── Multi-hop graphs ─────────────────────────────────────────────────────────

_ALL_VIABLE = {
    'A_B': {'cost': 2.0, 'p_success': 0.95}, 'B_E': {'cost': 2.0, 'p_success': 0.95},
    'A_C': {'cost': 1.5, 'p_success': 0.85}, 'C_E': {'cost': 1.5, 'p_success': 0.85},
    'A_D': {'cost': 1.0, 'p_success': 0.70}, 'D_E': {'cost': 1.0, 'p_success': 0.70},
}
_ONLY_SAFE_VIABLE = {
    'A_B': {'cost': 2.0, 'p_success': 0.95}, 'B_E': {'cost': 0.85, 'p_success': 0.85},
    'A_C': {'cost': 1.5, 'p_success': 0.0},  'C_E': {'cost': 1.5, 'p_success': 0.85},
    'A_D': {'cost': 1.0, 'p_success': 0.0},  'D_E': {'cost': 1.0, 'p_success': 0.70},
}

MULTI_HOP_SCENARIOS = {
    "all_paths_viable": {"edges": _ALL_VIABLE, "expected_safe_ratio": 0.5},
    "only_safe_viable": {"edges": _ONLY_SAFE_VIABLE, "expected_safe_ratio": 0.95},
}


def test_multi_hop_basic_path_planning():
    graph = multi_hop_graph(_ALL_VIABLE)
    mcts = MCTSPlanner(actions_for(graph, locations=graph.nodes))
    action = mcts(start_state(graph), F("at r1 E"), max_iterations=1000, c=500)
    assert action != "NONE"
    assert any(f"A {n}" in action for n in "BCDE"), f"should step toward E, got {action}"


@pytest.mark.parametrize("scenario_name", list(MULTI_HOP_SCENARIOS))
def test_multi_hop_safe_vs_risky(scenario_name):
    scenario = MULTI_HOP_SCENARIOS[scenario_name]
    graph = multi_hop_graph(scenario["edges"])
    state = start_state(graph)
    mcts = MCTSPlanner(actions_for(graph, locations=graph.nodes))

    num_trials = 20
    first = [mcts(state, F("at r1 E"), max_iterations=100000, c=500, heuristic_multiplier=0)
             for _ in range(num_trials)]
    chose_b = sum("A B" in a for a in first)
    chose_d = sum("A D" in a for a in first)

    ratio = scenario["expected_safe_ratio"]
    assert chose_b >= ratio * num_trials, \
        f"should prefer A->B->E at least {ratio:.0%} (got {chose_b}/{num_trials})"
    assert chose_b > chose_d, f"should prefer safe over risky (got {chose_b} vs {chose_d})"


# ── Multi-robot ──────────────────────────────────────────────────────────────

TWO_ROBOT_GRAPH = two_arm_graph(cost_b=2.0, p_b=0.90, cost_c=1.0, p_c=0.50)


@pytest.mark.parametrize("goal, label", [
    (F("at r1 B") | F("at r1 C") | F("at r2 B") | F("at r2 C"), "either robot, either node"),
    (F("at r1 B") | F("at r2 B"), "either robot reaches the safe node"),
])
def test_two_robot_planning(goal, label):
    robots = ("r1", "r2")
    state = start_state(TWO_ROBOT_GRAPH, robots=robots)
    mcts = MCTSPlanner(actions_for(TWO_ROBOT_GRAPH, robots=robots))
    action = mcts(state, goal, max_iterations=1000, c=500)

    assert action != "NONE", f"should find an action for {label}"
    assert "traverse_edge" in action
    assert "r1" in action or "r2" in action
