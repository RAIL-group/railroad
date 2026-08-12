# What the two relaxations are supposed to do on graphs whose right answer is known by hand, and
# where they currently do something else. Each graph is small enough to check with a pencil.
#
# The passing tests pin behaviour that is already correct. The xfail(strict) ones are open defects:
# they record the intended behaviour and will flip to a failure the moment one is fixed, which is
# the signal to delete the marker.
#
# Companion to test_assignment.py, which covers .assign() on risk-free graphs only.

import math
import random

import pytest

from railroad import operators as rr_operators
from railroad._bindings import State
from railroad.core import Fluent as F
from railroad.environment import SymbolicEnvironment

from resilient_mrp.experiments.mission import run_episode
from resilient_mrp.planning.baselines import (CautiousPolicy, OptimisticPolicy,
                                              best_assignment, build_route_table,
                                              cautious_weight, optimistic_weight)
from resilient_mrp.planning.core import (ResilientGraph, create_risk_move_operator,
                                         create_safely_visited_operator,
                                         parse_available_paths)
from resilient_mrp.planning.value_function import RiskAwareCostToGo

C_FAIL = 500.0

# terrain "bad" is where the profile bites: survival is exactly 1 - hazard for these robots
PERFECT = {"r1": {"clear": 1.0, "bad": 0.0}}


# ---------------------------------------------------------------------------- helpers


# Episodes here drive the environment's probabilistic branches through the global RNG, and several
# MCTS tests elsewhere in this package are sensitive to where that RNG is left. Put it back.
@pytest.fixture(autouse=True)
def _restore_global_rng():
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


def _operators(graph, profiles, blocks=True):
    return [create_risk_move_operator(graph, profiles, blocks_on_failure=blocks),
            create_safely_visited_operator(),
            rr_operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)]


def _fluents(graph, profiles, goal_sites, start="start"):
    fl = set()
    for robot in profiles:
        fl |= {F(f"at {robot} {start}"), F(f"free {robot}"), F(f"operational {robot}")}
    fl |= graph.get_edge_fluents()
    fl |= graph.get_available_path_fluents()
    fl |= {F(f"is_goal {g}") for g in goal_sites}
    return fl


def _env(graph, profiles, goal_sites, start="start", blocks=True):
    return SymbolicEnvironment(
        state=State(0.0, _fluents(graph, profiles, goal_sites, start), []),
        objects_by_type={"robot": set(profiles), "location": set(graph.nodes)},
        operators=_operators(graph, profiles, blocks))


def _goal(goal_sites):
    goal = F(f"safely_visited {goal_sites[0]}")
    for site in goal_sites[1:]:
        goal = goal & F(f"safely_visited {site}")
    return goal


# One mission. Returns (mission succeeded, makespan), scored the way TrialOutcome does.
def _episode(graph, profiles, goal_sites, policy_cls, seed, max_steps=60, start="start",
             blocks=True):
    random.seed(seed)
    env = _env(graph, profiles, goal_sites, start, blocks)
    policy = policy_cls(graph, list(goal_sites), profiles)
    visited, _ = run_episode(env, _goal(goal_sites), 0, 0, list(goal_sites),
                             max_steps=max_steps, route_policy=policy, graph=graph)
    return visited == len(goal_sites), env.state.time


# Mean of trial_cost over draws: makespan when every goal was reached, flat C_fail otherwise.
# Also the mean makespan of the runs that finished, which is what shows an assignment stacking.
def _sweep(graph, profiles, goal_sites, policy_cls, draws=200, max_steps=60, blocks=True):
    runs = [_episode(graph, profiles, goal_sites, policy_cls, s, max_steps, blocks=blocks)
            for s in range(draws)]
    wins = [t for ok, t in runs if ok]
    mean_cost = sum(t if ok else C_FAIL for ok, t in runs) / len(runs)
    makespan = sum(wins) / len(wins) if wins else float("nan")
    return len(wins) / len(runs), mean_cost, makespan


# ---------------------------------------------------------------------------- fixtures


# A diamond with two ways to the goal: 10 units through risky ground, 30 units through safe.
# The detour is declared last, which is how you would naturally write it.
@pytest.fixture
def diamond() -> ResilientGraph:
    g = ResilientGraph()
    g.add_edge("start", "fast", cost=5.0, terrain_type="bad", hazard_severity=0.3)
    g.add_edge("fast", "g", cost=5.0, terrain_type="clear", hazard_severity=0.0)
    g.add_edge("start", "slow", cost=15.0, terrain_type="clear", hazard_severity=0.0)
    g.add_edge("slow", "g", cost=15.0, terrain_type="clear", hazard_severity=0.0)
    return g


# Same shape, but nothing is risk-free: the safe legs carry a small hazard rather than none.
# That one change is the difference between the cautious policy working and livelocking.
@pytest.fixture
def diamond_no_free_lunch() -> ResilientGraph:
    g = ResilientGraph()
    g.add_edge("start", "g", cost=10.0, terrain_type="bad", hazard_severity=0.5)
    g.add_edge("start", "m", cost=15.0, terrain_type="bad", hazard_severity=0.02)
    g.add_edge("m", "g", cost=15.0, terrain_type="bad", hazard_severity=0.02)
    return g


# Two goals on one side. Splitting them finishes at 10, one robot walking both finishes at 19.
@pytest.fixture
def side_by_side() -> ResilientGraph:
    g = ResilientGraph()
    g.add_edge("start", "gA", cost=10.0, terrain_type="t", hazard_severity=1.0)
    g.add_edge("start", "gB", cost=10.0, terrain_type="t", hazard_severity=1.0)
    g.add_edge("gA", "gB", cost=9.0, terrain_type="t", hazard_severity=1.0)
    return g


# Robots start apart, no edge between the goals. r1 is nearer both, but r2 covers gA cheaply
# enough that splitting beats stacking: r1->gB at 11 and r2->gA at 12 gives makespan 12.
# Stacking on r1 costs 10 + 21 = 31.
@pytest.fixture
def two_depots() -> ResilientGraph:
    g = ResilientGraph()
    g.add_edge("p1", "gA", cost=10.0, terrain_type="t", hazard_severity=0.0)
    g.add_edge("p1", "gB", cost=11.0, terrain_type="t", hazard_severity=0.0)
    g.add_edge("p2", "gA", cost=12.0, terrain_type="t", hazard_severity=0.0)
    g.add_edge("p2", "gB", cost=100.0, terrain_type="t", hazard_severity=0.0)
    return g


# ------------------------------------------------------------------ routing: what works


# Optimistic prices a route by travel time alone, so it takes the short risky leg.
def test_optimistic_routes_by_time(diamond):
    table = build_route_table(diamond, PERFECT, "r1", "g", optimistic_weight)
    assert table["start"].travel_cost == pytest.approx(10.0)
    assert table["start"].survival == pytest.approx(0.7)
    policy = OptimisticPolicy(diamond, ["g"], PERFECT)
    assert policy.step_toward("start", "r1", "g") == "risk_move r1 start fast"


# Cautious prices it by -log(survival), so it takes the long safe leg and pays the time.
def test_cautious_routes_by_survival(diamond):
    table = build_route_table(diamond, PERFECT, "r1", "g", cautious_weight)
    assert table["start"].survival == pytest.approx(1.0)
    assert table["start"].travel_cost == pytest.approx(30.0)
    policy = CautiousPolicy(diamond, ["g"], PERFECT)
    assert policy.step_toward("start", "r1", "g") == "risk_move r1 start slow"


# The trade the cautious baseline exists to make: three times the travel for double the odds of
# arriving. Over a mission it has to come out ahead on cost, not just on survival.
def test_cautious_buys_survival_with_time(diamond_no_free_lunch):
    p_opt, cost_opt, _ = _sweep(diamond_no_free_lunch, PERFECT, ["g"], OptimisticPolicy)
    p_cau, cost_cau, _ = _sweep(diamond_no_free_lunch, PERFECT, ["g"], CautiousPolicy)

    assert p_opt == pytest.approx(0.5, abs=0.1), "direct leg should get through about half the time"
    assert p_cau > 0.9, f"cautious detour is nearly risk-free but only survived {p_cau}"
    assert cost_cau < cost_opt, (
        f"cautious paid {cost_cau:.1f} against optimistic {cost_opt:.1f}; the detour is 30 units "
        f"and C_fail is {C_FAIL}, so the safe route should win on cost")


# Optimistic's assignment search is a real uniform-cost search over the joint space, so it finds
# the split even when one robot is nearer to both goals.
def test_optimistic_assignment_minimises_makespan(two_depots):
    profiles = {"r1": {"t": 1.0}, "r2": {"t": 1.0}}
    policy = OptimisticPolicy(two_depots, ["gA", "gB"], profiles)
    assert policy.assign({"r1": "p1", "r2": "p2"}, set()) == {"r1": ["gB"], "r2": ["gA"]}


# ------------------------------------------------------------------ routing: open defects


# cautious_weight used to be -log(survival) alone, which is exactly 0.0 on any edge a robot is
# certain to cross. A risk-free region was then a plateau in the route table, and step_toward
# descends it with a strict <, so a hop back the way it came tied with the hop toward the goal and
# won on whichever was declared first: start->slow->start->slow until max_steps ran out. The
# _EPS_TIME term makes every real edge weigh something, which breaks the tie the right way.
def test_cautious_makes_progress_across_risk_free_ground(diamond):
    policy = CautiousPolicy(diamond, ["g"], PERFECT)
    assert policy.step_toward("slow", "r1", "g") == "risk_move r1 slow g", (
        "from slow the only sane hop is the goal itself")

    reached, _ = _episode(diamond, PERFECT, ["g"], CautiousPolicy, seed=0)
    assert reached, "there is a route to g that cannot fail, so cautious must always arrive"


# Between two routes that are equally likely to survive, the shorter one is strictly better. The
# survival terms tie here, so this is entirely decided by _EPS_TIME. Before it, whichever node the
# heap happened to settle first won: the table recorded the 1000-unit route while step_toward walked
# the 10-unit one, and best_assignment priced the leg against a path nobody took.
def test_cautious_prefers_the_shorter_of_two_equally_safe_routes():
    g = ResilientGraph()
    g.add_edge("start", "s", cost=5.0, terrain_type="clear", hazard_severity=0.0)
    g.add_edge("s", "g", cost=5.0, terrain_type="clear", hazard_severity=0.0)
    g.add_edge("start", "L", cost=500.0, terrain_type="clear", hazard_severity=0.0)
    g.add_edge("L", "g", cost=500.0, terrain_type="clear", hazard_severity=0.0)

    table = build_route_table(g, PERFECT, "r1", "g", cautious_weight)
    assert table["start"].survival == pytest.approx(1.0)
    assert table["start"].travel_cost == pytest.approx(10.0), (
        "both routes are certain, so the cheaper one is the route")


# ------------------------------------------------- assignment: what cautious is, by design


# best_assignment prices a joint state by weigh(makespan, team survival). Under cautious that is
# -log(survival) plus a term too small to matter at this scale, so the makespan argument has no real
# say and stacking both goals on the safest robot costs nothing in the search. Cautious therefore
# concentrates work on whichever robot is least likely to die, and leaves the others standing.
#
# That is deliberate. Cautious is one end of a bracket -- the relaxation that cares only about
# getting there, as optimistic is the one that cares only about when -- and a cautious baseline that
# balanced load would stop being a naive extreme and start competing with the planner it exists to
# measure. These tests pin the behaviour so it stays a decision rather than becoming folklore.
#
# The price, measured over 2000 draws on this graph with p(r1)=0.90 and p(r2)=0.60: makespan 20.0
# against optimistic's 13.4, for a survival rate of 0.901 against 0.911. Cautious pays about half
# again in time and buys nothing, because the product of leg survivals it maximises is not the
# mission success probability of a system that reassigns a dead robot's goal to a survivor.
@pytest.mark.parametrize("p1,p2", [(0.99, 0.98), (0.95, 0.90), (0.90, 0.50)])
def test_cautious_concentrates_goals_on_the_safest_robot(side_by_side, p1, p2):
    profiles = {"r1": {"t": p1}, "r2": {"t": p2}}
    policy = CautiousPolicy(side_by_side, ["gA", "gB"], profiles)
    queues = policy.assign({"r1": "start", "r2": "start"}, set())
    # the visiting order within the tour is settled by the time term and is not the point here
    assert sorted(queues["r1"]) == ["gA", "gB"], f"the safer robot should take both: {queues}"
    assert queues["r2"] == [], f"the riskier robot should stand: {queues}"


# The same choice seen end to end, where it shows up as makespan: splitting finishes at 10, one
# robot walking both finishes at 19. The gap is stable across draws in a way the survival rates
# are not, which is why this pins the makespan rather than the success rate.
def test_cautious_pays_makespan_to_concentrate_risk(side_by_side):
    profiles = {"r1": {"t": 0.90}, "r2": {"t": 0.60}}
    _, _, makespan_opt = _sweep(side_by_side, profiles, ["gA", "gB"], OptimisticPolicy)
    _, _, makespan_cau = _sweep(side_by_side, profiles, ["gA", "gB"], CautiousPolicy)
    assert makespan_cau > makespan_opt + 3.0, (
        f"cautious finished at {makespan_cau:.1f} against optimistic {makespan_opt:.1f}; if these "
        f"have converged, cautious is no longer the risk-averse extreme of the bracket")


# ------------------------------------------------------------- the leaf estimate: defects


# risk_move retracts "at" the moment a robot departs, so parse_state recovers the robot from the
# arrival branch of the pending effect and reports it already standing on its destination. The leaf
# then charges it no travel and no risk for the edge it is halfway across. Committing r1 to a
# 100-unit coin-flip drops the estimate from 350 to 1, which is a reward for taking the gamble.
def test_leaf_charges_for_the_edge_a_robot_is_crossing():
    g = ResilientGraph()
    g.add_edge("start", "gA", cost=100.0, terrain_type="t", hazard_severity=0.5)
    g.add_edge("start", "gB", cost=1.0, terrain_type="t", hazard_severity=0.0)
    profiles = {"r1": {"t": 0.0}, "r2": {"t": 0.0}}
    leaf = RiskAwareCostToGo(g, ["gA", "gB"], profiles, C_FAIL)

    env = _env(g, profiles, ["gA", "gB"])
    before = leaf(env.state)
    env.act(next(a for a in env.get_actions() if a.name == "risk_move r1 start gA"))
    after = leaf(env.state)

    assert after > 0.5 * before, (
        f"estimate fell from {before:.1f} to {after:.1f} on a move that resolves nothing: "
        f"100 units of travel and a coin flip are still outstanding")


# The failure branch of risk_move retracts path_available both ways, so a failure can cut a goal
# off entirely. Both baselines re-read the open edges in .observe(); the leaf builds its tables
# once with open_edges=None and never looks again, so it keeps quoting a route through a shut edge.
def test_leaf_returns_failure_cost_when_the_goal_is_cut_off():
    g = ResilientGraph()
    g.add_edge("start", "mid", cost=10.0, terrain_type="t", hazard_severity=0.5)
    g.add_edge("mid", "g", cost=10.0, terrain_type="t", hazard_severity=0.5)
    profiles = {"r1": {"t": 0.0}}
    leaf = RiskAwareCostToGo(g, ["g"], profiles, C_FAIL)

    fluents = _fluents(g, profiles, ["g"])
    shut = fluents - {F("path_available mid g"), F("path_available g mid")}
    assert leaf(State(0.0, shut, [])) == pytest.approx(C_FAIL), (
        "the only corridor to g is closed, so the mission is already lost")


def _two_depot_estimate(two_depots, order):
    profiles = {"r1": {"t": 1.0}, "r2": {"t": 1.0}}
    leaf = RiskAwareCostToGo(two_depots, order, profiles, C_FAIL)
    fluents = _fluents(two_depots, profiles, order, start="p1")
    fluents = (fluents - {F("at r2 p1")}) | {F("at r2 p2")}
    return leaf(State(0.0, fluents, []))


# The leaf used to hand the outstanding goals out in one pass and never revisit a choice, which
# made the answer depend on the order it walked them: 31 one way round, 12 the other, where 12 is
# the optimum. It searches the orderings now, so the estimate is a function of the state rather
# than of how the caller happened to build the goal list -- and it is the right number, which
# matters because the makespan term is supposed to be the optimistic side of the estimate and 31
# overshoots rather than under-shoots.
@pytest.mark.parametrize("order", [["gA", "gB"], ["gB", "gA"]])
def test_leaf_finds_the_optimal_makespan_whatever_the_goal_order(two_depots, order):
    assert _two_depot_estimate(two_depots, order) == pytest.approx(12.0)


# Time a robot already owes has to reach the assignment, not just the makespan added at the end.
# r1 is nearer gA (10 against 12), so with everyone standing still it takes it. Give r1 a 30-unit
# crossing still to finish and that stops being true: r2 should be sent instead. An assignment that
# starts every robot at zero cannot see the difference.
def test_assignment_accounts_for_time_a_robot_already_owes(two_depots):
    profiles = {"r1": {"t": 1.0}, "r2": {"t": 1.0}}
    policy = OptimisticPolicy(two_depots, ["gA"], profiles)
    positions = {"r1": "p1", "r2": "p2"}

    def taker(initial_loads):
        legs = best_assignment(["gA"], positions, set(), policy.route_to,
                               optimistic_weight, initial_loads=initial_loads)
        assert legs, "gA is reachable from both depots"
        return legs[0][0]

    assert taker(None) == "r1", "standing still, the nearer robot goes"
    assert taker({"r1": 30.0}) == "r2", "r1 owes 30 before it can start, so r2 is sooner"


# A pair of goals both robots can reach from where they stand, on ground that kills half the
# robots that cross it. Losing one robot does not lose the mission here: the other can be sent to
# whatever the first was carrying.
@pytest.fixture
def both_can_cover() -> ResilientGraph:
    g = ResilientGraph()
    g.add_edge("start", "gA", cost=10.0, terrain_type="t", hazard_severity=0.5)
    g.add_edge("start", "gB", cost=10.0, terrain_type="t", hazard_severity=0.5)
    g.add_edge("gA", "gB", cost=10.0, terrain_type="t", hazard_severity=0.5)
    return g


# The mission is lost when the robots still standing cannot cover what is left, not when any one
# robot is lost. With two robots that can each reach both goals, the only outcome that loses the
# mission is losing both -- so the estimate must be far below the one-robot case, not equal to it.
def test_leaf_counts_a_survivor_taking_over(both_can_cover):
    # profile 0.0 on hazard 0.5 makes survival exactly 0.5 per leg
    profiles = {"r1": {"t": 0.0}, "r2": {"t": 0.0}}
    goals = ["gA", "gB"]

    alone = RiskAwareCostToGo(both_can_cover, goals, {"r1": profiles["r1"]}, C_FAIL)
    lone_state = State(0.0, _fluents(both_can_cover, {"r1": profiles["r1"]}, goals), [])

    pair = RiskAwareCostToGo(both_can_cover, goals, profiles, C_FAIL)
    pair_state = State(0.0, _fluents(both_can_cover, profiles, goals), [])

    assert pair(pair_state) < alone(lone_state) - 50.0, (
        f"two robots that can each cover both goals estimated at {pair(pair_state):.1f} against "
        f"{alone(lone_state):.1f} for one; the second robot's cover is not being counted")


# The other half of the same claim: with nobody to take over, one robot's odds are the mission's.
def test_leaf_reports_a_lone_robots_odds_as_the_missions(both_can_cover):
    profiles = {"r1": {"t": 0.0}}
    leaf = RiskAwareCostToGo(both_can_cover, ["gA"], profiles, C_FAIL)
    state = State(0.0, _fluents(both_can_cover, profiles, ["gA"]), [])
    # one leg at survival 0.5, so half the failure cost plus the 10 units of travel
    assert leaf(state) == pytest.approx(10.0 + 0.5 * C_FAIL)


# A robot that cannot reach a goal cannot cover it, however healthy it is. Here r2 sits behind a
# closed edge, so it is no insurance at all and the estimate should match r1 working alone.
def test_leaf_does_not_count_cover_a_robot_cannot_reach(both_can_cover):
    profiles = {"r1": {"t": 0.0}, "r2": {"t": 0.0}}
    leaf = RiskAwareCostToGo(both_can_cover, ["gA"], profiles, C_FAIL)

    fluents = _fluents(both_can_cover, profiles, ["gA"])
    # strand r2 at gB with the only ways out shut
    fluents = (fluents - {F("at r2 start"),
                          F("path_available gB gA"), F("path_available gA gB"),
                          F("path_available gB start"), F("path_available start gB")}
               ) | {F("at r2 gB")}

    stranded = leaf(State(0.0, fluents, []))
    assert stranded == pytest.approx(10.0 + 0.5 * C_FAIL), (
        f"r2 is walled off and cannot cover gA, so the estimate should be r1's alone: {stranded:.1f}")


# --------------------------------------------------- more goals than robots to carry them


# Goals evenly spaced on a circle around the depot, every pair connected. best_assignment does
# chain several goals onto one robot, so this shape is representable; what it costs is the point.
def _ring(n_goals, hazard=0.15):
    import numpy as np
    g = ResilientGraph()
    coords = {"start": np.array([0.0, 0.0])}
    for i in range(n_goals):
        angle = 2 * math.pi * i / n_goals
        coords[f"g{i}"] = np.array([10 * math.cos(angle), 10 * math.sin(angle)])
    names = list(coords)
    for i, u in enumerate(names):
        for v in names[i + 1:]:
            g.add_edge(u, v, cost=float(np.linalg.norm(coords[u] - coords[v])),
                       terrain_type="t", hazard_severity=hazard)
    return g


# The assignment does produce multi-goal tours, which is worth pinning: the formulation is not
# structurally limited to one goal per robot.
@pytest.mark.parametrize("n_goals", [3, 5])
def test_optimistic_builds_multi_goal_tours(n_goals):
    goals = [f"g{i}" for i in range(n_goals)]
    profiles = {"r1": {"t": 0.9}, "r2": {"t": 0.9}}
    policy = OptimisticPolicy(_ring(n_goals), goals, profiles)
    queues = policy.assign({"r1": "start", "r2": "start"}, set())
    assert sum(len(q) for q in queues.values()) == n_goals, "every goal must be handed out once"
    assert max(len(q) for q in queues.values()) > 1, "with more goals than robots, someone tours"
    assert all(q for q in queues.values()), f"optimistic left a robot idle: {queues}"


# The same concentration, scaled: with the makespan argument having no real say, the cautious
# weight sees no reason not to walk its safest robot through the entire mission however long the
# tour gets. Documented rather than fixed, for the reason above.
@pytest.mark.parametrize("n_goals", [3, 5, 7])
def test_cautious_walks_one_robot_through_every_goal(n_goals):
    goals = [f"g{i}" for i in range(n_goals)]
    profiles = {"r1": {"t": 0.9}, "r2": {"t": 0.85}}
    policy = CautiousPolicy(_ring(n_goals), goals, profiles)
    queues = policy.assign({"r1": "start", "r2": "start"}, set())
    assert len(queues["r1"]) == n_goals, f"r1 should take the whole tour: {queues}"
    assert queues["r2"] == [], f"r2 should stand: {queues}"


# ------------------------------------------------------- does a wreck block the edge?


# One goal, two robots, and a choice: a shortcut that kills half the robots that take it, or a
# detour costing three times as much that almost never does. The reserve robot only gets the job
# after the first one is lost, which is where the two world models come apart.
@pytest.fixture
def deathtrap_with_detour() -> ResilientGraph:
    g = ResilientGraph()
    g.add_edge("start", "g", cost=10.0, terrain_type="bad", hazard_severity=0.5)
    g.add_edge("start", "m", cost=15.0, terrain_type="bad", hazard_severity=0.02)
    g.add_edge("m", "g", cost=15.0, terrain_type="bad", hazard_severity=0.02)
    return g


TWO_BLIND = {"r1": {"bad": 0.0}, "r2": {"bad": 0.0}}


# A single corridor with nothing beside it: whether a wreck shuts the edge decides whether the
# rest of the team can follow at all.
@pytest.fixture
def corridor() -> ResilientGraph:
    g = ResilientGraph()
    g.add_edge("start", "mid", cost=10.0, terrain_type="bad", hazard_severity=0.3)
    g.add_edge("mid", "g", cost=10.0, terrain_type="bad", hazard_severity=0.3)
    return g


# The toggle has to reach the operator, not just the signature. On a raw operator the branches of
# a probabilistic effect are still (probability, [effects]) tuples, so this walks them directly
# rather than through effect_fluents, which expects the grounded form.
@pytest.mark.parametrize("blocks", [True, False])
def test_blocks_on_failure_controls_what_the_wreck_retracts(corridor, blocks):
    operator = create_risk_move_operator(corridor, TWO_BLIND, blocks_on_failure=blocks)
    retracted = set()
    for effect in operator.effects:
        for _probability, branch in getattr(effect, "prob_effects", ()) or ():
            for sub in branch:
                retracted |= {f.name for f in sub.resulting_fluents if f.negated}
    assert {"free", "operational"} <= retracted, "a wreck always costs the team the robot"
    assert ("path_available" in retracted) is blocks


# The knob has to survive the trip from Spec through build_operators, since that is the only path
# the benchmark takes. Without this, sweeping the failure model would silently sweep nothing.
@pytest.mark.parametrize("blocks", [True, False])
def test_spec_carries_the_failure_model_to_the_operator(blocks):
    from resilient_mrp.experiments import Spec, build_instance
    inst = build_instance(Spec(graph_type="small_scale", num_robots=2,
                               blocks_on_failure=blocks))
    move = inst.operators[0]
    retracted = {f.name
                 for effect in move.effects
                 for _probability, branch in getattr(effect, "prob_effects", ()) or ()
                 for sub in branch for f in sub.resulting_fluents if f.negated}
    assert ("path_available" in retracted) is blocks


# With the edge left open, no failure can ever change the set of available paths. This is the
# premise the leaf estimate is already built on: it fixes its route tables at construction with
# open_edges=None and never re-reads them, which is exactly right here and wrong when edges close.
def test_open_edge_set_is_invariant_when_failures_do_not_block(corridor):
    for seed in range(30):
        random.seed(seed)
        env = _env(corridor, TWO_BLIND, ["g"], blocks=False)
        before = parse_available_paths(env.state)
        policy = OptimisticPolicy(corridor, ["g"], TWO_BLIND)
        run_episode(env, _goal(["g"]), 0, 0, ["g"], max_steps=40,
                    route_policy=policy, graph=corridor)
        assert parse_available_paths(env.state) == before, f"an edge closed on seed {seed}"


# The asymmetry itself. On seed 0 the first robot dies on the shortcut. When the wreck shuts that
# edge, the reserve robot has no choice but to take the detour and the mission finishes. When the
# edge stays open, the optimistic policy prices by time alone, routes the reserve down the very
# same shortcut, and loses it too.
def test_a_wreck_prunes_the_route_that_killed_it(deathtrap_with_detour):
    blocked, _ = _episode(deathtrap_with_detour, TWO_BLIND, ["g"], OptimisticPolicy,
                          seed=0, blocks=True)
    open_, _ = _episode(deathtrap_with_detour, TWO_BLIND, ["g"], OptimisticPolicy,
                        seed=0, blocks=False)
    assert blocked, "the closed shortcut should have forced the reserve robot onto the detour"
    assert not open_, "with the shortcut still open the reserve robot repeats the fatal choice"


# Aggregated, that asymmetry is large and one-sided: blocking is a substantial help to the policy
# that would otherwise keep choosing the route that kills robots.
def test_blocking_helps_the_policy_that_does_not_learn(deathtrap_with_detour):
    _, cost_blocked, _ = _sweep(deathtrap_with_detour, TWO_BLIND, ["g"], OptimisticPolicy,
                                blocks=True)
    _, cost_open, _ = _sweep(deathtrap_with_detour, TWO_BLIND, ["g"], OptimisticPolicy,
                             blocks=False)
    assert cost_blocked < cost_open - 50.0, (
        f"optimistic cost {cost_blocked:.1f} with blocking against {cost_open:.1f} without; "
        f"the wreck is doing the policy's exploration for it")

    # cautious already avoids the shortcut, so closing it can only ever take something away
    _, safe_blocked, _ = _sweep(deathtrap_with_detour, TWO_BLIND, ["g"], CautiousPolicy,
                                blocks=True)
    _, safe_open, _ = _sweep(deathtrap_with_detour, TWO_BLIND, ["g"], CautiousPolicy,
                             blocks=False)
    assert safe_open <= safe_blocked + 1.0, (
        f"cautious cost {safe_open:.1f} open against {safe_blocked:.1f} blocked")


# On a graph with no alternative route the sign flips: a wreck severs the only corridor, and no
# number of surviving robots can help. Adding a third robot buys nothing under blocking and a
# great deal when the edge stays open.
@pytest.mark.parametrize("team", [2, 3])
def test_blocking_severs_a_graph_with_no_alternative(corridor, team):
    profiles = {f"r{i + 1}": {"bad": 0.0} for i in range(team)}
    p_blocked, _, _ = _sweep(corridor, profiles, ["g"], OptimisticPolicy, blocks=True)
    p_open, _, _ = _sweep(corridor, profiles, ["g"], OptimisticPolicy, blocks=False)
    assert p_open > p_blocked + 0.15, (
        f"team of {team}: P(success) {p_open:.3f} open against {p_blocked:.3f} blocked")


# The blocking model decides which baseline looks better, which makes it a confound rather than a
# detail. Same graph, same draws, same policies: blocking flatters optimistic enough to overturn
# the four-fold advantage cautious holds when the map stays fixed.
def test_the_blocking_model_decides_which_baseline_wins(deathtrap_with_detour):
    def cost(policy_cls, blocks):
        return _sweep(deathtrap_with_detour, TWO_BLIND, ["g"], policy_cls, blocks=blocks)[1]

    assert cost(OptimisticPolicy, True) < cost(CautiousPolicy, True), (
        "with wrecks closing edges, optimistic should come out ahead here")
    assert cost(CautiousPolicy, False) < cost(OptimisticPolicy, False), (
        "with the map fixed, cautious should come out ahead on the same graph")


# ------------------------------------------------------------------------ shared premise


# Both baselines and the leaf read survival off compute_p_success, so the whole comparison rests
# on it staying inside [0, 1] and on -log of it being finite.
@pytest.mark.parametrize("hazard", [0.0, 0.3, 0.9, 1.0])
def test_cautious_weight_is_finite_and_non_negative(hazard):
    from resilient_mrp.planning.core import compute_p_success
    for phi in (0.0, 0.5, 1.0):
        survival = compute_p_success({"t": phi}, "t", hazard)
        assert 0.0 <= survival <= 1.0
        assert math.isfinite(cautious_weight(1.0, survival))
        assert cautious_weight(1.0, survival) >= 0.0
