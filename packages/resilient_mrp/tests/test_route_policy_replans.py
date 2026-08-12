# A failure closes the edge it happened on. The route policies have to notice, or they keep steering
# through a closed edge and walk the same one back and forth until the step budget runs out.

import pytest
from dataclasses import replace

from railroad._bindings import State
from railroad.core import Fluent as F
from resilient_mrp.experiments import Spec, build_instance, start_trial
from resilient_mrp.experiments.instance import planner_setup
from resilient_mrp.experiments.mission import _route_policy_action
from resilient_mrp.planning.baselines import CautiousPolicy
from resilient_mrp.planning.core import parse_available_paths

# two robots, two goals and real failure odds; with any of those turned off nothing closes
SPEC = Spec(graph_type="sctp_random", graph_size=10, num_robots=2, num_goals=2,
            max_steps=50, risk_scale=3.0)


# longest run of the same robot walking u->v then v->u
def _max_alternations(moves):
    best = run = 0
    for (r0, u0, v0), (r1, u1, v1) in zip(moves, moves[1:]):
        run = run + 1 if (r0 == r1 and u0 == v1 and v0 == u1) else 0
        best = max(best, run)
    return best


def _episode(planner, topo_seed, exec_seed):
    inst = build_instance(replace(SPEC, seed=topo_seed))
    env = start_trial(inst, exec_seed)
    *_, policy = planner_setup(inst, planner)
    moves = []
    for _ in range(SPEC.max_steps):
        if inst.goal_fluent.evaluate(env.state.fluents):
            break
        actions = env.get_actions()
        if not actions:
            break
        action = _route_policy_action(env, actions, inst.goal_sites, policy)
        if action is None:
            break
        if action.name.startswith("risk_move"):
            _, robot, u, v = action.name.split()
            moves.append((robot, u, v))
        env.act(action)
    return moves


# seed 43 draw 101 was the original repro: 42 alternations, makespan 982, mission never ended
@pytest.mark.parametrize("planner", ["optimistic", "cautious"])
@pytest.mark.parametrize("topo_seed", [43, 44, 45, 46])
@pytest.mark.parametrize("exec_seed", [100, 101, 103, 104, 105])
def test_route_policy_does_not_walk_an_edge_back_and_forth(planner, topo_seed, exec_seed):
    moves = _episode(planner, topo_seed, exec_seed)
    assert _max_alternations(moves) < 3, (
        f"{planner} on graph {topo_seed} draw {exec_seed} retraced an edge "
        f"{_max_alternations(moves)} times in a row: {moves[-8:]}")


# the tables the policy routes on must match the edges the state still has
def test_policy_rebuilds_its_routes_when_an_edge_closes():
    inst = build_instance(replace(SPEC, seed=43))
    env = start_trial(inst, 101)
    policy = CautiousPolicy(inst.graph, inst.goal_sites, inst.profiles)
    policy.observe(env.state)
    before = set(policy._open or ())
    tables_before = policy._routes

    # close one edge, exactly as the failure branch of risk_move does
    (u, v), _ = next(iter(inst.graph.edges.items()))
    closed = State(env.state.time,
                   {f for f in env.state.fluents if f != F(f"path_available {u} {v}")}, [])
    policy.observe(closed)

    open_now = policy._open
    assert open_now is not None
    assert (u, v) in before
    assert open_now == parse_available_paths(closed)
    assert (u, v) not in open_now
    # the tables themselves get rebuilt, and the committed assignment is dropped with them
    assert policy._routes is not tables_before
    assert policy._assignment_cache == {}
