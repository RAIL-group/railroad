import random
from dataclasses import dataclass, field
from typing import Any

from railroad._bindings import State
from railroad.core import Fluent as F
from railroad.environment import SymbolicEnvironment
from railroad.planner import seed_planner_rng
from railroad import operators as rr_operators

# Offset between the two streams a trial drives, both derived from exec_seed. They are kept apart
# because Python's `random` and the planner's mt19937 are both Mersenne Twister: handing them the
# same integer invites the failure draws and the search's own coin flips to move together, which
# would tie what the world does to what the planner happened to sample. Any fixed offset does; the
# value is not tuned and nothing should depend on it.
_PLANNER_SEED_OFFSET = 1_000_003

from resilient_mrp.planning.core import (
    RobotProfile,
    ResilientGraph,
    create_risk_move_operator,
    create_safely_visited_operator,
)
from resilient_mrp.planning.value_function import RiskAwareCostToGo
from resilient_mrp.planning.baselines import OptimisticPolicy, CautiousPolicy
from resilient_mrp.experiments.mission import run_episode
import resilient_mrp.scenarios.small_scale as _small
import resilient_mrp.scenarios.sctp_scenario as _sctp

ALL_PLANNERS = ("optimistic", "cautious",
                "failure_aware_ff", "failure_aware_split")


# Everything that defines one run: which instance to build and how hard to search it. The demos and
# the benchmark both vary these, so they live in one place rather than three parameter lists.
@dataclass(frozen=True)
class Spec:
    graph_type: str = "sctp_random"
    graph_size: int = 20
    num_robots: int = 2
    num_goals: int = 2
    risk_scale: float = 1.0
    c_fail: float = 500.0   # what failing the mission costs, same number for the whole experiment
    # Whether a wreck shuts the edge it happened on. Not a detail: it decides which baseline wins.
    # On a graph with a risky shortcut beside a safe detour, blocking deletes the shortcut after the
    # first robot dies on it, which rescues the optimistic policy from repeating its own choice --
    # optimistic 33.4 against cautious 39.1 with blocking, 127.1 against 31.4 without. So the
    # benchmark sweeps it rather than picking one.
    blocks_on_failure: bool = True
    mcts_iterations: int = 10000
    max_depth: int = 40
    max_steps: int = 100
    seed: int | None = None
    robot_types: tuple[str, ...] | None = None


# One problem: the terrain graph, where the goals are, who is on the team, and the failure cost
# derived from all three. Deterministic given a Spec, so it can be checked without running anything.
@dataclass(frozen=True)
class Instance:
    spec: Spec
    graph: ResilientGraph
    goal_sites: list[str]
    profiles: dict[str, RobotProfile]
    robots: list[str]
    operators: list = field(repr=False)
    c_fail: float
    goal_fluent: F = field(repr=False)

    # Fresh environment per trial; every planner has to start from the same state.
    #
    # The seed is threaded explicitly because SymbolicEnvironment samples its probabilistic
    # branches from its own random.Random(seed), not from the global module. Left None it seeds
    # from OS entropy, and then which robots die is unreproducible and differs between planners on
    # the same trial -- which would silently undo the paired comparison the benchmark is built on.
    def new_env(self, seed: int | None = None) -> SymbolicEnvironment:
        initial_fluents: set = set()
        for robot in self.robots:
            initial_fluents.add(F(f"at {robot} start"))
            initial_fluents.add(F(f"free {robot}"))
            initial_fluents.add(F(f"operational {robot}"))
        initial_fluents.update(self.graph.get_edge_fluents())
        initial_fluents.update(self.graph.get_available_path_fluents())
        for g in self.goal_sites:
            initial_fluents.add(F(f"is_goal {g}"))
        return SymbolicEnvironment(
            state=State(0.0, initial_fluents, []),
            objects_by_type={"robot": set(self.robots), "location": set(self.graph.nodes)},
            operators=self.operators,
            seed=seed,
        )


# What one trial produced. env is kept so callers can read final fluents and makespan.
@dataclass
class TrialOutcome:
    visited: int
    travel: float
    env: SymbolicEnvironment = field(repr=False)
    num_goals: int
    c_fail: float = 0.0

    @property
    def success(self) -> bool:
        return self.visited == self.num_goals

    @property
    def makespan(self) -> float:
        return self.env.state.time

    # What this trial cost: its makespan if the mission finished, otherwise C_fail flat. Failing is
    # one price and does not depend on when, so every failed run in a case scores the same.
    @property
    def trial_cost(self) -> float:
        return self.makespan if self.success else self.c_fail


# Return the scenario module backing a graph type.
def _scenario(name: str):
    return _sctp if name in ("sctp_random", "sctp_island") else _small


# the true-model operators, shared by execution and search; failure carries no cost in the model,
# C_fail is applied once at mission failure in the metric and the split value function
def build_operators(graph, profiles, blocks_on_failure: bool = True) -> list:
    move_op    = create_risk_move_operator(graph, profiles, blocks_on_failure=blocks_on_failure)
    visited_op = create_safely_visited_operator()
    no_op      = rr_operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
    return [move_op, visited_op, no_op]


# Build a robot team: explicit archetype mix (e.g. ["r1","r2","r3","r1"]) or num_robots distinct (cycled).
def select_robots(num_robots: int = 2,
                  archetypes: tuple[str, ...] | None = None) -> dict[str, RobotProfile]:
    base = sorted(_sctp.ROBOT_PROFILES.keys())
    if archetypes is None:
        archetypes = tuple(base[i % len(base)] for i in range(num_robots))
    return {f"r{i + 1}": dict(_sctp.ROBOT_PROFILES[a]) for i, a in enumerate(archetypes)}


# Graph + goals for a graph_type: small_scale is hand-built, the sctp ones are generated.
def generate_graph(graph_type: str, graph_size: int, risk_scale: float,
                   seed: int | None = None, num_goals: int = 2):
    if graph_type == "small_scale":
        return _small.create_graph(risk_scale=risk_scale), list(_small.GOAL_SITES)
    return _sctp.create_graph(graph_type, graph_size, risk_scale, seed, num_goals)


def make_goal(goal_sites) -> F:
    goal = F(f"safely_visited {goal_sites[0]}")
    for site in goal_sites[1:]:
        goal = goal & F(f"safely_visited {site}")
    return goal


# The construction every driver shares: graph, team, operators, C_fail, goal.
def build_instance(spec: Spec) -> Instance:
    graph, goal_sites = generate_graph(spec.graph_type, spec.graph_size,
                                       spec.risk_scale, spec.seed, spec.num_goals)
    sc = _scenario(spec.graph_type)
    if spec.graph_type == "small_scale":
        profiles = {r: dict(sc.ROBOT_PROFILES[r])
                    for r in sorted(sc.ROBOT_PROFILES)[:spec.num_robots]}
    else:
        profiles = select_robots(spec.num_robots, spec.robot_types)
    robots = sorted(profiles.keys())
    return Instance(
        spec=spec,
        graph=graph,
        goal_sites=goal_sites,
        profiles=profiles,
        robots=robots,
        operators=build_operators(graph, profiles, spec.blocks_on_failure),
        c_fail=spec.c_fail,
        goal_fluent=make_goal(goal_sites),
    )


# Returns (plan_ops, heuristic_fn, heuristic_mult, unreachable_penalty, dead_end_penalty,
# route_policy). Both leaves are already in cost units, so their weight is 1.0 and the multiplier is
# applied to the search heuristic only.
#
# dead_end_penalty is what makes losing cost the same in the search as it does in the score. A
# branch the relaxation proves cannot reach the goal is charged a flat c_fail: not c_fail plus the
# clock, not c_fail plus the extra cost already spent. That matters because TrialOutcome.trial_cost
# charges exactly c_fail for a failed run however long it took, so without the flat treatment the
# search would be minimising a different objective than the one being reported -- it would prefer
# to lose quickly, a preference the metric does not express. Left None (the C++ default) a dead end
# scores 0, which is *better* than any reachable state and actively draws the search into it.
def planner_setup(inst: Instance, planner: str, *, heuristic_mult: float | None = None,
                  unreachable_penalty: float | None = None
                  ) -> tuple[Any, Any, float, float, float | None, Any]:
    graph, goals, profiles = inst.graph, inst.goal_sites, inst.profiles
    if planner == "optimistic":
        return None, None, 1.0, 0.0, None, OptimisticPolicy(graph, goals, profiles)
    if planner == "cautious":
        return None, None, 1.0, 0.0, None, CautiousPolicy(graph, goals, profiles)

    mult = 1.0 if heuristic_mult is None else heuristic_mult
    penalty = inst.c_fail if unreachable_penalty is None else unreachable_penalty
    if planner == "failure_aware_ff":
        return inst.operators, None, mult, penalty, inst.c_fail, None
    if planner == "failure_aware_split":
        # The leaf is told whether edges can close rather than inferring it from the state: when
        # they cannot, `path_available` never changes, and the planner projects unchanging fluents
        # out of the states it searches -- including the ones it hands the leaf.
        leaf = RiskAwareCostToGo(graph, goals, profiles, inst.c_fail,
                                 edges_can_close=inst.spec.blocks_on_failure)
        return inst.operators, leaf, mult, penalty, inst.c_fail, None
    raise ValueError(f"unknown planner: {planner}")


# Fresh env for a trial, with the same execution draws every time when exec_seed is given. Separate
# from run_trial because the dashboard has to be handed the env before the run starts.
def start_trial(inst: Instance, exec_seed: int | None = None) -> SymbolicEnvironment:
    if exec_seed is not None:
        random.seed(exec_seed)
        # The search samples too, from the planner's own thread-local mt19937, which otherwise
        # seeds itself from random_device. Left alone, two runs of the same trial searched
        # differently -- failure_aware_split gave makespans 169.5, 66.2, 161.3 on one graph and one
        # exec_seed -- so a trial was not reproducible and the planners were not being compared on
        # equal terms. Seeding it here makes trial t mean the same thing for every configuration.
        # The benchmark runs trials in separate processes, so a per-trial global seed is safe.
        seed_planner_rng(exec_seed + _PLANNER_SEED_OFFSET)
    # exec_seed also goes to the environment itself: seeding the global module does not reach
    # SymbolicEnvironment's own RNG, which is what decides who survives.
    return inst.new_env(seed=exec_seed)


# One trial. The single place run_episode is called, so the demos and the benchmark cannot drift
# apart on its arguments the way max_steps did.
def run_trial(inst: Instance, planner: str, env: SymbolicEnvironment, *,
              dashboard: Any = None, heuristic_mult: float | None = None,
              unreachable_penalty: float | None = None) -> TrialOutcome:
    spec = inst.spec
    plan_ops, heuristic_fn, mult, penalty, dead_end, policy = planner_setup(
        inst, planner, heuristic_mult=heuristic_mult, unreachable_penalty=unreachable_penalty)
    visited, travel = run_episode(
        env, inst.goal_fluent, spec.mcts_iterations, spec.max_depth, inst.goal_sites,
        planning_operators=plan_ops, c=500, max_steps=spec.max_steps,
        heuristic_fn=heuristic_fn, heuristic_multiplier=mult,
        unreachable_penalty=penalty, dead_end_penalty=dead_end,
        route_policy=policy, dashboard=dashboard, graph=inst.graph,
    )
    return TrialOutcome(visited=visited, travel=travel, env=env,
                        num_goals=len(inst.goal_sites), c_fail=inst.c_fail)
