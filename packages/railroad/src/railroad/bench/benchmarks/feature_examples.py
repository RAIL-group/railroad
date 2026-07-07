"""
Exemplar benchmarks for recently added planning features.

Each benchmark here is deliberately tiny: it exists to show how a feature is
*defined* and to verify the planner honors it end-to-end.

1. ``extra_cost_route_choice`` — per-action scalar costs beyond time
   (``Operator(extra_cost=...)``), charged in the MCTS objective and
   estimated by the FF heuristic.
2. ``conditional_effects_briefcase`` — native conditional effect branches:
   sub-effects that fire only when their condition fluents hold at
   effect-fire time, quantified over the object universe with
   ``ForallEffect`` (the native forall+when; per-object branches can also
   be listed explicitly via ``Effect(cond_effects=...)``).
3. ``pddl_converter_features`` — converting and solving PDDL/PPDDL problem
   text via ``railroad.pddl_converter`` (action costs, probabilistic
   effects, and conditional effects expressed in PDDL).
"""

import random
import time
from dataclasses import dataclass
from typing import Dict, Optional

from railroad.bench import benchmark, BenchmarkCase
from railroad.core import (
    Effect,
    Fluent as F,
    ForallEffect,
    Operator,
    State,
    get_action_by_name,
    transition,
)
from railroad.planner import MCTSPlanner


def _plan_and_execute(actions, initial_state, goal, *, seed=0, max_steps=50,
                      max_iterations=1000):
    """Minimal plan/act loop over grounded actions.

    Returns (final_state, executed_action_names, total_extra_cost).
    """
    planner = MCTSPlanner(actions)
    rng = random.Random(seed)
    state = initial_state
    executed = []
    total_extra_cost = 0.0
    for _ in range(max_steps):
        if goal.evaluate(state.fluents):
            break
        action_name = planner(state, goal, max_iterations=max_iterations, c=100)
        if action_name == "NONE":
            break
        action = get_action_by_name(actions, action_name)
        successors = transition(state, action)
        state = rng.choices(
            successors, weights=[prob for _, prob in successors]
        )[0][0]
        executed.append(action_name)
        total_extra_cost += action.extra_cost
    return state, executed, total_extra_cost


# ============================================================================
# 1. extra_cost: action costs beyond time
# ============================================================================


@benchmark(
    name="extra_cost_route_choice",
    description="Two routes to the goal: a fast toll road with an extra_cost "
                "and a slow free road. The planner should minimize "
                "time + extra_cost, so the best route flips with the toll.",
    tags=["feature-example", "extra-cost"],
    repeat=1,
    timeout=60.0,
)
def bench_extra_cost_route_choice(case: BenchmarkCase):
    # The toll road is fast (duration 2) but charges `extra_cost`; the back
    # road is slow (duration 6) and free. The MCTS objective is
    # -(completion time + accumulated extra_cost), and since this branch the
    # FF heuristic also folds extra_cost into its estimates.
    toll_road_op = Operator(
        name="take-toll-road",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r"), F("at ?r start")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r start")}),
            Effect(time=2.0, resulting_fluents={F("free ?r"), F("at ?r destination")}),
        ],
        extra_cost=case.toll_cost,
    )
    back_road_op = Operator(
        name="take-back-road",
        parameters=[("?r", "robot")],
        preconditions=[F("free ?r"), F("at ?r start")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r start")}),
            Effect(time=6.0, resulting_fluents={F("free ?r"), F("at ?r destination")}),
        ],
    )

    objects_by_type = {"robot": {"robot1"}}
    actions = [
        a
        for op in (toll_road_op, back_road_op)
        for a in op.instantiate(objects_by_type)
    ]
    initial_state = State(0.0, {F("free robot1"), F("at robot1 start")}, [])
    goal = F("at robot1 destination")

    start_time = time.perf_counter()
    state, executed, extra_cost = _plan_and_execute(actions, initial_state, goal)
    wall_time = time.perf_counter() - start_time

    expected_action = f"take-{case.expected_route}-road robot1"
    return {
        "success": goal.evaluate(state.fluents) and executed == [expected_action],
        "wall_time": wall_time,
        "plan_cost": float(state.time) + extra_cost,
        "actions_count": len(executed),
        "actions": executed,
    }


bench_extra_cost_route_choice.add_cases([
    # Expensive toll (2 + 10 > 6): the slow free road wins.
    {"toll_cost": 10.0, "expected_route": "back"},
    # Cheap toll (2 + 1 < 6): the fast toll road wins.
    {"toll_cost": 1.0, "expected_route": "toll"},
])


# ============================================================================
# 2. Conditional effects (Effect.cond_effects)
# ============================================================================


@benchmark(
    name="conditional_effects_briefcase",
    description="Classic briefcase domain: moving the briefcase also moves "
                "exactly the objects inside it, via a ForallEffect "
                "(native forall+when) expanded per item at grounding time.",
    tags=["feature-example", "conditional-effects"],
    repeat=1,
    timeout=60.0,
)
def bench_conditional_effects_briefcase(case: BenchmarkCase):
    items = [f"item{i}" for i in range(case.num_items)]

    put_in_op = Operator(
        name="put-in",
        parameters=[("?obj", "item"), ("?loc", "location")],
        preconditions=[
            F("free briefcase"), F("at briefcase ?loc"), F("at ?obj ?loc"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={F("not free briefcase")}),
            Effect(time=1.0, resulting_fluents={F("free briefcase"), F("in ?obj")}),
        ],
    )
    take_out_op = Operator(
        name="take-out",
        parameters=[("?obj", "item")],
        preconditions=[F("free briefcase"), F("in ?obj")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free briefcase")}),
            Effect(time=1.0, resulting_fluents={F("free briefcase"), F("not in ?obj")}),
        ],
    )
    # The universally quantified conditional effect (PDDL forall+when):
    # at Operator.instantiate() time, ForallEffect expands into one
    # conditional branch per object of type "item" — each applied only if
    # that item is in the briefcase when the move completes. Conditions are
    # evaluated against the state BEFORE the effect's own fluents apply.
    move_op = Operator(
        name="move",
        parameters=[("?from", "location"), ("?to", "location")],
        preconditions=[F("free briefcase"), F("at briefcase ?from")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free briefcase")}),
            Effect(
                time=2.0,
                resulting_fluents={
                    F("free briefcase"),
                    F("at briefcase ?to"),
                    F("not at briefcase ?from"),
                },
                forall_effects=[ForallEffect(
                    variables=[("?obj", "item")],
                    conditions={F("in ?obj")},
                    effects=[Effect(time=0, resulting_fluents={
                        F("at ?obj ?to"),
                        F("not at ?obj ?from"),
                    })],
                )],
            ),
        ],
    )

    objects_by_type = {
        "item": set(items),
        "location": {"home", "office"},
    }
    actions = [
        a
        for op in (put_in_op, take_out_op, move_op)
        for a in op.instantiate(objects_by_type)
    ]
    initial_fluents = {F("free briefcase"), F("at briefcase home")}
    initial_fluents.update(F(f"at {item} home") for item in items)
    initial_state = State(0.0, initial_fluents, [])

    # Bring item0 to the office; everything else must STAY home, so the
    # planner must rely on the conditional firing only for item0.
    goal = F("at item0 office")
    for item in items[1:]:
        goal = goal & F(f"at {item} home")

    start_time = time.perf_counter()
    state, executed, _ = _plan_and_execute(actions, initial_state, goal)
    wall_time = time.perf_counter() - start_time

    # Optimal: put-in item0 home, move home office (the other items' branches
    # do not fire because their `in` condition is false).
    return {
        "success": goal.evaluate(state.fluents) and len(executed) == 2,
        "wall_time": wall_time,
        "plan_cost": float(state.time),
        "actions_count": len(executed),
        "actions": executed,
    }


bench_conditional_effects_briefcase.add_cases([
    {"num_items": 3},
])


# ============================================================================
# 3. PDDL converter (railroad.pddl_converter)
# ============================================================================

@dataclass(frozen=True)
class _PDDLVariant:
    planner: str
    domain: str
    problem: str
    expected_cost: Optional[float]


_PDDL_VARIANTS: Dict[str, _PDDLVariant] = {
    # :action-costs -> durations; minimize (total-cost). Optimal cost is
    # pick (1) + move (5) + drop (1) = 7.
    "action-costs": _PDDLVariant(
        planner="mcts",
        domain="""
            (define (domain gripper-costs)
              (:requirements :strips :typing :action-costs)
              (:types room ball)
              (:predicates (at-robby ?r - room) (at ?b - ball ?r - room)
                           (carry ?b - ball) (hand-empty))
              (:functions (total-cost) - number
                          (move-cost ?from - room ?to - room) - number)
              (:action move
                 :parameters (?from - room ?to - room)
                 :precondition (at-robby ?from)
                 :effect (and (not (at-robby ?from)) (at-robby ?to)
                              (increase (total-cost) (move-cost ?from ?to))))
              (:action pick
                 :parameters (?b - ball ?r - room)
                 :precondition (and (at ?b ?r) (at-robby ?r) (hand-empty))
                 :effect (and (carry ?b) (not (at ?b ?r)) (not (hand-empty))
                              (increase (total-cost) 1)))
              (:action drop
                 :parameters (?b - ball ?r - room)
                 :precondition (and (carry ?b) (at-robby ?r))
                 :effect (and (at ?b ?r) (hand-empty) (not (carry ?b))
                              (increase (total-cost) 1))))
        """,
        problem="""
            (define (problem gripper-costs-1)
              (:domain gripper-costs)
              (:objects rooma roomb - room ball1 - ball)
              (:init (at-robby rooma) (at ball1 rooma) (hand-empty)
                     (= (total-cost) 0)
                     (= (move-cost rooma roomb) 5) (= (move-cost roomb rooma) 5))
              (:goal (at ball1 roomb))
              (:metric minimize (total-cost)))
        """,
        expected_cost=7.0,
    ),
    # PPDDL probabilistic effects: pickup may slip (70% success).
    "probabilistic": _PDDLVariant(
        planner="mcts",
        domain="""
            (define (domain slippery-blocks)
              (:requirements :strips :probabilistic-effects)
              (:predicates (on-table ?b) (holding ?b) (hand-empty) (delivered ?b))
              (:action pickup
                 :parameters (?b)
                 :precondition (and (on-table ?b) (hand-empty))
                 :effect (and (not (hand-empty))
                              (probabilistic
                                 7/10 (and (holding ?b) (not (on-table ?b)))
                                 0.3  (hand-empty))))
              (:action deliver
                 :parameters (?b)
                 :precondition (holding ?b)
                 :effect (and (not (holding ?b)) (hand-empty) (delivered ?b))))
        """,
        problem="""
            (define (problem slippery-1)
              (:domain slippery-blocks)
              (:objects b1 b2)
              (:init (on-table b1) (on-table b2) (hand-empty))
              (:goal (and (delivered b1) (delivered b2))))
        """,
        expected_cost=None,  # stochastic: retries lengthen the plan
    ),
    # forall+when (miconic-style elevator): a single `stop` action boards
    # and deboards every matching passenger through universally quantified
    # conditional effects — the construct that used to be the dominant
    # blocker across IPC/IPPC domains. The converter expands the forall over
    # the passengers and attaches one conditional branch each.
    "forall-when": _PDDLVariant(
        planner="mcts",
        domain="""
            (define (domain mini-miconic)
              (:requirements :strips :typing :conditional-effects)
              (:types passenger floor)
              (:predicates (lift-at ?f - floor)
                           (origin ?p - passenger ?f - floor)
                           (destin ?p - passenger ?f - floor)
                           (boarded ?p - passenger)
                           (served ?p - passenger))
              (:action move
                 :parameters (?from - floor ?to - floor)
                 :precondition (lift-at ?from)
                 :effect (and (not (lift-at ?from)) (lift-at ?to)))
              (:action stop
                 :parameters (?f - floor)
                 :precondition (lift-at ?f)
                 :effect (and
                    (forall (?p - passenger)
                      (when (and (boarded ?p) (destin ?p ?f))
                            (and (not (boarded ?p)) (served ?p))))
                    (forall (?p - passenger)
                      (when (and (origin ?p ?f) (not (served ?p)))
                            (boarded ?p))))))
        """,
        problem="""
            (define (problem mini-miconic-2)
              (:domain mini-miconic)
              (:objects p1 p2 - passenger f1 f2 - floor)
              (:init (lift-at f1)
                     (origin p1 f1) (destin p1 f2)
                     (origin p2 f2) (destin p2 f1))
              (:goal (and (served p1) (served p2))))
        """,
        # stop f1 (board p1), move to f2, stop f2 (serve p1, board p2),
        # move back, stop f1 (serve p2).
        expected_cost=5.0,
    ),
    # Conditional effects: dropping breaks exactly the fragile items.
    # Dropping the unpadded vase is an irreversible mistake (the goal needs
    # it unbroken). The greedy planner avoids it because the FF heuristic
    # returns h=inf for the broken state; MCTS currently clamps h=inf to
    # HEURISTIC_CANNOT_FIND_GOAL_PENALTY=0, which makes dead ends look
    # attractive, so it is the wrong tool for this variant.
    "conditional-effects": _PDDLVariant(
        planner="greedy",
        domain="""
            (define (domain fragile-delivery)
              (:requirements :strips :conditional-effects)
              (:predicates (holding ?x) (fragile ?x) (padded ?x)
                           (delivered ?x) (broken ?x))
              (:action pad
                 :parameters (?x)
                 :precondition (holding ?x)
                 :effect (padded ?x))
              (:action drop-off
                 :parameters (?x)
                 :precondition (holding ?x)
                 :effect (and (not (holding ?x)) (delivered ?x)
                              (when (and (fragile ?x) (not (padded ?x)))
                                    (broken ?x)))))
        """,
        problem="""
            (define (problem fragile-1)
              (:domain fragile-delivery)
              (:objects vase brick)
              (:init (holding vase) (holding brick) (fragile vase))
              (:goal (and (delivered vase) (delivered brick)
                          (not (broken vase)))))
        """,
        # pad vase, drop-off vase, drop-off brick (order-insensitive).
        expected_cost=3.0,
    ),
}


@benchmark(
    name="pddl_converter_features",
    description="Convert inline PDDL/PPDDL text (action costs, probabilistic "
                "effects, conditional effects) with railroad.pddl_converter "
                "and solve it with MCTS.",
    tags=["feature-example", "pddl-converter"],
    repeat=1,
    timeout=60.0,
)
def bench_pddl_converter_features(case: BenchmarkCase):
    from railroad import pddl_converter as pc

    variant = _PDDL_VARIANTS[case.variant]
    problem = pc.convert_texts(variant.domain, variant.problem)

    start_time = time.perf_counter()
    result = pc.solve(
        problem, seed=case.seed, max_iterations=2000, planner=variant.planner
    )
    wall_time = time.perf_counter() - start_time

    success = result.success
    if variant.expected_cost is not None:
        success = success and abs(result.sim_time - variant.expected_cost) < 1e-6
    return {
        "success": success,
        "wall_time": wall_time,
        "plan_cost": float(result.sim_time),
        "actions_count": len(result.plan),
        "actions": result.plan,
    }


bench_pddl_converter_features.add_cases([
    {"variant": "action-costs", "seed": 0},
    {"variant": "probabilistic", "seed": 0},
    {"variant": "forall-when", "seed": 0},
    {"variant": "conditional-effects", "seed": 0},
])
