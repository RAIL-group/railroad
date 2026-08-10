"""Decentralized planning methods: each robot plans in isolation over only
its own actions; a coordination protocol (different per method) glues the
robots together at execution time.

Three methods, differing only in how a robot learns about another robot's
resource use and what it does when that knowledge turns out to be wrong or
incomplete:

- plan_reactive: optimistic broadcast of other robots' committed future
  effects; discovers unavailability only at real execution time; recovers
  by blocking + retrying whenever poked by another robot's completion.
- plan_reservation: explicit predicted release-time bookkeeping, baked into
  a real wait_for_resource action with a corrected exact duration — no
  execution-time surprises.
- plan_no_op_blind: knows a resource is reserved (a real fluent) but
  predicts no timing at all — fixed-interval no_op + full-goal replan from
  scratch, including abandoning an in-flight plan invalidated mid-execution.

All three take a `Domain` (see common.py) and return a `PlanResult`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from railroad.core import Fluent as F
from railroad import operators

from common import (  # pyrefly: ignore [missing-import]
    Domain,
    PlanResult,
    run_decentralized,
    Reservation,
    predict_release_time,
    current_reservation,
    construct_wait_for_resource_operator,
    correct_wait_duration,
)


def _reserved_pick_place(domain: Domain):
    """Pick/place that fold a global `reserved`/`reserved_by` fluent into
    their own effects: picking something up also reserves it, placing it
    down clears the reservation. Used by plan_reservation and
    plan_no_op_blind — the two methods that need pick to be gated on
    `~reserved ?obj` — harmless no-op bookkeeping for any uncontested object.
    """
    pick_op = operators.construct_pick_operator_blocking(pick_time=domain.pick_time)
    pick_op.preconditions.append(~F("reserved ?obj"))
    pick_op.effects[1].resulting_fluents |= {F("reserved ?obj"), F("reserved_by ?r ?obj")}

    place_op = operators.construct_place_operator_blocking(place_time=domain.place_time)
    place_op.effects[1].resulting_fluents |= {~F("reserved ?obj"), ~F("reserved_by ?r ?obj")}
    return pick_op, place_op


# ---------------------------------------------------------------------------
# 1. Reactive: optimistic broadcast + blocked/retry-on-event
# ---------------------------------------------------------------------------


def plan_reactive(domain: Domain, verbose: bool = False) -> PlanResult:
    pick_op = operators.construct_pick_operator_blocking(pick_time=domain.pick_time)
    place_op = operators.construct_place_operator_blocking(place_time=domain.place_time)
    my_operators = list(domain.base_operators) + [pick_op, place_op]

    def broadcast_state_view(world_state, robot_plans, t, robot, ctx):
        """World state augmented with the committed future effects of every
        robot's plan — lets a newly-planning robot see resources that are
        currently held but will be released, without knowing *when*.
        """
        state = set(world_state)
        for plan in robot_plans.values():
            for action in plan:
                for d in action.del_effects:
                    state.discard(d)
                for a in action.add_effects:
                    state.add(a)
        return state

    def plain_exec_view(world_state, robot_plans, t, robot, ctx):
        """Real current state, not the broadcast projection — this is what
        actually lets unavailability be discovered only at execution time:
        planning is optimistic, but starting an action checks reality.
        """
        return set(world_state)

    return run_decentralized(
        domain, my_operators,
        state_view_fn=broadcast_state_view,
        exec_state_view_fn=plain_exec_view,
        on_plan_failed="fail",
        on_precondition_failed="block",
        wake_blocked_on_action_end=True,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# 2. Reservation: predicted release time + exact wait_for_resource
# ---------------------------------------------------------------------------


def plan_reservation(domain: Domain, verbose: bool = False) -> PlanResult:
    pick_op, place_op = _reserved_pick_place(domain)
    my_operators = list(domain.base_operators) + [pick_op, place_op]

    def state_view(world_state, robot_plans, t, robot, ctx):
        """world_state overlaid with any reservation the tracker knows about
        that real world_state fluents haven't caught up to yet.
        """
        reservations = ctx.setdefault("reservations", {})
        state = set(world_state)
        for obj in list(reservations.keys()):
            res = current_reservation(reservations, obj, t)
            if res is not None and res.holder != robot:
                state.add(f"reserved {obj}")
                state.add(f"reserved_by {res.holder} {obj}")
        return state

    def select_operators(t, robot, ctx):
        reservations = ctx.setdefault("reservations", {})
        ops = my_operators
        for obj, loc in domain.contested_resources.items():
            res = current_reservation(reservations, obj, t)
            if res is not None and res.holder != robot:
                wait_op = construct_wait_for_resource_operator(obj, loc, res.release_time, t)
                ops = ops + [wait_op]
        return ops

    def on_plan_found(t, robot, plan, ctx):
        reservations = ctx.setdefault("reservations", {})
        for obj, loc in domain.contested_resources.items():
            res = current_reservation(reservations, obj, t)
            if res is not None and res.holder != robot:
                correct_wait_duration(plan, t, res.release_time)
            release = predict_release_time(plan, t, obj, loc)
            if release is not None:
                reservations.setdefault(obj, []).append(Reservation(holder=robot, release_time=release))

    return run_decentralized(
        domain, my_operators,
        state_view_fn=state_view,
        select_operators_fn=select_operators,
        on_plan_found_fn=on_plan_found,
        on_plan_failed="fail",
        on_precondition_failed="raise",
        wake_blocked_on_action_end=False,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# 3. Aware-but-blind: real `reserved` fluent, no timing prediction, no_op polling
# ---------------------------------------------------------------------------


def plan_no_op_blind(domain: Domain, no_op_time: float = 5.0, verbose: bool = False) -> PlanResult:
    pick_op, place_op = _reserved_pick_place(domain)
    my_operators = list(domain.base_operators) + [pick_op, place_op]

    def plain_state_view(world_state, robot_plans, t, robot, ctx):
        return set(world_state)

    return run_decentralized(
        domain, my_operators,
        state_view_fn=plain_state_view,
        on_plan_failed="no_op",
        on_precondition_failed="no_op",
        wake_blocked_on_action_end=False,
        no_op_time=no_op_time,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# 4. Myopic reactive: plans against the literal current state only (no
#    broadcast of other robots' committed future effects, unlike
#    plan_reactive), and fails that *one robot's* task outright the first
#    time a resource turns out to be unavailable at execution time, rather
#    than blocking/retrying. Other robots are unaffected — see
#    PlanResult.robot_results for each robot's own success/failure/cost.
# ---------------------------------------------------------------------------


def plan_myopic_reactive(domain: Domain, verbose: bool = False) -> PlanResult:
    pick_op = operators.construct_pick_operator_blocking(pick_time=domain.pick_time)
    place_op = operators.construct_place_operator_blocking(place_time=domain.place_time)
    my_operators = list(domain.base_operators) + [pick_op, place_op]

    def current_state_view(world_state, robot_plans, t, robot, ctx):
        """Literal current state only — contrast plan_reactive's
        broadcast_state_view, which layers in every robot's committed
        future plan effects so a robot can see a contested resource will
        *eventually* free up. A robot planning here has no such visibility:
        it only ever sees what's true right now, so it can easily plan to
        use a resource another robot is about to take first — that race is
        exactly what on_precondition_failed="terminate" catches (for that
        robot alone; it doesn't stop everyone else).
        """
        return set(world_state)

    return run_decentralized(
        domain, my_operators,
        state_view_fn=current_state_view,
        on_plan_failed="fail",
        on_precondition_failed="terminate",
        wake_blocked_on_action_end=False,
        verbose=verbose,
    )
