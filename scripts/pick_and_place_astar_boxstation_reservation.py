"""Gift wrap — box-station domain, reservation-based baseline.

Same proactive reservation mechanism as pick_and_place_astar_reservation.py
(an explicit reservation on the scissors lets the *other* robot's planner
bake a wait-then-pick sequence directly into its own plan), but for the
box-station domain (pick_and_place_astar_boxstation.py), where box
collection is genuinely independent of scissors — a robot always has real
work it can do whether or not scissors are available.

That independence is exactly what pick_and_place_astar_reservation.py's wait
mechanism couldn't handle: its wait duration was computed once, assuming
*exactly one* deterministic move precedes it (a documented, domain-specific
assumption — see construct_wait_for_resource_operator there). Here, the
search is free to schedule box-collection before, after, or split around the
wait, so the amount of real time preceding the wait step is no longer known
until a full plan comes back.

The fix: a Numeric duration function only ever sees grounded action
parameters, never the search's own elapsed time (confirmed — no such hook
exists in the engine, see core.py's Operator.instantiate and the C++ state
transition code), so the wait step is given a provisional worst-case
duration guess at construction time (assume nothing precedes it), which is
enough for the search to have *some* concrete number to reason about — the
wait's own duration doesn't gate feasibility, only its cost accounting.
Once a full plan is returned, `correct_wait_duration` patches that one
number in place, using the sum of the (already fully deterministic, already
fixed) durations of whatever actions the search actually put before it —
however many, and whatever they are, including box-collection.

Shares the world definition and non-contended operators (move/cut_paper/
wrap_gift/cut_ribbon/complete_job) with pick_and_place_astar_boxstation.py,
and the reservation-tracking primitives (Reservation/predict_release_time/
current_reservation — fully domain-agnostic) with
pick_and_place_astar_reservation.py.

Run directly: `uv run scripts/pick_and_place_astar_boxstation_reservation.py`
"""

import heapq
import os
import random
import re
import sys
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner_interface import call_planner, SimpleAction  # pyrefly: ignore [missing-import]
from railroad.core import Operator, Effect, Fluent as F
from railroad.operators._utils import Numeric
from railroad import operators

from pick_and_place_astar_boxstation import (
    PLANNER,
    expected_cost,
    objects_by_type,
    initial_world_state,
    ROBOT_TASKS,
    move_op,
    cut_paper_op,
    wrap_gift_op,
    cut_ribbon_op,
    complete_job_op,
    Event,
    check_preconditions,
    make_logger,
)
from pick_and_place_astar_reservation import Reservation, predict_release_time, current_reservation

# ---------------------------------------------------------------------------
# Reservation-based baseline — alternate operators, separate from
# pick_and_place_astar_boxstation.my_operators.
#
# Reused unmodified from the base box-station domain: cut_paper/wrap_gift/
# cut_ribbon/complete_job only reference `holding ?r scissors`/hand-full, no
# `reserved` bookkeeping needed on them. Only pick/place fold `reserved` into
# their own effects — applies uniformly to any ?obj (scissors *and* gift
# boxes), which is harmless: nothing ever queries a box's `reserved` fluent
# since boxes are never contended between robots.
# ---------------------------------------------------------------------------

pick_op_reserved  = operators.construct_pick_operator_blocking(pick_time=1.0)
pick_op_reserved.preconditions.append(~F("reserved ?obj"))
pick_op_reserved.effects[1].resulting_fluents |= {F("reserved ?obj"), F("reserved_by ?r ?obj")}

place_op_reserved = operators.construct_place_operator_blocking(place_time=1.0)
place_op_reserved.effects[1].resulting_fluents |= {~F("reserved ?obj"), ~F("reserved_by ?r ?obj")}

my_operators_reserved = [
    move_op, pick_op_reserved, place_op_reserved,
    cut_paper_op, wrap_gift_op, cut_ribbon_op, complete_job_op,
]


def construct_wait_for_resource_operator(
    obj: str, loc: str, release_time: float, t_now: float, epsilon: float = 0.1,
) -> Operator:
    """Wait until `obj` is predicted to be back at `loc`, then trust it's there.

    `t_now` here is a provisional guess only — the caller passes the current
    planning time (i.e. assumes nothing precedes the wait) purely so the
    search has a concrete number for cost accounting. It is deliberately
    *not* an attempt to predict how much independent work (e.g. box
    collection) the search will schedule first — that's unknowable until a
    full plan exists, and a Numeric duration function has no visibility into
    the search's own elapsed time. See `correct_wait_duration`, which
    overwrites this guess with the real value once a plan is in hand.

    Still targets release_time + epsilon (never exactly release_time), for
    the same reason as before: the real holder's release fires at exactly
    release_time, and an identical-timestamp tie in the event queue isn't
    guaranteed to resolve holder-releases-before-waiter-checks.
    """
    def _wait_time(_r: str) -> float:
        return max(0.0, release_time + epsilon - t_now)

    return Operator(
        name="wait_for_resource",
        parameters=[("?r", "robot")],
        preconditions=[F(f"at ?r {loc}"), F("free ?r"), F(f"reserved {obj}")],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(
                time=(Numeric(_wait_time), ["?r"]),
                resulting_fluents={F("free ?r"), F(f"at {obj} {loc}"), ~F(f"reserved {obj}")},
            ),
        ],
    )


def correct_wait_duration(
    plan: List[SimpleAction], t_start: float, release_time: float, epsilon: float = 0.1,
) -> None:
    """Overwrite the plan's wait_for_resource step's duration in place so it
    ends at exactly release_time + epsilon, no matter how many actions (or
    which ones — box collection, moves, anything) the search chose to
    schedule before it.

    Safe to patch after the fact: every other action's duration in this
    domain is a deterministic constant or distance function, independent of
    the wait, so the real elapsed time up to the wait step is exactly the
    sum of those actions' own (already-correct) durations. Fluent
    preconditions only care about relative order, not absolute time, so
    changing this one number doesn't invalidate anything else in the plan —
    it just shifts every later action's real start time to match.
    """
    elapsed = 0.0
    for action in plan:
        if action.name.startswith("wait_for_resource"):
            action.duration = max(0.0, release_time + epsilon - (t_start + elapsed))
            return
        elapsed += action.duration

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def main_reservation_based() -> None:
    """Proactive baseline for the box-station domain: an explicit reservation
    on the scissors lets the *other* robot's planner bake a wait-then-pick
    sequence directly into its own plan. Unlike the base reservation domain,
    the waiting robot may also have independent box-collection work the
    search can freely schedule around the wait — see correct_wait_duration.
    """
    world_state: Set[str] = set(initial_world_state)
    event_queue: List[Event] = []

    robot_plans:        Dict[str, List[SimpleAction]] = {r: [] for r in objects_by_type["robot"]}
    robot_tasks:        Dict[str, Optional[str]]      = {r: None for r in objects_by_type["robot"]}
    robot_task_queue:   Dict[str, List[str]]          = {r: list(tasks) for r, tasks in ROBOT_TASKS.items()}

    # Persistent reservation tracker: obj -> queue of Reservations, in commit
    # order (empty/absent means free). See pick_and_place_astar_reservation
    # for the full rationale (FIFO by commit order, lazily expired entries).
    reservations: Dict[str, List[Reservation]] = {}

    first_robot = random.choice(["robot1", "robot2"])
    second_robot = "robot2" if first_robot == "robot1" else "robot1"
    print(f"Coin toss: {first_robot} plans first at t=0")

    for order, robot in enumerate([first_robot, second_robot]):
        goal = robot_task_queue[robot].pop(0)
        heapq.heappush(event_queue, Event(time=0.0, robot_id=robot,
                                          event_type="TASK_ARRIVAL", data=goal,
                                          order=order))

    log = make_logger(list(objects_by_type["robot"]))

    def state_with_reservations(t: float, robot: str) -> Set[str]:
        """world_state overlaid with any reservation the tracker knows about
        that real world_state fluents haven't caught up to yet. See
        pick_and_place_astar_reservation.main_reservation_based for the full
        rationale.
        """
        state = set(world_state)
        for obj in list(reservations.keys()):
            res = current_reservation(reservations, obj, t)
            if res is not None and res.holder != robot:
                state.add(f"reserved {obj}")
                state.add(f"reserved_by {res.holder} {obj}")
        return state

    def plan_and_start(t: float, robot: str, goal: str) -> None:
        robot_tasks[robot] = goal
        robot_gifts = {
            m.group() for task in ROBOT_TASKS[robot]
            for m in [re.search(r'gift\d+', task)] if m
        }
        task_objects = (
            {**objects_by_type, "gift": robot_gifts, "object": robot_gifts | {"scissors"}}
            if robot_gifts else objects_by_type
        )

        ops_for_call = my_operators_reserved
        res = current_reservation(reservations, "scissors", t)
        if res is not None and res.holder != robot:
            wait_op = construct_wait_for_resource_operator("scissors", "tool_space", res.release_time, t)
            ops_for_call = my_operators_reserved + [wait_op]

        plan = call_planner(state_with_reservations(t, robot), goal, task_objects,
                            ops_for_call, robot, planner_cls=PLANNER,
                            extra_cost_fn=expected_cost)
        if plan is None:
            log(t, robot, "goal already satisfied")
        elif not plan:
            log(t, robot, f"FAILED — cannot plan for '{goal}'")
        else:
            if res is not None and res.holder != robot:
                correct_wait_duration(plan, t, res.release_time)
            robot_plans[robot] = plan
            release = predict_release_time(plan, t, "scissors")
            if release is not None:
                reservations.setdefault("scissors", []).append(Reservation(holder=robot, release_time=release))
            try_start_next(t, robot)

    def try_start_next(t: float, robot: str) -> None:
        if not robot_plans[robot]:
            if robot_task_queue[robot]:
                goal = robot_task_queue[robot].pop(0)
                log(t, robot, f"idle — starting next queued task '{goal}'")
                plan_and_start(t, robot, goal)
            else:
                log(t, robot, "finished — idle")
            return

        action = robot_plans[robot][0]
        fail = check_preconditions(action, state_with_reservations(t, robot))

        if fail is not None:
            # Pure trust, no fallback: the reservation's predicted release
            # time is exact in this deterministic simulation (and now
            # corrected for however much real work precedes the wait), so a
            # failed precondition here means the prediction was wrong.
            raise RuntimeError(
                f"{robot}'s precondition '{fail}' unexpectedly false at t={t:.1f} "
                f"for {action.name} — reservation prediction was wrong"
            )

        for d in action.del_effects:
            world_state.discard(d)
        log(t, robot, f"START  {action.name}")
        heapq.heappush(event_queue, Event(
            time=t + action.duration, robot_id=robot,
            event_type="ACTION_END", data=action,
        ))

    final_time = 0.0
    while event_queue:
        event = heapq.heappop(event_queue)
        t, robot = event.time, event.robot_id
        final_time = max(final_time, t)

        if event.event_type == "TASK_ARRIVAL":
            goal = event.data
            log(t, robot, f"TASK_ARRIVAL → '{goal}'")
            plan_and_start(t, robot, goal)

        elif event.event_type == "ACTION_END":
            action: SimpleAction = event.data
            for a in action.add_effects:
                world_state.add(a)
            log(t, robot, f"END    {action.name}")
            robot_plans[robot].pop(0)
            try_start_next(t, robot)

    print("\n--- Final world state ---")
    for f in sorted(world_state):
        print(f"  {f}")

    print(f"\nAll 6 tasks completed at t={final_time:.1f} (total time to finish all tasks)")


if __name__ == "__main__":
    main_reservation_based()
