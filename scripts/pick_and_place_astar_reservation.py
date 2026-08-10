"""Gift wrap — reservation-based baseline.

Proactive alternative to the reactive blocked/retry mechanism in
pick_and_place_astar.main(): an explicit reservation on the scissors (who
holds it, predicted release time) lets the *other* robot's planner bake a
wait-then-pick sequence directly into its own plan, instead of reactively
retrying on every other robot's ACTION_END.

Shares the world definition (locations, objects, per-robot task queues) and
the non-contended operators (move/cut_paper/fold_paper/cut_ribbon/wrap_paper)
with pick_and_place_astar.py, imported from there. Everything specific to the
reservation mechanism — the pick/place operator variants that fold `reserved`
into their own effects, the wait operator, the reservation tracker, and the
simulation loop — lives here so it can never accidentally affect the reactive
baseline.

Run directly: `uv run scripts/pick_and_place_astar_reservation.py`
"""

import heapq
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner_interface import call_planner, SimpleAction, CoordinationAStarPlanner  # pyrefly: ignore [missing-import]
from railroad.core import Operator, Effect, Fluent as F
from railroad.operators._utils import Numeric
from railroad import operators

from pick_and_place_astar import (
    PLANNER,
    _move_time,
    expected_cost,
    objects_by_type,
    initial_world_state,
    ROBOT_TASKS,
    move_op,
    cut_paper_op,
    fold_paper_op,
    cut_ribbon_op,
    wrap_paper_op,
    Event,
    check_preconditions,
    make_logger,
)

# ---------------------------------------------------------------------------
# Reservation-based baseline — alternate operators, separate from
# pick_and_place_astar.my_operators.
#
# pick_op/place_op in pick_and_place_astar.py are shared with the reactive
# (default) simulation and must never be mutated. pick_op_reserved/
# place_op_reserved are distinct Operator instances that fold a global
# (non-robot-specific) `reserved` fluent into pick/place's own effects, so
# picking something up also reserves it and placing it down clears the
# reservation — no separate reserve/unreserve actions needed, since in this
# domain "reserved" and "physically held by someone" are the same thing.
# ---------------------------------------------------------------------------

pick_op_reserved  = operators.construct_pick_operator_blocking(pick_time=1.0)
pick_op_reserved.preconditions.append(~F("reserved ?obj"))
pick_op_reserved.effects[1].resulting_fluents |= {F("reserved ?obj"), F("reserved_by ?r ?obj")}

place_op_reserved = operators.construct_place_operator_blocking(place_time=1.0)
place_op_reserved.effects[1].resulting_fluents |= {~F("reserved ?obj"), ~F("reserved_by ?r ?obj")}

my_operators_reserved = [
    move_op, pick_op_reserved, place_op_reserved,
    cut_paper_op, fold_paper_op, cut_ribbon_op, wrap_paper_op,
]


def construct_wait_for_resource_operator(
    obj: str, loc: str, release_time: float, t_now: float, epsilon: float = 0.1,
) -> Operator:
    """Wait until `obj` is predicted to be back at `loc`, then trust it's there.

    Built fresh per plan_and_start call, closing over already-known concrete
    release_time/t_now values (not a live global read at grounding time).
    Modeled on operators.construct_no_op_operator's time=0-then-time=(fn,[params])
    shape. Only offered when `obj` is actually reserved by someone else (see
    the `reserved {obj}` precondition) — its terminal effect directly asserts
    `at {obj} {loc}` and clears `reserved {obj}`, which is what makes
    pick_op_reserved's two preconditions (at ?obj ?loc, ~reserved ?obj) both
    become satisfiable in the waiting robot's own isolated search.

    The target is release_time + epsilon, deliberately never exactly
    release_time: the real holder's placing action's ACTION_END is scheduled
    at exactly release_time, and event-queue ties at an identical timestamp
    aren't guaranteed to resolve holder-releases-before-waiter-checks. Ending
    the wait strictly after (not exactly at) release_time guarantees the real
    release event is popped from the event queue first, so this robot's own
    precondition check always sees the real, physical result — matching the
    domain's existing 0.1s buffers for just-moved/just-picked/just-placed.
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

# ---------------------------------------------------------------------------
# Reservation tracking
# ---------------------------------------------------------------------------

@dataclass
class Reservation:
    holder: str
    release_time: float


def predict_release_time(
    plan: List[SimpleAction], t_start: float, obj: str, loc: str = "tool_space",
) -> Optional[float]:
    """Absolute sim time at which `plan` will place `obj` at `loc`, assuming
    it starts executing at `t_start`. None if the plan never does.

    Only ever called on a freshly committed plan (t_start = the exact moment
    it was computed, before any of its actions have run), so no correction
    for an already-in-flight action is needed.

    Takes the *last* matching action, not the first: if this plan itself
    starts with a wait_for_resource step (waiting on someone else's
    reservation), that step's own terminal effect also asserts
    `at {obj} {loc}` (that's how it lets the search past pick_op_reserved's
    precondition) — which would otherwise be mistaken for this robot's own
    final release, wildly underestimating when this plan actually returns
    the object after doing its own full task.
    """
    total = 0.0
    release: Optional[float] = None
    for action in plan:
        total += action.duration
        if f"at {obj} {loc}" in action.add_effects:
            release = t_start + total
    return release


def _robot_location(state: Set[str], robot: str) -> str:
    prefix = f"at {robot} "
    for s in state:
        if s.startswith(prefix):
            return s[len(prefix):]
    raise ValueError(f"no 'at {robot} ...' fluent found in state")


def current_reservation(
    reservations: Dict[str, List[Reservation]], obj: str, t: float,
) -> Optional[Reservation]:
    """The reservation currently "at the front of the queue" for `obj`, or
    None if nobody currently claims it.

    Every successful plan-commit *appends* its own predicted release to the
    back of `obj`'s queue (see main_reservation_based.plan_and_start) rather
    than overwriting a single slot — a robot queued behind another must not
    erase the current holder's own claim. This pops entries whose predicted
    release_time has already passed (as a side effect, since it's always
    safe/correct to drop a claim once its time is up), so the front entry is
    always the genuinely current one.
    """
    queue = reservations.get(obj)
    if not queue:
        return None
    while queue and queue[0].release_time <= t:
        queue.pop(0)
    return queue[0] if queue else None

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def main_reservation_based() -> None:
    """Proactive baseline: an explicit reservation on the scissors (who holds
    it, predicted release time) lets the *other* robot's planner bake a
    wait-then-pick sequence directly into its own plan, instead of reactively
    retrying on every other robot's ACTION_END like pick_and_place_astar.main()
    does.

    Uses my_operators_reserved (pick_op_reserved/place_op_reserved) rather
    than pick_and_place_astar.my_operators, so main()'s behavior there is
    completely unaffected by this.
    """
    world_state: Set[str] = set(initial_world_state)
    event_queue: List[Event] = []

    robot_plans:        Dict[str, List[SimpleAction]] = {r: [] for r in objects_by_type["robot"]}
    robot_tasks:        Dict[str, Optional[str]]      = {r: None for r in objects_by_type["robot"]}
    robot_task_queue:   Dict[str, List[str]]          = {r: list(tasks) for r, tasks in ROBOT_TASKS.items()}

    # Persistent reservation tracker: obj -> queue of Reservations, in commit
    # order (empty/absent means free). Every successful plan-commit appends
    # its own predicted release to the back (see plan_and_start) — a robot
    # queued behind another must never overwrite/erase the current holder's
    # own claim, which a single dict slot per object cannot represent.
    # Entries expire (get popped) lazily, in current_reservation, once their
    # release_time has passed.
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
        that real world_state fluents haven't caught up to yet — a robot's
        reservation is recorded the moment its plan is committed, which is
        earlier than when its physical pick action actually executes.

        Deliberately does *not* try to synthesize `at {obj} {loc}`/clear
        `reserved` once a reservation's release_time has passed: the wait
        operator's target is release_time + epsilon (see
        construct_wait_for_resource_operator), strictly after the real
        holder's placing ACTION_END at exactly release_time, so by the time
        any waiter's own precondition check runs, real world_state has
        already caught up for real — no synthetic trust needed there, and
        asserting it early (this was tried) creates a real race letting two
        robots pick up the same object at once.
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
            {**objects_by_type, "gift": robot_gifts}
            if robot_gifts else objects_by_type
        )

        ops_for_call = my_operators_reserved
        res = current_reservation(reservations, "scissors", t)
        if res is not None and res.holder != robot:
            # The wait's duration is baked in via a closure over (release_time,
            # t_now) at construction time — a Numeric duration function only
            # ever sees grounded *parameters*, never the search's internal
            # elapsed time, so it can't itself account for however many
            # actions precede it in whatever plan the search settles on. Since
            # this robot must first physically move to tool_space before it
            # can wait there, and that's the only action that will precede
            # the wait in this domain, fold that known, deterministic travel
            # time into t_now here so the wait ends at the right absolute time.
            current_loc = _robot_location(world_state, robot)
            travel_time = 0.0 if current_loc == "tool_space" else _move_time(robot, current_loc, "tool_space") + 0.1
            wait_op = construct_wait_for_resource_operator("scissors", "tool_space", res.release_time, t + travel_time)
            ops_for_call = my_operators_reserved + [wait_op]

        plan = call_planner(state_with_reservations(t, robot), goal, task_objects,
                            ops_for_call, robot, planner_cls=PLANNER,
                            extra_cost_fn=expected_cost)
        if plan is None:
            log(t, robot, "goal already satisfied")
        elif not plan:
            log(t, robot, f"FAILED — cannot plan for '{goal}'")
        else:
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
            # time is exact in this deterministic simulation, so a failed
            # precondition here means the prediction was wrong — fail loudly
            # rather than silently stall (nothing would ever wake this robot).
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
            # No explicit reservation-clearing step needed here — expired
            # entries are popped lazily by current_reservation whenever the
            # queue is next queried (see state_with_reservations/plan_and_start).
            log(t, robot, f"END    {action.name}")
            robot_plans[robot].pop(0)
            try_start_next(t, robot)

    print("\n--- Final world state ---")
    for f in sorted(world_state):
        print(f"  {f}")

    print(f"\nAll 6 tasks completed at t={final_time:.1f} (total time to finish all tasks)")


if __name__ == "__main__":
    main_reservation_based()
