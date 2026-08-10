"""Box-station domain — no_op fallback baseline (third point in the design space).

Two other baselines already exist for this domain:
  - pick_and_place_astar_boxstation.py: reactive. Plans optimistically against
    a *broadcast* snapshot (other robots' committed future effects applied as
    if already true), discovers unavailability only at real execution time,
    and recovers by blocking + retrying whenever poked by another robot's
    ACTION_END.
  - pick_and_place_astar_boxstation_reservation.py: proactive. An explicit
    Python-side reservation queue predicts *when* a contended resource will
    be released, and bakes a precisely-timed wait_for_resource action
    directly into the plan.

This file is a third, deliberately dumber baseline: the robot *does* know
scissors is reserved (a real `reserved`/`reserved_by` fluent, set/cleared by
pick_op_reserved/place_op_reserved's own effects — reused unmodified from the
reservation file) — but there is no predicted release time anywhere, no
Python-side bookkeeping at all. Planning is always done against the current,
real world_state directly (no broadcast optimism either). Whenever a full
plan can't be found (or, once committed, turns out to be wrong mid-flight —
see below), the only fallback is a fixed-duration, no-op operator
(operators.construct_no_op_operator), exactly the pattern
multi_robot_breakfast.py uses for its toaster contention: idle for a fixed
chunk, then try the whole goal again from scratch.

Only one gift per robot (not the full 3-gift queue) — this file exists to
observe *how the contention gets handled*, not to benchmark full completion.

Run directly: `uv run scripts/test_boxstation.py`
"""

import heapq
import os
import sys
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner_interface import call_planner, SimpleAction, CoordinationAStarPlanner, _action_to_simple  # pyrefly: ignore [missing-import]
from railroad.core import State, get_action_by_name
from railroad.environment.symbolic import SymbolicEnvironment
from railroad import operators

from pick_and_place_astar_boxstation import (
    objects_by_type,
    initial_world_state,
    move_op,
    cut_paper_op,
    wrap_gift_op,
    cut_ribbon_op,
    complete_job_op,
    Event,
    check_preconditions,
    make_logger,
)
from pick_and_place_astar_boxstation_reservation import pick_op_reserved, place_op_reserved

PLANNER = CoordinationAStarPlanner
NO_OP_TIME = 5.0

my_operators = [
    move_op, pick_op_reserved, place_op_reserved,
    cut_paper_op, wrap_gift_op, cut_ribbon_op, complete_job_op,
]

# Not offered to the search (see module docstring / plan_and_start): grounded
# once per robot, up front, purely as a fallback the *outer loop* executes
# directly when planning comes up empty.
no_op_op = operators.construct_no_op_operator(no_op_time=NO_OP_TIME)


def _ground_no_op(robot: str) -> SimpleAction:
    env = SymbolicEnvironment(State(0.0, set()), {"robot": {robot}}, [no_op_op])
    action = get_action_by_name(env.get_actions(), f"no_op {robot}")
    return _action_to_simple(action)


# One gift per robot — enough to force immediate scissors contention, without
# the overhead of a full multi-task queue.
ROBOT_GOALS: Dict[str, str] = {
    "robot1": "wrapped_gift robot1 gift1 & at scissors tool_space",
    "robot2": "wrapped_gift robot2 gift4 & at scissors tool_space",
}


def main() -> None:
    world_state: Set[str] = set(initial_world_state)
    event_queue: List[Event] = []

    robot_plans: Dict[str, List[SimpleAction]] = {r: [] for r in objects_by_type["robot"]}
    no_op_actions: Dict[str, SimpleAction] = {r: _ground_no_op(r) for r in objects_by_type["robot"]}

    for order, robot in enumerate(sorted(ROBOT_GOALS)):
        heapq.heappush(event_queue, Event(time=0.0, robot_id=robot,
                                          event_type="TASK_ARRIVAL", data=ROBOT_GOALS[robot],
                                          order=order))

    log = make_logger(list(objects_by_type["robot"]))

    def _start_no_op(t: float, robot: str, reason: str) -> None:
        no_op = no_op_actions[robot]
        log(t, robot, f"{reason} — no_op {NO_OP_TIME:.1f}s, will retry whole goal")
        heapq.heappush(event_queue, Event(
            time=t + no_op.duration, robot_id=robot,
            event_type="NO_OP_END", data=ROBOT_GOALS[robot],
        ))

    def plan_and_start(t: float, robot: str, goal: str) -> None:
        gift_id = goal.split()[2]  # "wrapped_gift robot1 gift1 & ..." -> "gift1"
        task_objects = {**objects_by_type, "gift": {gift_id}, "object": {gift_id, "scissors"}}

        # Plan against the *real, current* world_state only — no broadcast of
        # other robots' future effects (unlike the reactive baseline), and no
        # predicted release time fed in anywhere (unlike the reservation
        # baseline). reserved/reserved_by are real fluents pick_op_reserved/
        # place_op_reserved set and clear; this is the only thing the robot
        # knows about contention, and it only ever reflects *right now*.
        plan = call_planner(world_state, goal, task_objects, my_operators, robot, planner_cls=PLANNER)

        if plan is None:
            log(t, robot, "goal already satisfied")
        elif not plan:
            # Full plan unreachable right now — pick_op_reserved's own
            # ~reserved precondition can never become true through this
            # robot's own actions alone, so A* can't find *any* plan while
            # scissors is held elsewhere. No notion of *when* it frees up, so
            # just idle for a fixed chunk and re-plan the whole goal from
            # scratch (which will naturally pick up wherever real progress
            # already happened, since it's always planned fresh off
            # world_state).
            _start_no_op(t, robot, "scissors reserved, no full plan found")
        else:
            robot_plans[robot] = plan
            try_start_next(t, robot)

    def try_start_next(t: float, robot: str) -> None:
        if not robot_plans[robot]:
            log(t, robot, "finished — idle")
            return

        action = robot_plans[robot][0]
        fail = check_preconditions(action, world_state)

        if fail is not None:
            # The plan was committed optimistically (scissors looked free at
            # planning time) but the *other* robot won the race in the
            # meantime — discovered only now, mid-flight, since nothing here
            # ever predicted a release time to guard against it. Unlike the
            # reservation baseline (where this would be a bug, RuntimeError),
            # here it's an expected outcome of never modeling time at all:
            # abandon whatever's left of this plan and fall back to the same
            # no_op-then-replan-from-scratch response.
            robot_plans[robot] = []
            _start_no_op(t, robot, f"'{fail}' unexpectedly false — plan invalidated")
            return

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

        elif event.event_type == "NO_OP_END":
            goal = event.data
            log(t, robot, "END    no_op — replanning")
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

    print(f"\nBoth gifts completed at t={final_time:.1f}")


if __name__ == "__main__":
    main()
