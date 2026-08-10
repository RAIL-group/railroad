"""Gift wrap — box-station domain.

Extends the base gift-wrap scenario (pick_and_place_astar.py) with a second
shared, non-contended resource: a box_station holding all 6 gift boxes.
Collecting a box (pick it up, carry it to your own workstation) never
touches scissors, so it's genuinely independent of the scissors contention —
unlike the base domain, where the very first step of every task needs
scissors, here a robot always has real, useful work it could do while
scissors are unavailable. This is the domain used to explore generalizing
the reservation-based wait mechanism (see pick_and_place_astar_reservation.py)
beyond its current single-preceding-move assumption.

Layout (1 unit = 1 second of travel):
  workstation1 ----5---- tool_space ----5---- workstation2
  box_station is off that axis, positioned exactly 10s from *both*
  workstations (Euclidean distance) — symmetric like tool_space, just farther.

Scenario:
  robot1 (at workstation1) — tasks queued in sequence: gift1, gift2, gift3
  robot2 (at workstation2) — tasks queued in sequence: gift4, gift5, gift6
  All 6 gift boxes start at box_station.

Both robots' first task "arrives" at t=0 concurrently; a coin toss decides
which robot gets to plan first. Each robot's remaining tasks are not on a
fixed schedule — the next task in a robot's queue is only planned once the
previous one finishes — but the two robots otherwise act concurrently in
the environment.

Gift wrap workflow per robot, per gift:
  1. move to box_station → pick gift box
  2. move to own workstation → place gift box     (independent of scissors)
  3. move to tool_space → pick scissors → move to own workstation
  4. cut_paper   (20s, scissors in hand, box present)
  5. place scissors at workstation
  6. wrap_gift   (10s, empty hands — duration not specified in the domain
                  spec, assumed here; tune via WRAP_GIFT_TIME)
  7. pick scissors again
  8. cut_ribbon  (20s, scissors in hand)
  9. place scissors at workstation
  10. complete_job (10s, empty hands) → wrapped_gift
  11. pick scissors → move to tool_space → place scissors back

Robots cannot enter each other's workstations ('accessible' fluent).
No replanning or retries: if scissors is unavailable when a task arrives
the task fails immediately. (Reactive baseline — mirrors
pick_and_place_astar.main(); a reservation-based variant of this domain can
be built the same way pick_and_place_astar_reservation.py extends the base.)

Run directly: `uv run scripts/pick_and_place_astar_boxstation.py`
"""

import heapq
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner_interface import call_planner, SimpleAction, CoordinationAStarPlanner  # pyrefly: ignore [missing-import]
from railroad.core import Operator, Effect, Fluent as F
from railroad.operators._utils import Numeric
from railroad import operators

from pick_and_place_astar import Event, check_preconditions, make_logger

PLANNER = CoordinationAStarPlanner

# Not specified in the domain description — assumed equal to the base
# domain's wrap_paper duration. Tune here if needed.
WRAP_GIFT_TIME = 10.0


def expected_cost(planner, action, successor) -> float:
    """Extra coordination cost added to h during A* search.

    For the robot taking `action`, actually plans (full A*) each of its queued
    tasks from `successor` and takes the resulting plan's total time, then
    averages those times across all of the robot's tasks — see
    pick_and_place_astar.expected_cost for the full rationale.
    """
    robot = action.name.split()[1]
    tasks = ROBOT_TASKS[robot]
    times = [planner.simulate_time_to_goal(successor, goal) for goal in tasks]
    times = [t for t in times if t is not None]
    if not times:
        return 0.0
    return sum(times) / len(times)

# ---------------------------------------------------------------------------
# Locations & move time
# ---------------------------------------------------------------------------

# box_station sits off the workstation1<->workstation2 axis so it can be
# exactly 10s from *both* stations (Euclidean distance) while tool_space
# stays on-axis, 5s from each, unchanged from the base domain.
_BOX_STATION_Y = float(np.sqrt(10.0**2 - 5.0**2))  # ~8.66

LOCATION_COORDS: Dict[str, Any] = {
    "workstation1": np.array([0.0, 0.0]),
    "tool_space":   np.array([5.0, 0.0]),
    "workstation2": np.array([10.0, 0.0]),
    "box_station":  np.array([5.0, _BOX_STATION_Y]),
}

def _move_time(_robot: str, loc_from: str, loc_to: str) -> float:
    return float(np.linalg.norm(LOCATION_COORDS[loc_from] - LOCATION_COORDS[loc_to]))

move_time = Numeric(_move_time)  # supports arithmetic (e.g. move_time + 0.1)

# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

objects_by_type: Dict[str, Set[str]] = {
    "robot":    {"robot1", "robot2"},
    "location": set(LOCATION_COORDS.keys()),
    "object":   {"scissors", "gift1", "gift2", "gift3", "gift4", "gift5", "gift6"},
    "gift":     {"gift1", "gift2", "gift3", "gift4", "gift5", "gift6"},
}

initial_world_state: Set[str] = {
    "free robot1", "at robot1 workstation1",
    "free robot2", "at robot2 workstation2",
    "at scissors tool_space",
    "at gift1 box_station", "at gift2 box_station", "at gift3 box_station",
    "at gift4 box_station", "at gift5 box_station", "at gift6 box_station",
    # Workstation access control — each robot can only enter its own workstation;
    # tool_space and box_station are shared/common to both robots.
    "accessible robot1 workstation1", "accessible robot1 tool_space", "accessible robot1 box_station",
    "accessible robot2 workstation2", "accessible robot2 tool_space", "accessible robot2 box_station",
    # Needed by work operator preconditions to distinguish workstations from tool_space/box_station
    "is_workstation workstation1",
    "is_workstation workstation2",
}

# Per-robot task queues, in the order each robot must work through them.
# Goal includes returning scissors so the other robot can acquire it.
ROBOT_TASKS: Dict[str, List[str]] = {
    "robot1": [
        "wrapped_gift robot1 gift1 & at scissors tool_space",
        "wrapped_gift robot1 gift2 & at scissors tool_space",
        "wrapped_gift robot1 gift3 & at scissors tool_space",
    ],
    "robot2": [
        "wrapped_gift robot2 gift4 & at scissors tool_space",
        "wrapped_gift robot2 gift5 & at scissors tool_space",
        "wrapped_gift robot2 gift6 & at scissors tool_space",
    ],
}

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def construct_accessible_move_operator() -> Operator:
    """Move with workstation exclusivity enforced by the 'accessible' precondition."""
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[
            F("at ?r ?from"), F("free ?r"), ~F("just-moved ?r"),
            F("accessible ?r ?to"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("at ?r ?to"), F("just-moved ?r")},
            ),
            Effect(
                time=(move_time + 0.1, ["?r", "?from", "?to"]),
                resulting_fluents={~F("just-moved ?r")},
            ),
        ],
    )

def construct_cut_paper_operator() -> Operator:
    return Operator(
        name="cut_paper",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), F("holding ?r scissors"),
            F("at ?gift ?ws"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
            ~F("paper_cut ?r ?gift"),
        ],
        effects=[
            Effect(time=0,    resulting_fluents={~F("free ?r")}),
            Effect(time=20.0, resulting_fluents={F("free ?r"), F("paper_cut ?r ?gift")}),
        ],
    )

def construct_wrap_gift_operator() -> Operator:
    return Operator(
        name="wrap_gift",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), ~F("hand-full ?r"),
            F("paper_cut ?r ?gift"), F("at ?gift ?ws"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
            ~F("paper_wrapped ?r ?gift"),
        ],
        effects=[
            Effect(time=0,               resulting_fluents={~F("free ?r")}),
            Effect(time=WRAP_GIFT_TIME,  resulting_fluents={F("free ?r"), F("paper_wrapped ?r ?gift")}),
        ],
    )

def construct_cut_ribbon_operator() -> Operator:
    return Operator(
        name="cut_ribbon",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), F("holding ?r scissors"),
            F("paper_wrapped ?r ?gift"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
            ~F("ribbon_cut ?r ?gift"),
        ],
        effects=[
            Effect(time=0,    resulting_fluents={~F("free ?r")}),
            Effect(time=20.0, resulting_fluents={F("free ?r"), F("ribbon_cut ?r ?gift")}),
        ],
    )

def construct_complete_job_operator() -> Operator:
    return Operator(
        name="complete_job",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), ~F("hand-full ?r"),
            F("ribbon_cut ?r ?gift"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
        ],
        effects=[
            Effect(time=0,    resulting_fluents={~F("free ?r")}),
            Effect(time=10.0, resulting_fluents={F("free ?r"), F("wrapped_gift ?r ?gift")}),
        ],
    )

move_op         = construct_accessible_move_operator()
pick_op         = operators.construct_pick_operator_blocking(pick_time=1.0)
place_op        = operators.construct_place_operator_blocking(place_time=1.0)
cut_paper_op    = construct_cut_paper_operator()
wrap_gift_op    = construct_wrap_gift_operator()
cut_ribbon_op   = construct_cut_ribbon_operator()
complete_job_op = construct_complete_job_operator()

my_operators = [
    move_op, pick_op, place_op,
    cut_paper_op, wrap_gift_op, cut_ribbon_op, complete_job_op,
]

# ---------------------------------------------------------------------------
# Simulation (reactive baseline — mirrors pick_and_place_astar.main())
# ---------------------------------------------------------------------------

def main() -> None:
    world_state: Set[str] = set(initial_world_state)
    event_queue: List[Event] = []

    robot_plans:        Dict[str, List[SimpleAction]] = {r: [] for r in objects_by_type["robot"]}
    robot_tasks:        Dict[str, Optional[str]]      = {r: None for r in objects_by_type["robot"]}
    robot_blocked:      Dict[str, bool]               = {r: False for r in objects_by_type["robot"]}
    robot_task_queue:   Dict[str, List[str]]          = {r: list(tasks) for r, tasks in ROBOT_TASKS.items()}

    first_robot = random.choice(["robot1", "robot2"])
    second_robot = "robot2" if first_robot == "robot1" else "robot1"
    print(f"Coin toss: {first_robot} plans first at t=0")

    for order, robot in enumerate([first_robot, second_robot]):
        goal = robot_task_queue[robot].pop(0)
        heapq.heappush(event_queue, Event(time=0.0, robot_id=robot,
                                          event_type="TASK_ARRIVAL", data=goal,
                                          order=order))

    log = make_logger(list(objects_by_type["robot"]))

    def broadcast_state() -> Set[str]:
        """World state augmented with the committed future effects of every robot's plan.

        Applying each robot's remaining actions in sequence produces the state the
        world will reach once all current plans finish.  This lets a newly-planning
        robot see resources that are currently held but will be released.
        """
        state = set(world_state)
        for other_plan in robot_plans.values():
            for action in other_plan:
                for d in action.del_effects:
                    state.discard(d)
                for a in action.add_effects:
                    state.add(a)
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
        plan = call_planner(broadcast_state(), goal, task_objects,
                            my_operators, robot, planner_cls=PLANNER,
                            extra_cost_fn=expected_cost)
        if plan is None:
            log(t, robot, "goal already satisfied")
        elif not plan:
            log(t, robot, f"FAILED — cannot plan for '{goal}'")
        else:
            robot_plans[robot] = plan
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
        fail = check_preconditions(action, world_state)

        if fail is None:
            robot_blocked[robot] = False
            for d in action.del_effects:
                world_state.discard(d)
            log(t, robot, f"START  {action.name}")
            heapq.heappush(event_queue, Event(
                time=t + action.duration, robot_id=robot,
                event_type="ACTION_END", data=action,
            ))
        else:
            robot_blocked[robot] = True
            log(t, robot, f"WAITING — '{fail}' not yet true, will retry")

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
            # Wake any robots that were blocked waiting for world-state changes
            for other in list(objects_by_type["robot"]):
                if other != robot and robot_blocked[other]:
                    log(t, other, f"retrying after {robot} completed {action.name}")
                    try_start_next(t, other)

    print("\n--- Final world state ---")
    for f in sorted(world_state):
        print(f"  {f}")

    print(f"\nAll 6 tasks completed at t={final_time:.1f} (total time to finish all tasks)")


if __name__ == "__main__":
    main()
