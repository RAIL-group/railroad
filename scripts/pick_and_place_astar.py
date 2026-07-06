"""Gift wrap — decentralized concurrent task planning.

Two robots share a single pair of scissors (the contended resource).
Each robot has an exclusive workstation; scissors live at tool_space.

Layout (1 unit = 1 second of travel):
  workstation1 ----5---- tool_space ----5---- workstation2

Scenario:
  robot1 (at workstation1) — tasks at t=0, t=80, t=160  (gifts 1, 4, 5)
  robot2 (at workstation2) — tasks at t=20, t=70         (gifts 2, 3)

Each task targets a unique gift ID so completed work is never confused with
a new task.  No replanning or retries: if scissors is unavailable when a task
arrives the task fails immediately.

Gift wrap workflow per robot:
  1. move to tool_space → pick scissors
  2. move to own workstation
  3. cut_paper  (10 s, scissors in hand)
  4. cut_ribbon  (5 s, scissors in hand)
  5. place scissors temporarily at workstation
  6. fold_paper  (20 s, empty hands)
  7. wrap_paper  (10 s, empty hands)
  8. pick scissors → move to tool_space → place scissors back

Robots cannot enter each other's workstations ('accessible' fluent).
robot2 must wait for robot1 to return scissors before it can begin.
"""

import heapq
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner_interface import call_planner, SimpleAction, CoordinationAStarPlanner  # pyrefly: ignore [missing-import]
from railroad.core import Operator, Effect, Fluent as F
from railroad.operators._utils import Numeric
from railroad import operators

PLANNER = CoordinationAStarPlanner


def expected_cost(_action, successor) -> float:
    """Extra coordination cost added to h during A* search.

    Penalises states where scissors are resting at any location other than
    tool_space.  This steers the planner away from plans that stash scissors
    at a workstation (forcing robot2 to wait) when returning them to tool_space
    directly is possible.

    No penalty is applied while scissors are being carried (no 'at scissors'
    fluent exists in that case), only when they are set down somewhere wrong.
    """
    for f in successor.fluents:
        if f.name == "at" and len(f.args) >= 2 and f.args[0] == "scissors":
            if f.args[1] != "tool_space":
                return 100.0
    return 0.0

# ---------------------------------------------------------------------------
# Locations & move time
# ---------------------------------------------------------------------------

LOCATION_COORDS: Dict[str, Any] = {
    "workstation1": np.array([0.0, 0.0]),
    "tool_space":   np.array([5.0, 0.0]),
    "workstation2": np.array([10.0, 0.0]),
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
    "object":   {"scissors"},
    "gift":     {"gift1", "gift2", "gift3", "gift4", "gift5"},
}

initial_world_state: Set[str] = {
    "free robot1", "at robot1 workstation1",
    "free robot2", "at robot2 workstation2",
    "at scissors tool_space",
    # Workstation access control — each robot can only enter its own workstation
    "accessible robot1 workstation1", "accessible robot1 tool_space",
    "accessible robot2 workstation2", "accessible robot2 tool_space",
    # Needed by work operator preconditions to distinguish workstations from tool_space
    "is_workstation workstation1",
    "is_workstation workstation2",
}

# Tasks: (arrival_time, robot_id, goal_string)
# Each task names a unique gift so the planner always has fresh work to do.
# Goal includes returning scissors so the other robot can acquire it.
# If a task arrives while the robot is busy, planning is deferred until the
# current action completes.
TASKS: List[Tuple[float, str, str]] = [
    (  0.0, "robot1", "wrapped_gift robot1 gift1 & at scissors tool_space"),
    ( 20.0, "robot2", "wrapped_gift robot2 gift2 & at scissors tool_space"),
    # ( 70.0, "robot2", "wrapped_gift robot2 gift3 & at scissors tool_space"),
    # ( 80.0, "robot1", "wrapped_gift robot1 gift4 & at scissors tool_space"),
    # (160.0, "robot1", "wrapped_gift robot1 gift5 & at scissors tool_space"),
]

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
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
            ~F("paper_cut ?r ?gift"),
        ],
        effects=[
            Effect(time=0,    resulting_fluents={~F("free ?r")}),
            Effect(time=10.0, resulting_fluents={F("free ?r"), F("paper_cut ?r ?gift")}),
        ],
    )

def construct_fold_paper_operator() -> Operator:
    return Operator(
        name="fold_paper",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), ~F("hand-full ?r"),
            F("paper_cut ?r ?gift"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
            ~F("paper_folded ?r ?gift"),
        ],
        effects=[
            Effect(time=0,    resulting_fluents={~F("free ?r")}),
            Effect(time=20.0, resulting_fluents={F("free ?r"), F("paper_folded ?r ?gift")}),
        ],
    )

def construct_cut_ribbon_operator() -> Operator:
    return Operator(
        name="cut_ribbon",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), F("holding ?r scissors"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
            ~F("ribbon_cut ?r ?gift"),
        ],
        effects=[
            Effect(time=0,   resulting_fluents={~F("free ?r")}),
            Effect(time=5.0, resulting_fluents={F("free ?r"), F("ribbon_cut ?r ?gift")}),
        ],
    )

def construct_wrap_paper_operator() -> Operator:
    return Operator(
        name="wrap_paper",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), ~F("hand-full ?r"),
            F("paper_cut ?r ?gift"), F("paper_folded ?r ?gift"), F("ribbon_cut ?r ?gift"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
        ],
        effects=[
            Effect(time=0,    resulting_fluents={~F("free ?r")}),
            Effect(time=10.0, resulting_fluents={F("free ?r"), F("wrapped_gift ?r ?gift")}),
        ],
    )

move_op       = construct_accessible_move_operator()
pick_op       = operators.construct_pick_operator_blocking(pick_time=1.0)
place_op      = operators.construct_place_operator_blocking(place_time=1.0)
cut_paper_op  = construct_cut_paper_operator()
fold_paper_op = construct_fold_paper_operator()
cut_ribbon_op = construct_cut_ribbon_operator()
wrap_paper_op = construct_wrap_paper_operator()

my_operators = [move_op, pick_op, place_op, cut_paper_op, fold_paper_op, cut_ribbon_op, wrap_paper_op]

# ---------------------------------------------------------------------------
# Event queue
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Event:
    time: float
    robot_id: str
    event_type: str = field(compare=False)
    data: Any       = field(compare=False)


def check_preconditions(action: SimpleAction, state: Set[str]) -> Optional[str]:
    for pre in action.preconditions:
        if pre.startswith("not "):
            if pre[4:] in state:
                return pre
        else:
            if pre not in state:
                return pre
    return None

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def main() -> None:
    world_state: Set[str] = set(initial_world_state)
    event_queue: List[Event] = []

    robot_plans:        Dict[str, List[SimpleAction]] = {r: [] for r in objects_by_type["robot"]}
    robot_tasks:        Dict[str, Optional[str]]      = {r: None for r in objects_by_type["robot"]}
    robot_busy:         Dict[str, bool]               = {r: False for r in objects_by_type["robot"]}
    robot_blocked:      Dict[str, bool]               = {r: False for r in objects_by_type["robot"]}
    robot_pending_goal: Dict[str, Optional[str]]      = {r: None for r in objects_by_type["robot"]}

    for arrival_t, robot, goal in TASKS:
        heapq.heappush(event_queue, Event(time=arrival_t, robot_id=robot,
                                          event_type="TASK_ARRIVAL", data=goal))

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
        gift_match = re.search(r'gift\d+', goal)
        task_objects = (
            {**objects_by_type, "gift": {gift_match.group()}}
            if gift_match else objects_by_type
        )
        # Plan against the broadcast (future) state so the planner knows what
        # other robots are committed to doing and can sequence around them.
        plan = call_planner(broadcast_state(), goal, task_objects,
                            my_operators, robot, planner_cls=PLANNER,
                            extra_cost_fn=expected_cost)
        if plan is None:
            print(f"  t={t:06.1f} | {robot} goal already satisfied")
        elif not plan:
            print(f"  t={t:06.1f} | {robot} FAILED — cannot plan for '{goal}'")
        else:
            robot_plans[robot] = plan
            try_start_next(t, robot)

    def try_start_next(t: float, robot: str) -> None:
        if not robot_plans[robot]:
            robot_busy[robot] = False
            if robot_pending_goal[robot] is not None:
                goal = robot_pending_goal[robot]
                robot_pending_goal[robot] = None
                print(f"  t={t:06.1f} | {robot} idle — planning deferred task '{goal}'")
                plan_and_start(t, robot, goal)
            else:
                print(f"  t={t:06.1f} | {robot} finished — idle")
            return

        action = robot_plans[robot][0]
        fail = check_preconditions(action, world_state)

        if fail is None:
            robot_blocked[robot] = False
            for d in action.del_effects:
                world_state.discard(d)
            robot_busy[robot] = True
            print(f"  t={t:06.1f} | {robot} START  {action.name}")
            heapq.heappush(event_queue, Event(
                time=t + action.duration, robot_id=robot,
                event_type="ACTION_END", data=action,
            ))
        else:
            robot_busy[robot] = False
            robot_blocked[robot] = True
            print(f"  t={t:06.1f} | {robot} WAITING — '{fail}' not yet true, will retry")

    while event_queue:
        event = heapq.heappop(event_queue)
        t, robot = event.time, event.robot_id

        if event.event_type == "TASK_ARRIVAL":
            goal = event.data
            print(f"\nt={t:06.1f} | [TASK_ARRIVAL] {robot} → '{goal}'")
            # Defer planning if robot already has committed work (busy or blocked)
            if robot_plans[robot]:
                print(f"  t={t:06.1f} | {robot} has active plan — deferring until plan completes")
                robot_pending_goal[robot] = goal
            else:
                plan_and_start(t, robot, goal)

        elif event.event_type == "ACTION_END":
            action: SimpleAction = event.data
            for a in action.add_effects:
                world_state.add(a)
            print(f"  t={t:06.1f} | {robot} END    {action.name}")
            robot_plans[robot].pop(0)
            try_start_next(t, robot)
            # Wake any robots that were blocked waiting for world-state changes
            for other in list(objects_by_type["robot"]):
                if other != robot and robot_blocked[other]:
                    print(f"  t={t:06.1f} | {other} retrying after {robot} completed {action.name}")
                    try_start_next(t, other)

    print("\n--- Final world state ---")
    for f in sorted(world_state):
        print(f"  {f}")


if __name__ == "__main__":
    main()
