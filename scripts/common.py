"""Domain-agnostic building blocks shared by centralized.py and decentralized.py.

Consolidates infrastructure that was previously duplicated across
pick_and_place_astar_boxstation.py, ..._reservation.py, test_boxstation.py,
joint_astar_boxstation.py and joint_mcts_boxstation.py: the event-queue
simulation engine, the columnar per-robot logger, the reservation-tracking
primitives, and the action-timing fix needed to replay a plan's real
start/end times correctly.

Nothing here is specific to the box-station gift-wrap domain — a caller
builds a `Domain` describing *their* problem (locations, objects, work
operators, per-robot goals) and hands it to one of the planning functions in
decentralized.py / centralized.py.
"""

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from railroad.core import Operator, Effect, Fluent as F
from railroad.operators._utils import Numeric

from planner_interface import (  # pyrefly: ignore [missing-import]
    call_planner,
    SimpleAction,
    CoordinationAStarPlanner,
    _str_to_fluent,
    _action_duration,
)

__all__ = [
    "Domain",
    "PlanStep",
    "PlanResult",
    "Event",
    "check_preconditions",
    "make_logger",
    "free_again_duration",
    "Reservation",
    "predict_release_time",
    "current_reservation",
    "construct_wait_for_resource_operator",
    "correct_wait_duration",
    "run_decentralized",
    "call_planner",
    "SimpleAction",
    "CoordinationAStarPlanner",
    "restrict_objects_by_type",
]

# ---------------------------------------------------------------------------
# Domain contract + uniform result types
# ---------------------------------------------------------------------------


@dataclass
class Domain:
    """Everything a planning method needs, supplied by the caller.

    Deliberately excludes pick/place/no_op/wait operators: those implement
    *coordination mechanisms*, not the domain itself, and each planning
    method constructs whatever variant it needs (plain vs "reserved"-folded
    pick/place, an explicit wait/no_op/reserve action, ...) on its own.

    `robot_goals` is a *queue* per robot (matching the original
    ROBOT_TASKS structure in pick_and_place_astar_boxstation.py): a robot
    only gets its next goal once the previous one is actually satisfied.
    The decentralized methods (run_decentralized) work through each robot's
    queue automatically within a single call. Centralized methods only plan
    for the *first* queued goal per robot — see centralized.py's
    _combined_goal — since joint search over even a single goal per robot
    is already at the edge of tractable (see plan_joint_astar).
    """

    objects_by_type: Dict[str, Set[str]]
    initial_state: Set[str]
    robots: List[str]
    base_operators: List[Operator]
    robot_goals: Dict[str, List[str]]
    contested_resources: Dict[str, str] = field(default_factory=dict)
    pick_time: float = 1.0
    place_time: float = 1.0
    extra_cost_fn: Optional[Callable] = None


@dataclass
class PlanStep:
    robot: str
    action: str
    start: float
    end: float


@dataclass
class PlanResult:
    success: bool
    cost: Optional[float]  # makespan (max end across all steps); None if no plan
    steps: List[PlanStep]  # chronologically sorted; empty if failed
    message: str = ""
    # Per-robot outcome -- {"success": bool, "cost": Optional[float], "message": str}.
    # `cost` is that robot's own finish time on success, or the time it hit a
    # failure (e.g. on_precondition_failed="terminate") on failure -- unlike
    # the aggregate `cost` above, this is set even when success is False, so
    # a caller can see how far an individual robot got. Empty dict for
    # planning methods that don't track this (e.g. the centralized ones).
    robot_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Per-robot list of one entry per planning attempt (a robot's task queue
    # can hold several goals, and a single goal can be replanned more than
    # once under on_plan_failed="no_op"/no_op retries) --
    # {"goal": str, "states_expanded": Optional[int], "states_generated": Optional[int]}.
    # states_* are None where the underlying search wasn't a
    # CoordinationAStarPlanner (see call_planner's stats_out) or the goal was
    # already satisfied (no search ran at all). Empty dict for planning
    # methods that don't track this.
    search_stats: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event queue primitives (decentralized simulation)
# ---------------------------------------------------------------------------


@dataclass(order=True)
class Event:
    time: float
    robot_id: str = field(compare=False)
    event_type: str = field(compare=False)
    data: Any = field(compare=False)
    # Tie-break for events at the same time. Lower order is processed first.
    order: int = 0


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
# Logging
# ---------------------------------------------------------------------------


def make_logger(robots: List[str], col_width: int = 44):
    """One column per robot so concurrent events line up side-by-side."""
    robots = sorted(robots)

    def log(t: float, robot: str, msg: str) -> None:
        cols = [msg.ljust(col_width) if r == robot else " " * col_width for r in robots]
        print(f"t={t:07.2f} | " + " | ".join(cols))

    header = "  time   | " + " | ".join(r.ljust(col_width) for r in robots)
    print(header)
    print("-" * len(header))
    return log


def _null_logger(_t: float, _robot: str, _msg: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Timing correction — the fix from joint_*_boxstation.py
# ---------------------------------------------------------------------------


def free_again_duration(action) -> float:
    """When this action's robot is actually available for a new decision.

    `_action_duration` reports the time of an action's *last* effect, which
    overstates this for move/pick/place (the "_blocking" constructors): they
    have a trailing bookkeeping-only effect — clearing just-moved/
    just-picked/just-placed — that fires *after* `free ?r` is already true
    again (move sets `free`/`at` at `move_time`, then only clears
    `just-moved` 0.1s later; that just blocks an immediate re-move, it
    doesn't keep the robot busy). The correct end is whichever effect
    actually sets `free {robot}` back to true; falls back to
    `_action_duration` for operators without that trailing effect (their
    last effect already coincides with setting `free` anyway).
    """
    robot = action.name.split()[1]
    free_fluent = F(f"free {robot}")
    free_times = [
        eff.time
        for eff in action.effects
        if any(f == free_fluent and not f.negated for f in eff.resulting_fluents)
    ]
    return min(free_times) if free_times else _action_duration(action)


# ---------------------------------------------------------------------------
# Reservation-based coordination primitives
# ---------------------------------------------------------------------------


@dataclass
class Reservation:
    holder: str
    release_time: float


def predict_release_time(
    plan: List[SimpleAction], t_start: float, obj: str, loc: str,
) -> Optional[float]:
    """Absolute sim time at which `plan` will place `obj` at `loc`, assuming
    it starts executing at `t_start`. None if the plan never does.

    Takes the *last* matching action, not the first: if this plan itself
    starts with a wait_for_resource step (waiting on someone else's
    reservation), that step's own terminal effect also asserts
    `at {obj} {loc}` — which would otherwise be mistaken for this robot's
    own final release.
    """
    total = 0.0
    release: Optional[float] = None
    for action in plan:
        total += action.duration
        if f"at {obj} {loc}" in action.add_effects:
            release = t_start + total
    return release


def current_reservation(
    reservations: Dict[str, List[Reservation]], obj: str, t: float,
) -> Optional[Reservation]:
    """The reservation at the front of `obj`'s queue, or None if free.

    Every successful plan-commit *appends* its own predicted release to the
    back of the queue rather than overwriting a single slot, so a robot
    queued behind another doesn't erase the current holder's claim. Pops
    entries whose predicted release_time has already passed.
    """
    queue = reservations.get(obj)
    if not queue:
        return None
    while queue and queue[0].release_time <= t:
        queue.pop(0)
    return queue[0] if queue else None


def construct_wait_for_resource_operator(
    obj: str, loc: str, release_time: float, t_now: float, epsilon: float = 0.1,
) -> Operator:
    """Wait until `obj` is predicted to be back at `loc`, then trust it's there.

    `t_now` is a provisional guess only — pass the current planning time
    (i.e. assume nothing precedes the wait); a Numeric duration function
    only ever sees grounded parameters, never the search's own elapsed time,
    so it can't know how much real work (e.g. independent, order-free
    prep) the search puts before this step. `correct_wait_duration`
    overwrites this guess with the real value once a full plan is known.

    Targets release_time + epsilon (never exactly release_time): the real
    holder's release fires at exactly release_time, and an identical-
    timestamp tie in the event queue isn't guaranteed to resolve
    holder-releases-before-waiter-checks.
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
    which ones) the search scheduled before it.

    Safe to patch after the fact: every other action's duration is a
    deterministic constant or distance function, independent of the wait, so
    the real elapsed time up to the wait step is exactly the sum of those
    actions' own (already-correct) durations.
    """
    elapsed = 0.0
    for action in plan:
        if action.name.startswith("wait_for_resource"):
            action.duration = max(0.0, release_time + epsilon - (t_start + elapsed))
            return
        elapsed += action.duration


def restrict_objects_by_type(
    objects_by_type: Dict[str, Set[str]], goal: str, always_keep: Set[str],
) -> Dict[str, Set[str]]:
    """Shrink each object-like type's grounding set to just what this
    specific goal (plus always_keep) actually mentions by name, so a
    robot's per-task search doesn't waste time grounding pick/place/etc.
    for objects that are completely irrelevant to what it's doing right now
    (e.g. a sibling robot's own gift box). Only applied to "object"-like
    keys — "robot" and "location" stay unrestricted, since a plan may need
    to visit any location or reference other robots regardless of what's
    named in the goal string.

    This matters a lot in practice: grounding is done per goal, so with N
    robots each cycling through their own task queue, an unrestricted
    objects_by_type gets re-ground with every *other* robot's irrelevant
    objects on every single planning call — the cost compounds across a
    whole queue, not just once. Confirmed by measurement: without this,
    going from 2 total gift objects to 4 turned a sub-second per-task
    planning call into one that didn't return within 60s.

    This is a heuristic: it assumes anything the search *needs* to touch is
    literally named in the goal fluent string — true for these box/resource
    -style domains (goals are just target fluents over concrete object
    names) but not guaranteed for every possible domain. If your domain's
    goal doesn't name an object it still needs along the way, don't rely on
    this restriction — ground your own already-narrowed objects_by_type per
    call instead (e.g. via a custom select_operators_fn-style wrapper).
    """
    tokens = set(goal.replace("&", " ").replace("not ", " ").split())
    restricted: Dict[str, Set[str]] = {}
    for key, members in objects_by_type.items():
        if key in ("robot", "location"):
            restricted[key] = members
        else:
            restricted[key] = {m for m in members if m in tokens} | (members & always_keep)
    return restricted


# ---------------------------------------------------------------------------
# Shared decentralized event-loop engine
# ---------------------------------------------------------------------------

StateViewFn = Callable[[Set[str], Dict[str, List[SimpleAction]], float, str, dict], Set[str]]
SelectOperatorsFn = Callable[[float, str, dict], List[Operator]]
OnPlanFoundFn = Callable[[float, str, List[SimpleAction], dict], None]


def run_decentralized(
    domain: Domain,
    operators_list: List[Operator],
    *,
    state_view_fn: StateViewFn,
    exec_state_view_fn: Optional[StateViewFn] = None,
    select_operators_fn: Optional[SelectOperatorsFn] = None,
    on_plan_found_fn: Optional[OnPlanFoundFn] = None,
    on_plan_failed: str = "fail",  # "fail" | "no_op"
    on_precondition_failed: str = "block",  # "block" | "no_op" | "raise" | "terminate"
    wake_blocked_on_action_end: bool = False,
    no_op_time: float = 5.0,
    planner_cls: type = CoordinationAStarPlanner,
    verbose: bool = False,
) -> PlanResult:
    """Generic per-robot isolated-planning event loop.

    Each robot plans in isolation (call_planner), executes via a heapq event
    queue (TASK_ARRIVAL / ACTION_END / NO_OP_END). The three built-in
    decentralized methods (reactive, reservation, no_op-blind) all fit this
    shape and differ only in the hooks passed in — see decentralized.py.

    `state_view_fn` is what a robot plans *against* (may be optimistic,
    e.g. reactive's broadcast of other robots' committed future effects).
    `exec_state_view_fn` is what's checked when actually starting the next
    committed action — defaults to `state_view_fn` (reservation/no_op-blind
    plan and execute against the same view), but reactive needs these to
    differ: it plans optimistically yet must check *real* current state at
    execution time, which is precisely how it discovers unavailability only
    when an action is actually about to start.

    `on_precondition_failed="terminate"` stops *that one robot* (not the
    whole run) the first time this happens — used by plan_myopic_reactive,
    which has no broadcast/reservation mechanism at all, so a resource
    going missing between when a robot planned to use it and when it
    actually tries to is treated as an outright failure for that robot's
    task, rather than something to wait out or replan around. Other robots
    are unaffected and keep working toward their own goals — see
    `PlanResult.robot_results` for each robot's own outcome; the aggregate
    `success`/`cost`/`message` still reflect whether *every* robot finished,
    same as always.
    """
    exec_view = exec_state_view_fn if exec_state_view_fn is not None else state_view_fn
    world_state: Set[str] = set(domain.initial_state)
    event_queue: List[Event] = []
    ctx: Dict[str, Any] = {}

    robot_plans: Dict[str, List[SimpleAction]] = {r: [] for r in domain.robots}
    robot_blocked: Dict[str, bool] = {r: False for r in domain.robots}
    robot_done: Dict[str, bool] = {r: False for r in domain.robots}
    # Local mutable copy — domain.robot_goals itself is never mutated.
    robot_task_queue: Dict[str, List[str]] = {r: list(domain.robot_goals[r]) for r in domain.robots}
    # The goal each robot is *currently* working on — needed by the no_op
    # retry hooks below, since domain.robot_goals[r] is now a queue, not a
    # single string.
    robot_current_goal: Dict[str, Optional[str]] = {r: None for r in domain.robots}
    # Set per-robot by try_start_next when on_precondition_failed="terminate"
    # fires for that robot — {"t": ..., "action": ..., "fail": ...}. That
    # robot simply gets no further events scheduled; everyone else continues.
    robot_failures: Dict[str, Dict[str, Any]] = {}
    # Simulated time each robot actually finished all its queued goals, set
    # by _advance_or_finish when robot_done[r] becomes True — used for that
    # robot's own PlanResult.robot_results[r]["cost"].
    robot_finish_time: Dict[str, float] = {}
    # One entry per planning attempt per robot — see PlanResult.search_stats.
    task_search_stats: Dict[str, List[Dict[str, Any]]] = {r: [] for r in domain.robots}

    for order, robot in enumerate(sorted(domain.robots)):
        goal = robot_task_queue[robot].pop(0)
        heapq.heappush(
            event_queue,
            Event(time=0.0, robot_id=robot, event_type="TASK_ARRIVAL", data=goal, order=order),
        )

    log = make_logger(domain.robots) if verbose else _null_logger
    steps: List[PlanStep] = []

    def _select_operators(t: float, robot: str) -> List[Operator]:
        if select_operators_fn is not None:
            return select_operators_fn(t, robot, ctx)
        return operators_list

    def _start_no_op(t: float, robot: str, goal: str, reason: str) -> None:
        log(t, robot, f"{reason} — no_op {no_op_time:.1f}s, will retry")
        heapq.heappush(
            event_queue,
            Event(time=t + no_op_time, robot_id=robot, event_type="NO_OP_END", data=(goal, t)),
        )

    def _advance_or_finish(t: float, robot: str) -> None:
        """A robot's current goal is done — pull the next one off its own
        queue, or mark it finished if the queue is empty.
        """
        if robot_task_queue[robot]:
            goal = robot_task_queue[robot].pop(0)
            log(t, robot, f"idle — starting next queued task '{goal}'")
            plan_and_start(t, robot, goal)
        else:
            log(t, robot, "finished — idle")
            robot_done[robot] = True
            robot_finish_time[robot] = t

    def plan_and_start(t: float, robot: str, goal: str) -> None:
        robot_current_goal[robot] = goal
        state = state_view_fn(world_state, robot_plans, t, robot, ctx)
        ops = _select_operators(t, robot)
        task_objects = restrict_objects_by_type(
            domain.objects_by_type, goal, set(domain.contested_resources.keys()),
        )
        stats_out: Dict[str, int] = {}
        plan = call_planner(
            state, goal, task_objects, ops, robot,
            planner_cls=planner_cls, extra_cost_fn=domain.extra_cost_fn,
            stats_out=stats_out,
        )
        task_search_stats[robot].append({
            "goal": goal,
            "states_expanded": stats_out.get("states_expanded"),
            "states_generated": stats_out.get("states_generated"),
        })
        if plan is None:
            log(t, robot, "goal already satisfied")
            _advance_or_finish(t, robot)
        elif not plan:
            if on_plan_failed == "no_op":
                _start_no_op(t, robot, goal, "no full plan found")
            else:
                log(t, robot, f"FAILED — cannot plan for '{goal}'")
        else:
            if on_plan_found_fn is not None:
                on_plan_found_fn(t, robot, plan, ctx)
            robot_plans[robot] = plan
            try_start_next(t, robot)

    def try_start_next(t: float, robot: str) -> None:
        if not robot_plans[robot]:
            _advance_or_finish(t, robot)
            return

        action = robot_plans[robot][0]
        state = exec_view(world_state, robot_plans, t, robot, ctx)
        fail = check_preconditions(action, state)

        if fail is not None:
            if on_precondition_failed == "block":
                robot_blocked[robot] = True
                log(t, robot, f"WAITING — '{fail}' not yet true, will retry")
            elif on_precondition_failed == "no_op":
                robot_plans[robot] = []
                current_goal = robot_current_goal[robot]
                assert current_goal is not None
                _start_no_op(
                    t, robot, current_goal,
                    f"'{fail}' unexpectedly false — plan invalidated",
                )
            elif on_precondition_failed == "terminate":
                robot_failures[robot] = {"t": t, "action": action.name, "fail": fail}
                log(t, robot, f"FAILED — '{fail}' not available for {action.name}, stopping this robot")
            else:
                raise RuntimeError(
                    f"{robot}'s precondition '{fail}' unexpectedly false at t={t:.2f} "
                    f"for {action.name} — reservation prediction was wrong"
                )
            return

        robot_blocked[robot] = False
        for d in action.del_effects:
            world_state.discard(d)
        log(t, robot, f"START  {action.name}")
        heapq.heappush(
            event_queue,
            Event(time=t + action.duration, robot_id=robot, event_type="ACTION_END", data=action),
        )

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
            goal, start = event.data
            log(t, robot, "END    no_op — replanning")
            steps.append(PlanStep(robot=robot, action="no_op", start=start, end=t))
            plan_and_start(t, robot, goal)

        elif event.event_type == "ACTION_END":
            action: SimpleAction = event.data
            for a in action.add_effects:
                world_state.add(a)
            log(t, robot, f"END    {action.name}")
            steps.append(PlanStep(robot=robot, action=action.name, start=t - action.duration, end=t))
            robot_plans[robot].pop(0)
            try_start_next(t, robot)
            if wake_blocked_on_action_end:
                for other in domain.robots:
                    if other != robot and robot_blocked[other]:
                        log(t, other, f"retrying after {robot} completed {action.name}")
                        try_start_next(t, other)
    robot_results: Dict[str, Dict[str, Any]] = {}
    for r in domain.robots:
        if r in robot_failures:
            f = robot_failures[r]
            robot_results[r] = {
                "success": False,
                "cost": f["t"],
                "message": (
                    f"'{f['fail']}' was not available for {f['action']} "
                    f"— failed at t={f['t']:.2f}"
                ),
            }
        else:
            robot_results[r] = {
                "success": robot_done[r],
                "cost": robot_finish_time.get(r),
                "message": "" if robot_done[r] else "did not reach its goal",
            }

    success = all(robot_done[r] for r in domain.robots)
    steps.sort(key=lambda s: s.start)
    cost = final_time if success else None
    if success:
        message = ""
    else:
        message = "; ".join(
            f"{r}: {robot_results[r]['message']}"
            for r in domain.robots
            if not robot_results[r]["success"]
        )
    return PlanResult(
        success=success, cost=cost, steps=steps, message=message,
        robot_results=robot_results, search_stats=task_search_stats,
    )
