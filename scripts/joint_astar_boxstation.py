"""Box-station domain — joint/centralized temporal planner.

This is the "true joint temporal planner" contrasted against the
decentralized approaches (pick_and_place_astar_boxstation.py's reactive
retries, ..._reservation.py's proactive wait-with-known-duration,
test_boxstation.py's blind no_op polling): ONE single A* search over the
COMBINED action space of both robots, against a single AND-goal covering
both robots' tasks, with no per-robot isolation, no reservation fluent, no
wait/no_op operator at all.

Why this works "for free" with existing infrastructure: the C++ core
(railroad._bindings) already tracks per-state "upcoming effects" so that one
robot's long in-flight action doesn't block a *different* robot's action
from being taken concurrently (see State::advance_to_terminal) — this is
exactly the "concurrent multi-agent action" the engine was built for (see
CLAUDE.md, and the existing joint-planning examples
packages/railroad/src/railroad/examples/heterogeneous_robots.py and
.../bench/benchmarks/multi_object_search.py, which both already ground *all*
robots' actions unfiltered and run one MCTSPlanner over them).

Two things make this different from our other scripts here:
  1. Actions are NOT filtered to one robot (`SymbolicEnvironment.get_actions()`
     is used unfiltered) — the search can freely interleave robot1 and
     robot2 actions in a single sequence.
  2. It uses `railroad.planner.AStarPlanner` (native C++, backed by the
     unrestricted C++ `get_next_actions`), NOT
     planner_interface.CoordinationAStarPlanner — that hand-rolled Python
     search uses railroad.core's pure-Python `get_next_actions`, which has a
     documented bug restricting visible actions to a single "next free
     robot" and silently breaks joint multi-robot search.

No `reserved`/`reserved_by` fluent bookkeeping is needed either: mutual
exclusion on scissors falls straight out of the *shared* `at scissors ?loc`
fluent — once robot1 picks it up, it's no longer "at" anywhere, so robot2's
own pick action's precondition simply can't be satisfied until robot1
places it back down. A temporal/joint planner gets resource exclusivity for
free from the state model; a decentralized per-robot planner needs the
`reserved` trick specifically because its searches never see each other's
in-flight state at all.

Run directly: `uv run scripts/joint_astar_boxstation.py`
"""

import os
import sys
from typing import List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from railroad.core import Fluent as F, State, transition as _transition
from railroad._bindings import astar as _astar_cpp
from railroad.environment.symbolic import SymbolicEnvironment
from railroad.planner import AStarPlanner
from railroad import operators

from planner_interface import _str_to_fluent, _action_duration  # pyrefly: ignore [missing-import]


def _free_again_duration(action) -> float:
    """When this action's robot is actually available for a new decision.

    `_action_duration` reports the time of an action's *last* effect, which
    overstates this for move/pick/place (the "_blocking" constructors):
    they have a trailing bookkeeping-only effect — clearing just-moved/
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
        eff.time for eff in action.effects
        if any(f == free_fluent and not f.negated for f in eff.resulting_fluents)
    ]
    return min(free_times) if free_times else _action_duration(action)

from pick_and_place_astar_boxstation import (
    objects_by_type,
    initial_world_state,
    move_op,
    cut_paper_op,
    wrap_gift_op,
    cut_ribbon_op,
    complete_job_op,
    make_logger,
)

# Plain pick/place — no `reserved` fluent needed at all for a joint search
# (see module docstring): exclusivity comes for free from the shared
# `at ?obj ?loc` fluent both robots' searches see simultaneously.
pick_op = operators.construct_pick_operator_blocking(pick_time=1.0)
place_op = operators.construct_place_operator_blocking(place_time=1.0)

my_operators = [move_op, pick_op, place_op, cut_paper_op, wrap_gift_op, cut_ribbon_op, complete_job_op]

# Same single-gift-per-robot scope as test_boxstation.py — enough to observe
# the scissors handoff, without the extra grounding cost of all 6 gifts.
task_objects = {
    **objects_by_type,
    "gift": {"gift1", "gift4"},
    "object": {"scissors", "gift1", "gift4"},
}


def main() -> None:
    fluents = {_str_to_fluent(s) for s in initial_world_state}
    state = State(0.0, fluents)

    # Unfiltered: both robots' grounded actions in one list, handed to one
    # single search. Nothing here restricts the planner to "one robot's own
    # actions" — that restriction was always a *choice* our decentralized
    # scripts made in call_planner, not a property of the engine.
    env = SymbolicEnvironment(state, task_objects, my_operators)
    all_actions = env.get_actions()

    goal = (
        F("wrapped_gift robot1 gift1")
        & F("wrapped_gift robot2 gift4")
        & F("at scissors tool_space")
    )

    planner = AStarPlanner(all_actions)
    plan = planner.plan_sequence(state, goal)

    if not plan:
        print("No joint plan found.")
        return

    # _prepare applies the same negative-precondition -> positive-fluent
    # conversion plan_sequence used internally, giving a converted_state
    # consistent with the returned (also-converted) plan actions, so we can
    # replay via the same `transition` the search itself used.
    converted_state, _ = planner._prepare(state, goal)

    # Replay to recover each action's real start/end time. state.time after a
    # transition marks the *next* decision point across BOTH robots (possibly
    # a different robot whose own action finishes sooner) — not necessarily
    # when *this* action's own effects resolve. But the search only ever
    # offers an action to a robot the instant it's free, so state.time
    # *before* taking action[i] is exactly when action[i] starts; combined
    # with its fixed, known duration that's enough to get the correct end
    # time regardless of what the replay's state.time does afterward.
    s = converted_state
    timeline: List[Tuple[float, str, str]] = []
    end_times = []
    for action in plan:
        robot = action.name.split()[1]
        start = s.time
        duration = _free_again_duration(action)
        end = start + duration
        end_times.append(end)
        timeline.append((start, robot, f"START  {action.name}"))
        timeline.append((end, robot, f"END    {action.name}"))
        for successor, prob in _transition(s, action):
            if prob == 0.0:
                continue
            s = successor
            break

    timeline.sort(key=lambda entry: entry[0])
    robots = sorted({robot for _, robot, _ in timeline})
    log = make_logger(robots)
    for t, robot, msg in timeline:
        log(t, robot, msg)

    n_r1 = sum(1 for a in plan if a.name.split()[1] == "robot1")
    n_r2 = sum(1 for a in plan if a.name.split()[1] == "robot2")
    print(f"\nJoint plan: {len(plan)} actions total (robot1: {n_r1}, robot2: {n_r2})")
    print(f"Makespan: {max(end_times):.1f}")


if __name__ == "__main__":
    main()
