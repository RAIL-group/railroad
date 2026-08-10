"""Box-station domain — joint/centralized temporal planner, MCTS-backed.

joint_astar_boxstation.py attempted the "pure, provably optimal" version of
a centralized joint planner: one A* search over both robots' combined
action space, no per-robot isolation, no reservation fluent. It's correct
in principle (see that file's docstring) but empirically intractable even
on a *reduced* single-gift-per-robot version of this domain — the search's
open/closed sets grew past 6GB in under 30 seconds with no sign of
converging, before being killed. That's not a bug, it's the actual
scalability wall this whole conversation has been circling: a joint search
over N robots' combined action space blows up combinatorially, and this
domain (7 operators x 4 locations x 2 robots, ~17 real actions per robot)
is already enough to hit it hard under an *optimal*, exhaustive algorithm.

This file is the same idea — one search, unfiltered joint action space,
single AND-goal, no reservation trick needed (mutual exclusion on scissors
falls out of the shared `at scissors ?loc` fluent for free) — but backed by
MCTS instead of A*, which is exactly the choice this repo's own existing
joint-planning examples already make for this reason:
packages/railroad/src/railroad/examples/heterogeneous_robots.py and
.../bench/benchmarks/multi_object_search.py both ground *all* robots'
actions unfiltered and run MCTSPlanner, never a from-scratch optimal
search, over a genuinely large combined action space. MCTS trades the
optimality guarantee for tractability: bounded rollouts, satisficing, not
exhaustive — the same trade this codebase already made elsewhere for
multi-robot joint planning.

Two things had to be worked out empirically to get this to actually run,
both worth keeping in mind:
  - A momentary *zero-legal-actions* deadlock is possible even when some
    robot is nominally "free": e.g. right after a move, `just-moved` blocks
    another move for 0.1s, and there may be nothing at the robot's current
    location to pick. `no_op` (precondition is just `free ?r`) is the
    escape valve for that, same as multi_robot_breakfast.py/
    heterogeneous_robots.py — without it, MCTS has nothing legal to select
    and reports "NONE" even though the system isn't actually stuck.
  - The goal here is only `paper_cut` for both gifts (not the full
    `wrapped_gift`, which needs ~17 real actions per robot): MCTS is
    satisficing, not optimal, and the *combined* 2-robot AND-goal is
    considerably harder for its heuristic to guide well than either
    robot's subgoal alone — the full goal did eventually get "achieved" in
    testing, but only via a long, visibly wasteful plan (robot1 redundantly
    re-picking/re-placing its own gift box several times before settling).
    The shorter goal here still exercises everything interesting (box
    collection, scissors handoff, natural mutual exclusion) with a plan
    that stays legible.

Run directly: `uv run scripts/joint_mcts_boxstation.py`
"""

import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from railroad.core import Fluent as F, State, get_action_by_name
from railroad.environment.symbolic import SymbolicEnvironment
from railroad.planner import MCTSPlanner
from railroad import operators

from planner_interface import _str_to_fluent, _action_duration  # pyrefly: ignore [missing-import]


def _free_again_duration(action) -> float:
    """When this action's robot is actually available for a new decision.

    `_action_duration` reports the time of an action's *last* effect, which
    overstates this for move/pick/place (the "_blocking" constructors):
    they have a trailing bookkeeping-only effect — clearing just-moved/
    just-picked/just-placed — that fires *after* `free ?r` is already true
    again (e.g. move sets `free`/`at` at `move_time`, then only clears
    `just-moved` at `move_time + 0.1`; that extra 0.1s just blocks an
    immediate re-move, it doesn't keep the robot busy). Using the last
    effect's time as "when this action ends" made replayed logs show a
    later action starting before an earlier one "finished" — not a bug in
    the plan, just the wrong effect being used to mark completion. The
    correct one is whichever effect actually sets `free {robot}` back to
    true; if none does (cut_paper/wrap_gift/cut_ribbon/complete_job all
    have a clean two-effect shape where that already coincides with the
    last effect), fall back to `_action_duration`.
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
    make_logger,
)

# Plain pick/place — no `reserved` fluent needed (see module docstring):
# exclusivity on scissors comes for free from the shared `at ?obj ?loc`
# fluent both robots' actions read from the same joint state.
pick_op = operators.construct_pick_operator_blocking(pick_time=1.0)
place_op = operators.construct_place_operator_blocking(place_time=1.0)
no_op_op = operators.construct_no_op_operator(no_op_time=0.5, extra_cost=10.0)

# wrap_gift/cut_ribbon/complete_job deliberately left out: the goal below
# only needs paper_cut, and leaving their operators ungrounded keeps the
# joint action space (and therefore MCTS's branching factor) smaller.
my_operators = [move_op, pick_op, place_op, cut_paper_op, no_op_op]

# Same single-gift-per-robot scope as test_boxstation.py / joint_astar_boxstation.py.
task_objects = {
    **objects_by_type,
    "gift": {"gift1", "gift4"},
    "object": {"scissors", "gift1", "gift4"},
}


def main() -> None:
    fluents = {_str_to_fluent(s) for s in initial_world_state}
    initial_state = State(0.0, fluents)

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=task_objects,
        operators=my_operators,
    )

    goal = (
        F("paper_cut robot1 gift1")
        & F("paper_cut robot2 gift4")
        & F("at scissors tool_space")
    )

    log = make_logger(["robot1", "robot2"])
    timeline: List[Tuple[float, str, str]] = []

    max_steps = 80
    for step in range(max_steps):
        if goal.evaluate(env.state.fluents):
            break

        all_actions = env.get_actions()  # unfiltered: both robots' actions together
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(
            env.state, goal, max_iterations=5000, c=300, max_depth=25,
            heuristic_multiplier=3,
        )

        if action_name == "NONE":
            print(f"No action found at step {step}, t={env.state.time:.1f} — stuck.")
            break

        action = get_action_by_name(all_actions, action_name)
        robot = action.name.split()[1]
        start = env.state.time
        duration = _free_again_duration(action)
        timeline.append((start, robot, f"START  {action.name}"))
        timeline.append((start + duration, robot, f"END    {action.name}"))
        env.act(action)

    timeline.sort(key=lambda entry: entry[0])
    for t, robot, msg in timeline:
        log(t, robot, msg)

    if goal.evaluate(env.state.fluents):
        print(f"\nJoint goal achieved. {len(timeline) // 2} actions total.")
        print(f"Makespan: {max(t for t, _, _ in timeline):.1f}")
    else:
        print(f"\nGoal NOT achieved after {max_steps} steps (t={env.state.time:.1f}).")


if __name__ == "__main__":
    main()
