# Multi-robot coordination planners

Five ways to plan for multiple robots sharing contended resources, built on
top of `railroad`'s temporal planning core. Three files:

- **[common.py](common.py)** — the `Domain` contract, `PlanResult`/`PlanStep`,
  and shared infrastructure (event-queue engine, logger, timing fixes,
  reservation bookkeeping).
- **[decentralized.py](decentralized.py)** — 3 methods where each robot plans
  in isolation over only its own actions.
- **[centralized.py](centralized.py)** — 2 methods where one search runs over
  the combined action space of all robots.
- **[demo_planners.py](demo_planners.py)** — a runnable example wiring the
  box-station gift-wrap domain into all 5 methods.

Every method has the signature `plan_xxx(domain: Domain, verbose: bool = False, ...) -> PlanResult`.

```python
result = plan_reservation(domain, verbose=True)
result.success   # bool
result.cost      # float makespan, or None if no plan was found
result.steps     # List[PlanStep(robot, action, start, end)], chronologically sorted
result.message   # short status string, e.g. failure reason
```

`verbose=True` prints a live, columnar, one-column-per-robot log as the
simulation runs (or, for the centralized/single-step methods, as a
chronologically-merged replay) — see any run of `demo_planners.py` for what
this looks like.

## The 5 planners

| # | Function | File | Scope | Coordination mechanism | Search | Trade-off |
|---|---|---|---|---|---|---|
| 1 | `plan_reactive` | decentralized.py | per-robot | Optimistic: plans against a *broadcast* of other robots' committed future effects; discovers real unavailability only at execution time; recovers by blocking + retrying whenever another robot finishes an action | `CoordinationAStarPlanner` (full plan per goal) | Simple, no bookkeeping; gets independent work done "for free" before hitting a real conflict, since its optimistic plan is a complete valid sequence |
| 2 | `plan_reservation` | decentralized.py | per-robot | Proactive: a Python-side `Reservation` queue predicts release times from committed plans, baked into a real `wait_for_resource` action with an exact (post-hoc-corrected) duration | `CoordinationAStarPlanner` | No execution-time surprises; but the search has no cost signal rewarding overlap of independent work with the wait — only happens if the search stumbles into it |
| 3 | `plan_no_op_blind` | decentralized.py | per-robot | Aware-but-blind: a real `reserved`/`reserved_by` fluent, but *no* timing prediction anywhere — fixed-interval `no_op` + full-goal replan from scratch, including abandoning an in-flight plan invalidated mid-execution | `CoordinationAStarPlanner` | Simplest to reason about; worst performer — all-or-nothing replanning can't salvage partial progress, and an optimistic commit can be invalidated mid-flight |
| 4 | `plan_joint_astar` | centralized.py | joint | None needed — exclusivity falls out of the shared `at ?obj ?loc` fluent | `AStarPlanner`, optimal/exhaustive | Correct and makespan-optimal in principle; **empirically intractable** at any real scale (measured: 6GB+ memory and climbing within 30s on a reduced 2-robot goal, no convergence) |
| 5 | `plan_joint_mcts` | centralized.py | joint | None needed | `MCTSPlanner`, satisficing | Tractable (seconds); needs a `no_op` operator for momentary zero-legal-actions deadlocks (e.g. right after a move, `just-moved` blocks another move for 0.1s with nothing else to do); plan quality suffers — can include redundant/wasteful sub-plans |

Underlying point: every method sits on the same temporal-with-concurrent-effects
engine. What changes is only (a) whether the search sees one robot's actions
or all of them, (b) what protocol (if any) tells a robot about another
robot's resource timeline, and (c) whether the search is exhaustive (A*) or
sampling-based (MCTS). There's no free lunch in this table — every row buys
tractability, correctness, or plan quality by giving up one of the other two.

## Running it against a different domain

Nothing in `common.py`/`decentralized.py`/`centralized.py` is specific to the
box-station gift-wrap scenario — build a `Domain` describing your own problem
and hand it to whichever method(s) you want.

### 1. Define your world

```python
objects_by_type = {
    "robot":    {"robot1", "robot2"},
    "location": {"kitchen", "pantry", "table"},
    "object":   {"knife", "bowl"},          # anything pick/place-able
}

initial_state = {
    "at robot1 kitchen", "free robot1",
    "at robot2 kitchen", "free robot2",
    "at knife pantry",
    "at bowl pantry",
}
```

`initial_state` is a set of fluent strings, same format used throughout this
package (`"at robot1 kitchen"`, or `"not free robot1"` / negate with `~F(...)`
when constructing operators — see `planner_interface._str_to_fluent` for the
exact parsing rule).

### 2. Define your domain's own operators

Write your own `Operator`s for whatever work your robots do (mixing, moving,
searching — whatever isn't pick/place/wait/no_op/reserve, which the planning
methods construct for you automatically). Example move + a work operator:

```python
from railroad.core import Operator, Effect, Fluent as F
from railroad.operators._utils import Numeric

def _move_time(robot, loc_from, loc_to) -> float:
    return 5.0  # or a real distance function

move_op = Operator(
    name="move",
    parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
    preconditions=[F("at ?r ?from"), F("free ?r")],
    effects=[
        Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
        Effect(time=(Numeric(_move_time), ["?r", "?from", "?to"]),
               resulting_fluents={F("free ?r"), F("at ?r ?to")}),
    ],
)

mix_op = Operator(
    name="mix",
    parameters=[("?r", "robot"), ("?loc", "location")],
    preconditions=[F("at ?r ?loc"), F("free ?r"), F("holding ?r bowl")],
    effects=[
        Effect(time=0, resulting_fluents={~F("free ?r")}),
        Effect(time=10.0, resulting_fluents={F("free ?r"), F("mixed")}),
    ],
)

base_operators = [move_op, mix_op]
```

**Fluent conventions your operators must follow** — these are hardcoded into
the pick/place/no_op/wait/reserve operators the planning methods build, so
your own operators need to use the same names for the engine to interoperate
correctly:
- `free ?r` — robot is idle / available for a new decision. Every operator
  should clear it at the start (time 0) and set it again when the robot
  becomes available. This is also what the timing-correction helper
  (`free_again_duration`) keys off when replaying a plan for display.
- `at ?r ?loc` — robot location. `at ?obj ?loc` — object location (what
  pick/place read and write).
- `holding ?r ?obj` / `hand-full ?r` — set by pick, cleared by place; your
  own operators can *read* these as preconditions (e.g. "need scissors in
  hand") but shouldn't write them.

Anything else — `accessible ?r ?loc`, `is_workstation ?loc`, or whatever
domain-specific flags you invent — is entirely up to you; those are
box-station's own conventions, not engine requirements.

### 3. Describe goals and contested resources

```python
robot_goals = {
    "robot1": ["mixed & at bowl table"],
    "robot2": ["at knife table", "at knife pantry"],   # a queue — 2nd task
                                                        # only starts once
                                                        # the 1st is done
}
contested_resources = {"bowl": "pantry"}   # object -> its "home" location
```

`robot_goals` is a **queue per robot** (`Dict[str, List[str]]`), matching the
`ROBOT_TASKS` shape from `pick_and_place_astar_boxstation.py` — a robot only
starts its next task once the previous one is actually satisfied. Goal
strings support `&`-joined conjunctions of fluents (positive, or
`not `-prefixed for negation) — see `planner_interface._goal_from_str`.
`contested_resources` is only read by `plan_reservation`/`plan_no_op_blind`
(it's what lets them build a `wait_for_resource` action and predict release
times) — the other 3 methods ignore it; leave it as `{}` if you have no
shared resource to reserve.

**The decentralized methods** (`plan_reactive`/`plan_reservation`/
`plan_no_op_blind`) work through each robot's *entire* queue automatically
within a single call — you don't need to call the function once per task.

**The centralized methods** (`plan_joint_astar`/`plan_joint_mcts`) only ever
plan for each robot's *first* queued task — see `centralized.py`'s
`_combined_goal` — since joint search over even one goal per robot is
already at the edge of tractable (`plan_joint_astar`). If you need a
centralized plan for a robot's later queued tasks, call it again yourself
with a `Domain` whose `robot_goals` only contains the remaining tasks.

**Grounding is restricted per goal, automatically.** Every planning call
narrows `objects_by_type` down to just the objects literally named in the
goal string being planned right now (`common.restrict_objects_by_type`) —
`"robot"`/`"location"` stay unrestricted, but e.g. `"object"`/`"gift"` don't.
This matters a lot in practice, not just as a micro-optimization: with a
task queue, each robot re-grounds on every single queued task, so an
unrestricted `objects_by_type` re-grounds pick/place for *every other*
object in the whole domain (including every other robot's own gift boxes)
on every one of those calls. Measured effect: going from 2 total pickable
objects to 4 turned a sub-second planning call into one that didn't return
within 10 minutes, before this restriction was added. The restriction is a
heuristic — it assumes anything a plan needs to touch is named in the goal
fluent string, true here (goals are just target fluents over concrete
object names) but not guaranteed for every possible domain; if your domain's
goal doesn't name an object it still needs along the way, don't rely on it.

### 4. Build the Domain and call a planner

```python
from common import Domain
from decentralized import plan_reservation
from centralized import plan_joint_mcts

domain = Domain(
    objects_by_type=objects_by_type,
    initial_state=initial_state,
    robots=["robot1", "robot2"],
    base_operators=base_operators,
    robot_goals=robot_goals,
    contested_resources=contested_resources,
    pick_time=1.0,     # optional, defaults shown
    place_time=1.0,
)

result = plan_reservation(domain, verbose=True)
print(result.success, result.cost)
```

Run `uv run python scripts/demo_planners.py` for a complete, working example
(box-station domain, all 5 methods compared side by side). Copy its
`build_domain()` function as a template.

## Adding a new operator to an existing domain

Just add another `Operator` to the `base_operators` list you pass into
`Domain` — nothing else needs to change. For example, adding a `paint`
step to the mixing example above:

```python
paint_op = Operator(
    name="paint",
    parameters=[("?r", "robot"), ("?loc", "location")],
    preconditions=[F("at ?r ?loc"), F("free ?r"), F("mixed"), ~F("painted")],
    effects=[
        Effect(time=0, resulting_fluents={~F("free ?r")}),
        Effect(time=8.0, resulting_fluents={F("free ?r"), F("painted")}),
    ],
)

domain.base_operators.append(paint_op)   # or include it when building base_operators
```

Update `robot_goals` to require `F("painted")` if it should be part of the
goal. No changes are needed in `common.py`/`decentralized.py`/`centralized.py`
— all 5 planners ground whatever operators the `Domain` supplies at call
time via `SymbolicEnvironment`/`call_planner`, so a new operator is picked up
automatically as long as its preconditions/effects use fluent names
consistent with the rest of your domain (see the conventions above).

If the new operator introduces a *second* contended resource, add it to
`contested_resources` (e.g. `{"bowl": "pantry", "brush": "sink"}`) —
`plan_reservation`/`plan_no_op_blind` already loop over every entry in that
dict, so multiple resources should work, though only the single-resource
case (box-station's scissors) has actually been tested end-to-end.
