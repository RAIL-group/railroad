# `railroad` : Concurrent multi-agent, probabilistic planning framework

**Multi-Agent Task Planning, supporting concurrency and probabilistic effects. PDDL-inspired operators with a Python API**

The `railroad` planning framework is meant to support **concurrent multi-robot task planning under uncertainty**. Operators are PDDL-like and defined in Python, so that learned estimators can be used to specify timing, probabilities, and costs. Planning is C++-based for efficiency, and uses MCTS with an uncertainty-aware FF heuristic (_still a work in progress_) as its value function.

*Developed by the [Robot Anticipatory Intelligence & Learning (RAIL) Group @ GMU](https://people.cs.gmu.edu/~gjstein/), led by Prof. Gregory J. Stein.*

#### Key properties
- **States store both active fluents and upcoming effects**: actions add effects to a queue, which update the active fluents as time advances.
- **Concurrency**: state transitions advance time until an agent is marked `(free {agent_name})`, letting multiple robots act concurrently.
- **Probabilistic state transitions**: effects can be probabilistic
- **Conditional effects**: effects can branch on the state at the moment they fire (PDDL-style `when`), including universally quantified forms (`forall`+`when` via `ForallEffect`)
- **Action costs beyond time**: `Operator(extra_cost=...)` adds a scalar cost charged in the planning objective (time + Σ cost) and steered by the heuristic
- **Planning via MCTS**: planning via Monte Carlo Tree Search over joint action spaces
- **PDDL/PPDDL import**: `railroad pddl` downloads, converts, and runs International Planning Competition problems (see `railroad.pddl_converter`)

#### Multi-Robot Object Search Example

In this [ProcTHOR](https://procthor.allenai.org/)-generated household environment, the team is told to search for two objects, with proabilities correlated with their underlying locations, and deliver them at a destionation. The planner coordinates them to search effectively and split up, pirotizing search of the locations where the objects are likely to be.

<img src="assets/procthor-search-8616.jpeg" width="720" alt="Two-robots quickly searching for and delivering two objects in a ProcTHOR-generated home.">

#### Quickstart via the [`uv`](https://docs.astral.sh/uv/) package manager
```bash
mkdir railroad-env && cd railroad-env && uv init
uv add "railroad @ git+https://github.com/RAIL-group/railroad.git#subdirectory=packages/railroad"
uv run railroad example multi-object-search
```
*Note:* Linux systems require `python3.12-dev` and `build-essential`.

Use the optional benchmark suite
```bash
# Install with railroad[bench]
uv add "railroad[bench] @ git+https://github.com/RAIL-group/railroad.git#subdirectory=packages/railroad"
uv run railroad benchmarks run --dry-run  # Inspect what will run
uv run railroad benchmarks run  # Runs all
uv run railroad benchmarks run <filter>  # Runs all with <filter> string

# Run the interactive web dashboard (after starting/running benchmarks)
uv run railroad benchmarks dashboard
```

The dashboard answers on every interface, so viewing it from a phone or another
machine on a VPN needs no setup -- it prints the URLs that work, including your
tailnet address. `--host tailscale` binds only that address, `--host 127.0.0.1`
keeps it to this machine, and `--port` moves it.

ProcTHOR is an optional install via `railroad[procthor]`. To run the example from above and generate a video:
```bash
uv add "railroad[procthor] @ git+https://github.com/RAIL-group/railroad.git#subdirectory=packages/railroad"
uv run railroad example procthor-search --seed 8616 --save-video ./procthor-search-8616.mp4 --save-plot ./procthor-search-8616.jpg
```

`railroad` is is heavily type-hinted, and so can be used with a standard type checker, e.g., via `uv run ty check`.

## Quick Example: Two-Robot Object Search

*[Run this example in a Google Colab notebook](https://colab.research.google.com/drive/1jdUtZmKc9OA9LiCSeDdteCqZMGRHik2U?usp=sharing).*

Two robots concurrently move and search to find a Knife and a Cup in a five-room space.

```python
import numpy as np
from railroad.core import Fluent as F, get_action_by_name, State, Operator, Effect
import railroad.operators
from railroad.environment import ObjectSearchEnvironment
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard

locations = {
    "den":     np.array([5, 5]),
    "kitchen": np.array([0, 0]),
    "bedroom": np.array([10, 0]),
    "office":  np.array([0, 8]),
    "garage":  np.array([10, 8]),
}
objects_by_type = {
    "robot":    {"robot1", "robot2"},
    "location": set(locations),
    "object":   {"Knife", "Cup"},
}
# Ground truth object locations (unknown to robots initially)
true_object_locations = {"kitchen": {"Cup"}, "garage": {"Knife"}}

def move_time(robot, loc_from, loc_to):
    return float(np.linalg.norm(locations[loc_from] - locations[loc_to]))

move = Operator(
    name="move",
    parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
    preconditions=[F("at ?r ?from"), F("free ?r")],
    effects=[  # not free at t=0, free again at destination after move_time
        Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
        Effect(time=(move_time, ["?r", "?from", "?to"]),
               resulting_fluents={F("free ?r"), F("at ?r ?to")}),
    ],
)

@railroad.operators.numeric  # decorator to allow algebraic "1 - prob"
def object_find_prob(robot: str, loc: str, obj: str) -> float:
    objects_here = true_object_locations.get(loc, set())
    return 0.9 if obj in objects_here else 0.2

search = Operator(
    name="search",
    parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
    preconditions=[F("at ?r ?loc"), F("free ?r"), F("not found ?obj"),
                   F("not revealed ?loc"), F("not searched ?loc ?obj")],
    effects=[  # after 5s, location is searched, revealing if object is there
        Effect(time=0, resulting_fluents={F("not free ?r")}),
        Effect(time=5.0, resulting_fluents={F("free ?r"), F("searched ?loc ?obj")},
               prob_effects=[  # find object with p=object_find_prob
                   ((object_find_prob, ["?r", "?loc", "?obj"]),
                    [Effect(time=0, resulting_fluents={F("found ?obj"), F("at ?obj ?loc")})]),
                   ((1 - object_find_prob, ["?r", "?loc", "?obj"]), []),
               ]),
    ],
)

# Both robots start free in the den
initial_state = State(0.0, {
    F("free robot1"), F("free robot2"),
    F("at robot1 den"), F("at robot2 den"),
    F("revealed den"),
})

goal = F("found Knife") & F("found Cup")

env = ObjectSearchEnvironment(
    state=initial_state, objects_by_type=objects_by_type,
    operators=[move, search],
    true_object_locations=true_object_locations,
)

def fluent_filter(f):
    return any(kw in f.name for kw in ["at", "holding", "found"])

with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
    # Plan-act loop: replan whenever a robot becomes free
    for _ in range(20):
        if goal.evaluate(env.state.fluents):
            break

        actions = env.get_actions()
        planner = MCTSPlanner(actions)
        action_name = planner(env.state, goal, max_iterations=10000, c=200)
        action = get_action_by_name(actions, action_name)
        env.act(action)
        dashboard.update(planner, action_name)
```

The planner dispatches both robots in parallel. Sample dashboard output:

```
Actions Taken (5)
         |0.0                       12.1|
  robot1 |1                4            |
  robot2 |2             3           5   |

  1. move robot1 den kitchen
  2. move robot2 den garage
  3. search robot2 garage Knife
  4. search robot1 kitchen Cup
  5. move robot2 garage office

Goal:
  AND(✓(found Knife), ✓(found Cup))

Total cost: 12.1 (seconds)
```

## Built-in Examples

```bash
uv run railroad example <name>
```

- **`multi-object-search`** -- Search for and collect multiple objects with multiple robots
- **`clear-table`** -- Clear objects from a table (demonstrates negative goals)
- **`find-and-move-couch`** -- Cooperative task requiring two robots (demonstrates wait operators)
- **`heterogeneous-robots`** -- Drone, rover, and crawler with different speeds and capabilities
  - Add `--interruptible-moves` to allow rerouting robots mid-transit
- **`frontier-search`** -- Explore unknown space and search discovered sites for objects (requires `railroad[procthor]`)

## Guided Tutorial

An eight-step tour that builds a problem up from operators, driven from a single
editable file:

```bash
uv run railroad tutorial doctor   # check extras, scene cache, git, ffmpeg
uv run railroad tutorial init     # scaffold ./railroad-tutorial
uv run railroad tutorial watch    # the pane: r run, n next, b sweep, o dashboard
```

`demo.py` is the only file you open. `next` prints the diff to the following
step, then three-way merges it into whatever you currently have, so anything you
changed while explaining it survives. Each step carries its own benchmark sweep,
so the same edit drives both the single run and the distribution behind it, and
every sweep accumulates into one MLflow experiment.

The arc: the state semantics, clear-the-table, a second robot, hidden objects,
the per-room search lock, the heuristic knobs, a ProcTHOR home, and finally
swapping the hand-tuned find-probability for the packaged learned model. The
last two need `railroad[procthor]`.

## Key Concepts

- **Fluent** -- A fact about the world: `F("at robot1 kitchen")`, `F("free robot1")`
- **State** -- The current set of fluents, the time, and any upcoming effects
- **Operator** -- An action template with typed parameters (`move ?robot ?from ?to`), optionally carrying a scalar `extra_cost` added to the planning objective
- **Effect** -- A state change that happens at a specified time; can be probabilistic and/or conditional (branches that fire only if their condition fluents hold when the effect does)
- **Goal** -- A target condition to achieve, built from fluents with `&`, `|`, and `~`

## Goal Expressions

Goals compose with Python operators:

```python
from railroad.core import Fluent as F

F("found Knife") & F("found Fork")           # AND -- both must be true
F("at robot1 kitchen") | F("at robot1 bed")   # OR  -- at least one
~F("at Knife table")                          # NOT -- must not hold

# Combine freely
goal = (F("found Knife") | F("found Spoon")) & ~F("at Cup table")
```

## Conditional Effects

Effects can carry branches that fire only if their condition fluents hold
when the effect does (PDDL-style `when`), checked *before* the effect's own
fluents apply:

```python
from railroad.core import Effect, Fluent as F

# Dropping breaks the item -- but only if it's fragile and unpadded.
Effect(
    time=1.0,
    resulting_fluents={F("free ?r"), F("dropped ?x")},
    cond_effects=[
        ({F("fragile ?x"), ~F("padded ?x")},
         [Effect(time=0, resulting_fluents={F("broken ?x")})]),
    ],
)
```

`ForallEffect` is the universally quantified form (`forall`+`when`) — one
branch per object of the quantified type, expanded at `instantiate()` time,
for effects like "moving the briefcase relocates whatever is inside it".
Conditional and probabilistic branches nest freely in either order. The
`feature_examples` benchmarks demonstrate both end to end.

## Action Costs Beyond Time

Operators may carry a scalar `extra_cost`. The planner minimizes expected
completion time **plus** accumulated cost, and the FF heuristic accounts for
it when guiding search — so `Operator(..., extra_cost=10.0)` on a 2-second
toll road loses to a free 6-second back road.

## Importing PDDL / PPDDL Problems

International Planning Competition problems, classical and probabilistic, can
be downloaded, converted, and run directly:

```bash
uv run railroad pddl check ipc-2000     # per-domain compatibility report
uv run railroad pddl run --collection ipc-2000 --domain logistics-strips-typed --instance 1
```

See [the converter README](packages/railroad/src/railroad/pddl_converter/README.md)
for mapping semantics and the supported-feature list.

## Running via this Repo

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/RAIL-group/railroad.git
cd railroad
uv run railroad example multi-object-search   # builds automatically on first run
```

## Architecture

Railroad is organized as a monorepo. The core planning engine lives in `packages/railroad/`:

```
packages/railroad/
  include/               # C++ headers (A* search, MCTS, FF heuristic)
  src/railroad/
    _bindings.cpp        # pybind11 bridge
    core.py              # Fluent, State, Action, Operator, Effect, Goal
    planner.py           # MCTSPlanner (wraps C++ MCTS with automatic preprocessing)
    pddl_converter/      # IPC PDDL/PPDDL download + conversion (see its README)
    operators/           # Helper constructors for move, search, pick, place, wait
    navigation/          # Reusable grid navigation (theta* pathing, occupancy grid mixin)
    environment/
      environment.py     # Abstract Environment base class (subclass & override define_operators())
      symbolic.py        # SymbolicEnvironment: generic symbolic execution
      object_search.py   # ObjectSearchEnvironment: search-domain conventions
      skill/             # Skill protocols + navigation skill implementations
      procthor/          # Optional AI2-THOR/ProcTHOR 3D simulator integration
    experimental/        # Frontier-based unknown-space exploration
    examples/            # Built-in runnable examples
    bench/               # Benchmarking framework with MLflow + Plotly Dash
```

Additional packages:
- `packages/environments/` -- Extra environment backends (e.g. PyRoboSim)

## Development

```bash
uv run ty check              # type-check (fast, run first)
uv run pytest                # full test suite
uv run pytest -vk <filter>   # run specific tests
```

`uv run` automatically detects changes to source files (including C++) and rebuilds as needed.
