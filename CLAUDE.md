# CLAUDE.md

`railroad` is a research repository for concurrent multi-agent probabilistic planning via a PDDL-style interface. Actions, defined via operators, contain timestamped effects. The 'state' tracks both 'active fluents' (things that are true now) and 'upcoming effects', both deterministic and probabilistic, which are effects that will become true at a later time, allowing for multi-agent concurrent action in the environment. The system combines a C++-accelerated planning core with Python bindings, optionally integrated with AI2-THOR/ProcTHOR simulators for environment generation.

## Build System

This project uses `uv` as the package manager and build tool. The build system automatically handles C++ compilation when needed. Build is automatic via `uv run`, which detects changes to code (including C++) and rebuilds as necessary. Do not run `uv sync` unless explicitly needed.

- **Type checking**: `uv run ty check`. Type checking is fast and effective and should be run before tests, examples, or scripts.
- **Unit tests**: `uv run pytest`. Tests can be filtered via `uv run pytest -vk <filter>`. We have strong test coverage, so tests are a good way to validate the code is working. For quick passes use `uv run pytest -m 'not slow'`, which takes around 10 seconds instead of 1 minute.
- **Run an example planning problem**: `uv run railroad example <name>` (e.g., `clear-table`, `multi-object-search`, `heterogeneous-robots`)

## Architecture

The repository is organized as a monorepo with several interdependent packages:

#### Core Planning (`packages/railroad/`)

- **C++ Core** (`include/`, `src/railroad/_bindings.cpp`):
  - A* search, MCTS planning
  - State management and action grounding
  - FF-like heuristic for forward planning
- **Python Layer** (`src/railroad/`):
  - `core.py`: Main classes (`Fluent`, `State`, `Action`, `Effect`, `ForallEffect`, `Operator`, `Goal`) - re-exports C++ types and adds Python utilities
  - `operators/`: Helper functions to construct common operators (move, search, pick, place, wait)
  - `planner.py`: `MCTSPlanner` wrapper with automatic negative precondition handling
  - `pddl_converter/`: downloads IPC PDDL/PPDDL problems and converts them into railroad problems; CLI via `railroad pddl list/run/check` (see its README.md for mapping semantics and supported features)
- **Testing**: Tests in `tests/` including unit tests and integration tests

#### Environment Module (`packages/railroad/src/railroad/environment/`)

Provides abstractions for planning execution:

- **`environment.py`**: `Environment` abstract base class
  - Active skill tracking and time management
  - State assembly (fluents + upcoming effects)
  - Action instantiation from operators
  - The `act()` loop that executes until a robot is free
  - `ActiveSkill` protocol for skill execution
  - Subclass hooks: `define_operators()`, `set_robot_pose()`, `_should_interrupt_skills()`, `_cap_next_advance_time()`, `_after_skills_advanced()`

- **`symbolic.py`**: `SymbolicEnvironment` and skill implementations
  - `SymbolicEnvironment`: Concrete, domain-agnostic environment for symbolic execution; probabilistic branches sample from a seedable RNG (`seed=`), no fluent or action name has special meaning
  - `SymbolicSkill`: Standard skill implementation (non-interruptible)
  - `LocationRegistry`: Coordinates robot locations during interruptible moves

- **`object_search.py`**: `ObjectSearchEnvironment(SymbolicEnvironment)` — railroad's object-search domain conventions: ground-truth resolution of search effects (`true_object_locations`), revelation (`searched` → `revealed`/`found`/`at`), robot intermediate `{robot}_loc` locations, and move/place/search action hygiene filters. Use this (not the base) for robot search domains

- **`skill/`**: Skill protocols and navigation skill implementations
  - `protocols.py`: `MotionSkill` protocol, `SupportsMovePathEnvironment` contract
  - `navigation.py`: `NavigationMoveSkill` (path-following with occupancy-grid pathing), `InterruptibleNavigationMoveSkill` (interruptible variant)

- **`types.py`**: Shared types (`Pose`, `PoseLike` protocol)

- **`procthor/`**: ProcTHOR simulator interface (optional dependency)
  - `ThorInterface`: Main underlying interface to AI2-THOR/ProcTHOR scenes
  - `ProcTHORScene`: User-facing data provider for ProcTHOR, wrapping ThorInterface
  - `ProcTHOREnvironment`: Subclass of `ObjectSearchEnvironment` + `OccupancyGridPathingMixin`; subclasses must override `define_operators()`
  - `SceneGraph`: Scene graph representation

- **`railsim/`**: OpenGL visual simulator with procedural maze/office worlds (optional dependency, `railroad[railsim]`)
  - `Simulator`: Renders perspective and panoramic RGB/depth images at any meter-space pose (moderngl)
  - `RailsimScene`: Data provider mirroring `ProcTHORScene` (`.grid`, `.locations`, `.object_locations`); `RailsimScene.maze(seed=...)` / `.office(seed=...)` constructors; navigation `grid` is the raw occupancy grid inflated by `inflation_radius_m` while rendering uses raw geometry
  - `VisualUnknownSpaceEnvironment`: `UnknownSpaceEnvironment` subclass that renders a panorama at every laser sensor step into `pano_records` (list of `PanoRecord`)
  - Coordinates: railroad environments work in grid cells; railsim renders in meters (`meters = cells * map_data.resolution`)
  - Example: `uv run railroad example visual-frontier-search --env maze|office`; rendering requires a working GL context (`RAILSIM_GL_BACKEND` to pin one)

#### Navigation Module (`packages/railroad/src/railroad/navigation/`)

Reusable grid navigation primitives, independent of any specific environment:

- `pathing.py`: Theta\* any-angle pathfinding, Dijkstra cost grids, path interpolation, grid inflation
- `occupancy_grid_mixin.py`: `OccupancyGridPathingMixin` — mixin providing `estimate_move_time()` and `compute_move_path()` from an occupancy grid
- `plotting.py`: Occupancy grid visualization helpers
- `constants.py`: Grid cell values (`COLLISION_VAL`, `FREE_VAL`, `UNOBSERVED_VAL`, `OBSTACLE_THRESHOLD`)

#### Benchmarking (`packages/railroad/src/railroad/bench/`)
- `registry.py`: Benchmark registration via `@benchmark` decorator
- `runner.py`: Parallel benchmark execution with MLflow tracking
- `dashboard/`: Interactive Plotly Dash visualization
- `benchmarks/`: Benchmark definitions (multi-object search, movie night, etc.)

#### Experimental (`packages/railroad/src/railroad/experimental/`)
- **`unknown_search/`**: Frontier-based unknown-space exploration and object search
  - `UnknownSpaceEnvironment`: occupancy-grid environment with laser sensing, frontier detection, and navigation skills
  - Specialized operators for navigable moves, site search, and frontier search

#### External Packages
- **`packages/environments/`**: Additional environment implementations (PyRoboSim)

### Key Concepts

#### Static Preconditions and Grounding

Constraints on predicates no operator effect touches (e.g. `F("connected ?from ?to")`) are *static*: `railroad.core.ground_operators` checks them against the initial facts while enumerating bindings (backtracking with early pruning — this is what keeps IPC-scale domains tractable), then compiles them away by default: verified static preconditions are stripped from grounded actions, and static conjuncts of conditional-branch conditions are evaluated per grounding. It also raises on an operator naming an unknown parameter type or an unbound precondition variable — both would otherwise fail silently. `Eq`/`Neq` in an operator's preconditions are PDDL's `=` — grounding-time constraints, never runtime preconditions. `Environment.get_actions()` grounds through this path with caching and never mutates the environment's state; environments that mutate predicates outside operator effects must declare them via `runtime_mutated_predicates()`, and state captured inside operator callables (policies, registries) requires `invalidate_grounding()` after mutation.

#### Relevance Projection (planner-side)

Shrinking *states* is the planner's job, not the environment's. `railroad.core.relevant_predicates(actions, goal, upcoming_effects)` returns every predicate that can influence search — read by an action precondition, a conditional-branch condition (in actions *or* in the state's queued effects), the goal, or the core's name-keyed machinery (`free`/`waiting`/`at`/`found`). `MCTSPlanner` projects everything else out of searched states and out of action effects (`project_state`/`project_action`), which is bisimulation-preserving and typically several times faster, since every state hash walks the fluent set. Disable with `project_irrelevant=False` for debugging. This subsumes static-fact elimination and also catches *dynamic* write-only predicates. It is sound only for a caller holding the whole problem, which is why `Environment` does not attempt it: it cannot see the goal or the conditions on effects a state happens to carry.

#### Creating a New Planning Problem
1. Define objects by type: `{"robot": {"r1"}, "location": {"kitchen", "bedroom"}, "object": {"Knife"}}`
2. Define operators — either by subclassing `Environment`/`SymbolicEnvironment` (or `ObjectSearchEnvironment` for robot search domains) and overriding `define_operators()` (preferred), or by passing `operators=` to the constructor (deprecated)
3. Create the environment with initial state and ground truth object locations
4. Define goal: `goal = F("found Knife")`
5. Run planner: `action_name = planner(env.state, goal, max_iterations=1000)`
6. Execute: `env.act(action)`

#### `railroad` Planning Flow
1. Define `Operator` with parameters, preconditions, and `Effect`s
2. Instantiate operators with objects to create grounded `Action`s
3. `MCTSPlanner` searches over actions to reach goal
4. Planner automatically converts negative preconditions to positive equivalents

#### State and Fluents
- `Fluent`: Symbolic predicate like `F("at robot1 kitchen")` or `F("free robot1")`
- Negation: Use `~F(...)` or `F("not ...")`
- `State`: Collection of fluents + time + upcoming effects
- Effects modify state at specific times (supports probabilistic outcomes)

#### Actions and Effects
- Actions have preconditions (what must be true) and effects (what changes)
- Effects can be deterministic or probabilistic with multiple outcomes
- Effects happen at specified times (e.g., move takes time based on distance)
- Effects can carry conditional branches (`Effect(cond_effects=[(conditions, sub_effects), ...])`): sub-effects applied only if the condition fluents hold when the effect fires, evaluated before the effect's own fluents apply (PDDL `when`; negated conditions use negation-as-absence)
- Branching sub-effects (conditional and probabilistic) apply *after* their parent effect's own fluents, even at the same timestamp — deletes-before-adds holds within one effect, not across an effect and its branches
- `ForallEffect` is the universally quantified form (PDDL `forall`+`when`): expands into one conditional branch per object of the quantified type at `Operator.instantiate()` time; empty conditions give a plain universal effect
- Operators may carry a scalar `extra_cost`, charged in the MCTS objective (time + accumulated cost) and folded into the FF heuristic's estimates

#### Goals

Goals specify planning objectives. Use Python operators for simple cases:

```python
from railroad.core import Fluent as F

goal = F("found Knife") & F("found Fork")  # AND
goal = F("at robot1 kitchen") | F("at robot1 bedroom")  # OR
goal = ~F("at Knife table")  # NOT (knife must not be on table)
```

## Testing Strategy

Tests are organized by component:
- `packages/railroad/tests/`: Core PDDL functionality, planners, environment
- `packages/railroad/tests/environment/procthor/`: ProcTHOR integration tests (skipped if deps not installed)
- `packages/environments/tests/`, `packages/gridmap/tests/`: Component-specific tests

Key test patterns:
- Use fixtures for common test setups
- Parametrize tests with `@pytest.mark.parametrize` for multiple scenarios
