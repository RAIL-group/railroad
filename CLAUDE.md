# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RAIL_mrppddl_dev is a research repository for Multi-Robot Probabilistic Planning using PDDL (Planning Domain Definition Language) with a focus on embodied AI tasks in simulated indoor environments. The system combines a C++-accelerated PDDL planning core with Python bindings, integrated with AI2-THOR/ProcTHOR simulators for realistic 3D environment simulation.

## Build System

This project uses `uv` as the package manager and build tool. The build system automatically handles C++ compilation when needed. `uv run` handles all building and rebuilding any time the code is changed.

### Key Commands

- **Run all tests**: `uv run pytest`
- **Run tests matching filter**: `uv run pytest -vk <filter>`
- **Type checking**: `uv run ty check`
- **Run benchmarks**: `uv run railroad benchmarks run` (use `--dry-run` to preview, `-k <filter>` to filter)
- **Launch benchmark dashboard**: `uv run railroad benchmarks dashboard`
- **Run examples**: `uv run railroad example run <name>` (e.g., `clear-table`, `multi-object-search`, `heterogeneous-robots`)

### Important Build Notes

- Build is automatic via `uv run`, which will automatically detect changes to code (including the C++ code) and rebuild as necessary. Do not run `uv sync`, as it is not needed.

## Architecture

### Multi-Package Structure

The repository is organized as a monorepo with several interdependent packages:

#### Core Planning (`railroad/`)
- **C++ Core**: High-performance planning algorithms implemented in C++ (headers in `include/`, bindings in `src/railroad/_bindings.cpp`)
  - A* search, MCTS planning
  - State management and action grounding
  - FF heuristic for forward planning
- **Python Layer**: Python wrapper and utilities (`src/railroad/`)
  - `core.py`: Effect, Operator, Action, State, Fluent classes
  - `planner.py`: MCTSPlanner wrapper with automatic negative precondition handling
  - `helper.py`: Helper functions to construct common operators (move, search, pick, place, wait)
- **Testing**: Comprehensive tests in `tests/` including unit tests and integration tests

#### Environment Module (`railroad/environment/`)
The environment module provides abstractions for planning execution:

- **`Environment`** (`environment.py`): Abstract base class for all environments
  - Active skill tracking and time management
  - State assembly (fluents + upcoming effects)
  - Action instantiation from operators
  - The `act()` loop that executes until a robot is free

- **`SymbolicEnvironment`** (`symbolic.py`): Concrete implementation for symbolic execution
  - Manages fluents and derives object locations from state
  - Creates `SymbolicSkill` instances for action execution
  - Handles probabilistic effect resolution
  - Supports skill overrides for custom behavior (e.g., `InterruptableMoveSymbolicSkill`)

- **`ActiveSkill`** (`skill.py`): Protocol for skill execution
  - `SymbolicSkill`: Standard skill implementation
  - `InterruptableMoveSymbolicSkill`: Moves that can be interrupted mid-execution
  - `LocationRegistry`: Coordinates robot locations during interruptible moves

- **`procthor/`**: ProcTHOR simulator interface (optional dependency)
  - `ThorInterface`: Main interface to AI2-THOR/ProcTHOR scenes
  - `ProcTHORScene`: Data provider wrapping ThorInterface for planning
  - `ProcTHOREnvironment`: Full environment implementation
  - `SceneGraph`: Scene graph representation
  - Install with: `pip install railroad[procthor]`

- **Legacy classes** in `railroad.experimental.environment`:
  - `EnvironmentInterface`, `AbstractEnvironment`, `BaseEnvironment`
  - Preserved for backward compatibility

#### External Packages
- **`environments/`**: Additional environment implementations (PyRoboSim, etc.)
- **`gridmap/`**: Occupancy grid mapping and planning
- **`common/`**: Shared utilities (Pose class, etc.)

#### Benchmarking (`railroad/bench/`)
- `registry.py`: Benchmark registration via `@benchmark` decorator
- `runner.py`: Parallel benchmark execution with MLflow tracking
- `dashboard/`: Interactive Plotly Dash visualization
- `benchmarks/`: Benchmark definitions (multi-object search, movie night, etc.)

### Key Concepts

#### PDDL Planning Flow
1. Define `Operator` with parameters, preconditions, and `Effect`s
2. Instantiate operators with objects to create grounded `Action`s
3. `MCTSPlanner` or A* searches over actions to reach goal fluents
4. Planner automatically converts negative preconditions to positive equivalents

#### State and Fluents
- `Fluent`: Symbolic predicate like "at robot1 kitchen" or "free robot1"
- Negation: Use `~Fluent(...)` or `Fluent("not ...")`
- `State`: Collection of fluents representing world state
- Effects modify state at specific times (supports probabilistic outcomes)

#### Actions and Effects
- Actions have preconditions (what must be true) and effects (what changes)
- Effects can be deterministic or probabilistic with multiple outcomes
- Effects happen at specified times (e.g., move takes time based on distance)
- Effects can produce resulting fluents (additions/removals from state)

#### Goals and the Goal API
Goals specify planning objectives using complex logical expressions:

```python
from functools import reduce
from operator import and_, or_
from railroad.core import Fluent as F

# AND goal: all conditions must be true
goal = reduce(and_, [F("at robot1 kitchen"), F("found Knife")])

# OR goal: at least one condition must be true
goal = reduce(or_, [F("at robot1 kitchen"), F("at robot1 bedroom")])

# Negated goal: condition must be FALSE
goal = ~F("at Book table")  # Book must NOT be at table

# "None" pattern: no objects at location
goal = reduce(and_, [~F(f"at {obj} table") for obj in objects])
```

Key points:
- Use `reduce(and_, [...])` for "all" conditions, `reduce(or_, [...])` for "any"
- Use `~F(...)` for negative conditions (must be FALSE)
- MCTSPlanner handles negative goal fluents via automatic mapping conversion
- Use `ff_heuristic_goal` for heuristic computation with Goal objects
- Use `goal.evaluate(state.fluents)` to check if goal is satisfied

#### Environment Execution
The `SymbolicEnvironment` provides a clean API for executing plans:

```python
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State

env = SymbolicEnvironment(
    state=State(0.0, initial_fluents, []),
    objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
    operators=[move_op, search_op],
)

# Get available actions
actions = env.get_actions()

# Execute an action (returns when a robot is free)
new_state = env.act(action)

# Check goal
if env.is_goal_reached(goal_fluents):
    print("Done!")
```

## Testing Strategy

Tests are organized by component:
- `railroad/tests/`: Core PDDL functionality, planners, state transitions, environment
- `railroad/tests/environment/procthor/`: ProcTHOR integration tests (skipped if deps not installed)
- `environments/tests/`, `gridmap/tests/`: Component-specific tests

Key test patterns:
- Use fixtures for common test setups
- Parametrize tests with `@pytest.mark.parametrize` for multiple scenarios
- Test both symbolic (PDDL) and grounded (instantiated) levels
- Integration tests in `test_wait.py` demonstrate multi-agent coordination

## Example Workflows

### Running Examples
```bash
# List available examples
uv run railroad example

# Run an example
uv run railroad example run clear-table
uv run railroad example run multi-object-search
uv run railroad example run heterogeneous-robots
uv run railroad example run heterogeneous-robots --interruptible-moves

# ProcTHOR example (requires procthor dependencies)
uv run railroad example run procthor-search
```

### Running a Single Test
```bash
uv run pytest -vk test_fluent_equality
```

### Creating a New PDDL Problem
The typical flow:
1. Define environment with locations and objects
2. Create operators using helpers from `railroad.operators`
3. Create a `SymbolicEnvironment` with initial state and operators
4. Define goal using the Goal API: `goal = reduce(and_, [F(...), F(...)])`
5. Run planner with `env.state` and goal: `planner(env.state, goal, ...)`
6. Execute actions via `env.act(action)`

## Common Gotchas

- **Negative Preconditions**: MCTSPlanner automatically converts them - no manual handling needed
- **Negative Goals**: MCTSPlanner also handles negative goal fluents automatically by extending the mapping
- **Goal API**: Use `reduce(and_, [...])` or `reduce(or_, [...])` from `functools` and `operator` modules
- **ProcTHOR Dependencies**: Install with `pip install railroad[procthor]`. Check availability with `from railroad.environment.procthor import is_available`
- **Resource Downloads**: ProcTHOR auto-downloads resources on first import. Large downloads may take time
- **Cache**: ProcTHOR caches scenes in `resources/procthor-10k/cache/` for faster loading
- **Skill Interruptibility**: `SymbolicSkill` is non-interruptible by default. Use `InterruptableMoveSymbolicSkill` with `skill_overrides` for interrupt behavior
