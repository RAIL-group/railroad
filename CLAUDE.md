# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RAIL_mrppddl_dev is a research repository for Multi-Robot Probabilistic Planning using PDDL (Planning Domain Definition Language) with a focus on embodied AI tasks in simulated indoor environments. The system combines a C++-accelerated PDDL planning core with Python bindings, integrated with AI2-THOR/ProcTHOR simulators for realistic 3D environment simulation.

## Build System

This project uses `uv` as the package manager and build tool. The build system automatically handles C++ compilation when needed. `uv run` handles all building and rebuliding any time the code is changed.

### Key Commands

- **Run all tests**: `uv run pytest`
- **Run tests matching filter**: `uv run pytest -vk <filter>`
- **Type checking**: `uv run ty check`
- **Run benchmarks**: `uv run benchmarks-run` (use `--dry-run` to preview, `-k <filter>` to filter)
- **Launch benchmark dashboard**: `uv run benchmarks-dashboard`

### Important Build Notes

- Build is automatic via `uv run`, which will automatically detect changes to code (including the C++ code) and rebuild as necessary. Do not run `uv sync`, as it is not needed.

## Architecture

### Multi-Package Structure

The repository is organized as a monorepo with several interdependent packages:

#### Core Planning (`mrppddl/`)
- **C++ Core**: High-performance planning algorithms implemented in C++ (headers in `include/`, bindings in `src/mrppddl/_bindings.cpp`)
  - A* search, MCTS planning
  - State management and action grounding
  - FF heuristic for forward planning
- **Python Layer**: Python wrapper and utilities (`src/mrppddl/`)
  - `core.py`: Effect, Operator, Action, State, Fluent classes
  - `planner.py`: MCTSPlanner wrapper with automatic negative precondition handling
  - `helper.py`: Helper functions to construct common operators (move, search, pick, place, wait)
- **Testing**: Comprehensive tests in `tests/` including unit tests and integration tests

#### Environment Support
- **`procthor/`**: ProcTHOR simulator interface
  - `ThorInterface`: Main interface to AI2-THOR/ProcTHOR scenes
  - Scene graph construction, occupancy grids, caching
  - Automatic resource downloading on import (disable with `PROCTHOR_AUTO_DOWNLOAD=0`)

- **`environments/`**: Environment abstractions and execution
  - `BaseEnvironment`: Abstract interface for all environments
  - `Simulator` (in `environments/simulator/`): Execution wrapper that applies PDDL actions and deterministically reveals search outcomes
  - `OngoingAction` classes: Track action execution progress (move, search, pick, place)
  - Mapping between symbolic locations and simulator coordinates
  - Provides cost functions and perception interfaces

#### Utilities
- **`gridmap/`**: Occupancy grid mapping and planning
  - `laser.py`: Laser scan representation and ray casting
  - `mapping.py`: Occupancy grid construction from laser scans
  - `planning.py`: Dijkstra path planning with optional sparsification
  - `utils.py`: Obstacle inflation utilities

- **`common/`**: Shared utilities
  - `Pose` class for 2D robot poses with transforms
  - Path length computation utilities

#### Benchmarking (`src/bench/`)
- **`bench/`**: Benchmark harness for planning system evaluation
  - `registry.py`: Benchmark registration via `@benchmark` decorator
  - `runner.py`: Parallel benchmark execution with MLflow tracking
  - `dashboard/`: Interactive Plotly Dash visualization
  - `cli.py`: Command-line interface for running benchmarks
- **`benchmarks/`**: Benchmark definitions (multi-object search, movie night, etc.)

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
from mrppddl.core import Fluent as F

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

## Testing Strategy

Tests are organized by component:
- `mrppddl/tests/`: Core PDDL functionality, planners, state transitions
- `environments/tests/`, `procthor/tests/`, `gridmap/tests/`: Component-specific tests

Key test patterns:
- Use fixtures for common test setups
- Parametrize tests with `@pytest.mark.parametrize` for multiple scenarios
- Test both symbolic (PDDL) and grounded (instantiated) levels
- Integration tests in `test_mrppddl_wait.py` demonstrate multi-agent coordination

## Example Workflows

### Running a Single Test
```bash
uv run pytest -vk test_fluent_equality
```

### Creating a New PDDL Problem
The typical flow:
1. Define environment with locations and objects
2. Create operators using helpers from `mrppddl.helper`
3. Instantiate actions from operators with `operator.instantiate(objects_by_type)`
4. Define goal using the Goal API: `goal = reduce(and_, [F(...), F(...)])`
5. Run planner with initial state and goal: `planner(state, goal, ...)`
6. Execute actions in simulator (see `environments.Simulator` for execution wrapper)

## Common Gotchas

- **Negative Preconditions**: MCTSPlanner automatically converts them - no manual handling needed
- **Negative Goals**: MCTSPlanner also handles negative goal fluents automatically by extending the mapping
- **Goal API**: Use `reduce(and_, [...])` or `reduce(or_, [...])` from `functools` and `operator` modules
- **Resource Downloads**: ProcTHOR auto-downloads resources on first import. Large downloads may take time
- **Cache**: ProcTHOR caches scenes in `resources/procthor-10k/cache/` for faster loading
