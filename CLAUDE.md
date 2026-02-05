# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Railroad is a research repository for Multi-Robot Probabilistic Planning using PDDL (Planning Domain Definition Language). The system combines a C++-accelerated planning core with Python bindings, optionally integrated with AI2-THOR/ProcTHOR simulators for realistic 3D environment simulation.

## Build System

This project uses `uv` as the package manager and build tool. The build system automatically handles C++ compilation when needed.

### Key Commands

- **Run all tests**: `uv run pytest`
- **Run tests matching filter**: `uv run pytest -vk <filter>`
- **Type checking**: `uv run ty check`
- **Run benchmarks**: `uv run railroad benchmarks run` (use `--dry-run` to preview, `-k <filter>` to filter)
- **Launch benchmark dashboard**: `uv run railroad benchmarks dashboard`
- **Run examples**: `uv run railroad example run <name>` (e.g., `clear-table`, `multi-object-search`, `heterogeneous-robots`)

### Important Build Notes

- Build is automatic via `uv run`, which detects changes to code (including C++) and rebuilds as necessary. Do not run `uv sync` unless explicitly needed.

## Architecture

### Package Structure

The repository is organized as a monorepo with several interdependent packages:

#### Core Planning (`packages/railroad/`)

- **C++ Core** (`include/`, `src/railroad/_bindings.cpp`):
  - A* search, MCTS planning
  - State management and action grounding
  - FF heuristic for forward planning

- **Python Layer** (`src/railroad/`):
  - `core.py`: Main classes (`Fluent`, `State`, `Action`, `Effect`, `Operator`, `Goal`) - re-exports C++ types and adds Python utilities
  - `operators/`: Helper functions to construct common operators (move, search, pick, place, wait)
  - `planner.py`: `MCTSPlanner` wrapper with automatic negative precondition handling
  - `helper.py`: Goal formatting utilities

- **Testing**: Tests in `tests/` including unit tests and integration tests

#### Environment Module (`packages/railroad/src/railroad/environment/`)

Provides abstractions for planning execution:

- **`environment.py`**: `Environment` abstract base class
  - Active skill tracking and time management
  - State assembly (fluents + upcoming effects)
  - Action instantiation from operators
  - The `act()` loop that executes until a robot is free

- **`symbolic.py`**: `SymbolicEnvironment` and skill implementations
  - `SymbolicEnvironment`: Concrete environment for symbolic execution
  - `SymbolicSkill`: Standard skill implementation (non-interruptible)
  - `InterruptableMoveSymbolicSkill`: Moves that can be interrupted mid-execution
  - `LocationRegistry`: Coordinates robot locations during interruptible moves

- **`skill.py`**: `ActiveSkill` protocol defining the skill interface

- **`procthor/`**: ProcTHOR simulator interface (optional dependency)
  - `ThorInterface`: Main interface to AI2-THOR/ProcTHOR scenes
  - `ProcTHORScene`: Data provider wrapping ThorInterface
  - `ProcTHOREnvironment`: Full environment implementation
  - `SceneGraph`: Scene graph representation
  - Install with: `pip install railroad[procthor]`

- **Legacy classes** in `railroad.experimental.environment`:
  - `EnvironmentInterface`, `AbstractEnvironment`, `BaseEnvironment`
  - Preserved for backward compatibility

#### External Packages
- **`packages/environments/`**: Additional environment implementations (PyRoboSim)
- **`packages/gridmap/`**: Occupancy grid mapping and planning
- **`packages/common/`**: Shared utilities (Pose class, etc.)

#### Benchmarking (`packages/railroad/src/railroad/bench/`)
- `registry.py`: Benchmark registration via `@benchmark` decorator
- `runner.py`: Parallel benchmark execution with MLflow tracking
- `dashboard/`: Interactive Plotly Dash visualization
- `benchmarks/`: Benchmark definitions (multi-object search, movie night, etc.)

### Key Concepts

#### PDDL Planning Flow
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

#### Goals

Goals specify planning objectives. Use Python operators for simple cases:

```python
from railroad.core import Fluent as F

# Simple goals using & and |
goal = F("found Knife") & F("found Fork")  # AND
goal = F("at robot1 kitchen") | F("at robot1 bedroom")  # OR
goal = ~F("at Knife table")  # NOT (knife must not be on table)
```

For multiple fluents, use `reduce`:

```python
from functools import reduce
from operator import and_, or_

# All objects must be found
goal = reduce(and_, [F(f"found {obj}") for obj in ["Knife", "Fork", "Spoon"]])

# "Clear the table" - no objects at table
goal = reduce(and_, [~F(f"at {obj} table") for obj in objects])
```

Key points:
- MCTSPlanner handles negative goal fluents automatically
- Use `goal.evaluate(state.fluents)` to check if goal is satisfied
- Use `ff_heuristic_goal` for heuristic computation with Goal objects

#### Environment Execution

The `SymbolicEnvironment` provides a clean API for executing plans:

```python
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State
from railroad.core import Fluent as F

env = SymbolicEnvironment(
    state=State(0.0, initial_fluents, []),
    objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}, "object": {"Knife"}},
    operators=[move_op, search_op],
    objects_at_locations={"kitchen": {"Knife"}},  # Ground truth for probabilistic resolution
)

# Get available actions
actions = env.get_actions()

# Execute an action (returns when a robot is free)
new_state = env.act(action)

# Check goal
if goal.evaluate(env.state.fluents):
    print("Done!")
```

## Testing Strategy

Tests are organized by component:
- `packages/railroad/tests/`: Core PDDL functionality, planners, environment
- `packages/railroad/tests/environment/procthor/`: ProcTHOR integration tests (skipped if deps not installed)
- `packages/environments/tests/`, `packages/gridmap/tests/`: Component-specific tests

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
1. Define objects by type: `{"robot": {"r1"}, "location": {"kitchen", "bedroom"}, "object": {"Knife"}}`
2. Create operators using `railroad.operators` helpers
3. Create a `SymbolicEnvironment` with initial state, operators, and ground truth object locations
4. Define goal: `goal = F("found Knife")`
5. Run planner: `action_name = planner(env.state, goal, max_iterations=1000)`
6. Execute: `env.act(action)`

## Common Gotchas

- **Negative Preconditions**: MCTSPlanner automatically converts them - no manual handling needed
- **Negative Goals**: MCTSPlanner handles negative goal fluents automatically by extending the mapping
- **Operators Module**: Use `from railroad import operators` (not `helper.py`)
- **ProcTHOR Dependencies**: Install with `pip install railroad[procthor]`. Check availability with `from railroad.environment.procthor import is_available`
- **Resource Downloads**: ProcTHOR auto-downloads resources on first import (~2GB)
- **Skill Interruptibility**: `SymbolicSkill` is non-interruptible by default. Use `InterruptableMoveSymbolicSkill` with `skill_overrides` for interrupt behavior
- **State vs Fluents**: `State` includes time and upcoming effects; `state.fluents` is just the current facts
