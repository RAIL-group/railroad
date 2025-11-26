# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RAIL_mrppddl_dev is a research repository for Multi-Robot Probabilistic Planning using PDDL (Planning Domain Definition Language) with a focus on embodied AI tasks in simulated indoor environments. The system combines a C++-accelerated PDDL planning core with Python bindings, integrated with AI2-THOR/ProcTHOR simulators for realistic 3D environment simulation.

## Build System

This project uses `uv` as the package manager and build tool. The build system automatically handles C++ compilation when needed.

### Key Commands

- **Setup**: `uv sync` - Install all dependencies and build C++ extensions
- **Run tests**: `uv run pytest -vk <filter>` - Run tests matching filter pattern
- **Run specific test**: `uv run pytest -vk test_name` - Run a single test
- **Type checking**: `uv run pyright -w mrppddl/src/mrppddl mrppddl/tests`
- **Rebuild C++**: `make rebuild-cpp` - Force rebuild of C++ modules (rarely needed)
- **Clean**: `make clean-cpp` - Remove C++ build artifacts only

### Important Build Notes

- Build is now automatic via `uv sync` or `uv run ...`
- C++ modules only need manual rebuild after `make clean-cpp` or changes to `.cpp`/`.hpp` files
- If C++ bindings are missing, the system will prompt with: "To rebuild the package, run: `uv sync --reinstall-package mrppddl`"

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
  - `helper.py`: Helper functions to construct common operators (move, search, pick, place)
- **Testing**: Comprehensive tests in `tests/` including unit tests and integration tests

#### Environment Support
- **`procthor/`**: ProcTHOR simulator interface
  - `ThorInterface`: Main interface to AI2-THOR/ProcTHOR scenes
  - Scene graph construction, occupancy grids, caching
  - Automatic resource downloading on import (disable with `PROCTHOR_AUTO_DOWNLOAD=0`)

- **`environments/`**: Environment abstractions
  - `BaseEnvironment`: Abstract interface for all environments
  - Mapping between symbolic locations and simulator coordinates
  - Provides cost functions and perception interfaces

- **`mrppddl_env/`**: MRPPDDL-specific environment implementations
  - `Robot`, `Simulator`, `Environment` classes
  - Bridges symbolic PDDL planning with real/simulated environments

#### Utilities
- **`gridmap/`**: Occupancy grid mapping and planning
  - `laser.py`: Laser scan representation and ray casting
  - `mapping.py`: Occupancy grid construction from laser scans
  - `planning.py`: Dijkstra path planning with optional sparsification
  - `utils.py`: Obstacle inflation utilities

- **`common/`**: Shared utilities
  - `Pose` class for 2D robot poses with transforms
  - Path length computation utilities

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

### Type Checking Before Commit
```bash
uv run pyright -w mrppddl/src/mrppddl mrppddl/tests
```

### After Modifying C++ Code
```bash
make rebuild-cpp  # Or just: uv sync --reinstall-package mrppddl
uv run pytest -vk test_mrppddl  # Verify changes work
```

### Creating a New PDDL Problem
See `environment_wrapper.py` for a complete example. The typical flow:
1. Define environment with locations and objects
2. Create operators using helpers from `mrppddl.helper`
3. Instantiate actions from operators
4. Run planner with initial state and goal fluents
5. Execute actions in simulator

## Common Gotchas

- **Import Errors**: If you see "_bindings is missing", run `uv sync --reinstall-package mrppddl`
- **Negative Preconditions**: MCTSPlanner automatically converts them - no manual handling needed
- **Resource Downloads**: ProcTHOR auto-downloads resources on first import. Large downloads may take time
- **Test Filters**: Use `PYTEST_FILTER=pattern make test` or `uv run pytest -vk pattern`
- **Cache**: ProcTHOR caches scenes in `resources/procthor-10k/cache/` for faster loading
