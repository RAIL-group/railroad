# Railroad: Multi-Robot Probabilistic PDDL Planning

A high-performance planning framework for multi-robot systems that combines symbolic PDDL-style planning with probabilistic reasoning and realistic 3D simulation environments.

## Features

- **Fast C++ Planning Core**: A* and MCTS planners with Python bindings
- **Probabilistic Effects**: Handle uncertain action outcomes in planning
- **Multi-Robot Coordination**: Plan and execute coordinated multi-robot tasks
- **Environment Abstraction**: Clean `SymbolicEnvironment` API for plan execution
- **ProcTHOR Integration**: Optional realistic 3D indoor environment simulation via AI2-THOR
- **Occupancy Grid Planning**: Built-in mapping and path planning utilities

## Installation

This project requires Python 3.13+ and uses `uv` for dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd RAIL_mrppddl_dev

# Install all dependencies and build C++ extensions
uv sync
```

### Optional Dependencies

The `railroad` package supports optional dependency groups:

```bash
# Install with ProcTHOR support (AI2-THOR simulator)
pip install railroad[procthor]

# Install with benchmarking tools
pip install railroad[bench]

# Install everything
pip install railroad[all]
```

## Quick Start

### Running Built-in Examples

```bash
# List available examples
uv run railroad example

# Run examples
uv run railroad example run clear-table
uv run railroad example run multi-object-search
uv run railroad example run find-and-move-couch
uv run railroad example run heterogeneous-robots
uv run railroad example run heterogeneous-robots --interruptible-moves

# ProcTHOR example (requires: pip install railroad[procthor])
uv run railroad example run procthor-search
```

### Basic PDDL Planning Example

```python
from functools import reduce
from operator import and_

from railroad.core import Fluent as F
from railroad._bindings import State
from railroad import operators
from railroad.planner import MCTSPlanner

# Define initial state
initial_state = State(
    time=0,
    fluents={
        F("at robot1 bedroom"),
        F("at robot2 kitchen"),
        F("free robot1"),
        F("free robot2"),
    },
    upcoming_effects=[],
)

# Define goal using the Goal API
goal = reduce(and_, [
    F("found Knife"),
    F("found Notebook"),
])

# Define available objects and locations
objects_by_type = {
    "robot": {"robot1", "robot2"},
    "location": {"bedroom", "kitchen", "living_room"},
    "object": {"Knife", "Notebook"},
}

# Create operators
move_op = operators.construct_move_operator_blocking(
    move_time=lambda r, from_loc, to_loc: 5.0
)
search_op = operators.construct_search_operator(
    object_find_prob=lambda r, loc, obj: 0.8,
    search_time=lambda r, loc, obj: 3.0,
)

# Instantiate actions from operators
actions = list(move_op.instantiate(objects_by_type)) + list(search_op.instantiate(objects_by_type))

# Run MCTS planner
planner = MCTSPlanner(actions)
next_action_name = planner(initial_state, goal, max_iterations=10000, c=10)

print(f"Next action to execute: {next_action_name}")
```

### Using the SymbolicEnvironment

The `SymbolicEnvironment` provides a clean API for executing plans:

```python
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State
from railroad.core import Fluent as F, get_action_by_name

# Create environment
env = SymbolicEnvironment(
    state=State(0.0, initial_fluents, []),
    objects_by_type=objects_by_type,
    operators=[move_op, search_op],
    objects_at_locations={"kitchen": {"Knife"}, "bedroom": {"Notebook"}},
)

# Planning and execution loop
while not goal.evaluate(env.state.fluents):
    # Get next action from planner
    action_name = planner(env.state, goal, max_iterations=1000)
    action = get_action_by_name(env.get_actions(), action_name)

    # Execute action (returns when a robot is free)
    env.act(action)

print(f"Goal reached at time {env.time}!")
```

### Using with ProcTHOR Simulation

ProcTHOR integration requires optional dependencies:

```bash
pip install railroad[procthor]
```

```python
from railroad.environment.procthor import ProcTHOREnvironment, is_available

if is_available():
    env = ProcTHOREnvironment(scene_id="train_0")
    # ... use environment for planning and execution
```

Run the built-in example:
```bash
uv run railroad example run procthor-search
```

## Project Structure

The repository is organized as a monorepo:

- **`packages/`** - All packages:
    - **`railroad/`** - Core PDDL planning engine (C++ with Python bindings)
      - `src/railroad/environment/` - Environment abstractions (`SymbolicEnvironment`, `ActiveSkill`)
      - `src/railroad/environment/procthor/` - ProcTHOR simulator interface (optional)
      - `src/railroad/bench/` - Benchmarking harness with MLflow tracking
      - `src/railroad/examples/` - Built-in runnable examples
    - **`environments/`** - Additional environment implementations (PyRoboSim)
    - **`gridmap/`** - Occupancy grid mapping and planning utilities
    - **`common/`** - Shared utilities (Pose class, etc.)
- **`scripts/`** - Example scripts and demonstrations
- **`resources/`** - Downloaded resources (ProcTHOR data, models, etc.)

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest -vk test_name

# Run tests matching pattern
uv run pytest -vk search
```

### Type Checking

```bash
uv run ty check
```

### Running Benchmarks

```bash
# Run all benchmarks
uv run railroad benchmarks run

# Dry run to see what will execute
uv run railroad benchmarks run --dry-run

# Run with filters and options
uv run railroad benchmarks run -k movie_night --repeat-max 3 --parallel 4

# Launch interactive dashboard
uv run railroad benchmarks dashboard
```

### Rebuilding C++ Extensions

If you modify C++ code or encounter import errors:

```bash
uv sync --reinstall-package railroad
```

## Key Concepts

### Operators and Actions

- **Operators** are parameterized action templates (e.g., "move from ?from to ?to")
- **Actions** are grounded instances with specific objects (e.g., "move robot1 from kitchen to bedroom")

### Fluents and State

- **Fluents** are symbolic predicates representing facts (e.g., "at robot1 kitchen")
- **State** is a set of fluents representing the current world state
- Use `~Fluent(...)` for negation

### Goals and the Goal API

Goals specify what conditions must be satisfied to complete planning. The system supports complex goal expressions using Python operators.

#### Basic Goal Types

```python
from functools import reduce
from operator import and_, or_
from railroad.core import Fluent as F

# Single literal goal
goal = F("at robot1 kitchen")

# AND goal: all conditions must be true
goal = reduce(and_, [F("at robot1 kitchen"), F("found Knife")])

# OR goal: at least one condition must be true
goal = reduce(or_, [F("at robot1 kitchen"), F("at robot1 bedroom")])

# Negated goal: condition must be FALSE
goal = ~F("at Book table")  # Book must NOT be at table
```

#### Complex Goal Patterns

```python
# "Clear the table" - remove all objects from a location
objects_on_table = ["Book", "Mug", "Vase"]
goal = reduce(and_, [~F(f"at {obj} table") for obj in objects_on_table])

# "Move any object to destination"
objects = ["Book", "Mug"]
goal = reduce(or_, [F(f"at {obj} shelf") for obj in objects])

# Nested goals with AND and OR
from railroad._bindings import AndGoal, OrGoal, LiteralGoal
goal = AndGoal([
    OrGoal([LiteralGoal(F("at Remote den")), LiteralGoal(F("at Plate den"))]),
    OrGoal([LiteralGoal(F("at Cookie den")), LiteralGoal(F("at Couch den"))]),
])
```

#### Goal Methods

```python
# Check if goal is satisfied
if goal.evaluate(current_state.fluents):
    print("Goal achieved!")

# Get all literal fluents in the goal
all_literals = goal.get_all_literals()

# Get goal type
from railroad._bindings import GoalType
goal_type = goal.get_type()  # GoalType.AND, GoalType.OR, GoalType.LITERAL, etc.
```

### Environment and Skills

The `SymbolicEnvironment` manages plan execution:

- **Environment** owns ground truth fluents; state is assembled on demand
- **Skills** encapsulate action execution with time-based effect scheduling
- **Probabilistic effects** are resolved at execution time
- **Interruptible skills** allow moves to be interrupted when another robot becomes free

```python
from railroad.environment import (
    SymbolicEnvironment,
    InterruptableMoveSymbolicSkill,
    LocationRegistry,
)

# For interruptible moves, use LocationRegistry and skill_overrides
registry = LocationRegistry({"kitchen": np.array([0, 0]), "bedroom": np.array([10, 0])})
env = SymbolicEnvironment(
    ...,
    location_registry=registry,
    skill_overrides={"move": InterruptableMoveSymbolicSkill},
)
```

### Effects and Timing

- Effects can happen at different times (e.g., movement takes 5 seconds)
- Effects can be probabilistic with multiple possible outcomes
- The planner reasons about time and probability to find optimal plans

### Planning Algorithms

- **MCTS (Monte Carlo Tree Search)**: Good for probabilistic domains, handles uncertainty
- **A***: Optimal for deterministic domains with good heuristics

## Resources

- **AI2-THOR Documentation**: https://ai2thor.allenai.org/
- **ProcTHOR Dataset**: https://procthor.allenai.org/

## Troubleshooting

### Import Error: "_bindings is missing"

The C++ extension needs to be rebuilt:
```bash
uv sync --reinstall-package railroad
```

### ProcTHOR Dependencies Not Found

Install optional dependencies:
```bash
pip install railroad[procthor]
```

Check if available in code:
```python
from railroad.environment.procthor import is_available
if is_available():
    # ProcTHOR is ready to use
```

### ProcTHOR Resource Downloads

On first import, ProcTHOR automatically downloads required resources (scenes, models). This may take several minutes. To disable auto-download:
```bash
export PROCTHOR_AUTO_DOWNLOAD=0
```

### Tests Failing After Git Pull

Rebuild C++ extensions if headers or bindings changed:
```bash
uv sync --reinstall-package railroad
```

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```
