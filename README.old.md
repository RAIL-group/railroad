# Railroad: Multi-Robot Probabilistic PDDL Planning

A high-performance planning framework for multi-robot systems that combines symbolic PDDL-style planning with probabilistic reasoning and realistic 3D simulation environments.

## Quickstart

```bash
uv venv
uv pip install 'git+https://github.com/RAIL-group/railroad.git@main#subdirectory=packages/railroad'
uv run railroad example multi-object-search
```

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

# Clone and install
git clone https://github.com/RAIL-group/railroad.git
cd railroad
uv sync
```

### Optional Dependencies

```bash
# Install with ProcTHOR support (AI2-THOR simulator)
pip install railroad[procthor]

# Install with benchmarking tools
pip install railroad[bench]

# Install everything
pip install railroad[all]
```

## Quick Start

### 1. Run a Built-in Example

The fastest way to see Railroad in action:

```bash
# List available examples
uv run railroad example

# Run the clear-the-table example
uv run railroad example run clear-table
```

You'll see a live dashboard showing robots planning and executing tasks:

```
Planning: clear-table
Goal: Move all objects off the table

  0.0                                              42.0
r1,r2 |⠀⠀⠀⠀⠁⠀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀|
   r1  M·····Sp···M·····Pp
   r2  M·····Sp·········M·····Pp

Step 3 | Time: 42.0 | Heuristic: 0
```

Other examples to try:
```bash
uv run railroad example run multi-object-search
uv run railroad example run find-and-move-couch
uv run railroad example run heterogeneous-robots
uv run railroad example run heterogeneous-robots --interruptible-moves
```

### 2. Minimal Planning Example

Here's the simplest possible planning example:

```python
from railroad.core import Fluent as F
from railroad._bindings import State
from railroad import operators
from railroad.planner import MCTSPlanner

# Two robots need to find a knife
initial_state = State(
    time=0,
    fluents={F("at robot1 kitchen"), F("at robot2 bedroom"), F("free robot1"), F("free robot2")},
    upcoming_effects=[],
)
goal = F("found Knife")

# Create move and search operators
move_op = operators.construct_move_operator_blocking(move_time=5.0)
search_op = operators.construct_search_operator(object_find_prob=0.8, search_time=3.0)

# Instantiate actions
objects = {"robot": {"robot1", "robot2"}, "location": {"kitchen", "bedroom"}, "object": {"Knife"}}
actions = list(move_op.instantiate(objects)) + list(search_op.instantiate(objects))

# Plan!
planner = MCTSPlanner(actions)
print(planner(initial_state, goal, max_iterations=1000))
# Output: "search robot1 kitchen Knife" or "search robot2 bedroom Knife"
```

### 3. Planning with Execution Loop

Use `SymbolicEnvironment` to execute plans step-by-step:

```python
from railroad.core import Fluent as F, get_action_by_name
from railroad._bindings import State
from railroad import operators
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner

# Setup
initial_fluents = {F("at robot1 kitchen"), F("free robot1")}
objects = {"robot": {"robot1"}, "location": {"kitchen", "bedroom"}, "object": {"Knife"}}
goal = F("found Knife")

move_op = operators.construct_move_operator_blocking(move_time=5.0)
search_op = operators.construct_search_operator(object_find_prob=0.8, search_time=3.0)

# Create environment (knows where objects actually are)
env = SymbolicEnvironment(
    state=State(0.0, initial_fluents, []),
    objects_by_type=objects,
    operators=[move_op, search_op],
    objects_at_locations={"bedroom": {"Knife"}},  # Knife is in bedroom
)

# Create planner
planner = MCTSPlanner(list(move_op.instantiate(objects)) + list(search_op.instantiate(objects)))

# Plan and execute until goal reached
while not goal.evaluate(env.state.fluents):
    action_name = planner(env.state, goal, max_iterations=1000)
    action = get_action_by_name(env.get_actions(), action_name)
    print(f"t={env.time:.1f}: Executing {action_name}")
    env.act(action)

print(f"Goal reached at t={env.time:.1f}!")
```

Output:
```
t=0.0: Executing search robot1 kitchen Knife
t=3.0: Executing move robot1 kitchen bedroom
t=8.0: Executing search robot1 bedroom Knife
Goal reached at t=11.0!
```

### 4. Complex Goals

Use Python operators for complex goal expressions:

```python
from functools import reduce
from operator import and_, or_
from railroad.core import Fluent as F

# AND: all must be true
goal = F("found Knife") & F("found Fork")

# OR: at least one must be true
goal = F("at robot1 kitchen") | F("at robot1 bedroom")

# Combining multiple fluents
goal = reduce(and_, [F("found Knife"), F("found Fork"), F("found Spoon")])

# Negation: object must NOT be at location
goal = ~F("at Knife table")  # Knife must not be on table

# "Clear the table" pattern
objects = ["Knife", "Fork", "Spoon"]
goal = reduce(and_, [~F(f"at {obj} table") for obj in objects])
```

### 5. ProcTHOR 3D Simulation (Optional)

For realistic 3D environments, install ProcTHOR dependencies:

```bash
pip install railroad[procthor]
uv run railroad example run procthor-search
```

```python
from railroad.environment.procthor import ProcTHOREnvironment, is_available

if is_available():
    # Load a ProcTHOR scene
    env = ProcTHOREnvironment(scene_id="train_0")

    # Use like any other environment
    actions = env.get_actions()
    state = env.state
    # ... plan and execute
```

## Project Structure

```
railroad/
├── packages/
│   ├── railroad/          # Core planning engine
│   │   ├── src/railroad/
│   │   │   ├── environment/        # SymbolicEnvironment, skills
│   │   │   ├── environment/procthor/  # ProcTHOR integration (optional)
│   │   │   ├── bench/              # Benchmarking tools
│   │   │   └── examples/           # Built-in examples
│   │   └── include/                # C++ headers
│   ├── environments/      # Additional environments (PyRoboSim)
│   ├── gridmap/           # Occupancy grid utilities
│   └── common/            # Shared utilities
├── scripts/               # Standalone scripts
└── resources/             # Downloaded data (ProcTHOR scenes, etc.)
```

## Development

```bash
# Run tests
uv run pytest

# Run specific test
uv run pytest -vk test_name

# Type checking
uv run ty check

# Run benchmarks
uv run railroad benchmarks run --dry-run  # Preview
uv run railroad benchmarks run            # Run all
uv run railroad benchmarks dashboard      # Launch dashboard
```

### Rebuilding C++ Extensions

```bash
uv sync --reinstall-package railroad
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Fluent** | A fact about the world: `F("at robot1 kitchen")` |
| **State** | Set of fluents + time + pending effects |
| **Operator** | Action template: `move ?robot ?from ?to` |
| **Action** | Grounded operator: `move robot1 kitchen bedroom` |
| **Effect** | State change at a specific time (can be probabilistic) |
| **Goal** | Target condition(s) to achieve |

### Environment and Skills

The `SymbolicEnvironment` executes plans by managing skills:

```python
import numpy as np
from railroad.environment import (
    SymbolicEnvironment,
    InterruptibleNavigationMoveSkill,
    LocationRegistry,
)

# For interruptible moves (robot can be redirected mid-movement)
registry = LocationRegistry({
    "kitchen": np.array([0, 0]),
    "bedroom": np.array([10, 0]),
})

env = SymbolicEnvironment(
    state=initial_state,
    objects_by_type=objects,
    operators=[move_op],
    location_registry=registry,
    skill_overrides={"move": InterruptibleNavigationMoveSkill},
)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `_bindings` import error | `uv sync --reinstall-package railroad` |
| ProcTHOR not found | `pip install railroad[procthor]` |
| Slow first ProcTHOR import | Normal - downloads ~2GB of resources |
| Tests fail after git pull | Rebuild: `uv sync --reinstall-package railroad` |

## Resources

- [AI2-THOR Documentation](https://ai2thor.allenai.org/)
- [ProcTHOR Dataset](https://procthor.allenai.org/)
