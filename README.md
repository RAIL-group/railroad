# MRPPDDL: Multi-Robot Probabilistic PDDL Planning

A high-performance planning framework for multi-robot systems that combines symbolic PDDL-style planning with probabilistic reasoning and realistic 3D simulation environments.

## Features

- **Fast C++ Planning Core**: A* and MCTS planners with Python bindings
- **Probabilistic Effects**: Handle uncertain action outcomes in planning
- **Multi-Robot Coordination**: Plan and execute coordinated multi-robot tasks
- **ProcTHOR Integration**: Realistic 3D indoor environment simulation via AI2-THOR
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

## Quick Start

### Basic PDDL Planning Example

```python
from mrppddl.core import Fluent, State
from mrppddl.helper import construct_move_operator, construct_search_operator
from mrppddl.planner import MCTSPlanner

# Define initial state
initial_state = State(fluents={
    Fluent("at robot1 bedroom"),
    Fluent("at robot2 kitchen"),
    Fluent("free robot1"),
    Fluent("free robot2"),
})

# Define goal
goal_fluents = {
    Fluent("found Knife"),
    Fluent("found Notebook"),
}

# Define available objects and locations
objects_by_type = {
    "robot": ["robot1", "robot2"],
    "location": ["bedroom", "kitchen", "living_room"],
    "object": ["Knife", "Notebook"],
}

# Create operators and instantiate actions
move_op = construct_move_operator(move_time=lambda r, from_loc, to_loc: 5.0)
search_op = construct_search_operator(
    object_find_prob=lambda r, loc, obj: 0.8,  # 80% chance of finding object
    move_time=lambda r, from_loc, to_loc: 5.0,
    pick_time=3.0
)

actions = move_op.instantiate(objects_by_type) + search_op.instantiate(objects_by_type)

# Run MCTS planner
planner = MCTSPlanner(actions)
next_action_name = planner(initial_state, goal_fluents, max_iterations=10000, c=10)

print(f"Next action to execute: {next_action_name}")
```
This will yield:
> Next action to execute: search robot1 bedroom living_room Notebook


### Using with ProcTHOR Simulation

See `scripts/obf_door_example.py` for a complete example of planning and execution in a ProcTHOR environment with:
- Scene loading and caching
- Occupancy grid mapping
- Robot navigation
- Object search tasks

### Example: Multi-Robot Search Task

[TODO: Add the Multi-Robot Search Example.]

```python
from environments import Map, Location, SymbolicToRealSimulator, Robot
from mrppddl.helper import construct_search_operator
from mrppddl.core import Fluent, get_action_by_name
from mrppddl.planner import MCTSPlanner

# Initialize a random environment with 2 robots and 3 locations
map = Map(n_robots=2, max_locations=3, seed=1015)

# Create robots at starting positions
r1 = Robot(name='r1', start=map.robot_poses[0])
r2 = Robot(name='r2', start=map.robot_poses[1])
robots = [r1, r2]

# Define search goal
goal_fluents = {
    Fluent("found Knife"),
    Fluent("found Notebook"),
    Fluent("found Clock")
}

# Create simulator
simulator = SymbolicToRealSimulator(map, robots, goal_fluents)

# Plan and execute until goal is reached
while not simulator.is_goal():
    # Get available actions
    objects_by_type = {
        "robot": [r.name for r in simulator.robots],
        "location": [loc.name for loc in simulator.locations
                     if loc.name not in simulator.visited_locations],
        "object": simulator.object_of_interest,
    }

    search_actions = construct_search_operator(
        simulator.get_likelihood_of_object,
        simulator.get_move_cost,
        search_time=3
    ).instantiate(objects_by_type)

    # Plan next action
    mcts = MCTSPlanner(search_actions)
    action_name = mcts(simulator.state, goal_fluents, max_iterations=10000, c=10)
    action = get_action_by_name(search_actions, action_name)

    # Execute in simulator
    simulator.execute_action(action)
    print(f"Executed: {action.name}")
    print(f"Current state: {simulator.state}")
```

## Project Structure

The repository is organized as a monorepo with multiple packages:

- **`mrppddl/`** - Core PDDL planning engine (C++ with Python bindings)
- **`procthor/`** - ProcTHOR simulator interface and utilities
- **`environments/`** - Environment abstractions and implementations
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
uv run pyright -w mrppddl/src/mrppddl mrppddl/tests
```

### Rebuilding C++ Extensions

If you modify C++ code or encounter import errors:

```bash
make rebuild-cpp
# or equivalently:
uv sync --reinstall-package mrppddl
```

## Key Concepts

### Operators and Actions

- **Operators** are parameterized action templates (e.g., "move from ?from to ?to")
- **Actions** are grounded instances with specific objects (e.g., "move robot1 from kitchen to bedroom")

### Fluents and State

- **Fluents** are symbolic predicates representing facts (e.g., "at robot1 kitchen")
- **State** is a set of fluents representing the current world state
- Use `~Fluent(...)` for negation

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
uv sync --reinstall-package mrppddl
```

### ProcTHOR Resource Downloads

On first import, ProcTHOR automatically downloads required resources (scenes, models). This may take several minutes. To disable auto-download:
```bash
export PROCTHOR_AUTO_DOWNLOAD=0
```

### Tests Failing After Git Pull

Rebuild C++ extensions if headers or bindings changed:
```bash
make rebuild-cpp
```

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```
