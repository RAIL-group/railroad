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


## TODO Items

**FF Heuristic & Action Filtering**
- [X] *Combine `get_usable_actions` with `ff_heuristic`* There is considerable duplicated code here and they really have the same functionality under the hood. Can I try passing a pointer for `visited_actions` to `ff_heuristic` and use that to return values if desired? Then the `get_usable_actions` function could just be a wrapper around it. Also, I can have `ff_heuristic` only optionally take in a goal function, since it isn't needed for determining which actions are reachable from some initial state.
- [ ] **Feature** *Handle 'fluent outcome likelihood' correctly within the `ff_heuristic`.* Currently, the maximum probability of a fluent within `ff_heuristic.hpp` keeps only the most likley *per successor/outcome*. Instead, they should be aggregated on a per-action basis, to keep around the total probability of making that fluent true. (If one outcome is 60% likely and another is 40% likely but both have a fluent 'F' in their outcomes, it should register as 100% for that fluent.)
- [ ] **Feature** *Keep only K-most-likely actions.* In previous work dealing with probabilistic outcomes, I would keep around only the K actions most likely to make that fluent true. To implement a similar functionality we can use the following steps: (1) determine which fluents are probabilistic and (2) for each fluent, determine which actions may result in that fluent as an outcome and then sort those and select the top-K of those, and (3) aggregate all of those actions in addition to all the deterministic actions. I will also write some tests for this purpose, to confirm that only probabilistic actions are limited.
- [ ] **Feature** *Select the K-most-likely actions and the K-least-expensive-ways-to-get-a-fluent*. Let's say I have a fluent 'F'. I think we can use the heuristic calculation on a goal function that includes all the positive conditions for any action that has some probability of resulting in 'F' being true. We could use the result as an estimate of how much time it would take to complete that action. Then we could use both (1) cost-to-action + action-cost (a measure of how long it will then take to accomplish a particular fluent) and (2) the probability of making a particular fluent true. 
- [ ] **Bug** *Fix issue with 'wait' actions and action pruning.* When I enable action pruning via 'get_usable_actions' in `mcts` in planner.hpp, the wait tests fail for some reason. I suspect something about the relaxed graph either means they are pruned and/or that the heuristic calculation is not quite right. *EDIT: this seems somewhat unreliable anyway. I will need to look into it. I think the heuristic is no longer quite right.*

