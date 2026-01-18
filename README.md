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
from functools import reduce
from operator import and_

from mrppddl.core import Fluent as F, State
from mrppddl.helper import construct_move_operator, construct_search_operator
from mrppddl.planner import MCTSPlanner

# Define initial state
initial_state = State(fluents={
    F("at robot1 bedroom"),
    F("at robot2 kitchen"),
    F("free robot1"),
    F("free robot2"),
})

# Define goal using the Goal API
# reduce(and_, [...]) creates an AndGoal that requires ALL fluents to be true
goal = reduce(and_, [
    F("found Knife"),
    F("found Notebook"),
])

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
next_action_name = planner(initial_state, goal, max_iterations=10000, c=10)

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
- **`src/bench/`** - Benchmarking harness with MLflow tracking and dashboard
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
# Type check src/ packages (more coverage coming later)
uv run ty check src
```

### Running Benchmarks

```bash
# Run all benchmarks
uv run benchmarks-run

# Dry run to see what will execute
uv run benchmarks-run --dry-run

# Run with filters and options
uv run benchmarks-run -k movie_night --repeat-max 3 --parallel 4

# Launch interactive dashboard
uv run benchmarks-dashboard
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

### Goals and the Goal API

Goals specify what conditions must be satisfied to complete planning. The system supports complex goal expressions using Python operators.

#### Basic Goal Types

```python
from functools import reduce
from operator import and_, or_
from mrppddl.core import Fluent as F

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
from mrppddl._bindings import AndGoal, OrGoal, LiteralGoal
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
from mrppddl._bindings import GoalType
goal_type = goal.get_type()  # GoalType.AND, GoalType.OR, GoalType.LITERAL, etc.
```

#### Using Goals with Planners

```python
from mrppddl.planner import MCTSPlanner
from mrppddl._bindings import ff_heuristic_goal

# Plan with goal object
planner = MCTSPlanner(actions)
action_name = planner(state, goal, max_iterations=1000, c=10)

# Compute heuristic for goal
h_value = ff_heuristic_goal(state, goal, actions)
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

**Wait Actions**
- [ ] **Bug** There is an issue with how the Simulator handles 'wait' actions that I'm unsure how to fix. The way that 'wait' actions are processed within the transition function seems to cause issues with how the Simulator determines how much time has passed. Wait actions don't actually advance time, which is their entire intended purpose. Right now, 'wait' actions are incompatible with the Simulator. It is not entirely clear to me if this is actually a bug with the `transition` function, so we may need more tests to investigate.


**Complex Goals**
- [ ] The live visualization doesn't handle the new goal function and still expects the 'goal_fluents' to be passed. We should fix that.
- [X] I also want to be able to handle 'any' and 'all' and 'none' conditions in the goals. To do some of these, I may need to pass the set of available objects, but I think that's okay. *Implemented: Use `reduce(and_, [...])` for all, `reduce(or_, [...])` for any, `reduce(and_, [~F(...) for ...])` for none.*
- [X] The trick with some of the negative conditions will be to handle the 'mapping' correctly. I need to be sure that's handled in the new implementation and add some test for it. At the moment, I don't think those are handled correctly, but are important for getting a good value out of the heuristic function. *Fixed: MCTSPlanner dynamically extends the mapping when goals with negative fluents are encountered.*
- [X] It would also be nice to be able to use python built-ins to specify goal functions: 'and' and 'or' should be able to do most of the basics. It would be a bit 'abuse of notation' to overload the 'and' operator for two fluents, but I think that might be okay for usability. *Implemented: Use `&` and `|` operators on Fluent, or `reduce(and_, [...])` and `reduce(or_, [...])`.*
