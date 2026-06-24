# LLM Heuristic Integration Plan

## Goal
Enhance the existing C++ planning core (MCTS + FF heuristic) with an LLM-based heuristic or action reranker. By exposing the fast-to-compute "relaxed planning graph with costs" from the C++ FF implementation to Python, we can provide compact and structured debug output to an LLM to correct misleading heuristic estimates.

## Previous Phases (Completed ✅)
- **Phase 1**: Expose RPG + Cost Info to Python (Bound `get_relaxed_expected_costs` via Pybind11).
- **Phase 2**: Python Heuristic Context Builder.
- **Phase 3**: LLM Action Reranker (Root-only).
- **Phase 4**: Benchmarking and Integration.

## The New LLM Heuristic Generation Workflow
If we stick to Python generation for ease of integration, here is the step-by-step workflow to automate this process.

### Step 1: Context Assembly (The "Pre-Plan" Script) (Completed ✅)
Before hitting the LLM, the agent needs to run a setup script to gather the context:
- Extract the PDDL-like domain definitions (Operators, Effects, Fluents) from `core.py`.
- Extract the problem definition (initial state, goal).
- Run the planner for exactly 1 iteration on the initial state to trigger the bindings created in Phase 1 (`get_relaxed_expected_costs`).
- Serialize this output into a tight JSON or YAML block. This serves as the "Example Base FF Blindspot" data.

### Step 2: The LLM Prompt Strategy
Construct a heavy system prompt for the LLM. Instruct it to act as an expert AI planning researcher writing a heuristic function.
- **Inputs**: Domain rules, Task definition, and the RPG Example from Step 1.
- **Task**: Write a Python class inheriting from a base `Heuristic` class (you'll need to define a simple interface in `core.py`).
- **Guidelines**: The LLM must not hardcode initial state values. It must write a `__call__(self, state, goal, rpg)` method that parses the active fluents, examines the cheapest relaxed plan (`rpg`), and returns a float cost. Provide the exact Python API for parsing `state.fluents` and details on the `rpg` dictionary format (matching `context.json` cheapest relaxed plan structure).
- **Generation**: Ask the LLM to generate $n$ variations of the heuristic code (e.g., $n=5$ using a high temperature, to ensure diversity).

### Step 3: Sandbox Evaluation (The Tournament)
We do not want to blindly trust the first piece of code the LLM spits out.
- Have the agent save the 5 generated heuristic scripts into a temporary directory.
- Run a short benchmark loop: plug each heuristic into a scaled-down MCTS run (e.g., max 1000 iterations or a 5-second timeout).
- **Filter 1**: Automatically discard any heuristic that throws a syntax error or a Pybind11 type error.
- **Filter 2**: Select the surviving heuristic that reaches the goal in the fewest iterations or expands the fewest nodes.

### Step 4: Integration and Caching
Once the "winning" heuristic is selected:
- The agent writes it permanently into a new module (e.g., `packages/railroad/src/railroad/heuristics/llm_generated.py`).
- **MCTS Custom Heuristic Binding Support**:
  - Update `MCTSPlanner` in C++ to accept an optional `HeuristicFn` (i.e. `std::function<double(const State&)>` via pybind11) in both the constructor and `operator()` call. If provided, it overrides the default `ff_heuristic` or `det_ff_heuristic`.
  - In Python's `MCTSPlanner`, if a `custom_heuristic` is provided, wrap it inside a Python wrapper function:
    ```python
    def wrap_custom_heuristic(cpp_state: State) -> float:
        # Build RPG dictionary matching context.json using HeuristicContextBuilder
        rpg = context_builder.build_context(cpp_state, goal, actions)
        # Call the LLM-generated heuristic
        return custom_heuristic(cpp_state, goal, rpg)
    ```
    Pass this wrapper directly to the C++ planner's constructor/call operator.
  - This gives you a fully automated pipeline where the LLM reasons about the domain deeply once, writes the math/logic (with full access to state, goal, and the C++-extracted relaxed plan), and gets integrated directly into the C++ planning loop.
