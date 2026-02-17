# MRPPDDL C++ Core

This directory contains the C++ implementation of the Multi-Robot Probabilistic PDDL planning system.

## Files

- **core.hpp**: Core types (Fluent, Action, GroundedEffect, etc.)
- **state.hpp**: State representation and transition function
- **goal.hpp**: Goal representation (LiteralGoal, AndGoal, OrGoal, etc.)
- **ff_heuristic.hpp**: FF (Fast-Forward) heuristic implementation
- **planner.hpp**: MCTS planner implementation
- **constants.hpp**: Global constants
- **HEURISTIC_OVERVIEW.md**: Detailed walkthrough of heuristic pipeline and MCTS goal-shaping terms

## Heuristic Functions

### ff_heuristic (default)

The primary heuristic function for guiding MCTS search. Located in `ff_heuristic.hpp`.

**Signature:**
```cpp
double ff_heuristic(const State &input_state,
                    const GoalBase *goal,
                    const std::vector<Action> &all_actions,
                    FFMemory *ff_memory = nullptr);
```

**Python API:**
```python
from railroad._bindings import ff_heuristic

h = ff_heuristic(state, goal, all_actions, mode="default")
```

**Algorithm Overview:**

1. **Relaxed Transition (Fluents)**: Compute the union of all possible fluent outcomes from ongoing actions. This gives the set of fluents that *could* become true.

2. **Non-Relaxed Transition (Time)**: Use the actual time to first action completion. This provides a tighter lower bound than using relaxed time (which waits for all actions).

3. **Forward Phase**: Build a relaxed planning graph to find all reachable fluents and collect achievers for each fluent.

4. **Cost Computation (Delta-Based)**:
   - **Phase 1**: Compute D(f) for all fluents using fixed-point iteration. D(f) is the optimistic cost using deterministic achievers (or best probabilistic if none exists).
   - **Phase 2**: For fluents with probabilistic achievers, compute delta(f) = E_attempt - D_best, where E_attempt is the expected cost of trying achievers in efficiency order.

5. **Backward Phase**: Extract the relaxed plan via BFS from goals to initial fluents. Sum D(goal) + deltas for all fluents on the extraction path.

**Key Design Decisions:**

- **Deterministic-first**: If a fluent has both deterministic and probabilistic achievers, use the deterministic cost for D(f). This avoids inflating costs when a sure path exists.

- **Delta separation**: By computing deltas separately from D values, we avoid circular dependencies that would occur if we tried to compute expected costs recursively.

- **Efficiency ordering**: Probabilistic achievers are tried in order of efficiency (probability / exec_cost), which minimizes expected cost.

- **Non-relaxed time**: Using the time to first action completion (rather than all actions) gives a tighter lower bound, especially important for multi-robot scenarios where robots act in parallel.

### Future Heuristic Modes

The `mode` parameter is reserved for future heuristic variants. Currently only `"default"` is implemented. Potential future modes:

- Relaxed time variant (use relaxed transition for time)
- Additive heuristic (sum costs instead of max)
- Landmark-based heuristics

## Data Structures

### ProbabilisticAchiever

Represents an action that can achieve a fluent:
```cpp
struct ProbabilisticAchiever {
    const Action* action;
    double wait_cost;       // Time until preconditions satisfied (MAX of precondition costs)
    double exec_cost;       // Duration of action execution
    double probability;     // Probability of achieving target fluent (1.0 for deterministic)
};
```

### FFForwardResult

Result of the forward relaxed reachability phase:
```cpp
struct FFForwardResult {
    std::unordered_set<Fluent> known_fluents;      // All reachable fluents
    std::unordered_set<Fluent> initial_fluents;    // Fluents at t=0
    std::unordered_map<Fluent, const Action*> fact_to_action;  // First achiever
    std::unordered_map<Fluent, double> expected_cost;  // D(f) values
    std::unordered_map<Fluent, double> probabilistic_delta;  // delta(f) for prob fluents
    std::unordered_map<Fluent, std::vector<ProbabilisticAchiever>> achievers_by_fluent;
};
```

## Memoization

The heuristic uses `FFMemory` (a hash map from state hash to cost) to cache results. The key is the hash of the relaxed state with time=0, so states that differ only in time but have the same fluents share cached values.

## Usage in MCTS

The `MCTSPlanner` in `planner.hpp` uses `ff_heuristic` to evaluate leaf nodes during simulation. The heuristic value is multiplied by `heuristic_multiplier` (default 5.0) and added to the accumulated time cost to compute the reward signal for backpropagation.

---

## Heuristic Evolution: v1 (Bellman) vs v2 (Delta-Based)

This section documents the evolution of the probabilistic FF heuristic implementation, comparing the **v1 (Bellman)** approach with the **v2 (Delta-based)** approach introduced in commit `a5b50b9`.

### Problem Context

Both versions extend the classic FF heuristic to handle **probabilistic actions**—actions that may succeed with probability < 1.0. The challenge is computing the expected cost to achieve a fluent when multiple probabilistic achievers exist.

---

### Version 1: Bellman-Style Iteration

#### Core Idea

Compute `V(f) = min(D(f), E(f))` directly for each fluent via fixed-point iteration, where:
- **D(f)**: Minimum cost using deterministic achievers (p = 1.0)
- **E(f)**: Expected cost using probabilistic achievers, assuming optimal ordering

#### Data Structures

```cpp
struct ProbabilisticAchiever {
    double reach_cost;      // Cost to satisfy preconditions
    double action_cost;     // Duration of action itself
    double probability;

    double total_cost() const { return reach_cost + action_cost; }
    double efficiency() const { return probability / total_cost(); }
};
```

#### Algorithm

1. **Forward Phase**: Collect all achievers for each fluent
2. **Bellman Iteration**: For each fluent f:
   - Update `reach_cost` for each achiever from precondition V values
   - Compute D(f) from deterministic achievers
   - Compute E(f) by sorting probabilistic achievers by efficiency and computing expected cost
   - Set `V(f) = min(D(f), E(f))`
   - Repeat until convergence

#### E(f) Formula

```
E(f) = Σ_t [ ∏_{j<t}(1-p_j) × C_t ]
```

Where achievers are sorted by `efficiency = p / total_cost`.

#### Backward Phase

Simply sum `V(f)` for all goal fluents.

#### Limitations

1. **Efficiency metric couples wait and exec**: Using `p / total_cost` for ordering means the ranking depends on `reach_cost`, which in turn depends on V values of preconditions. This creates circular dependencies that can cause convergence issues.

2. **Time computation uses relaxed transition**: The time bound (`dtime`) comes from the relaxed transition, which may over-estimate how quickly the next decision point arrives.

---

### Version 2: Delta-Based Approach (Current)

#### Core Idea

Separate the cost into two components:
- **D(f)**: Optimistic cost (as if all achievers were deterministic)
- **δ(f)**: Delta capturing the additional expected cost due to probabilistic uncertainty

Total expected cost = D(goal) + Σ δ(f) for fluents on the extraction path.

#### Data Structures

```cpp
struct ProbabilisticAchiever {
    double wait_cost;       // Time until preconditions satisfied (MAX of prec costs)
    double exec_cost;       // Duration of action execution itself
    double probability;

    double attempt_cost() const { return wait_cost + exec_cost; }
    double efficiency() const { return probability / exec_cost; }  // exec only!
};

struct FFForwardResult {
    std::unordered_map<Fluent, double> expected_cost;       // D(f) values
    std::unordered_map<Fluent, double> probabilistic_delta; // δ(f) values
};
```

#### Key Insight: Decoupled Efficiency

The efficiency metric now uses **only `exec_cost`**:

```cpp
efficiency = probability / exec_cost
```

**Rationale**: When scheduling multiple probabilistic achievers for the same fluent, the `wait_cost` (time to satisfy preconditions) is paid regardless of which achiever we try first. Only the `exec_cost` is the incremental cost of each attempt. Therefore, the optimal ordering should maximize `p / exec_cost`.

#### Algorithm Structure

**Forward Phase: Compute D(f) (Optimistic Costs)**

Fixed-point iteration treating all achievers as deterministic:

```cpp
D(f) = min over achievers { wait_cost + exec_cost }
```

For fluents with only probabilistic achievers, use the best one (highest probability, then lowest cost).

**Backward Phase: Extraction with Lazy Delta Computation**

1. BFS from goal fluents back to initial fluents following `fact_to_action` links
2. Track all fluents on the extraction path
3. For each fluent on the path, **lazily compute δ(f)** on-demand:
   - Skip fluents with only deterministic achievers (δ = 0)
   - For probabilistic fluents, compute δ(f) = E_attempt - D_prob_best
   - Cache the result in `probabilistic_delta` for reuse across goal branches
4. Return: `Σ D(goal_fluent) + Σ δ(f)` for fluents on path

**Lazy Delta Computation (per fluent)**

For fluents with probabilistic achievers:

1. Sort achievers by `efficiency = p / exec_cost` (descending)
2. Compute `E_attempt` considering time advancement:
   ```
   E_attempt = Σ_t [ ∏_{j<t}(1-p_j) × (dtime_t + exec_t) ]
   ```
   where `dtime_t = max(0, wait_cost_t - current_time)`
3. Compute `D_prob_best = min(wait_cost + exec_cost)` over probabilistic achievers
4. Set `δ(f) = E_attempt - D_prob_best`

**Why Lazy?** Only fluents on the extraction path need their deltas computed. This avoids wasted computation for probabilistic fluents that aren't part of the relaxed plan. Caching ensures that if multiple OR goal branches share a prerequisite fluent, the delta is computed only once.

**Optimization**: The forward phase tracks which fluents have probabilistic achievers in `has_probabilistic_achiever`. The backward phase only calls delta computation for fluents in this set, completely skipping fluents with purely deterministic achievers (no iteration over achievers needed).

#### Time Computation: Non-Relaxed Transition

```cpp
// Relaxed transition for FLUENTS (union of all outcomes)
auto relaxed_result = transition(input_state, nullptr, true);

// Non-relaxed transition for TIME (first action completion)
auto nonrelaxed_result = transition(input_state, nullptr, false);
double dtime = nonrelaxed_result[0].first.time() - t0;
```

**Rationale**: The non-relaxed time gives a tighter lower bound since we can act again as soon as *any* robot finishes its current action, not all of them.

---

### Comparison Summary

| Aspect | v1 (Bellman) | v2 (Delta-based) |
|--------|--------------|------------------|
| **Cost representation** | Single V(f) value | D(f) + δ(f) decomposition |
| **Efficiency metric** | `p / (reach + exec)` | `p / exec` only |
| **Circular dependencies** | Possible (reach_cost affects ranking) | Avoided (δ computed after D converges) |
| **Delta computation** | Eager (all fluents) | Lazy (only fluents on extraction path) |
| **Time bound** | Relaxed transition | Non-relaxed transition |
| **Backward computation** | Sum of V(goal) | BFS extraction + lazy delta accumulation |

---

### Design Rationale

#### Why Decouple Wait and Exec?

Consider two achievers for fluent F:
- A1: wait=10, exec=1, p=0.5
- A2: wait=10, exec=5, p=0.9

Under v1 (`p / total_cost`):
- A1 efficiency: 0.5/11 ≈ 0.045
- A2 efficiency: 0.9/15 = 0.06
- Order: A2 first, then A1

Under v2 (`p / exec_cost`):
- A1 efficiency: 0.5/1 = 0.5
- A2 efficiency: 0.9/5 = 0.18
- Order: A1 first, then A2

The v2 ordering is superior: both achievers pay the same wait=10, so we should try the "cheap and fast" A1 first. If it fails, we still have A2 as backup.

#### Why Extract a Relaxed Plan?

Simply summing D(goal) might double-count shared preconditions. The BFS extraction identifies the actual fluents needed, ensuring deltas are counted exactly once per fluent on the path to the goal.

#### Why Non-Relaxed Time?

In multi-robot scenarios, the relaxed transition advances time to when *all* pending actions complete. But we can make a new decision as soon as *any* robot finishes. Using non-relaxed time provides a tighter (more admissible) lower bound.
