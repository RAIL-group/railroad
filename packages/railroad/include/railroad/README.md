# MRPPDDL C++ Core

This directory contains the C++ implementation of the Multi-Robot
Probabilistic PDDL planning system.

## Files

- **core.hpp**: Core types (Fluent, Action, GroundedEffect) plus the effect
  branches — `ProbBranchWrapper` for probabilistic outcomes,
  `CondBranchWrapper` for PDDL-`when`-style conditions
- **state.hpp**: State representation and the transition function, which
  resolves branches at fire time. Three ordering rules govern it: conditions
  see the state *before* the effect's own fluents apply, deletes apply before
  adds, and branch sub-effects apply *after* their parent effect
- **goal.hpp**: Goal representation (LiteralGoal, AndGoal, OrGoal, etc.)
- **heuristic_types.hpp**: shared FF data types — `Achiever`,
  `FFForwardResult`, `FFCacheKey`/`FFMemory`, and the `HeuristicFn` alias.
  Root of the heuristic header DAG.
- **heuristic_forward.hpp**: forward relaxed reachability
  (`ff_forward_phase`) and the optimistic-cost fixed point
  (`compute_optimistic_costs`).
- **heuristic_prob.hpp**: the per-fluent probabilistic retry delta
  (`get_or_compute_delta`) and its relaxed-plan sum
  (`relaxed_plan_prob_delta`).
- **heuristic_backward.hpp**: the `augment_at_with_found` ("at implies
  found") goal augmentation and the backward relaxed-plan extraction
  (`ff_backward_optimistic`).
- **heuristic.hpp**: umbrella header — includes the four above plus
  `goal.hpp`, and provides the public introspection helpers and the
  top-level `ff_heuristic` orchestrator (see "Header split" below).
- **heuristic_concurrent.hpp**: the concurrency-aware heuristic
  (`ConcurrentHeuristic`) -- a self-contained alternative to `ff_heuristic`
  for multi-robot problems (see "concurrent_heuristic" below).
- **planner.hpp**: MCTS planner implementation
- **constants.hpp**: Global constants

## Heuristic Functions

### ff_heuristic (default)

The primary heuristic for guiding MCTS search. Located in `heuristic.hpp`.

**Signature:**
```cpp
double ff_heuristic(const State &input_state,
                    const GoalBase *goal,
                    const std::vector<Action> &all_actions,
                    FFMemory *ff_memory   = nullptr,
                    double lambda_add     = 0.5,
                    double lambda_max     = 0.0,
                    double lambda_ff      = 0.5,
                    bool at_implies_found = true);
```

**Python API:**
```python
from railroad.core import ff_heuristic

h = ff_heuristic(state, goal, all_actions,
                 lambda_add=0.5, lambda_max=0.0, lambda_ff=0.5,
                 at_implies_found=True)
```

**Algorithm Overview:**

1. **Relaxed transition (fluents)**: union of all possible fluent outcomes
   from ongoing actions — the set of fluents that *could* become true.

2. **Non-relaxed transition (time)**: `dtime` = time until the *first*
   robot completes its current action. Tighter than relaxed time (which
   waits for *all* pending actions) and important for multi-robot parallel
   execution.

3. **Forward phase** (`ff_forward_phase`): build the relaxed planning graph
   — every reachable fluent and every achiever per fluent (with per-fluent
   aggregate success probability across an action's mutually-exclusive
   branches).

4. **Optimistic cost** (`compute_optimistic_costs`): fixed-point iteration
   filling `optimistic_cost[f]` — the cheapest single-attempt cost,
   preferring deterministic achievers, otherwise the best probabilistic one
   (charging a single attempt).

5. **Per goal branch** (DNF branches from `extract_or_branches`),
   `ff_backward_optimistic` walks back from the goal via
   `best_optimistic_achiever` (the achiever the optimistic-cost fixed point
   selected, so `h_ff` follows the same relaxed plan as `h_add`/`h_max`) and
   produces three component estimates plus the set of fluents on the relaxed
   plan:
   - `h_add`: Σ `optimistic_cost` over goal fluents (classic additive)
   - `h_max`: max `optimistic_cost` over goal fluents
   - `h_ff`:  Σ `action_duration` over unique actions on the relaxed plan
   - `on_path`: every fluent visited while walking back

6. **Mix + probabilistic delta**: per branch,
   `mixed = lambda_add·h_add + lambda_max·h_max + lambda_ff·h_ff`, then add
   `relaxed_plan_prob_delta(forward, on_path)` (the expected retry/fallback
   overhead for probabilistic fluents on the plan). Take the **minimum**
   over branches; the final value is `dtime + min_cost`.

The `lambda_*` weights are free-form (not normalized); defaults are an even
split between `h_add` and `h_ff` (`0.5, 0.0, 0.5`).

### concurrent_heuristic

`MCTSPlanner(..., heuristic="concurrent")`. Estimates the remaining time of
the *team* rather than of one sequential agent; the design notes at the top of
`heuristic_concurrent.hpp` explain each step. In brief:

1. **Timed relaxed state.** In-flight effects become available when they are
   scheduled; in-flight probabilistic outcomes are *pending achievers* with
   their branch probability (the FF heuristic applies all upcoming effects at
   time 0 and keeps one branch of each probabilistic effect, picked by hash
   order).
2. **Probability-aware costs.** Achievers ranked by `cost / rho^k`, `rho` the
   probability that the relaxed support succeeds (`prob_exponent`, default
   8); one Dijkstra-style pass.
3. **Per-agent relaxations** (agents = arguments of `free`), **route
   chaining** over each agent's location fluents (no teleporting between the
   places a task needs), and **LPT/EFT list scheduling** of goal tasks
   (`at X L` together with its implied `found X`) onto agents.
4. **Retry deltas** per task, with achievers that consume the same
   precondition counted as one attempt.
5. **Value** `lambda_add * sum_g C_g + lambda_ms * max_g C_g` over the
   scheduled completion times.

Compiled once per search into integer-indexed arrays, so evaluations are
several times faster than `ff_heuristic` on large grounded problems. Options
(`ConcurrentHeuristicOptions`) switch each component off for ablations.

Since this heuristic is calibrated (h tracks the remaining time), use it with
`heuristic_multiplier=1`: the leaf value is `-(t + w h)`, which for `w > 1`
improves with elapsed time along any decent path, so whichever branch is
searched deepest looks best. `backup="max"` (MaxUCT: decision nodes take their
best child, chance nodes the probability-weighted mean) keeps a few bad
coordination choices below a node from swamping its value.

### "at implies found" augmentation

When `at_implies_found` is true (default), `augment_at_with_found` adds a
`found <entity>` subgoal for every positive `at <entity> <loc>` fluent —
but only when `found <entity>` is reachable in the relaxed graph (so it
never introduces an unreachable subgoal; e.g. a robot, which no operator
can `found`, is silently skipped). This is applied to each goal branch and
to fluents discovered via action preconditions during backward extraction,
so the search cost of locating objects is reflected in `h_ff` and the
probabilistic delta even when `found` is not an explicit goal.

## Data Structures

### Achiever

An action that can produce a target fluent in the delete-relaxation:
```cpp
struct Achiever {
    const Action* action;
    double wait_cost;    // earliest time positive preconditions are achievable
    double exec_cost;    // the action's execution duration plus its extra_cost
    double probability;  // chance it produces the target fluent (1.0 = det.)

    double attempt_cost() const;  // wait_cost + exec_cost
    double efficiency() const;    // probability / exec_cost  (exec only!)
};
```

### FFForwardResult

Output of the forward relaxed reachability phase:
```cpp
struct FFForwardResult {
  std::unordered_set<Fluent> known_fluents;     // all reachable fluents
  std::unordered_set<Fluent> initial_fluents;   // t=0 seed fluents
  std::unordered_map<Fluent, std::vector<Achiever>> achievers_by_fluent;
  std::unordered_map<Fluent, const Action*> cheapest_achiever;   // smallest exec
  std::unordered_map<Fluent, const Action*> best_optimistic_achiever;
  std::unordered_map<const Action*, double> action_duration;     // max succ. time
  std::unordered_map<Fluent, double> optimistic_cost;            // 0 for initial
  std::unordered_set<Fluent> has_probabilistic_achiever;         // p < 1.0
  mutable std::unordered_map<Fluent, double> probabilistic_delta;// lazy δ cache
};
```

### FFBackwardResult

Output of the optimistic backward extraction (all three values are
infinite when any goal fluent is unreachable):
```cpp
struct FFBackwardResult {
  double h_add;                         // Σ optimistic_cost over goal fluents
  double h_max;                         // max optimistic_cost over goal fluents
  double h_ff;                          // Σ action_duration over plan actions
  std::unordered_set<Fluent> on_path;   // fluents visited while walking back
};
```

## Memoization

The heuristic uses `FFMemory` (a hash map from a composite key to cost). The
key includes the relaxed-state hash with `time = 0`, the goal hash, action-set
hash, `lambda_*` weights, and `at_implies_found` policy. States that differ
only in time but share the same relaxed fluents still reuse cached values, but
planner instances can safely be reused across different goals or filtered
action sets.

## Usage in MCTS

`MCTSPlanner` in `planner.hpp` uses `ff_heuristic` to estimate the
remaining cost-to-go at leaf nodes during simulation. The lambda mixing
weights are configurable on the planner wrapper
(`MCTSPlanner(..., lambda_add=, lambda_max=, lambda_ff=)`).

---

## Design notes

- **Deterministic-first optimistic cost.** If a fluent has both
  deterministic and probabilistic achievers, the optimistic cost uses the
  deterministic one — a sure path should not be inflated by retry math.
  The first time a fluent gains a deterministic achiever the fixed point
  force-adopts it even if it is higher than the prior probabilistic value.

- **Decoupled efficiency.** Achiever ordering for the retry delta uses
  `efficiency = probability / exec_cost` (exec only). When several
  probabilistic achievers target the same fluent the `wait_cost` is paid
  regardless of order; only `exec_cost` is the incremental cost of each
  attempt, so the optimal ordering maximizes `p / exec_cost`.

- **Optimistic core + delta separation.** `D(f)` (optimistic) and the
  probabilistic delta are computed separately. The delta is computed
  *lazily* per fluent, only for fluents on the extraction path whose selected
  optimistic achiever is probabilistic, and cached on the (mutable)
  `FFForwardResult` so OR branches that share a prerequisite compute it once.
  `has_probabilistic_achiever` lets the sum skip purely-deterministic fluents
  without iterating their achievers.

- **Relaxed-plan extraction.** Summing `optimistic_cost(goal)` directly
  would double-count shared preconditions; the BFS extraction identifies
  the actual fluents/actions needed so each delta is counted once.

- **Non-relaxed time bound.** `dtime` comes from the non-relaxed
  transition (first robot to finish), a tighter and more admissible lower
  bound than relaxed time in multi-robot scenarios.

### Header split

The heuristic is split across five self-contained headers along its
dependency DAG. Each has `#pragma once` and its own `#include`s (no
mid-file includes, one `namespace railroad` block per file):

```
heuristic_types.hpp      (core.hpp, state.hpp)
  └─ heuristic_forward.hpp
  └─ heuristic_prob.hpp
  └─ heuristic_backward.hpp
       └─ heuristic.hpp   (also includes the others + goal.hpp)
```

`heuristic_types.hpp` is the root: it owns `Achiever` / `FFForwardResult`,
so the forward, prob, and backward headers depend only on it (and on each
other not at all). `heuristic.hpp` is a thin umbrella that includes the
four plus `goal.hpp` — `goal.hpp` is only needed by `ff_heuristic` /
`extract_or_branches`, so including it at the top of the umbrella keeps it
out of the lower-level headers and removes the old circular-dependency
workaround. `augment_at_with_found` lives in `heuristic_backward.hpp` next
to its sole caller, `ff_backward_optimistic`. Consumers
(`planner.hpp`, `_bindings.cpp`) include only `heuristic.hpp`.
