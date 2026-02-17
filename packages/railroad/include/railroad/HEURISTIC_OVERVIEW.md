# FF Heuristic + Planner Integration Overview

This document describes how the heuristic in
`packages/railroad/include/railroad/ff_heuristic.hpp`
and the MCTS planner in
`packages/railroad/include/railroad/planner.hpp`
work together.

## 1. End-to-End Interaction Loop

`mcts(...)` integrates with the FF heuristic in this order:

1. Pre-prune the action set once at the root with `get_usable_actions(...)`.
2. Build `action_adds_map` from relaxed successors (only outcomes with `prob > 0`).
3. Run MCTS (selection, expansion, evaluation, backpropagation).
4. During evaluation of non-goal nodes, call `ff_heuristic(...)`.
5. Pick the root action with the highest visit count.

`MCTSPlanner` stores a shared `FFMemory` cache and passes it into each heuristic call, so relaxed-state values are reused across simulations and across planner invocations until `clear_cache()` is called.

## 2. `ff_heuristic(...)` (Probabilistic) Pipeline

### 2.1 Early exits

- `nullptr` or `TRUE_GOAL` -> `0.0`
- `FALSE_GOAL` -> `+infinity`

### 2.2 Dual transition views

The heuristic uses two transitions from the same input state:

1. `transition(state, nullptr, true)` for relaxed fluent reachability.
2. `transition(state, nullptr, false)` for near-term time lower bound.

Returned value:

`h(s) = dtime + min_branch_backward_cost`

where `dtime` is the first non-relaxed time increment.

### 2.3 Memoization key

After relaxed transition, the relaxed state's time is set to `0` before hashing.
That hash is the `FFMemory` key. This reuses heuristic work for states with the same relaxed fluent situation even if absolute times differ.

### 2.4 Forward relaxed phase (`ff_forward_phase`)

Forward phase computes:

- `known_fluents` (relaxed reachability closure),
- `achievers_by_fluent` (`ProbabilisticAchiever` entries),
- fallback achiever info (`fact_to_action`, `fact_to_probability`),
- `action_to_duration` (max relaxed successor time per action),
- `has_probabilistic_achiever`.

Each achiever stores:

- action pointer,
- `wait_cost` (filled later),
- `exec_cost` (action duration),
- success probability.

For probabilistic actions, achiever probability for `(action, fluent)` is computed
as the summed probability mass of outcomes where the fluent is present (clamped to 1.0).
So a fluent present in all branches is treated as deterministic (`p=1.0`), even if
branch timings differ.

### 2.5 Expected-cost fixed point (`compute_expected_costs`)

`D(f)` is iteratively updated (max 100 iterations):

1. `wait_cost` for each achiever becomes `max(D(preconditions))`.
2. `scan_achievers(...)` ranks achievers with deterministic preference and tie breaks.
3. `D(f)` choice:
   - use best deterministic cost if any deterministic achiever exists,
   - otherwise use best probabilistic candidate.
4. `best_achiever_action[f]` is recorded for backward extraction.

### 2.6 Backward extraction and base cost (`ff_backward_cost`)

For one DNF goal branch:

1. Validate branch fluents are reachable.
2. Backward extract achievers from goals to preconditions using:
   - `best_achiever_action` first,
   - `fact_to_action` as fallback.
3. Build relaxed arrival times for extracted fluents.
4. Compute hybrid base cost:

`base = max_goal_time + 0.5 * max(0, sum_goal_time - max_goal_time)`

The max term keeps critical-path signal; the additive term preserves gradient for multi-goal branches.

### 2.7 Probabilistic delta term

For each extracted fluent with probabilistic achievers:

- lazily compute `delta(f)` with `get_or_compute_delta(...)`,
- cache in `forward.probabilistic_delta`,
- add `PROBABILISTIC_DELTA_MULTIPLIER * delta(f)`.

Delta is:

`delta(f) = min_orderings(E_attempt) - D_prob_best`

Orderings tried:

1. efficiency (`p / exec_cost`) descending,
2. probability descending,
3. attempt cost ascending.

Small negative numerical artifacts are clamped to `0`.

### 2.8 OR goals / DNF branches

`extract_or_branches(goal)` uses `goal->get_dnf_branches()`.
The heuristic evaluates each branch independently via `ff_backward_cost` and returns the minimum finite branch cost.

## 3. Goal-Relevant Action Prioritization in `planner.hpp`

Before expansion, planner narrows and ranks applicable actions:

1. `get_unsatisfied_goal_literals(...)` picks one active DNF branch (best currently satisfied, then fewer missing literals, then smaller branch).
2. `get_goal_relevant_next_actions(...)` expands relevance backward through preconditions up to `relevance_depth` (default `2`).
3. Applicable actions not in the relevant set are filtered out if possible.
4. Remaining actions are scored:
   - `100 * adds_unsatisfied_goals + weighted_adds_relevant`.
5. Actions are sorted ascending by score, then popped from the back during expansion, so higher-score actions are expanded first.

This keeps expansion focused on actions that support the currently most promising goal branch while preserving fallback behavior if relevance filtering becomes too strict.

## 4. MCTS Evaluation and Reward Shaping

Per decision node:

- `goal_count = goal->goal_count(state.fluents())`
- `path_best_goal_count = max(parent_path_best_goal_count, goal_count)`

Shaping terms:

- `progress_bonus = LANDMARK_PROGRESS_REWARD * goal_count`
- `regression = max(0, path_best_goal_count - goal_count)`
- `regression_penalty = GOAL_REGRESSION_PENALTY * regression`

Reward equations:

- goal node:
  `reward = -time + SUCCESS_REWARD + progress_bonus - regression_penalty - accumulated_extra_cost`
- non-goal node:
  `reward = -time - heuristic_multiplier * h + progress_bonus - regression_penalty - accumulated_extra_cost`

If heuristic returns a very large value (`h > 1e10`), planner substitutes `HEURISTIC_CANNOT_FIND_GOAL_PENALTY`.

### 4.1 How `planner.hpp` uses `LANDMARK_PROGRESS_REWARD` and `GOAL_REGRESSION_PENALTY`

In `planner.hpp`, usage is direct in the simulation/evaluation block of `mcts(...)`:

1. `goal_count` and `path_best_goal_count` are populated by `set_goal_progress_fields(...)`.
2. Regression is measured as:
   `regression = max(0, path_best_goal_count - goal_count)`.
3. The constants are applied as:
   - `progress_bonus = LANDMARK_PROGRESS_REWARD * goal_count`
   - `regression_penalty = GOAL_REGRESSION_PENALTY * regression`

With current defaults, the shaping contribution at any evaluated node is:

`shaping = +25.0 * goal_count - 50.0 * regression`

This shaping term is added to both goal-node and non-goal-node reward equations, so it affects action selection and all backpropagated value estimates throughout the MCTS tree.

## 5. Current Constant Defaults (from `constants.hpp`)

- `HEURISTIC_MULTIPLIER = 5`
- `SUCCESS_REWARD = 0.0`
- `LANDMARK_PROGRESS_REWARD = 25.0`
- `GOAL_REGRESSION_PENALTY = 50.0`
- `PROBABILISTIC_DELTA_MULTIPLIER = 2.0`

`PROB_EXTRA_EXPLORE` is currently defined and a Bernoulli distribution is instantiated in `mcts(...)`, but that draw is not currently used to alter planner behavior in this header.
