# FF Heuristic and MCTS Goal-Shaping Overview

This document explains the current heuristic pipeline in:

- `packages/railroad/include/railroad/ff_heuristic.hpp`
- `packages/railroad/include/railroad/planner.hpp`

It also covers the new reward-shaping terms:

- `LANDMARK_PROGRESS_REWARD`
- `GOAL_REGRESSION_PENALTY`

## 1. High-Level Flow

The planner uses MCTS for action selection, and evaluates leaf states with the FF-style heuristic.

At each MCTS evaluation point:

1. Compute `h(s)` with `ff_heuristic(...)` (or `det_ff_heuristic(...)` if configured).
2. Convert that to reward with time, extra action cost, and goal-shaping terms.
3. Backpropagate that reward up the search tree.

## 2. FF Heuristic: End-to-End

## 2.1 Entry Conditions and Early Exits

`ff_heuristic` returns:

- `0.0` for null goal or `TRUE_GOAL`
- `+infinity` for `FALSE_GOAL`

## 2.2 Two Transition Views

The probabilistic FF heuristic uses two transition calls:

1. Relaxed transition (`transition(..., true)`) for fluent reachability.
2. Non-relaxed transition (`transition(..., false)`) for near-term time advance.

The returned heuristic is:

`h(s) = dtime + backward_cost`

where `dtime` is the first non-relaxed time increment and `backward_cost` is computed from the relaxed graph.

## 2.3 Memoization

After relaxed transition, time is zeroed (`relaxed.set_time(0)`), then the relaxed state hash is used as the memo key (`FFMemory`).

This lets states with the same relaxed fluent set reuse heuristic work even if wall-clock times differ.

## 2.4 Forward Phase: Reachability and Achievers

`ff_forward_phase(...)` computes:

- reachable fluents (`known_fluents`)
- achievers per fluent (`achievers_by_fluent`)
- fallback achiever map (`fact_to_action`)
- action duration map (`action_to_duration`)

Each achiever records:

- action pointer
- wait cost (filled later in fixed-point iteration)
- exec cost (action duration)
- success probability

## 2.5 Expected-Cost Fixed Point (`compute_expected_costs`)

For each fluent `f`, the code computes optimistic cost `D(f)` by iterating until convergence.

Key behavior:

- Preconditions contribute `wait_cost = max(D(preconditions))`.
- Achievers are scanned with deterministic-first policy:
  - If any deterministic achiever exists, `D(f)` uses best deterministic cost.
  - Otherwise it uses the best probabilistic achiever (probability-first tie break).
- The best achiever action is stored in `best_achiever_action[f]` for backward extraction.

## 2.6 Backward Extraction and Base Cost

`ff_backward_cost(...)` does backward extraction from goal fluents:

1. Build extraction frontier from goals to preconditions.
2. Choose achievers via `best_achiever_action`, with fallback to `fact_to_action`.
3. Build a relaxed arrival schedule `T(f)` over extracted fluents.

Base cost is a hybrid of makespan and additive pressure:

`base = max_goal_time + lambda * (sum_goal_time - max_goal_time)`, with `lambda = 0.5`.

This keeps critical-path sensitivity while still giving gradient for multiple goals.

## 2.7 Probabilistic Delta Term

For path fluents with probabilistic achievers, delta is computed lazily:

`delta(f) = E_attempt(f) - D_prob_best(f)`

where `E_attempt` is the best expected attempt cost among several achiever orderings.

Final backward cost:

`total = base + PROBABILISTIC_DELTA_MULTIPLIER * sum(delta(f) on extracted path)`

Default multiplier is currently `2.0`.

## 2.8 Goal Structure Handling

Complex goals are decomposed into DNF branches (`goal->get_dnf_branches()`), and heuristic cost is the minimum branch cost.

This supports nested AND/OR structure efficiently.

## 3. MCTS Reward and the New Goal-Shaping Terms

The planner now tracks per-node progress:

- `goal_count`: satisfied leaf progress at this node (`goal->goal_count(...)`)
- `path_best_goal_count`: best progress seen from root to this node

From those values:

- `progress_bonus = LANDMARK_PROGRESS_REWARD * goal_count`
- `regression = max(0, path_best_goal_count - goal_count)`
- `regression_penalty = GOAL_REGRESSION_PENALTY * regression`

Current defaults:

- `LANDMARK_PROGRESS_REWARD = 25.0`
- `GOAL_REGRESSION_PENALTY = 50.0`

Reward equations:

- Goal node:
  - `reward = -time + SUCCESS_REWARD + progress_bonus - regression_penalty - extra_cost`
- Non-goal node:
  - `reward = -time - heuristic_multiplier * h + progress_bonus - regression_penalty - extra_cost`

## 4. Intuition for the New Terms

`LANDMARK_PROGRESS_REWARD`:

- Gives positive signal before full goal completion.
- Helps MCTS value partial progress, not only terminal success.

`GOAL_REGRESSION_PENALTY`:

- Penalizes states that undo previously achieved goal progress on the sampled path.
- Reduces oscillatory behavior where search repeatedly gains and loses goal literals.

Together, they shape value estimates to prefer monotonic goal progress.

## 5. Practical Tuning Notes

1. If planning becomes overly conservative, reduce `GOAL_REGRESSION_PENALTY`.
2. If search ignores partial progress, increase `LANDMARK_PROGRESS_REWARD`.
3. If heuristic dominates too strongly, lower `HEURISTIC_MULTIPLIER`.
4. If probabilistic tasks become over-penalized, lower `PROBABILISTIC_DELTA_MULTIPLIER` toward `1.0`.

