#pragma once

// Shared data types for the FF heuristic.
//
// This header carries no algorithms — only the aliases and structs that the
// forward / probabilistic / backward phases all build on. It is the root of
// the heuristic header DAG: every other heuristic_*.hpp includes this one.

#include "railroad/core.hpp"
#include "railroad/state.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

// ============================================================================
//  Aliases
// ============================================================================

using HeuristicFn = std::function<double(const State &)>;

struct FFCacheKey {
  std::size_t relaxed_state_hash;
  std::size_t goal_hash;
  std::size_t actions_hash;
  std::size_t lambda_add_hash;
  std::size_t lambda_max_hash;
  std::size_t lambda_ff_hash;
  bool at_implies_found;

  bool operator==(const FFCacheKey& other) const {
    return relaxed_state_hash == other.relaxed_state_hash &&
           goal_hash == other.goal_hash &&
           actions_hash == other.actions_hash &&
           lambda_add_hash == other.lambda_add_hash &&
           lambda_max_hash == other.lambda_max_hash &&
           lambda_ff_hash == other.lambda_ff_hash &&
           at_implies_found == other.at_implies_found;
  }
};

struct FFCacheKeyHash {
  std::size_t operator()(const FFCacheKey& key) const {
    std::size_t h = key.relaxed_state_hash;
    hash_combine(h, key.goal_hash);
    hash_combine(h, key.actions_hash);
    hash_combine(h, key.lambda_add_hash);
    hash_combine(h, key.lambda_max_hash);
    hash_combine(h, key.lambda_ff_hash);
    hash_combine(h, std::hash<bool>{}(key.at_implies_found));
    return h;
  }
};

using FFMemory = std::unordered_map<FFCacheKey, double, FFCacheKeyHash>;

// ============================================================================
//  Core data types
// ============================================================================

// An action that can produce a target fluent in the delete-relaxation.
//   wait_cost: earliest time all positive preconditions are achievable
//              (zero in the forward phase, filled in by compute_optimistic_costs)
//   exec_cost: the action's own execution duration
//   probability: chance the action actually produces the target fluent
//                (1.0 = deterministic achiever)
struct Achiever {
    const Action* action;
    double wait_cost;
    double exec_cost;
    double probability;

    // Earliest time we'd hold the fluent if this achiever runs to completion.
    double attempt_cost() const { return wait_cost + exec_cost; }

    // Ranking key for ordering achievers when computing retry overhead.
    double efficiency() const {
        return (exec_cost > 1e-9) ? probability / exec_cost : probability * 1e9;
    }
};

// Output of the forward relaxed reachability phase.
//
// The "optimistic cost" of a fluent f is a lower bound on the time needed to
// achieve f in the delete-relaxation: pick the cheapest single-action
// achiever (deterministic preferred; otherwise the best probabilistic one,
// charging only one attempt) and recurse on its preconditions.
struct FFForwardResult {
  // Reachable fluents and the t=0 inputs that seeded the reachability search.
  std::unordered_set<Fluent> known_fluents;
  std::unordered_set<Fluent> initial_fluents;

  // For each fluent: every action that could produce it (wait/exec/prob).
  std::unordered_map<Fluent, std::vector<Achiever>> achievers_by_fluent;

  // For each fluent: the achiever with the smallest exec_cost. Filled during
  // forward reachability and kept as a fallback/debug aid.
  std::unordered_map<Fluent, const Action*> cheapest_achiever;

  // For each fluent: the achiever selected by compute_optimistic_costs().
  // Backward extraction uses this so h_ff follows the same relaxed plan as
  // h_add/h_max.
  std::unordered_map<Fluent, const Action*> best_optimistic_achiever;

  // Per-action exec_cost, taken as the max successor time.
  std::unordered_map<const Action*, double> action_duration;

  // Optimistic cost of each fluent (see comment above the struct).
  // Populated by compute_optimistic_costs(); 0 for initial fluents.
  std::unordered_map<Fluent, double> optimistic_cost;

  // Fluents with at least one strictly probabilistic achiever (p < 1.0).
  // Lets the prob extension skip purely deterministic fluents quickly.
  std::unordered_set<Fluent> has_probabilistic_achiever;

  // Cache of probabilistic deltas, populated lazily by heuristic_prob.hpp.
  // Mutable so it can be filled during const backward extraction; reused
  // across goal branches that share the same forward result.
  mutable std::unordered_map<Fluent, double> probabilistic_delta;
};

}  // namespace railroad
