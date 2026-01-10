#pragma once

#include "mrppddl/core.hpp"
#include "mrppddl/state.hpp"

#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mrppddl {

using HeuristicFn = std::function<double(const State &)>;
using FFMemory = std::unordered_map<std::size_t, double>;

// Result of the forward relaxed reachability phase (goal-independent)
struct FFForwardResult {
  std::unordered_set<Fluent> known_fluents;      // All reachable fluents
  std::unordered_set<Fluent> initial_fluents;    // Fluents at t=0 (after relaxed transition)
  std::unordered_map<Fluent, const Action*> fact_to_action;
  std::unordered_map<const Action*, double> action_to_duration;
  std::unordered_map<Fluent, double> fact_to_probability;
  double dtime;   // Time delta from relaxed transition
  bool valid;     // false if relaxed transition failed
};

// Forward relaxed reachability phase - goal independent
// Extracts the forward loop from ff_heuristic for reuse
FFForwardResult ff_forward_phase(
    const State &input_state,
    const std::vector<Action> &all_actions) {

  FFForwardResult result;
  result.valid = false;
  result.dtime = 0.0;

  const double t0 = input_state.time();

  // Step 1: Relaxed transition
  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) {
    return result;  // invalid
  }
  State relaxed = relaxed_result[0].first;

  result.dtime = relaxed.time() - t0;
  result.initial_fluents = std::unordered_set<Fluent>(
      relaxed.fluents().begin(), relaxed.fluents().end());
  result.known_fluents = result.initial_fluents;

  std::unordered_set<Fluent> newly_added = result.known_fluents;
  std::unordered_set<const Action *> visited_actions;
  std::unordered_set<const Action *> all_actions_set;
  for (const auto &a : all_actions) {
    all_actions_set.insert(&a);
  }

  // Forward relaxed reachability loop
  while (!newly_added.empty()) {
    std::unordered_set<Fluent> next_new;
    State state_all_known(0.0, result.known_fluents);

    for (const Action *a : all_actions_set) {
      if (visited_actions.count(a)) continue;
      if (!state_all_known.satisfies_precondition(*a, true)) continue;

      const auto& succs = a->get_relaxed_successors();
      visited_actions.insert(a);
      if (succs.empty()) continue;

      double duration = 0;
      for (const auto &[succ_state, succ_prob] : succs) {
        duration = std::max(succ_state.time(), duration);
        if (succ_prob <= 0.0) continue;

        for (const auto &f : succ_state.fluents()) {
          if (!result.known_fluents.count(f)) {
            result.known_fluents.insert(f);
            next_new.insert(f);
            result.fact_to_action[f] = a;
            result.fact_to_probability[f] = succ_prob;
          } else {
            result.fact_to_probability[f] =
                std::max(result.fact_to_probability[f], succ_prob);
            if (duration < result.action_to_duration[result.fact_to_action[f]]) {
              result.fact_to_action[f] = a;
            }
          }
        }
      }
      result.action_to_duration[a] = duration;
    }

    newly_added = std::move(next_new);
    for (const Action *a : visited_actions) {
      all_actions_set.erase(a);
    }
  }

  result.valid = true;
  return result;
}

// Backward cost computation given forward results and goal fluents
// Note: Negative goals should already be converted to positive equivalents
// by the Python layer before calling the heuristic.
double ff_backward_cost(
    const FFForwardResult &forward,
    const std::unordered_set<Fluent> &goal_fluents) {

  if (!forward.valid) {
    return std::numeric_limits<double>::infinity();
  }

  // Empty goal set means trivially satisfied (TrueGoal)
  if (goal_fluents.empty()) {
    return 0.0;
  }

  // Check reachability of all goal fluents
  for (const auto& gf : goal_fluents) {
    if (!forward.known_fluents.count(gf)) {
      return std::numeric_limits<double>::infinity();
    }
  }

  // Create a simple goal check function for ablation
  auto check_goal = [&goal_fluents](const std::unordered_set<Fluent>& fluents) {
    for (const auto& gf : goal_fluents) {
      if (!fluents.count(gf)) return false;
    }
    return true;
  };

  // Determine required fluents via ablation
  std::unordered_set<Fluent> required_fluents;
  for (const auto &f : forward.known_fluents) {
    std::unordered_set<Fluent> test_set(forward.known_fluents);
    test_set.erase(f);
    if (!check_goal(test_set)) {
      required_fluents.insert(f);
    }
  }

  // Backward relaxed plan
  std::unordered_set<Fluent> needed = required_fluents;
  std::unordered_set<const Action *> used_actions;
  double total_duration = 0.0;

  while (!needed.empty()) {
    Fluent f = *needed.begin();
    needed.erase(needed.begin());

    if (forward.initial_fluents.count(f)) continue;

    auto it = forward.fact_to_action.find(f);
    if (it == forward.fact_to_action.end()) continue;

    const Action *a = it->second;
    if (used_actions.count(a)) continue;
    used_actions.insert(a);

    auto dur_it = forward.action_to_duration.find(a);
    double base_duration = (dur_it != forward.action_to_duration.end()) ? dur_it->second : 0.0;
    total_duration += base_duration;

    for (const Fluent &p : a->pos_preconditions()) {
      if (!forward.initial_fluents.count(p)) {
        needed.insert(p);
      }
    }
  }

  return total_duration;
}

// Get usable actions via forward relaxed reachability
const std::vector<Action> get_usable_actions(const State &input_state,
					     const std::vector<Action> &all_actions) {
  // Use ff_forward_phase to get all reachable actions
  auto forward = ff_forward_phase(input_state, all_actions);

  if (!forward.valid) {
    return {};
  }

  // Collect all actions whose preconditions are satisfied by known fluents
  // This is simpler and more reliable than tracking specific action contributions
  std::vector<Action> feasible_actions;
  State state_all_known(0.0, forward.known_fluents);
  for (const auto& a : all_actions) {
    if (state_all_known.satisfies_precondition(a, true)) {
      feasible_actions.push_back(a);
    }
  }

  return feasible_actions;
}

} // namespace mrppddl

// Include goal.hpp here to get the full definition of GoalBase
// This is placed after the namespace closes to avoid circular dependencies
#include "mrppddl/goal.hpp"

namespace mrppddl {

// Extract DNF branches from a goal for efficient heuristic computation.
// Uses the goal's cached get_dnf_branches() method which properly handles
// nested OR inside AND by distributing: AND(A, OR(B,C)) -> [{A,B}, {A,C}]
inline const std::vector<std::unordered_set<Fluent>>& extract_or_branches(const GoalBase* goal) {
  static const std::vector<std::unordered_set<Fluent>> empty_branches;
  if (!goal) return empty_branches;
  return goal->get_dnf_branches();
}

// FF heuristic for complex goals with OR branches
// Runs forward phase ONCE, then computes backward cost for each branch
inline double ff_heuristic(const State &input_state,
                           const GoalBase *goal,
                           const std::vector<Action> &all_actions,
                           FFMemory *ff_memory = nullptr) {
  // Handle null goal
  if (!goal) return 0.0;

  GoalType type = goal->get_type();

  // Handle trivial cases without forward computation
  if (type == GoalType::TRUE_GOAL) return 0.0;
  if (type == GoalType::FALSE_GOAL) {
    return std::numeric_limits<double>::infinity();
  }

  // Run forward phase ONCE
  auto forward = ff_forward_phase(input_state, all_actions);
  if (!forward.valid) {
    return std::numeric_limits<double>::infinity();
  }

  // Extract branches based on goal structure
  auto branches = extract_or_branches(goal);
  if (branches.empty()) {
    return std::numeric_limits<double>::infinity();  // FalseGoal case
  }

  // Compute minimum cost across all branches
  double min_cost = std::numeric_limits<double>::infinity();

  for (const auto& branch_fluents : branches) {
    double backward_cost = ff_backward_cost(forward, branch_fluents);
    if (backward_cost < std::numeric_limits<double>::infinity()) {
      double total = forward.dtime + backward_cost;
      min_cost = std::min(min_cost, total);
    }
  }

  return min_cost;
}

} // namespace mrppddl
