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
  // Track which fluents can be deleted by reachable actions (for negative goals)
  std::unordered_map<Fluent, const Action*> deletable_fluent_to_action;
  double dtime;   // Time delta from relaxed transition
  bool valid;     // false if relaxed transition failed
};

class GoalFn {
private:
    std::unordered_set<Fluent> goal_fluents_;

public:
    // Constructor - replaces make_goal_fn
    GoalFn(const std::unordered_set<Fluent> &goal_fluents)
        : goal_fluents_(goal_fluents) {}

    // Make it callable - checks if all goal fluents are present
    bool operator()(const std::unordered_set<Fluent> &fluents) const {
        for (const auto &gf : goal_fluents_) {
            if (!fluents.count(gf))
                return false;
        }
        return true;
    }

    // Count how many goal fluents are present in the active set
    int goal_count(const std::unordered_set<Fluent> &active_fluents) const {
        int count = 0;
        for (const auto &gf : goal_fluents_) {
            if (active_fluents.count(gf)) {
                count++;
            }
        }
        return count;
    }
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

      // Track which fluents this action can delete (for negative goal support)
      for (const auto& eff : a->effects()) {
        for (const auto& deleted_fluent : eff->flipped_neg_fluents()) {
          // Record that this action can delete this fluent
          if (result.deletable_fluent_to_action.find(deleted_fluent) ==
              result.deletable_fluent_to_action.end()) {
            result.deletable_fluent_to_action[deleted_fluent] = a;
          }
        }
      }
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
// Handles both positive fluents (must be achieved) and negative fluents (must be absent)
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

  // Separate positive and negative goal fluents
  std::unordered_set<Fluent> positive_goals;
  std::unordered_set<Fluent> negative_goals;

  for (const auto& gf : goal_fluents) {
    if (gf.is_negated()) {
      negative_goals.insert(gf);
    } else {
      positive_goals.insert(gf);
    }
  }

  // Check reachability of positive goal fluents
  for (const auto& gf : positive_goals) {
    if (!forward.known_fluents.count(gf)) {
      return std::numeric_limits<double>::infinity();
    }
  }

  // Handle negative goal fluents:
  // A negative goal ~P is satisfied if P is not in the state.
  // In relaxed planning, we never delete fluents, so:
  // - If P is not in initial_fluents, ~P is already satisfied (cost 0)
  // - If P is in initial_fluents, we need an action that deletes P
  std::unordered_set<Fluent> negative_goals_needing_delete;
  for (const auto& neg_gf : negative_goals) {
    Fluent positive_fluent = neg_gf.invert();  // Get P from ~P

    // If P is not in initial state, ~P is already satisfied
    if (!forward.initial_fluents.count(positive_fluent)) {
      continue;  // This negative goal is free
    }

    // P is in initial state - we need to delete it
    // Check if any reachable action can delete this fluent
    auto it = forward.deletable_fluent_to_action.find(positive_fluent);
    if (it == forward.deletable_fluent_to_action.end()) {
      // No reachable action can delete this fluent
      return std::numeric_limits<double>::infinity();
    }

    // Mark this fluent as needing deletion - we'll add the delete action's cost
    negative_goals_needing_delete.insert(positive_fluent);
  }

  // Create a simple goal check function for ablation (positive goals only)
  auto check_goal = [&positive_goals](const std::unordered_set<Fluent>& fluents) {
    for (const auto& gf : positive_goals) {
      if (!fluents.count(gf)) return false;
    }
    return true;
  };

  // Determine required fluents via ablation (for positive goals)
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

  // First, add delete actions for negative goals
  for (const auto& fluent_to_delete : negative_goals_needing_delete) {
    auto it = forward.deletable_fluent_to_action.find(fluent_to_delete);
    if (it != forward.deletable_fluent_to_action.end()) {
      const Action* delete_action = it->second;
      if (!used_actions.count(delete_action)) {
        used_actions.insert(delete_action);

        // Add delete action's duration
        auto dur_it = forward.action_to_duration.find(delete_action);
        double duration = (dur_it != forward.action_to_duration.end()) ? dur_it->second : 0.0;
        total_duration += duration;

        // Add delete action's preconditions to needed set
        for (const Fluent& p : delete_action->pos_preconditions()) {
          if (!forward.initial_fluents.count(p)) {
            needed.insert(p);
          }
        }
      }
    }
  }

  // Then, backward plan for positive goals and delete action preconditions
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

double ff_heuristic(const State &input_state, const GoalFn *is_goal_fn,
                    const std::vector<Action> &all_actions,
                    FFMemory *ff_memory = nullptr,
                    std::unordered_set<const Action *> *output_visited_actions = nullptr) {

  const double t0 = input_state.time();
  // Step 1: Relaxed transition
  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty())
    return std::numeric_limits<double>::infinity();
  State relaxed = relaxed_result[0].first;

  double dtime = relaxed.time() - t0;
  const auto &initial_fluents = relaxed.fluents();
  std::unordered_set<Fluent> known_fluents(initial_fluents.begin(),
                                           initial_fluents.end());

  // Memoization check (only if goal function provided)
  relaxed.set_time(0);
  if (is_goal_fn && ff_memory && ff_memory->count(relaxed.hash())) {
    return dtime + ff_memory->at(relaxed.hash());
  }

  std::unordered_set<Fluent> newly_added = known_fluents;
  std::unordered_map<Fluent, const Action *> fact_to_action;
  std::unordered_map<const Action *, double> action_to_duration;
  // NEW: Track the probability of achieving each fluent
  std::unordered_map<Fluent, double> fact_to_probability;
  std::unordered_set<const Action *> visited_actions;
  std::unordered_set<const Action *> all_actions_set;
  for (const auto &a : all_actions) {
    all_actions_set.insert(&a);
  }

  // Step 1: Forward relaxed reachability
  while (!newly_added.empty()) {
    std::unordered_set<Fluent> next_new;
    State state_all_known(0.0, known_fluents); // dummy state to test preconditions

    for (const Action *a : all_actions_set) {
      if (visited_actions.count(a))
        continue;
      if (!state_all_known.satisfies_precondition(*a, /*relax=*/true))
        continue;

      const auto& succs = a->get_relaxed_successors();
      visited_actions.insert(a);
      if (succs.empty())
        continue;

      double duration = 0;

      // In relaxed planning, consider fluents from ALL probabilistic outcomes
      // NEW: But now track the maximum probability of achieving each fluent
      for (const auto &[succ_state, succ_prob] : succs) {
	duration = std::max(succ_state.time(), duration);
	if (succ_prob <= 0.0) {
	  continue;
	}
        for (const auto &f : succ_state.fluents()) {
          if (!known_fluents.count(f)) {  // If 'f' not in known_fluents
            known_fluents.insert(f);
            next_new.insert(f);
            fact_to_action[f] = a;
            // NEW: Record the probability of achieving this fluent
            fact_to_probability[f] = succ_prob;
          } else {  // If we've seen 'f' before
	    // FIXME: I don't think this will actually work...
            fact_to_probability[f] = std::max(fact_to_probability[f], succ_prob);

	    // Count the minimum time to reach a place
	    if (duration < action_to_duration[fact_to_action[f]]) {
	      fact_to_action[f] = a;
	    }
	  }
        }
      } // for over successor states
      action_to_duration[a] = duration;
    }  // for over actions

    newly_added = std::move(next_new);
    for (const Action *a : visited_actions) {
      all_actions_set.erase(a);
    }
  }

  // If output_visited_actions requested, populate it
  if (output_visited_actions) {
    *output_visited_actions = visited_actions;
  }

  // If no goal function provided, skip backward planning and return 0
  if (!is_goal_fn) {
    return 0.0;
  }

  if (!(*is_goal_fn)(known_fluents)) {
    return std::numeric_limits<double>::infinity(); // unreachable goal
  }

  // Step 2: Determine required goal fluents via ablation
  std::unordered_set<Fluent> required_fluents;
  for (const auto &f : known_fluents) {
    std::unordered_set<Fluent> test_set(known_fluents);
    test_set.erase(f);
    if (!(*is_goal_fn)(test_set)) {
      required_fluents.insert(f);
    }
  }

  // Step 3: Backward relaxed plan
  std::unordered_set<Fluent> needed = required_fluents;
  std::unordered_set<const Action *> used_actions;
  // NEW: Track which fluents are needed from each action for probability scaling
  std::unordered_map<const Action *, std::vector<Fluent>> action_to_needed_fluents;
  double total_duration = 0.0;

  while (!needed.empty()) {
    Fluent f = *needed.begin();
    needed.erase(needed.begin());

    if (initial_fluents.count(f))
      continue;

    auto it = fact_to_action.find(f);
    if (it == fact_to_action.end())
      continue;
    const Action *a = it->second;

    // NEW: Track which fluent(s) this action is being used for
    action_to_needed_fluents[a].push_back(f);

    if (used_actions.count(a)) {
      continue;
    }

    used_actions.insert(a);

    // NEW: Scale duration by probability of achieving needed fluent
    double base_duration = action_to_duration[a];
    double prob = fact_to_probability.count(f) ? fact_to_probability[f] : 1.0;

    // Cap the probability scaling to avoid extreme values
    // If prob < 0.01, treat as if prob = 0.01 (max 100x scaling)
    const double MIN_PROB = 0.01;
    prob = std::max(prob, MIN_PROB);

    // Expected attempts = 1 / probability
    double expected_duration = base_duration / prob;
    // total_duration += expected_duration;
    total_duration += base_duration;

    for (const Fluent &p : a->pos_preconditions()) {
      if (!initial_fluents.count(p)) {
        needed.insert(p);
      }
    }
  }

  if (ff_memory) {
    (*ff_memory)[relaxed.hash()] = total_duration;
  }

  return dtime + total_duration;
}

HeuristicFn make_ff_heuristic(GoalFn is_goal_fn,
                              std::vector<Action> all_actions,
                              FFMemory *memory = nullptr) {
  return [=](const State &s) -> double {
    return ff_heuristic(s, &is_goal_fn, all_actions, memory);
  };
}

const std::vector<Action> get_usable_actions(const State &input_state,
					     const std::vector<Action> &all_actions) {
  // Use ff_heuristic with nullptr goal function to get visited actions
  std::unordered_set<const Action *> visited_actions;
  ff_heuristic(input_state, nullptr, all_actions, nullptr, &visited_actions);

  // Convert set of action pointers to vector of actions
  std::vector<Action> feasible_actions;
  feasible_actions.reserve(visited_actions.size());
  for (const Action *action_ptr : visited_actions) {
    feasible_actions.push_back(*action_ptr);
  }

  return feasible_actions;
}

const std::vector<Action> get_usable_actions_fluent_list(const State &input_state,
					     const std::unordered_set<Fluent> &goal_fluents,
					     const std::vector<Action> &all_actions) {
 return get_usable_actions(input_state, all_actions);
}

} // namespace mrppddl

// Include goal.hpp here to get the full definition of GoalBase
// This is placed after the namespace closes to avoid circular dependencies
#include "mrppddl/goal.hpp"

namespace mrppddl {

// Extract OR branches from a goal for efficient heuristic computation
// Top-level OR only: each child becomes a branch
// Everything else: single branch with all literals
inline std::vector<std::unordered_set<Fluent>> extract_or_branches(const GoalBase* goal) {
  if (!goal) return {};

  GoalType type = goal->get_type();

  // TrueGoal: trivially satisfied, cost = 0 (empty fluent set)
  if (type == GoalType::TRUE_GOAL) {
    return {{}};
  }

  // FalseGoal: impossible to satisfy (no valid branches)
  if (type == GoalType::FALSE_GOAL) {
    return {};
  }

  // Top-level OR: each child is a branch
  if (type == GoalType::OR) {
    std::vector<std::unordered_set<Fluent>> branches;
    for (const auto& child : goal->children()) {
      branches.push_back(child->get_all_literals());
    }
    return branches;
  }

  // Everything else (LITERAL, AND, AND with nested OR): single branch
  return {goal->get_all_literals()};
}

// Efficient FF heuristic for complex goals with OR branches
// Runs forward phase ONCE, then computes backward cost for each branch
inline double ff_heuristic_for_goal(const State &input_state,
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
