#pragma once

#include "mrppddl/core.hpp"
#include "mrppddl/state.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mrppddl {

using HeuristicFn = std::function<double(const State &)>;
using FFMemory = std::unordered_map<std::size_t, double>;

// Information about an action that can achieve a fluent
struct ProbabilisticAchiever {
    const Action* action;
    double reach_cost;      // Cost to satisfy preconditions
    double action_cost;     // Duration/cost of action itself
    double probability;     // Probability of achieving target fluent (1.0 for deterministic)

    double total_cost() const { return reach_cost + action_cost; }

    // Efficiency for ordering: higher = try first
    double efficiency() const {
        double c = total_cost();
        return (c > 1e-9) ? probability / c : probability * 1e9;
    }
};

// Result of the forward relaxed reachability phase (goal-independent)
struct FFForwardResult {
  std::unordered_set<Fluent> known_fluents;      // All reachable fluents
  std::unordered_set<Fluent> initial_fluents;    // Fluents at t=0 (input to forward phase)
  std::unordered_map<Fluent, const Action*> fact_to_action;
  std::unordered_map<const Action*, double> action_to_duration;
  std::unordered_map<Fluent, double> fact_to_probability;

  // New fields for expected cost computation
  std::unordered_map<Fluent, std::vector<ProbabilisticAchiever>> achievers_by_fluent;
  std::unordered_map<Fluent, double> expected_cost;  // V(f) values from Bellman iteration
};

// Forward relaxed reachability phase - goal independent
// Takes initial fluents (after relaxed transition) and computes reachability
// The caller is responsible for doing the relaxed transition first
FFForwardResult ff_forward_phase(
    const std::unordered_set<Fluent> &initial_fluents,
    const std::vector<Action> &all_actions) {

  FFForwardResult result;
  result.initial_fluents = initial_fluents;
  result.known_fluents = initial_fluents;

  // Initialize expected costs for initial fluents
  for (const auto& f : initial_fluents) {
    result.expected_cost[f] = 0.0;
  }

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
          // Always record this as an achiever for f
          result.achievers_by_fluent[f].push_back({
              a, 0.0, duration, succ_prob  // reach_cost computed in phase 2
          });

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

  return result;
}

// Compute expected costs for fluents using Bellman-style iteration
inline void compute_expected_costs(FFForwardResult& result) {
  // Initialize: V(f) = 0 for initial, +infinity otherwise
  for (const auto& [f, achievers] : result.achievers_by_fluent) {
    if (!result.initial_fluents.count(f)) {
      result.expected_cost[f] = std::numeric_limits<double>::infinity();
    }
  }

  bool changed = true;
  int iteration = 0;
  const int MAX_ITERATIONS = 100;
  const double TOLERANCE = 1e-9;

  while (changed && iteration < MAX_ITERATIONS) {
    changed = false;
    iteration++;

    for (auto& [f, achievers] : result.achievers_by_fluent) {
      if (result.initial_fluents.count(f)) continue;

      // Update reach_cost for each achiever from precondition costs
      for (auto& achiever : achievers) {
        double max_prec_cost = 0.0;
        for (const auto& prec : achiever.action->pos_preconditions()) {
          auto it = result.expected_cost.find(prec);
          if (it != result.expected_cost.end()) {
            max_prec_cost = std::max(max_prec_cost, it->second);
          }
        }
        achiever.reach_cost = max_prec_cost;
      }

      // Compute D(f) - min cost from deterministic achievers (p >= 1.0 - epsilon)
      double D_f = std::numeric_limits<double>::infinity();
      for (const auto& achiever : achievers) {
        if (achiever.probability >= 1.0 - TOLERANCE) {
          D_f = std::min(D_f, achiever.total_cost());
        }
      }

      // Compute E(f) - expected cost from probabilistic achievers
      // Sort by efficiency (p/cost) descending
      std::vector<ProbabilisticAchiever> prob_achievers;
      for (const auto& a : achievers) {
        if (a.probability > TOLERANCE && a.probability < 1.0 - TOLERANCE) {
          prob_achievers.push_back(a);
        }
      }

      double E_f = std::numeric_limits<double>::infinity();
      if (!prob_achievers.empty()) {
        std::sort(prob_achievers.begin(), prob_achievers.end(),
            [](const ProbabilisticAchiever& a, const ProbabilisticAchiever& b) {
              return a.efficiency() > b.efficiency();
            });

        // E(f) = sum_{t} [ prod_{j<t}(1-p_j) * C_t ]
        E_f = 0.0;
        double prob_all_failed = 1.0;
        for (const auto& achiever : prob_achievers) {
          E_f += prob_all_failed * achiever.total_cost();
          prob_all_failed *= (1.0 - achiever.probability);
        }
      }

      // V(f) = min(D(f), E(f))
      double new_cost = std::min(D_f, E_f);
      if (std::abs(new_cost - result.expected_cost[f]) > TOLERANCE) {
        result.expected_cost[f] = new_cost;
        changed = true;
      }
    }
  }
}

// Backward cost computation given forward results and goal fluents
// Note: Negative goals should already be converted to positive equivalents
// by the Python layer before calling the heuristic.
double ff_backward_cost(
    const FFForwardResult &forward,
    const std::unordered_set<Fluent> &goal_fluents) {

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
  std::unordered_set<const Action*> feasible_action_set;

  // Step 1: Relaxed transition (processes upcoming effects)
  auto relaxed_result = transition(input_state, nullptr, true);
  if (!relaxed_result.empty()) {
    State relaxed = relaxed_result[0].first;

    // Get initial fluents from relaxed state
    std::unordered_set<Fluent> initial_fluents(
        relaxed.fluents().begin(), relaxed.fluents().end());

    // Use ff_forward_phase to get all reachable fluents
    auto forward = ff_forward_phase(initial_fluents, all_actions);

    // Collect all actions whose preconditions are satisfied by known fluents
    State state_all_known(0.0, forward.known_fluents);
    for (const auto& a : all_actions) {
      if (state_all_known.satisfies_precondition(a, true)) {
        feasible_action_set.insert(&a);
      }
    }
  }

  // Step 2: Also consider current fluents WITHOUT processing upcoming effects
  // This handles cases where upcoming effects would preclude valid actions
  // (e.g., another robot can still move to a location before it's marked visited)
  {
    std::unordered_set<Fluent> current_fluents(
        input_state.fluents().begin(), input_state.fluents().end());

    auto forward_current = ff_forward_phase(current_fluents, all_actions);

    State state_current_known(0.0, forward_current.known_fluents);
    for (const auto& a : all_actions) {
      if (state_current_known.satisfies_precondition(a, true)) {
        feasible_action_set.insert(&a);
      }
    }
  }

  // Convert set to vector
  std::vector<Action> feasible_actions;
  feasible_actions.reserve(feasible_action_set.size());
  for (const Action* a : feasible_action_set) {
    feasible_actions.push_back(*a);
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
// Does relaxed transition first, then uses memoization to avoid recomputation
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

  const double t0 = input_state.time();

  // Step 1: Relaxed transition
  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  State relaxed = relaxed_result[0].first;

  double dtime = relaxed.time() - t0;

  // Memoization check: use hash of relaxed state (with time=0)
  relaxed.set_time(0);
  if (ff_memory && ff_memory->count(relaxed.hash())) {
    return dtime + ff_memory->at(relaxed.hash());
  }

  // Get initial fluents from relaxed state
  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  // Run forward phase
  auto forward = ff_forward_phase(initial_fluents, all_actions);

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
      min_cost = std::min(min_cost, backward_cost);
    }
  }

  // Store in memory (memoize the cost AFTER relaxed transition)
  if (ff_memory) {
    (*ff_memory)[relaxed.hash()] = min_cost;
  }

  return dtime + min_cost;
}

} // namespace mrppddl
