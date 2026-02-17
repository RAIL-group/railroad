#pragma once

#include "railroad/constants.hpp"
#include "railroad/core.hpp"
#include "railroad/state.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

using HeuristicFn = std::function<double(const State &)>;
using FFMemory = std::unordered_map<std::size_t, double>;

// Information about an action that can achieve a fluent
struct ProbabilisticAchiever {
    const Action* action;
    double wait_cost;       // Time until preconditions satisfied (MAX of precondition costs)
    double exec_cost;       // Duration of action execution itself
    double probability;     // Probability of achieving target fluent (1.0 for deterministic)

    // Total cost for a single attempt (wait + execute)
    double attempt_cost() const { return wait_cost + exec_cost; }

    // Efficiency for ordering: higher = try first
    // Use probability / exec_cost since wait_cost is paid regardless
    double efficiency() const {
        return (exec_cost > 1e-9) ? probability / exec_cost : probability * 1e9;
    }
};

// Result of the forward relaxed reachability phase (goal-independent)
struct FFForwardResult {
  std::unordered_set<Fluent> known_fluents;      // All reachable fluents
  std::unordered_set<Fluent> initial_fluents;    // Fluents at t=0 (input to forward phase)
  std::unordered_map<Fluent, const Action*> fact_to_action;
  std::unordered_map<const Action*, double> action_to_duration;
  std::unordered_map<Fluent, double> fact_to_probability;

  // Fields for expected cost computation
  std::unordered_map<Fluent, std::vector<ProbabilisticAchiever>> achievers_by_fluent;
  std::unordered_map<Fluent, double> expected_cost;  // D(f) - optimistic cost

  // Fluents that have at least one probabilistic achiever (p < 1.0)
  // Used to skip delta computation for purely deterministic fluents
  std::unordered_set<Fluent> has_probabilistic_achiever;

  // Lazily computed delta(f) = E_attempt - D for probabilistic fluents
  // Mutable so it can be populated on-demand during const backward extraction
  mutable std::unordered_map<Fluent, double> probabilistic_delta;
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
              a, 0.0, duration, succ_prob  // wait_cost=0 (computed in phase 2), exec_cost=duration
          });

          // Track fluents with probabilistic achievers for lazy delta computation
          if (succ_prob < 1.0 - 1e-9) {
            result.has_probabilistic_achiever.insert(f);
          }

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

// Compute D(f) for all fluents (optimistic, treating all achievers as deterministic)
// Deltas for probabilistic fluents are computed lazily during backward extraction
inline void compute_expected_costs(FFForwardResult& result) {
  const double TOLERANCE = 1e-9;
  const int MAX_ITERATIONS = 100;

  // Initialize: D(f) = 0 for initial, +infinity otherwise
  for (const auto& [f, achievers] : result.achievers_by_fluent) {
    if (!result.initial_fluents.count(f)) {
      result.expected_cost[f] = std::numeric_limits<double>::infinity();
    }
  }

  // Track which fluents have deterministic achievers available
  std::unordered_set<Fluent> has_det_achiever;

  // Phase 1: Compute D(f) for all fluents using fixed-point iteration
  // D is the optimistic cost using deterministic achievers (or best probabilistic if none)
  bool changed = true;
  int iteration = 0;

  while (changed && iteration < MAX_ITERATIONS) {
    changed = false;
    iteration++;

    for (auto& [f, achievers] : result.achievers_by_fluent) {
      if (result.initial_fluents.count(f)) continue;

      // Update wait_cost for each achiever using current D values
      for (auto& achiever : achievers) {
        double max_prec_cost = 0.0;
        for (const auto& prec : achiever.action->pos_preconditions()) {
          auto it = result.expected_cost.find(prec);
          if (it != result.expected_cost.end()) {
            max_prec_cost = std::max(max_prec_cost, it->second);
          }
        }
        achiever.wait_cost = max_prec_cost;
      }

      // D(f) = min of deterministic achievers (p >= 1.0)
      // If no deterministic achievers, use best probabilistic (highest p, then lowest cost)
      double D_det = std::numeric_limits<double>::infinity();
      double D_prob = std::numeric_limits<double>::infinity();
      double best_prob = 0.0;

      for (const auto& achiever : achievers) {
        double cost = achiever.wait_cost + achiever.exec_cost;
        if (achiever.probability >= 1.0 - TOLERANCE) {
          D_det = std::min(D_det, cost);
        } else if (achiever.probability > TOLERANCE) {
          // For probabilistic, prefer higher probability, then lower cost
          if (achiever.probability > best_prob ||
              (achiever.probability >= best_prob - TOLERANCE && cost < D_prob)) {
            D_prob = cost;
            best_prob = achiever.probability;
          }
        }
      }

      // Determine which value to use
      double D_f;
      bool use_det = D_det < std::numeric_limits<double>::infinity();

      if (use_det) {
        D_f = D_det;
        // If we're switching from prob to det, force update
        bool was_prob = !has_det_achiever.count(f) &&
                        result.expected_cost[f] < std::numeric_limits<double>::infinity();
        if (was_prob) {
          has_det_achiever.insert(f);
          result.expected_cost[f] = D_f;
          changed = true;
          continue;
        }
        has_det_achiever.insert(f);
      } else {
        D_f = D_prob;
      }

      if (D_f < result.expected_cost[f] - TOLERANCE) {
        result.expected_cost[f] = D_f;
        changed = true;
      }
    }
  }
}

// Lazily compute delta for a single fluent (called during backward extraction)
// Returns the delta value, caching it in forward.probabilistic_delta for reuse
// Returns 0.0 if the fluent has no probabilistic achievers
inline double get_or_compute_delta(const FFForwardResult& forward, const Fluent& f) {
  const double TOLERANCE = 1e-9;

  // Check cache first
  auto cached_it = forward.probabilistic_delta.find(f);
  if (cached_it != forward.probabilistic_delta.end()) {
    return cached_it->second;
  }

  // Skip initial fluents
  if (forward.initial_fluents.count(f)) {
    return 0.0;
  }

  // Get achievers for this fluent
  auto achievers_it = forward.achievers_by_fluent.find(f);
  if (achievers_it == forward.achievers_by_fluent.end()) {
    return 0.0;
  }

  const auto& achievers = achievers_it->second;

  // Collect probabilistic achievers only
  std::vector<ProbabilisticAchiever> prob_achievers;
  for (const auto& a : achievers) {
    if (a.probability > TOLERANCE && a.probability < 1.0 - TOLERANCE) {
      prob_achievers.push_back(a);
    }
  }

  // No probabilistic achievers means delta = 0
  if (prob_achievers.empty()) {
    forward.probabilistic_delta[f] = 0.0;
    return 0.0;
  }

  // Helper lambda to compute E_attempt for a given ordering of achievers
  auto compute_E_attempt = [](const std::vector<ProbabilisticAchiever>& ordered_achievers) {
    double E_attempt = 0.0;
    double prob_all_failed = 1.0;
    double time = 0.0;

    for (const auto& achiever : ordered_achievers) {
      double dtime = std::max(achiever.wait_cost - time, 0.0);
      double this_attempt_cost = dtime + achiever.exec_cost;

      E_attempt += prob_all_failed * this_attempt_cost;
      prob_all_failed *= (1.0 - achiever.probability);
      time = std::max(time, achiever.wait_cost);
    }
    return E_attempt;
  };

  // Try multiple orderings and pick the minimum E_attempt
  double min_E_attempt = std::numeric_limits<double>::infinity();

  // Ordering 1: Sort by efficiency (probability / exec_cost) - DESCENDING
  std::sort(prob_achievers.begin(), prob_achievers.end(),
      [](const ProbabilisticAchiever& a, const ProbabilisticAchiever& b) {
        return a.efficiency() > b.efficiency();
      });
  min_E_attempt = std::min(min_E_attempt, compute_E_attempt(prob_achievers));

  // Ordering 2: Sort by probability - DESCENDING (highest probability first)
  std::sort(prob_achievers.begin(), prob_achievers.end(),
      [](const ProbabilisticAchiever& a, const ProbabilisticAchiever& b) {
        return a.probability > b.probability;
      });
  min_E_attempt = std::min(min_E_attempt, compute_E_attempt(prob_achievers));

  // Ordering 3: Sort by attempt_cost - ASCENDING (lowest cost first)
  std::sort(prob_achievers.begin(), prob_achievers.end(),
      [](const ProbabilisticAchiever& a, const ProbabilisticAchiever& b) {
        return a.attempt_cost() < b.attempt_cost();
      });
  min_E_attempt = std::min(min_E_attempt, compute_E_attempt(prob_achievers));

  // D_best for probabilistic achievers (the optimistic cost)
  double D_prob_best = std::numeric_limits<double>::infinity();
  for (const auto& a : prob_achievers) {
    D_prob_best = std::min(D_prob_best, a.wait_cost + a.exec_cost);
  }

  // delta = E_attempt - D_best (extra cost due to probabilistic uncertainty)
  double delta = min_E_attempt - D_prob_best;
  if (delta < TOLERANCE) {
    delta = 0.0;
  }

  // Cache and return
  forward.probabilistic_delta[f] = delta;
  return delta;
}

// Backward cost computation given forward results and goal fluents
// Extracts the relaxed plan and sums D(f) + deltas for probabilistic fluents
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

  // Extract the relaxed plan: BFS from goals back to initial fluents
  // Track which fluents are on the extraction path
  std::unordered_set<Fluent> on_path;
  std::unordered_set<Fluent> frontier = goal_fluents;
  std::unordered_map<Fluent, const Action*> extracted_achiever;
  constexpr double TOLERANCE = 1e-9;

  auto pick_best_achiever = [&](const Fluent& f) -> const Action* {
    const Action* best_action = nullptr;
    double best_cost = std::numeric_limits<double>::infinity();
    bool best_is_deterministic = false;
    double best_probability = -1.0;

    auto ach_it = forward.achievers_by_fluent.find(f);
    if (ach_it != forward.achievers_by_fluent.end()) {
      for (const auto& ach : ach_it->second) {
        if (ach.probability <= TOLERANCE) continue;
        double cost = ach.wait_cost + ach.exec_cost;
        if (cost >= std::numeric_limits<double>::infinity()) continue;

        bool is_deterministic = ach.probability >= 1.0 - TOLERANCE;
        bool better = false;

        if (cost + TOLERANCE < best_cost) {
          better = true;
        } else if (cost <= best_cost + TOLERANCE) {
          if (is_deterministic && !best_is_deterministic) {
            better = true;
          } else if (is_deterministic == best_is_deterministic &&
                     ach.probability > best_probability + TOLERANCE) {
            better = true;
          } else if (is_deterministic == best_is_deterministic &&
                     ach.probability >= best_probability - TOLERANCE &&
                     best_action &&
                     ach.exec_cost < forward.action_to_duration.at(best_action) - TOLERANCE) {
            better = true;
          }
        }

        if (better) {
          best_action = ach.action;
          best_cost = cost;
          best_is_deterministic = is_deterministic;
          best_probability = ach.probability;
        }
      }
    }

    if (best_action) return best_action;

    auto fallback = forward.fact_to_action.find(f);
    if (fallback != forward.fact_to_action.end()) {
      return fallback->second;
    }
    return nullptr;
  };

  while (!frontier.empty()) {
    std::unordered_set<Fluent> next_frontier;

    for (const auto& f : frontier) {
      if (on_path.count(f) || forward.initial_fluents.count(f)) continue;
      on_path.insert(f);

      // Add preconditions of the achieving action to the frontier
      const Action* best_action = pick_best_achiever(f);
      if (best_action) {
        extracted_achiever[f] = best_action;
        for (const auto& prec : best_action->pos_preconditions()) {
          next_frontier.insert(prec);
        }
      }
    }

    frontier = std::move(next_frontier);
  }

  // Build a relaxed schedule over extracted achievers.
  // T(f): earliest arrival time for fluent f along extracted achiever links.
  std::unordered_map<Fluent, double> fluent_arrival;
  for (const auto& f : on_path) {
    fluent_arrival[f] = std::numeric_limits<double>::infinity();
  }

  bool changed = true;
  std::size_t iter = 0;
  const std::size_t max_iter = on_path.size() + 2;

  while (changed && iter < max_iter) {
    changed = false;
    ++iter;

    for (const auto& f : on_path) {
      auto ach_it = extracted_achiever.find(f);
      if (ach_it == extracted_achiever.end()) continue;

      const Action* a = ach_it->second;
      double start_time = 0.0;
      bool preconditions_known = true;

      for (const auto& p : a->pos_preconditions()) {
        double p_time = 0.0;
        if (forward.initial_fluents.count(p)) {
          p_time = 0.0;
        } else {
          auto pit = fluent_arrival.find(p);
          if (pit != fluent_arrival.end() &&
              pit->second < std::numeric_limits<double>::infinity()) {
            p_time = pit->second;
          } else {
            // Fallback to D(p) if p was not explicitly extracted.
            auto dit = forward.expected_cost.find(p);
            if (dit != forward.expected_cost.end()) {
              p_time = dit->second;
            } else {
              preconditions_known = false;
              break;
            }
          }
        }
        start_time = std::max(start_time, p_time);
      }

      if (!preconditions_known) continue;

      auto dur_it = forward.action_to_duration.find(a);
      double duration = (dur_it != forward.action_to_duration.end()) ? dur_it->second : 0.0;
      double finish_time = start_time + duration;

      if (finish_time < fluent_arrival[f]) {
        fluent_arrival[f] = finish_time;
        changed = true;
      }
    }
  }

  // Hybrid base cost:
  // makespan term preserves reach+execute critical path,
  // additive term restores gradient across multiple goals.
  constexpr double GOAL_SUM_LAMBDA = 0.5;
  double max_goal_time = 0.0;
  double sum_goal_time = 0.0;
  for (const auto& gf : goal_fluents) {
    if (forward.initial_fluents.count(gf)) continue;

    double goal_time = std::numeric_limits<double>::infinity();
    auto fit = fluent_arrival.find(gf);
    if (fit != fluent_arrival.end()) {
      goal_time = fit->second;
    } else {
      auto dit = forward.expected_cost.find(gf);
      if (dit != forward.expected_cost.end()) {
        goal_time = dit->second;
      }
    }

    if (goal_time >= std::numeric_limits<double>::infinity()) {
      return std::numeric_limits<double>::infinity();
    }
    max_goal_time = std::max(max_goal_time, goal_time);
    sum_goal_time += goal_time;
  }
  double total_cost =
      max_goal_time + GOAL_SUM_LAMBDA * std::max(0.0, sum_goal_time - max_goal_time);

  // Lazily compute and add probabilistic deltas for fluents on the extraction path
  // Only compute for fluents that have probabilistic achievers (skip purely deterministic)
  // Results are cached in forward.probabilistic_delta for reuse across goal branches
  for (const auto& f : on_path) {
    if (forward.has_probabilistic_achiever.count(f)) {
      total_cost += PROBABILISTIC_DELTA_MULTIPLIER * get_or_compute_delta(forward, f);
    }
  }

  return total_cost;
}

// Deterministic backward cost computation (classic FF heuristic)
// Uses ablation to find required goal fluents, then sums action durations
// in the backward relaxed plan. No probabilistic cost adjustments.
double det_ff_backward_cost(
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

// Compute relaxed expected costs for all reachable fluents from a given state
// Returns a map from fluent to expected cost (0 for initial fluents, computed for others)
inline std::unordered_map<Fluent, double> get_relaxed_expected_costs(
    const State &input_state,
    const std::vector<Action> &all_actions) {

  // Step 1: Relaxed transition
  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) {
    return {};
  }
  State relaxed = relaxed_result[0].first;

  // Get initial fluents from relaxed state
  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  // Run forward phase
  auto forward = ff_forward_phase(initial_fluents, all_actions);

  // Compute expected costs via Bellman iteration
  compute_expected_costs(forward);

  return forward.expected_cost;
}

// Debug function: Get achiever information for a fluent
// Returns vector of tuples: (action_name, wait_cost, exec_cost, probability)
inline std::vector<std::tuple<std::string, double, double, double>> get_achievers_for_fluent(
    const State &input_state,
    const Fluent &fluent,
    const std::vector<Action> &all_actions) {

  std::vector<std::tuple<std::string, double, double, double>> achiever_info;

  // Step 1: Relaxed transition
  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) {
    return achiever_info;
  }
  State relaxed = relaxed_result[0].first;

  // Get initial fluents from relaxed state
  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  // Run forward phase
  auto forward = ff_forward_phase(initial_fluents, all_actions);

  // Compute expected costs via Bellman iteration
  compute_expected_costs(forward);

  // Get achievers for the target fluent
  auto it = forward.achievers_by_fluent.find(fluent);
  if (it != forward.achievers_by_fluent.end()) {
    for (const auto& achiever : it->second) {
      achiever_info.emplace_back(
          achiever.action->name(),
          achiever.wait_cost,
          achiever.exec_cost,
          achiever.probability);
    }
  }

  return achiever_info;
}

// Get the relaxed expected cost for a single fluent
// Returns infinity if fluent is unreachable, 0 if already true, otherwise the computed cost
inline double get_relaxed_expected_cost(
    const State &input_state,
    const Fluent &fluent,
    const std::vector<Action> &all_actions) {

  auto costs = get_relaxed_expected_costs(input_state, all_actions);

  auto it = costs.find(fluent);
  if (it != costs.end()) {
    return it->second;
  }

  // Fluent not found - check if it's in initial state (cost 0) or unreachable
  auto relaxed_result = transition(input_state, nullptr, true);
  if (!relaxed_result.empty()) {
    const auto& relaxed_fluents = relaxed_result[0].first.fluents();
    if (relaxed_fluents.count(fluent)) {
      return 0.0;
    }
  }

  return std::numeric_limits<double>::infinity();
}

} // namespace railroad

// Include goal.hpp here to get the full definition of GoalBase
// This is placed after the namespace closes to avoid circular dependencies
#include "railroad/goal.hpp"

namespace railroad {

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

  // Step 1: Relaxed transition for FLUENTS (union of all possible outcomes)
  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  State relaxed = relaxed_result[0].first;

  // Step 2: Non-relaxed transition for TIME (first action completion)
  // This gives a better lower bound since we can act again as soon as any robot finishes
  auto nonrelaxed_result = transition(input_state, nullptr, false);
  double dtime = 0.0;
  if (!nonrelaxed_result.empty()) {
    dtime = nonrelaxed_result[0].first.time() - t0;
  }

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

  // Compute expected costs via Bellman iteration
  compute_expected_costs(forward);

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

// Deterministic FF heuristic (classic fast-forward)
// Uses ablation-based backward cost and relaxed transition time.
// No probabilistic cost adjustments - pure action duration summing.
inline double det_ff_heuristic(const State &input_state,
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

  // Compute minimum cost across all branches using deterministic backward cost
  double min_cost = std::numeric_limits<double>::infinity();

  for (const auto& branch_fluents : branches) {
    double backward_cost = det_ff_backward_cost(forward, branch_fluents);
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

} // namespace railroad
