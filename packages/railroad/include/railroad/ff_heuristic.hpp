#pragma once

#include "railroad/constants.hpp"
#include "railroad/core.hpp"
#include "railroad/state.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

using HeuristicFn = std::function<double(const State &)>;
using FFMemory = std::unordered_map<std::size_t, double>;
constexpr double FF_TOLERANCE = 1e-9;

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

struct AchieverScanResult {
  const Action* best_action = nullptr;
  double best_cost = std::numeric_limits<double>::infinity();
  bool best_is_deterministic = false;
  double best_probability = -1.0;
  double best_exec_cost = std::numeric_limits<double>::infinity();
  double det_cost = std::numeric_limits<double>::infinity();
  double prob_cost = std::numeric_limits<double>::infinity();
  double prob_best_probability = 0.0;
};

inline AchieverScanResult scan_achievers(
    const std::vector<ProbabilisticAchiever>& achievers,
    double tolerance = FF_TOLERANCE) {
  AchieverScanResult result;

  for (const auto& ach : achievers) {
    if (ach.probability <= tolerance) continue;

    const double cost = ach.wait_cost + ach.exec_cost;
    if (cost >= std::numeric_limits<double>::infinity()) continue;

    const bool is_deterministic = ach.probability >= 1.0 - tolerance;
    if (is_deterministic) {
      result.det_cost = std::min(result.det_cost, cost);
    } else if (ach.probability > result.prob_best_probability ||
               (ach.probability >= result.prob_best_probability - tolerance &&
                cost < result.prob_cost)) {
      result.prob_cost = cost;
      result.prob_best_probability = ach.probability;
    }

    bool better = false;
    if (cost + tolerance < result.best_cost) {
      better = true;
    } else if (cost <= result.best_cost + tolerance) {
      if (is_deterministic && !result.best_is_deterministic) {
        better = true;
      } else if (is_deterministic == result.best_is_deterministic &&
                 ach.probability > result.best_probability + tolerance) {
        better = true;
      } else if (is_deterministic == result.best_is_deterministic &&
                 ach.probability >= result.best_probability - tolerance &&
                 ach.exec_cost + tolerance < result.best_exec_cost) {
        better = true;
      }
    }

    if (better) {
      result.best_action = ach.action;
      result.best_cost = cost;
      result.best_is_deterministic = is_deterministic;
      result.best_probability = ach.probability;
      result.best_exec_cost = ach.exec_cost;
    }
  }

  return result;
}

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
  std::unordered_map<Fluent, const Action*> best_achiever_action;

  // Fluents that have at least one probabilistic achiever (p < 1.0)
  // Used to skip delta computation for purely deterministic fluents
  std::unordered_set<Fluent> has_probabilistic_achiever;

  // Lazily computed delta(f) = E_attempt - D for probabilistic fluents
  // Mutable so it can be populated on-demand during const backward extraction
  mutable std::unordered_map<Fluent, double> probabilistic_delta;
};

inline const Action* select_best_achiever_action(
    const std::vector<ProbabilisticAchiever>& achievers,
    double tolerance = FF_TOLERANCE) {
  return scan_achievers(achievers, tolerance).best_action;
}

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
  std::unordered_set<Fluent> has_deterministic_achiever;
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
      std::unordered_map<Fluent, double> fluent_prob_mass;
      for (const auto &[succ_state, succ_prob] : succs) {
        duration = std::max(succ_state.time(), duration);
        if (succ_prob <= 0.0) continue;

        for (const auto &f : succ_state.fluents()) {
          fluent_prob_mass[f] += succ_prob;
        }
      }

      for (const auto& [f, prob_mass] : fluent_prob_mass) {
        const double achievement_probability =
            std::min(1.0, std::max(0.0, prob_mass));
        if (achievement_probability <= FF_TOLERANCE) continue;

        // Record one achiever per (action, fluent), with cumulative success probability
        // across all action outcomes that contain the fluent.
        result.achievers_by_fluent[f].push_back({
            a, 0.0, duration, achievement_probability
        });

        if (achievement_probability >= 1.0 - FF_TOLERANCE) {
          has_deterministic_achiever.insert(f);
        } else {
          result.has_probabilistic_achiever.insert(f);
        }

        if (!result.known_fluents.count(f)) {
          result.known_fluents.insert(f);
          next_new.insert(f);
          result.fact_to_action[f] = a;
          result.fact_to_probability[f] = achievement_probability;
        } else {
          result.fact_to_probability[f] =
              std::max(result.fact_to_probability[f], achievement_probability);
          if (duration < result.action_to_duration[result.fact_to_action[f]]) {
            result.fact_to_action[f] = a;
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

  // If any deterministic achiever exists for a fluent, do not treat it as
  // probabilistic for delta computation.
  for (const auto& f : has_deterministic_achiever) {
    result.has_probabilistic_achiever.erase(f);
  }

  return result;
}

// Compute D(f) for all fluents (optimistic, treating all achievers as deterministic)
// Deltas for probabilistic fluents are computed lazily during backward extraction
inline void compute_expected_costs(FFForwardResult& result) {
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
  result.best_achiever_action.clear();

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

      AchieverScanResult scan = scan_achievers(achievers);
      if (scan.best_action) {
        result.best_achiever_action[f] = scan.best_action;
      } else {
        result.best_achiever_action.erase(f);
      }

      // Determine which value to use
      double D_f;
      bool use_det = scan.det_cost < std::numeric_limits<double>::infinity();

      if (use_det) {
        D_f = scan.det_cost;
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
        D_f = scan.prob_cost;
      }

      if (D_f < result.expected_cost[f] - FF_TOLERANCE) {
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

  // If a deterministic achiever exists, uncertainty adds no extra penalty.
  for (const auto& a : achievers) {
    if (a.probability >= 1.0 - FF_TOLERANCE) {
      forward.probabilistic_delta[f] = 0.0;
      return 0.0;
    }
  }

  // Collect probabilistic achievers only
  std::vector<ProbabilisticAchiever> prob_achievers;
  for (const auto& a : achievers) {
    if (a.probability > FF_TOLERANCE && a.probability < 1.0 - FF_TOLERANCE) {
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
  if (delta < FF_TOLERANCE) {
    delta = 0.0;
  }
  delta = min_E_attempt;

  // Cache and return
  forward.probabilistic_delta[f] = delta;
  return delta;
}

struct FFBackwardExtractionResult {
  std::unordered_set<Fluent> effective_goals;
  std::unordered_set<Fluent> on_path;
  std::unordered_map<Fluent, const Action*> extracted_achiever;
  std::unordered_set<Fluent> auto_added_found_landmarks;
};

inline const Action* pick_best_achiever_for_fluent(
    const FFForwardResult& forward,
    const Fluent& f) {
  auto best_it = forward.best_achiever_action.find(f);
  if (best_it != forward.best_achiever_action.end()) {
    return best_it->second;
  }

  auto fallback = forward.fact_to_action.find(f);
  if (fallback != forward.fact_to_action.end()) {
    return fallback->second;
  }
  return nullptr;
}

inline void collect_object_candidates_from_at_fluent(
    const Fluent& f,
    std::unordered_set<std::string>& objects) {
  const auto& args = f.args();
  if (f.name() == "at" && args.size() >= 2) objects.insert(args[0]);
}

inline bool has_finite_found_cost(
    const FFForwardResult& forward,
    const Fluent& found_fluent) {
  if (forward.initial_fluents.count(found_fluent)) {
    return true;
  }
  auto it = forward.expected_cost.find(found_fluent);
  return it != forward.expected_cost.end() &&
         it->second < std::numeric_limits<double>::infinity();
}

inline FFBackwardExtractionResult build_backward_extraction(
    const FFForwardResult& forward,
    const std::unordered_set<Fluent>& goal_fluents) {
  FFBackwardExtractionResult result;
  result.effective_goals = goal_fluents;

  auto recompute_extraction =
      [&](const std::unordered_set<Fluent>& goals,
          std::unordered_set<Fluent>& on_path,
          std::unordered_map<Fluent, const Action*>& extracted_achiever) {
        on_path.clear();
        extracted_achiever.clear();
        std::unordered_set<Fluent> frontier = goals;
        while (!frontier.empty()) {
          std::unordered_set<Fluent> next_frontier;
          for (const auto& f : frontier) {
            if (on_path.count(f) || forward.initial_fluents.count(f)) continue;
            on_path.insert(f);
            const Action* best_action = pick_best_achiever_for_fluent(forward, f);
            if (best_action) {
              extracted_achiever[f] = best_action;
              for (const auto& prec : best_action->pos_preconditions()) {
                next_frontier.insert(prec);
              }
            }
          }
          frontier = std::move(next_frontier);
        }
      };

  constexpr std::size_t MAX_FOUND_AUGMENT_ITERS = 8;
  for (std::size_t iter = 0; iter < MAX_FOUND_AUGMENT_ITERS; ++iter) {
    recompute_extraction(result.effective_goals, result.on_path, result.extracted_achiever);

    std::unordered_set<std::string> object_candidates;
    for (const auto& f : result.effective_goals) {
      collect_object_candidates_from_at_fluent(f, object_candidates);
    }
    for (const auto& f : result.on_path) {
      collect_object_candidates_from_at_fluent(f, object_candidates);
    }

    bool added_any = false;
    for (const auto& object_name : object_candidates) {
      Fluent found_fluent("found " + object_name);
      if (result.effective_goals.count(found_fluent)) continue;
      if (!forward.known_fluents.count(found_fluent)) continue;
      if (!has_finite_found_cost(forward, found_fluent)) continue;

      result.effective_goals.insert(found_fluent);
      result.auto_added_found_landmarks.insert(found_fluent);
      added_any = true;
    }
    if (!added_any) {
      return result;
    }
  }

  recompute_extraction(result.effective_goals, result.on_path, result.extracted_achiever);
  return result;
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

  FFBackwardExtractionResult extraction =
      build_backward_extraction(forward, goal_fluents);

  for (const auto& gf : extraction.effective_goals) {
    if (!forward.known_fluents.count(gf)) {
      return std::numeric_limits<double>::infinity();
    }
  }

  const auto& on_path = extraction.on_path;
  const auto& extracted_achiever = extraction.extracted_achiever;

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
  // additive term adds pressure from the extracted relaxed plan volume.
  constexpr double ACTION_SUM_LAMBDA = 1.0;
  double max_goal_time = 0.0;
  double sum_goal_time = 0.0;
  for (const auto& gf : extraction.effective_goals) {
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

  // Sum execution durations of unique extracted achiever actions.
  // Counting each action once approximates relaxed-plan size while avoiding
  // double-counting when one action supports multiple extracted fluents.
  std::unordered_set<const Action*> unique_extracted_actions;
  double extracted_action_exec_sum = 0.0;
  for (const auto& [_, action] : extracted_achiever) {
    if (!action) continue;
    if (!unique_extracted_actions.insert(action).second) continue;
    auto dur_it = forward.action_to_duration.find(action);
    double duration =
        (dur_it != forward.action_to_duration.end()) ? dur_it->second : 0.0;
    extracted_action_exec_sum += duration;
  }

  double total_cost =
      max_goal_time + ACTION_SUM_LAMBDA *
                          std::max(0.0, extracted_action_exec_sum - max_goal_time);
  total_cost = (extracted_action_exec_sum + max_goal_time)/2;
  total_cost = (extracted_action_exec_sum + sum_goal_time)/2;

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

inline std::string fluent_to_debug_string(const Fluent& f) {
  std::ostringstream out;
  if (f.is_negated()) out << "not ";
  out << f.name();
  for (const auto& arg : f.args()) {
    out << " " << arg;
  }
  return out.str();
}

inline std::vector<Fluent> sort_fluents_for_debug(
    const std::unordered_set<Fluent>& fluents) {
  std::vector<Fluent> out(fluents.begin(), fluents.end());
  std::sort(out.begin(), out.end(), [](const Fluent& a, const Fluent& b) {
    return fluent_to_debug_string(a) < fluent_to_debug_string(b);
  });
  return out;
}

inline std::string ff_heuristic_debug_report(
    const State& input_state,
    const GoalBase* goal,
    const std::vector<Action>& all_actions,
    FFMemory* ff_memory = nullptr) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3);

  if (!goal) {
    out << "FF Debug: null goal -> 0.0\n";
    return out.str();
  }
  GoalType type = goal->get_type();
  if (type == GoalType::TRUE_GOAL) {
    out << "FF Debug: TRUE_GOAL -> 0.0\n";
    return out.str();
  }
  if (type == GoalType::FALSE_GOAL) {
    out << "FF Debug: FALSE_GOAL -> inf\n";
    return out.str();
  }

  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) {
    out << "No relaxed transition outcomes. heuristic=inf\n";
    return out.str();
  }
  State relaxed = relaxed_result[0].first;

  auto nonrelaxed_result = transition(input_state, nullptr, false);
  double dtime = 0.0;
  if (!nonrelaxed_result.empty()) {
    dtime = nonrelaxed_result[0].first.time() - input_state.time();
  }

  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  relaxed.set_time(0);

  const std::size_t memo_key = relaxed.hash();
  bool memo_hit = false;
  double memo_value = std::numeric_limits<double>::infinity();
  if (ff_memory) {
    auto it = ff_memory->find(memo_key);
    if (it != ff_memory->end()) {
      memo_hit = true;
      memo_value = it->second;
    }
  }

  auto forward = ff_forward_phase(initial_fluents, all_actions);
  compute_expected_costs(forward);

  out << "FF Debug Report\n";
  out << "state.time=" << input_state.time()
      << ", base_fluents=" << input_state.fluents().size()
      << ", upcoming_effects=" << input_state.upcoming_effects().size() << "\n";
  out << "memo_key=" << memo_key << ", memo_hit=" << (memo_hit ? "true" : "false");
  if (memo_hit) {
    out << ", memo_value=" << memo_value;
  }
  out << "\n";
  out << "dtime_addend=" << dtime << "\n";
  out << "initial_seed_fluents=" << initial_fluents.size()
      << ", reachable_fluents=" << forward.known_fluents.size() << "\n";
  out << "seed_fluents (t=0 in relaxed view):\n";
  for (const auto& f : sort_fluents_for_debug(initial_fluents)) {
    out << "  - " << fluent_to_debug_string(f) << "\n";
  }

  auto branches = extract_or_branches(goal);
  if (branches.empty()) {
    out << "No DNF branches (unsatisfiable goal)\n";
    return out.str();
  }

  double min_cost = std::numeric_limits<double>::infinity();
  std::size_t min_branch = 0;

  std::size_t branch_index = 0;
  for (const auto& branch_fluents : branches) {
    double branch_cost = ff_backward_cost(forward, branch_fluents);
    bool reachable = branch_cost < std::numeric_limits<double>::infinity();
    if (reachable && branch_cost < min_cost) {
      min_cost = branch_cost;
      min_branch = branch_index;
    }

    out << "\nBranch[" << branch_index << "] ";
    out << (reachable ? "reachable" : "unreachable");
    if (reachable) out << ", backward_cost=" << branch_cost;
    out << "\n";

    FFBackwardExtractionResult extraction =
        build_backward_extraction(forward, branch_fluents);

    auto sorted_goals = sort_fluents_for_debug(branch_fluents);
    out << "  goals (original branch):\n";
    for (const auto& gf : sorted_goals) {
      out << "    - " << fluent_to_debug_string(gf) << "\n";
    }

    auto sorted_auto_added = sort_fluents_for_debug(extraction.auto_added_found_landmarks);
    out << "  auto_added_found_landmarks:\n";
    if (sorted_auto_added.empty()) {
      out << "    - (none)\n";
    } else {
      for (const auto& f : sorted_auto_added) {
        out << "    - " << fluent_to_debug_string(f) << "\n";
      }
    }

    auto sorted_effective_goals = sort_fluents_for_debug(extraction.effective_goals);
    out << "  goals (effective):\n";
    for (const auto& gf : sorted_effective_goals) {
      out << "    - " << fluent_to_debug_string(gf) << "\n";
    }

    const auto& on_path = extraction.on_path;
    const auto& extracted_achiever = extraction.extracted_achiever;
    auto sorted_path = sort_fluents_for_debug(on_path);
    out << "  extracted_path_fluents:\n";
    for (const auto& f : sorted_path) {
      out << "    - " << fluent_to_debug_string(f);
      auto ea_it = extracted_achiever.find(f);
      if (ea_it != extracted_achiever.end() && ea_it->second) {
        out << " <- " << ea_it->second->name();
      }
      out << "\n";
    }

    std::unordered_set<Fluent> delta_fluents;
    std::unordered_map<Fluent, double> weighted_delta_by_fluent;
    double delta_sum = 0.0;
    for (const auto& f : on_path) {
      if (!forward.has_probabilistic_achiever.count(f)) continue;
      delta_fluents.insert(f);
      double d = get_or_compute_delta(forward, f);
      double weighted = PROBABILISTIC_DELTA_MULTIPLIER * d;
      weighted_delta_by_fluent[f] = weighted;
      delta_sum += weighted;
    }

    auto sorted_delta_fluents = sort_fluents_for_debug(delta_fluents);
    out << "  probabilistic_delta_fluents:\n";
    if (sorted_delta_fluents.empty()) {
      out << "    - (none)\n";
    } else {
      for (const auto& f : sorted_delta_fluents) {
        out << "    - " << fluent_to_debug_string(f)
            << ", weighted_delta=" << weighted_delta_by_fluent[f] << "\n";
      }
    }
    out << "  total_weighted_delta=" << delta_sum << "\n";

    std::unordered_set<Fluent> achiever_targets = on_path;
    achiever_targets.insert(branch_fluents.begin(), branch_fluents.end());
    auto sorted_targets = sort_fluents_for_debug(achiever_targets);

    out << "  achievers_considered:\n";
    for (const auto& f : sorted_targets) {
      out << "    " << fluent_to_debug_string(f) << ":\n";
      auto ach_it = forward.achievers_by_fluent.find(f);
      if (ach_it == forward.achievers_by_fluent.end() || ach_it->second.empty()) {
        out << "      - (seed/no achiever required)\n";
        continue;
      }

      std::vector<ProbabilisticAchiever> achievers = ach_it->second;
      std::sort(achievers.begin(), achievers.end(),
                [](const ProbabilisticAchiever& a, const ProbabilisticAchiever& b) {
                  if (a.probability != b.probability) return a.probability > b.probability;
                  if (a.exec_cost != b.exec_cost) return a.exec_cost < b.exec_cost;
                  return a.action->name() < b.action->name();
                });
      for (const auto& ach : achievers) {
        out << "      - action=" << ach.action->name()
            << ", p=" << ach.probability
            << ", wait=" << ach.wait_cost
            << ", exec=" << ach.exec_cost;
        if (ach.probability >= 1.0 - FF_TOLERANCE) {
          out << " (det)";
        } else {
          out << " (prob)";
        }
        out << "\n";
      }
    }

    ++branch_index;
  }

  if (min_cost < std::numeric_limits<double>::infinity()) {
    out << "\nmin_branch=" << min_branch << ", branch_cost=" << min_cost
        << ", dtime_addend=" << dtime
        << ", heuristic_total=" << (dtime + min_cost) << "\n";
    if (ff_memory) {
      (*ff_memory)[memo_key] = min_cost;
    }
  } else {
    out << "\nNo reachable branch. heuristic=inf\n";
  }

  return out.str();
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

} // namespace railroad
