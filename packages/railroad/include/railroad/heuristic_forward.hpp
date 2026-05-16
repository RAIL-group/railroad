#pragma once

// Forward relaxed reachability + optimistic-cost fixed point for the FF
// heuristic. Depends only on the shared data types.

#include "railroad/heuristic_types.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

// ============================================================================
//  Forward relaxed reachability
// ============================================================================

// Forward relaxed reachability: discover every fluent reachable from
// `initial_fluents` (the post-relaxed-transition state) and record every
// achiever for each fluent. Does not compute optimistic_cost — call
// compute_optimistic_costs() afterwards if you need it.
inline FFForwardResult ff_forward_phase(
    const std::unordered_set<Fluent> &initial_fluents,
    const std::vector<Action> &all_actions) {

  FFForwardResult result;
  result.initial_fluents = initial_fluents;
  result.known_fluents = initial_fluents;
  result.achievers_by_fluent.reserve(initial_fluents.size() + all_actions.size());
  result.cheapest_achiever.reserve(all_actions.size());
  result.best_optimistic_achiever.reserve(all_actions.size());
  result.action_duration.reserve(all_actions.size());
  result.optimistic_cost.reserve(initial_fluents.size() + all_actions.size());
  result.has_probabilistic_achiever.reserve(all_actions.size());

  for (const auto& f : initial_fluents) {
    result.optimistic_cost[f] = 0.0;
  }

  std::unordered_map<Fluent, std::vector<const Action*>> actions_by_missing_precondition;
  actions_by_missing_precondition.reserve(all_actions.size());
  std::unordered_map<const Action*, std::size_t> unmet_preconditions;
  unmet_preconditions.reserve(all_actions.size());
  std::vector<const Action*> ready_actions;
  ready_actions.reserve(all_actions.size());

  for (const auto &a : all_actions) {
    std::size_t unmet = 0;
    for (const auto& prec : a.pos_preconditions()) {
      if (!result.known_fluents.count(prec)) {
        ++unmet;
        actions_by_missing_precondition[prec].push_back(&a);
      }
    }
    if (unmet == 0) {
      ready_actions.push_back(&a);
    } else {
      unmet_preconditions[&a] = unmet;
    }
  }

  for (std::size_t ready_index = 0; ready_index < ready_actions.size(); ++ready_index) {
    const Action *a = ready_actions[ready_index];
    const auto& succs = a->get_relaxed_successors();
    if (succs.empty()) continue;

    // Aggregate per-fluent achievement probability across this action's
    // branches: branches are mutually exclusive, so the chance the action
    // produces fluent f is the sum of branch probabilities that contain f.
    // If that sum is ~1, the action is an effectively deterministic
    // achiever of f even when every individual branch has prob < 1.
    std::unordered_map<Fluent, double> fluent_prob;
    fluent_prob.reserve(succs.size() * 4);
    double duration = 0;
    for (const auto &[succ_state, succ_prob] : succs) {
      duration = std::max(succ_state.time(), duration);
      if (succ_prob <= 0.0) continue;
      for (const auto &f : succ_state.fluents()) {
        fluent_prob[f] += succ_prob;
      }
    }
    result.action_duration[a] = duration;

    for (const auto& [f, total_prob] : fluent_prob) {
      // Clamp to 1.0 to absorb floating-point overshoot when branches sum to 1.
      double prob = std::min(total_prob, 1.0);

      // wait_cost set to 0 here; compute_optimistic_costs fills it in.
      result.achievers_by_fluent[f].push_back({a, 0.0, duration, prob});

      if (prob < 1.0 - 1e-9) {
        result.has_probabilistic_achiever.insert(f);
      }

      auto insert_result = result.known_fluents.insert(f);
      if (!result.initial_fluents.count(f)) {
        auto cheapest_it = result.cheapest_achiever.find(f);
        if (cheapest_it == result.cheapest_achiever.end() ||
            duration < result.action_duration[cheapest_it->second]) {
          result.cheapest_achiever[f] = a;
        }
      }

      if (insert_result.second) {
        auto waiting_it = actions_by_missing_precondition.find(f);
        if (waiting_it == actions_by_missing_precondition.end()) continue;

        for (const Action* waiting_action : waiting_it->second) {
          auto unmet_it = unmet_preconditions.find(waiting_action);
          if (unmet_it == unmet_preconditions.end()) continue;
          --unmet_it->second;
          if (unmet_it->second == 0) {
            ready_actions.push_back(waiting_action);
            unmet_preconditions.erase(unmet_it);
          }
        }
      }
    }
  }

  return result;
}

// ============================================================================
//  Optimistic cost fixed point
// ============================================================================

// Fixed-point iteration that fills in result.optimistic_cost.
//
// For each fluent we update every achiever's wait_cost to the max
// optimistic_cost of its positive preconditions, then pick:
//   - the cheapest deterministic achiever (p >= 1) if any exists, or
//   - the best probabilistic achiever (highest p, then lowest cost).
// The first time a fluent gains a deterministic achiever we force-adopt the
// deterministic value even if it's higher than the prior probabilistic one,
// since the optimistic estimate prefers retry-free achievers.
inline void compute_optimistic_costs(FFForwardResult& result) {
  const double TOLERANCE = 1e-9;
  const int MAX_ITERATIONS = 100;

  for (const auto& [f, _achievers] : result.achievers_by_fluent) {
    if (!result.initial_fluents.count(f)) {
      result.optimistic_cost[f] = std::numeric_limits<double>::infinity();
    }
  }

  std::unordered_set<Fluent> has_det_achiever;

  bool changed = true;
  int iteration = 0;
  while (changed && iteration < MAX_ITERATIONS) {
    changed = false;
    iteration++;

    for (auto& [f, achievers] : result.achievers_by_fluent) {
      if (result.initial_fluents.count(f)) continue;

      // Refresh wait_cost using the latest optimistic_cost values.
      for (auto& achiever : achievers) {
        double max_prec_cost = 0.0;
        for (const auto& prec : achiever.action->pos_preconditions()) {
          auto it = result.optimistic_cost.find(prec);
          if (it != result.optimistic_cost.end()) {
            max_prec_cost = std::max(max_prec_cost, it->second);
          }
        }
        achiever.wait_cost = max_prec_cost;
      }

      double cost_det = std::numeric_limits<double>::infinity();
      double cost_prob = std::numeric_limits<double>::infinity();
      double best_prob = 0.0;
      const Action* best_det_action = nullptr;
      const Action* best_prob_action = nullptr;

      for (const auto& achiever : achievers) {
        double cost = achiever.attempt_cost();
        if (achiever.probability >= 1.0 - TOLERANCE) {
          if (cost < cost_det) {
            cost_det = cost;
            best_det_action = achiever.action;
          }
        } else if (achiever.probability > TOLERANCE) {
          if (achiever.probability > best_prob ||
              (achiever.probability >= best_prob - TOLERANCE && cost < cost_prob)) {
            cost_prob = cost;
            best_prob = achiever.probability;
            best_prob_action = achiever.action;
          }
        }
      }

      bool use_det = cost_det < std::numeric_limits<double>::infinity();
      double new_cost = use_det ? cost_det : cost_prob;
      const Action* selected_action = use_det ? best_det_action : best_prob_action;
      if (selected_action) {
        result.best_optimistic_achiever[f] = selected_action;
      }

      if (use_det) {
        // First time this fluent has a deterministic achiever — adopt it
        // regardless of prior (probabilistic) cost, then mark and continue.
        bool was_prob_only = !has_det_achiever.count(f) &&
                             result.optimistic_cost[f] < std::numeric_limits<double>::infinity();
        has_det_achiever.insert(f);
        if (was_prob_only) {
          result.optimistic_cost[f] = new_cost;
          changed = true;
          continue;
        }
      }

      if (new_cost < result.optimistic_cost[f] - TOLERANCE) {
        result.optimistic_cost[f] = new_cost;
        changed = true;
      }
    }
  }
}

}  // namespace railroad
