#pragma once

// Probabilistic retry delta for the FF heuristic. Self-contained: depends
// only on the shared data types (Achiever / FFForwardResult).
//
// The optimistic base cost (heuristic_forward.hpp) assumes the cheapest
// single achiever attempt succeeds first try. When probabilistic achievers
// are involved, that under-counts the expected time. The delta added here is:
//
//   delta(f) = E[time to first success across all achievers of f, executed
//                in some order] - min single-attempt cost across achievers
//
// All achievers participate — deterministic ones (p == 1) act as a
// guaranteed fallback in the ordering, so a cheap-but-flaky probabilistic
// attempt followed by a slower deterministic one is naturally modeled.
//
// Contents:
//   - selected_optimistic_achiever(): the achiever the relaxed plan picked.
//   - get_or_compute_delta(): the probabilistic retry delta for one fluent.
//   - relaxed_plan_prob_delta(): sum of that delta over a relaxed plan.

#include "railroad/heuristic_types.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

// ============================================================================
//  Probabilistic retry delta
// ============================================================================

inline const Achiever* selected_optimistic_achiever(const FFForwardResult& forward,
                                                   const Fluent& f) {
  auto selected_it = forward.best_optimistic_achiever.find(f);
  if (selected_it == forward.best_optimistic_achiever.end()) return nullptr;

  auto achievers_it = forward.achievers_by_fluent.find(f);
  if (achievers_it == forward.achievers_by_fluent.end()) return nullptr;

  for (const auto& achiever : achievers_it->second) {
    if (achiever.action == selected_it->second) {
      return &achiever;
    }
  }
  return nullptr;
}

// Lazily compute (and cache) the probabilistic delta for fluent f.
// Returns 0 for initial fluents and for fluents with no achievers. The
// cache lives on the (mutable) FFForwardResult so the same forward
// result can be reused across goal branches.
inline double get_or_compute_delta(const FFForwardResult& forward, const Fluent& f) {
  const double TOLERANCE = 1e-9;

  auto cached_it = forward.probabilistic_delta.find(f);
  if (cached_it != forward.probabilistic_delta.end()) {
    return cached_it->second;
  }

  if (forward.initial_fluents.count(f)) {
    forward.probabilistic_delta[f] = 0.0;
    return 0.0;
  }

  auto achievers_it = forward.achievers_by_fluent.find(f);
  if (achievers_it == forward.achievers_by_fluent.end()) {
    forward.probabilistic_delta[f] = 0.0;
    return 0.0;
  }

  // Keep every achiever with non-zero success probability. Deterministic
  // achievers (p == 1) are retained so they can act as a guaranteed fallback
  // when a cheaper probabilistic attempt may fail.
  std::vector<Achiever> achievers;
  for (const auto& a : achievers_it->second) {
    if (a.probability > TOLERANCE) {
      achievers.push_back(a);
    }
  }
  if (achievers.empty()) {
    forward.probabilistic_delta[f] = 0.0;
    return 0.0;
  }

  const Achiever* selected = selected_optimistic_achiever(forward, f);
  double selected_attempt = std::numeric_limits<double>::infinity();
  if (selected) {
    selected_attempt = selected->attempt_cost();
  }

  // Expected time to first success when achievers are tried in the given order.
  // Each attempt contributes its cost weighted by the probability that all
  // earlier attempts failed; `time` accumulates so we don't double-count waits.
  auto expected_time_to_success = [](const std::vector<Achiever>& ordered) {
    double total = 0.0;
    double prob_all_failed = 1.0;
    double time = 0.0;
    for (const auto& a : ordered) {
      double dwait = std::max(a.wait_cost - time, 0.0);
      double attempt = dwait + a.exec_cost;
      total += prob_all_failed * attempt;
      prob_all_failed *= (1.0 - a.probability);
      time = std::max(time, a.wait_cost);
    }
    return total;
  };

  // Try a few cheap orderings and take the best. The optimal ordering is
  // problem-dependent; these three cover the common cases.
  double best_E = std::numeric_limits<double>::infinity();

  std::sort(achievers.begin(), achievers.end(),
      [](const Achiever& a, const Achiever& b) { return a.efficiency() > b.efficiency(); });
  best_E = std::min(best_E, expected_time_to_success(achievers));

  std::sort(achievers.begin(), achievers.end(),
      [](const Achiever& a, const Achiever& b) { return a.probability > b.probability; });
  best_E = std::min(best_E, expected_time_to_success(achievers));

  std::sort(achievers.begin(), achievers.end(),
      [](const Achiever& a, const Achiever& b) { return a.attempt_cost() < b.attempt_cost(); });
  best_E = std::min(best_E, expected_time_to_success(achievers));

  // Optimistic base cost: the selected optimistic achiever attempt, assuming
  // it succeeds. Subtracting the same selected cost keeps this correction
  // aligned with the relaxed plan used by h_add/h_max/h_ff.
  if (selected_attempt == std::numeric_limits<double>::infinity()) {
    for (const auto& a : achievers) {
      selected_attempt = std::min(selected_attempt, a.attempt_cost());
    }
  }

  double delta = best_E - selected_attempt;
  if (delta < TOLERANCE) delta = 0.0;

  forward.probabilistic_delta[f] = delta;
  return delta;
}

// Sum the probabilistic retry delta over every fluent on a relaxed plan.
// Fluents with only deterministic achievers contribute nothing, so we skip
// them via has_probabilistic_achiever before paying for get_or_compute_delta.
inline double relaxed_plan_prob_delta(
    const FFForwardResult& forward,
    const std::unordered_set<Fluent>& on_path) {
  double delta_total = 0.0;
  for (const auto& f : on_path) {
    if (!forward.has_probabilistic_achiever.count(f)) continue;

    const Achiever* selected = selected_optimistic_achiever(forward, f);
    if (selected && selected->probability < 1.0 - 1e-9) {
      delta_total += get_or_compute_delta(forward, f);
    }
  }
  return delta_total;
}

} // namespace railroad
