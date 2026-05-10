#pragma once

// Probabilistic-delta extension to the FF heuristic.
//
// This header is included by ff_heuristic.hpp *after* FFForwardResult and
// Achiever are defined; it must not include ff_heuristic.hpp itself.
//
// The optimistic base cost in ff_heuristic.hpp assumes the cheapest single
// achiever attempt succeeds first try. When probabilistic achievers are
// involved, that under-counts the expected time. The delta added here is:
//
//   delta(f) = E[time to first success across all achievers of f, executed
//                in some order] - min single-attempt cost across achievers
//
// All achievers participate — deterministic ones (p == 1) act as a
// guaranteed fallback in the ordering, so a cheap-but-flaky probabilistic
// attempt followed by a slower deterministic one is naturally modeled.

#include <algorithm>
#include <limits>
#include <vector>

namespace railroad {

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

  // Optimistic base cost: the cheapest single attempt across all achievers,
  // assuming it succeeds. Subtracting it leaves only the expected retry /
  // fallback overhead introduced by probabilistic outcomes.
  double min_attempt = std::numeric_limits<double>::infinity();
  for (const auto& a : achievers) {
    min_attempt = std::min(min_attempt, a.attempt_cost());
  }

  double delta = best_E - min_attempt;
  if (delta < TOLERANCE) delta = 0.0;

  forward.probabilistic_delta[f] = delta;
  return delta;
}

} // namespace railroad
