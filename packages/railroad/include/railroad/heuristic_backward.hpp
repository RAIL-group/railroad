#pragma once

// Backward relaxed-plan extraction for the FF heuristic, plus the
// "at implies found" goal augmentation it relies on.
//
// augment_at_with_found() is a goal-augmentation concern (it needs only
// FFForwardResult + Fluent, not the probabilistic machinery), so it lives
// here with its sole caller rather than in heuristic_prob.hpp.

#include "railroad/heuristic_types.hpp"

#include <limits>
#include <unordered_set>
#include <vector>

namespace railroad {

// ============================================================================
//  Goal augmentation ("at implies found")
// ============================================================================

// "at implies found": for each positive `at <entity> <loc>` fluent in
// `fluents`, also require `found <entity>` -- but only when `found <entity>`
// is reachable in the relaxed planning graph. An entity whose `found` fluent
// is unreachable (e.g. a robot, which no operator can `found`) is silently
// skipped, so this never introduces an unreachable subgoal.
inline void augment_at_with_found(std::unordered_set<Fluent>& fluents,
                                  const FFForwardResult& forward) {
  std::vector<Fluent> to_add;
  for (const auto& f : fluents) {
    if (f.is_negated()) continue;
    if (f.name() != "at") continue;
    const auto& args = f.args();
    if (args.empty()) continue;
    Fluent found("found", {args[0]});
    if (forward.known_fluents.count(found)) {
      to_add.push_back(std::move(found));
    }
  }
  for (auto& f : to_add) fluents.insert(std::move(f));
}

// ============================================================================
//  Backward relaxed-plan extraction
// ============================================================================

// Result of the optimistic backward extraction.
// All three values are infinite when any goal fluent is unreachable.
struct FFBackwardResult {
  double h_add;                         // sum of optimistic_cost over goal_fluents
  double h_max;                         // max of optimistic_cost over goal_fluents
  double h_ff;                          // sum of action_duration over unique actions on relaxed plan
  std::unordered_set<Fluent> on_path;   // every fluent visited while walking back
};

// Walk back from `goal_fluents` via best_optimistic_achiever, computing three
// relaxed-plan estimates in a single BFS:
//   h_add: Σ optimistic_cost[gf] over goal fluents (classic additive)
//   h_max: max optimistic_cost[gf] over goal fluents
//   h_ff:  Σ action_duration[a] over unique actions visited via best_optimistic_achiever
// `on_path` is the set of fluents the BFS visited (used by caller for
// probabilistic-delta retries). Returns +inf for all three values if any goal
// fluent is unreachable.
inline FFBackwardResult ff_backward_optimistic(
    const FFForwardResult &forward,
    const std::unordered_set<Fluent> &goal_fluents,
    bool at_implies_found = true) {

  FFBackwardResult result{0.0, 0.0, 0.0, {}};
  if (goal_fluents.empty()) return result;

  // Local, possibly-augmented copy of the goal branch. augment_at_with_found
  // only adds reachable fluents, so the unreachability check below stays
  // correct and h_add/h_max/h_ff all see the added `found` subgoal(s).
  std::unordered_set<Fluent> goals = goal_fluents;
  if (at_implies_found) augment_at_with_found(goals, forward);

  for (const auto& gf : goals) {
    if (!forward.known_fluents.count(gf)) {
      double inf = std::numeric_limits<double>::infinity();
      return {inf, inf, inf, {}};
    }
  }

  std::unordered_set<Fluent>& on_path = result.on_path;
  std::unordered_set<const Action*> actions_on_path;
  std::unordered_set<Fluent> frontier = goals;
  while (!frontier.empty()) {
    std::unordered_set<Fluent> next_frontier;
    for (const auto& f : frontier) {
      if (on_path.count(f) || forward.initial_fluents.count(f)) continue;
      on_path.insert(f);

      auto it = forward.best_optimistic_achiever.find(f);
      if (it != forward.best_optimistic_achiever.end()) {
        actions_on_path.insert(it->second);
        for (const auto& prec : it->second->pos_preconditions()) {
          next_frontier.insert(prec);
        }
      }
    }
    // Objects that only appear via an action precondition still imply a
    // `found` subgoal, so the search cost is reflected in h_ff / the
    // probabilistic delta even when `found` is not an explicit goal.
    if (at_implies_found) augment_at_with_found(next_frontier, forward);
    frontier = std::move(next_frontier);
  }

  for (const auto& gf : goals) {
    if (forward.initial_fluents.count(gf)) continue;
    auto it = forward.optimistic_cost.find(gf);
    if (it == forward.optimistic_cost.end()) continue;
    result.h_add += it->second;
    result.h_max = std::max(result.h_max, it->second);
  }
  for (const Action* a : actions_on_path) {
    auto it = forward.action_duration.find(a);
    if (it != forward.action_duration.end()) {
      result.h_ff += it->second;
    }
  }

  return result;
}

}  // namespace railroad
