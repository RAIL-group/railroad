#pragma once

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

  // For each fluent: the achiever with the smallest exec_cost. Used by the
  // backward extraction to walk the relaxed plan one step at a time.
  std::unordered_map<Fluent, const Action*> cheapest_achiever;

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

  for (const auto& f : initial_fluents) {
    result.optimistic_cost[f] = 0.0;
  }

  std::unordered_set<Fluent> newly_added = result.known_fluents;
  std::unordered_set<const Action *> visited_actions;
  std::unordered_set<const Action *> all_actions_set;
  for (const auto &a : all_actions) {
    all_actions_set.insert(&a);
  }

  while (!newly_added.empty()) {
    std::unordered_set<Fluent> next_new;
    State state_all_known(0.0, result.known_fluents);

    for (const Action *a : all_actions_set) {
      if (visited_actions.count(a)) continue;
      if (!state_all_known.satisfies_precondition(*a, true)) continue;

      const auto& succs = a->get_relaxed_successors();
      visited_actions.insert(a);
      if (succs.empty()) continue;

      // Aggregate per-fluent achievement probability across this action's
      // branches: branches are mutually exclusive, so the chance the action
      // produces fluent f is the sum of branch probabilities that contain f.
      // If that sum is ~1, the action is an effectively deterministic
      // achiever of f even when every individual branch has prob < 1.
      std::unordered_map<Fluent, double> fluent_prob;
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

        if (!result.known_fluents.count(f)) {
          result.known_fluents.insert(f);
          next_new.insert(f);
          result.cheapest_achiever[f] = a;
        } else if (duration < result.action_duration[result.cheapest_achiever[f]]) {
          result.cheapest_achiever[f] = a;
        }
      }
    }

    newly_added = std::move(next_new);
    for (const Action *a : visited_actions) {
      all_actions_set.erase(a);
    }
  }

  return result;
}

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

      for (const auto& achiever : achievers) {
        double cost = achiever.attempt_cost();
        if (achiever.probability >= 1.0 - TOLERANCE) {
          cost_det = std::min(cost_det, cost);
        } else if (achiever.probability > TOLERANCE) {
          if (achiever.probability > best_prob ||
              (achiever.probability >= best_prob - TOLERANCE && cost < cost_prob)) {
            cost_prob = cost;
            best_prob = achiever.probability;
          }
        }
      }

      bool use_det = cost_det < std::numeric_limits<double>::infinity();
      double new_cost = use_det ? cost_det : cost_prob;

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

// Result of the optimistic backward extraction.
struct FFBackwardResult {
  double cost;                          // sum of optimistic_cost over goal_fluents
  std::unordered_set<Fluent> on_path;   // every fluent visited while walking back
};

// Walk back from `goal_fluents` via cheapest_achiever, summing the optimistic
// cost of each goal fluent and recording every fluent on the relaxed plan.
// Returns +inf cost if any goal fluent is unreachable.
inline FFBackwardResult ff_backward_optimistic(
    const FFForwardResult &forward,
    const std::unordered_set<Fluent> &goal_fluents) {

  if (goal_fluents.empty()) {
    return {0.0, {}};
  }

  for (const auto& gf : goal_fluents) {
    if (!forward.known_fluents.count(gf)) {
      return {std::numeric_limits<double>::infinity(), {}};
    }
  }

  std::unordered_set<Fluent> on_path;
  std::unordered_set<Fluent> frontier = goal_fluents;
  while (!frontier.empty()) {
    std::unordered_set<Fluent> next_frontier;
    for (const auto& f : frontier) {
      if (on_path.count(f) || forward.initial_fluents.count(f)) continue;
      on_path.insert(f);

      auto it = forward.cheapest_achiever.find(f);
      if (it != forward.cheapest_achiever.end()) {
        for (const auto& prec : it->second->pos_preconditions()) {
          next_frontier.insert(prec);
        }
      }
    }
    frontier = std::move(next_frontier);
  }

  double total = 0.0;
  for (const auto& gf : goal_fluents) {
    if (forward.initial_fluents.count(gf)) continue;
    auto it = forward.optimistic_cost.find(gf);
    if (it != forward.optimistic_cost.end()) {
      total += it->second;
    }
  }

  return {total, std::move(on_path)};
}

// Get usable actions via forward relaxed reachability.
inline const std::vector<Action> get_usable_actions(const State &input_state,
                                                    const std::vector<Action> &all_actions) {
  std::unordered_set<const Action*> feasible_action_set;

  // Pass 1: relaxed transition (processes upcoming effects).
  auto relaxed_result = transition(input_state, nullptr, true);
  if (!relaxed_result.empty()) {
    State relaxed = relaxed_result[0].first;
    std::unordered_set<Fluent> initial_fluents(
        relaxed.fluents().begin(), relaxed.fluents().end());

    auto forward = ff_forward_phase(initial_fluents, all_actions);
    State state_all_known(0.0, forward.known_fluents);
    for (const auto& a : all_actions) {
      if (state_all_known.satisfies_precondition(a, true)) {
        feasible_action_set.insert(&a);
      }
    }
  }

  // Pass 2: also consider current fluents WITHOUT processing upcoming
  // effects — handles cases where upcoming effects would mask actions that
  // remain valid for other robots (e.g., another robot can still move to a
  // location before it is marked visited).
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

  std::vector<Action> feasible_actions;
  feasible_actions.reserve(feasible_action_set.size());
  for (const Action* a : feasible_action_set) {
    feasible_actions.push_back(*a);
  }
  return feasible_actions;
}

// Compute optimistic costs for every reachable fluent from a given state.
// Returns a map from fluent to optimistic cost (0 for initial fluents).
inline std::unordered_map<Fluent, double> get_relaxed_optimistic_costs(
    const State &input_state,
    const std::vector<Action> &all_actions) {

  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) return {};
  State relaxed = relaxed_result[0].first;

  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  auto forward = ff_forward_phase(initial_fluents, all_actions);
  compute_optimistic_costs(forward);
  return forward.optimistic_cost;
}

// Debug helper: list the achievers (action_name, wait_cost, exec_cost, prob)
// for `fluent` from the relaxed reachability of `input_state`.
inline std::vector<std::tuple<std::string, double, double, double>> get_achievers_for_fluent(
    const State &input_state,
    const Fluent &fluent,
    const std::vector<Action> &all_actions) {

  std::vector<std::tuple<std::string, double, double, double>> info;

  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) return info;
  State relaxed = relaxed_result[0].first;

  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  auto forward = ff_forward_phase(initial_fluents, all_actions);
  compute_optimistic_costs(forward);

  auto it = forward.achievers_by_fluent.find(fluent);
  if (it != forward.achievers_by_fluent.end()) {
    for (const auto& a : it->second) {
      info.emplace_back(a.action->name(), a.wait_cost, a.exec_cost, a.probability);
    }
  }
  return info;
}

// Optimistic cost for a single fluent. +inf if unreachable, 0 if already true.
inline double get_relaxed_optimistic_cost(
    const State &input_state,
    const Fluent &fluent,
    const std::vector<Action> &all_actions) {

  auto costs = get_relaxed_optimistic_costs(input_state, all_actions);
  auto it = costs.find(fluent);
  if (it != costs.end()) return it->second;

  // Either already in the initial relaxed state (cost 0) or unreachable.
  auto relaxed_result = transition(input_state, nullptr, true);
  if (!relaxed_result.empty() &&
      relaxed_result[0].first.fluents().count(fluent)) {
    return 0.0;
  }
  return std::numeric_limits<double>::infinity();
}

} // namespace railroad

// Probabilistic delta extension. Included here (not at top) because it
// depends on FFForwardResult and Achiever defined above.
#include "railroad/heuristic_prob.hpp"

// Goal definitions are pulled in after the heuristic primitives so that
// `ff_heuristic` can dispatch on goal type. (Goal API has a circular
// dependency on Fluent/State that we side-step by including it last.)
#include "railroad/goal.hpp"

namespace railroad {

// Pull the cached DNF branches off a goal. Distribution of OR over AND
// (e.g., AND(A, OR(B,C)) -> [{A,B}, {A,C}]) is handled by the goal itself.
inline const std::vector<std::unordered_set<Fluent>>& extract_or_branches(const GoalBase* goal) {
  static const std::vector<std::unordered_set<Fluent>> empty_branches;
  if (!goal) return empty_branches;
  return goal->get_dnf_branches();
}

// Main FF heuristic.
//
// Layout:
//   1. Relaxed transition for fluents (union over outcomes).
//   2. Non-relaxed transition just to read out the time of the next robot
//      completion — gives a tighter dtime lower bound than the relaxed step.
//   3. Forward reachability + optimistic costs.
//   4. For each DNF branch of the goal: optimistic backward cost, then add
//      probabilistic-delta retries for any fluents on the relaxed plan that
//      have probabilistic achievers. Take the minimum across branches.
inline double ff_heuristic(const State &input_state,
                           const GoalBase *goal,
                           const std::vector<Action> &all_actions,
                           FFMemory *ff_memory = nullptr) {
  if (!goal) return 0.0;

  GoalType type = goal->get_type();
  if (type == GoalType::TRUE_GOAL) return 0.0;
  if (type == GoalType::FALSE_GOAL) {
    return std::numeric_limits<double>::infinity();
  }

  const double t0 = input_state.time();

  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  State relaxed = relaxed_result[0].first;

  auto nonrelaxed_result = transition(input_state, nullptr, false);
  double dtime = 0.0;
  if (!nonrelaxed_result.empty()) {
    dtime = nonrelaxed_result[0].first.time() - t0;
  }

  // Memoization key: relaxed-state fluents at time 0.
  relaxed.set_time(0);
  if (ff_memory && ff_memory->count(relaxed.hash())) {
    return dtime + ff_memory->at(relaxed.hash());
  }

  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  auto forward = ff_forward_phase(initial_fluents, all_actions);
  compute_optimistic_costs(forward);

  auto branches = extract_or_branches(goal);
  if (branches.empty()) {
    return std::numeric_limits<double>::infinity();  // FalseGoal-like
  }

  double min_cost = std::numeric_limits<double>::infinity();
  for (const auto& branch : branches) {
    auto opt = ff_backward_optimistic(forward, branch);
    if (opt.cost == std::numeric_limits<double>::infinity()) continue;

    double delta_total = 0.0;
    for (const auto& f : opt.on_path) {
      if (forward.has_probabilistic_achiever.count(f)) {
        delta_total += get_or_compute_delta(forward, f);
      }
    }
    min_cost = std::min(min_cost, opt.cost + delta_total);
  }

  if (ff_memory) {
    (*ff_memory)[relaxed.hash()] = min_cost;
  }

  return dtime + min_cost;
}

} // namespace railroad
