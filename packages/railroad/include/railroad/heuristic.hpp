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

// ============================================================================
//  Aliases
// ============================================================================

using HeuristicFn = std::function<double(const State &)>;

struct FFCacheKey {
  std::size_t relaxed_state_hash;
  std::size_t goal_hash;
  std::size_t actions_hash;
  std::size_t lambda_add_hash;
  std::size_t lambda_max_hash;
  std::size_t lambda_ff_hash;
  bool at_implies_found;

  bool operator==(const FFCacheKey& other) const {
    return relaxed_state_hash == other.relaxed_state_hash &&
           goal_hash == other.goal_hash &&
           actions_hash == other.actions_hash &&
           lambda_add_hash == other.lambda_add_hash &&
           lambda_max_hash == other.lambda_max_hash &&
           lambda_ff_hash == other.lambda_ff_hash &&
           at_implies_found == other.at_implies_found;
  }
};

struct FFCacheKeyHash {
  std::size_t operator()(const FFCacheKey& key) const {
    std::size_t h = key.relaxed_state_hash;
    hash_combine(h, key.goal_hash);
    hash_combine(h, key.actions_hash);
    hash_combine(h, key.lambda_add_hash);
    hash_combine(h, key.lambda_max_hash);
    hash_combine(h, key.lambda_ff_hash);
    hash_combine(h, std::hash<bool>{}(key.at_implies_found));
    return h;
  }
};

using FFMemory = std::unordered_map<FFCacheKey, double, FFCacheKeyHash>;

// ============================================================================
//  Core data types
// ============================================================================

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

  // For each fluent: the achiever with the smallest exec_cost. Filled during
  // forward reachability and kept as a fallback/debug aid.
  std::unordered_map<Fluent, const Action*> cheapest_achiever;

  // For each fluent: the achiever selected by compute_optimistic_costs().
  // Backward extraction uses this so h_ff follows the same relaxed plan as
  // h_add/h_max.
  std::unordered_map<Fluent, const Action*> best_optimistic_achiever;

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

// ============================================================================
//  Probabilistic extension
// ============================================================================
// Pulled in here rather than at the top of the file: it depends on `Achiever`
// and `FFForwardResult` defined above, and it provides augment_at_with_found(),
// used by the backward extraction immediately below. It must not include this
// header back.
#include "railroad/heuristic_prob.hpp"

namespace railroad {

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

// ============================================================================
//  Python / introspection query helpers
// ============================================================================

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

// Goal definitions are pulled in after the heuristic primitives so that
// `ff_heuristic` can dispatch on goal type. (Goal API has a circular
// dependency on Fluent/State that we side-step by including it last.)
#include "railroad/goal.hpp"

namespace railroad {

// ============================================================================
//  Goal API + main entry point
// ============================================================================

// Pull the cached DNF branches off a goal. Distribution of OR over AND
// (e.g., AND(A, OR(B,C)) -> [{A,B}, {A,C}]) is handled by the goal itself.
inline const std::vector<std::unordered_set<Fluent>>& extract_or_branches(const GoalBase* goal) {
  static const std::vector<std::unordered_set<Fluent>> empty_branches;
  if (!goal) return empty_branches;
  return goal->get_dnf_branches();
}

inline std::size_t hash_action_set_for_heuristic(const std::vector<Action>& all_actions) {
  std::size_t h = all_actions.size();
  std::size_t xor_hash = 0;
  std::size_t sum_hash = 0;

  for (const auto& action : all_actions) {
    std::size_t action_hash = action.hash();
    hash_combine(action_hash, 0);
    xor_hash ^= action_hash;
    sum_hash += action_hash;
  }

  hash_combine(h, xor_hash);
  hash_combine(h, sum_hash);
  return h;
}

inline FFCacheKey make_ff_cache_key(const State& relaxed,
                                    const GoalBase* goal,
                                    const std::vector<Action>& all_actions,
                                    double lambda_add,
                                    double lambda_max,
                                    double lambda_ff,
                                    bool at_implies_found) {
  return {
      relaxed.hash(),
      goal ? goal->hash() : 0,
      hash_action_set_for_heuristic(all_actions),
      std::hash<double>{}(lambda_add),
      std::hash<double>{}(lambda_max),
      std::hash<double>{}(lambda_ff),
      at_implies_found,
  };
}

// Main FF heuristic.
//
// The relaxed-plan extraction produces three component values:
//   h_add: Σ optimistic_cost over goal fluents (classic additive)
//   h_max: max optimistic_cost over goal fluents
//   h_ff:  Σ action_duration over unique actions on the relaxed plan
// These are mixed via the lambda_* weights (free-form, not normalized).
// The probabilistic-retry delta is added once per branch *after* mixing.
// Defaults are an even split between h_add and h_ff (0.5, 0.0, 0.5).
//
// Layout:
//   1. Relaxed transition for fluents (union over outcomes).
//   2. Non-relaxed transition just to read out the time of the next robot
//      completion — gives a tighter dtime lower bound than the relaxed step.
//   3. Forward reachability + optimistic costs.
//   4. For each DNF branch of the goal: backward extraction of all three
//      component values, mix with lambdas, add probabilistic-delta retries
//      for fluents on the relaxed plan that have probabilistic achievers.
//      Take the minimum across branches.
inline double ff_heuristic(const State &input_state,
                           const GoalBase *goal,
                           const std::vector<Action> &all_actions,
                           FFMemory *ff_memory = nullptr,
                           double lambda_add = 0.5,
                           double lambda_max = 0.0,
                           double lambda_ff  = 0.5,
                           bool at_implies_found = true) {
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

  // Memoization key: relaxed-state fluents at time 0. The cached value is the
  // already-mixed branch minimum. Include the goal, action universe, lambda
  // weights, and augmentation policy because all of them affect that value.
  relaxed.set_time(0);
  std::optional<FFCacheKey> cache_key;
  if (ff_memory) {
    cache_key = make_ff_cache_key(relaxed, goal, all_actions,
                                  lambda_add, lambda_max, lambda_ff,
                                  at_implies_found);
    auto cached_it = ff_memory->find(*cache_key);
    if (cached_it != ff_memory->end()) {
      return dtime + cached_it->second;
    }
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
    auto opt = ff_backward_optimistic(forward, branch, at_implies_found);
    if (opt.h_add == std::numeric_limits<double>::infinity()) continue;  // unreachable branch

    double delta_total = relaxed_plan_prob_delta(forward, opt.on_path);
    double mixed = lambda_add * opt.h_add
                 + lambda_max * opt.h_max
                 + lambda_ff  * opt.h_ff;
    min_cost = std::min(min_cost, mixed + delta_total);
  }

  if (ff_memory && cache_key) {
    (*ff_memory)[*cache_key] = min_cost;
  }

  return dtime + min_cost;
}

} // namespace railroad
