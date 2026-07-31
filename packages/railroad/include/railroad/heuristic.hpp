#pragma once

// FF heuristic umbrella header.
//
// The heuristic is split across self-contained headers along its dependency
// DAG; this header pulls them together in order and provides the public
// entry point (ff_heuristic) plus the Python/introspection query helpers.
//
//   heuristic_types.hpp    aliases, Achiever, FFForwardResult
//   heuristic_forward.hpp  ff_forward_phase, compute_optimistic_costs
//   heuristic_prob.hpp     probabilistic retry delta
//   heuristic_backward.hpp augment_at_with_found, ff_backward_optimistic
//   goal.hpp               GoalBase (only ff_heuristic / extract_or_branches
//                          need it; included here so it stays out of the
//                          lower-level heuristic headers)

#include "railroad/heuristic_types.hpp"
#include "railroad/heuristic_forward.hpp"
#include "railroad/heuristic_prob.hpp"
#include "railroad/heuristic_backward.hpp"
#include "railroad/goal.hpp"

#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

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

// Above this many DNF branches, ff_heuristic picks one greedily instead of
// enumerating (see select_cheapest_branch). Not a slow path being avoided:
// the DNF is a *product*, so past a few thousand branches materialising them
// exhausts memory. Below the cap the exhaustive minimum remains the contract.
inline constexpr std::size_t MAX_ENUMERATED_DNF_BRANCHES = 1024;

// Greedily choose one conjunctive goal set without building the DNF: union
// the children at each AND, keep the cheapest disjunct at each OR. `out` is
// both the accumulator and the set of literals already committed.
//
// Returns false when the goal is unreachable. That verdict is *exact*: the
// exhaustive minimum is infinite iff every branch holds an unreachable
// literal, which -- the choices at each OR being independent -- holds iff
// some OR has all its disjuncts unreachable. The reachability test
// (known_fluents) is the one ff_backward_optimistic applies.
//
// Disjuncts are ranked by *marginal* h_add: the summed optimistic_cost of the
// literals they add beyond what is already committed. Scoring in isolation
// would charge full price for literals a sibling conjunct already requires,
// which is the main way a per-disjunct score misranks. Summed (not max) cost
// is what makes this h_add exactly, and h_add is the component the result
// weights (lambda_max defaults to 0).
//
// Only the *selection* is greedy: the chosen conjunction still goes through
// the ordinary backward pass, so shared actions are counted once.
inline bool select_cheapest_branch(const GoalBase* goal,
                                   const FFForwardResult& forward,
                                   std::unordered_set<Fluent>& out) {
  if (!goal) return false;

  // Cost of the literals in `candidate` that `committed` does not already
  // hold. `candidate` is always a superset of `committed`, so this is the
  // price of the disjunct that produced it.
  auto marginal_cost = [&forward](const std::unordered_set<Fluent>& candidate,
                                  const std::unordered_set<Fluent>& committed) {
    double total = 0.0;
    for (const auto& f : candidate) {
      if (committed.count(f)) continue;  // already paid for by a sibling
      auto it = forward.optimistic_cost.find(f);
      if (it == forward.optimistic_cost.end())
        return std::numeric_limits<double>::infinity();
      total += it->second;
    }
    return total;
  };

  switch (goal->get_type()) {
    case GoalType::TRUE_GOAL:
      return true;  // contributes no requirements
    case GoalType::FALSE_GOAL:
      return false;
    case GoalType::LITERAL: {
      const auto& lits = goal->get_all_literals();
      for (const auto& f : lits) {
        if (!forward.known_fluents.count(f)) return false;
      }
      out.insert(lits.begin(), lits.end());
      return true;
    }
    case GoalType::AND: {
      // Two passes so that choice-free children commit first. A subtree with
      // dnf_branch_count() == 1 contributes the same literals no matter what
      // is picked elsewhere, so committing it up front costs nothing and
      // gives every OR decision below the largest possible committed set to
      // measure its marginal cost against.
      for (int pass = 0; pass < 2; ++pass) {
        for (const auto& child : goal->children()) {
          bool choice_free = child->dnf_branch_count() <= 1;
          if (choice_free != (pass == 0)) continue;
          if (!select_cheapest_branch(child.get(), forward, out)) return false;
        }
      }
      return true;
    }
    case GoalType::OR: {
      std::unordered_set<Fluent> best;
      double best_cost = std::numeric_limits<double>::infinity();
      bool found = false;
      for (const auto& child : goal->children()) {
        // Seed from `out` so nested ORs inside this disjunct see the same
        // committed set and score their own choices against it too.
        std::unordered_set<Fluent> candidate = out;
        if (!select_cheapest_branch(child.get(), forward, candidate)) continue;
        double cost = marginal_cost(candidate, out);
        if (!found || cost < best_cost) {
          best = std::move(candidate);
          best_cost = cost;
          found = true;
        }
      }
      if (!found) return false;  // every disjunct unreachable
      out = std::move(best);     // already contains everything `out` held
      return true;
    }
  }
  return false;
}

// The branches a heuristic pass should walk: the full DNF while that is
// affordable, otherwise the single greedily-chosen conjunction. Points at the
// goal's cached DNF in the common case, so the small path copies nothing.
//
// Above the cap, callers that take a *minimum* over branches (ff_heuristic)
// get an upper bound on the true minimum, and callers that take a *union*
// over branches (the introspection helpers) get a subset of the true union.
// Both are degradations that only apply to goals which are otherwise
// impossible to evaluate at all.
class BranchView {
public:
  BranchView(const GoalBase* goal, const FFForwardResult& forward) {
    if (!goal) { source_ = &empty(); return; }
    const std::size_t count = goal->dnf_branch_count();

    // Zero branches is FalseGoal-like: the DNF is empty, so there is nothing
    // to build. This must be tested *before* the cap, not folded into it.
    // AND's count is a saturating product and sat_mul annihilates on a zero,
    // so `AND(<huge subtree>, FalseGoal)` counts 0 -- under the cap -- while
    // the subtree it multiplies against is arbitrarily large. Taking the
    // enumerating path there would call get_dnf_branches(), which
    // materialises children in declaration order and would expand that
    // subtree in full before the FalseGoal short-circuits it, which is
    // exactly the out-of-memory abort the cap exists to prevent.
    if (count == 0) { source_ = &empty(); return; }

    if (count <= MAX_ENUMERATED_DNF_BRANCHES) {
      source_ = &goal->get_dnf_branches();
      return;
    }
    std::unordered_set<Fluent> chosen;
    if (select_cheapest_branch(goal, forward, chosen)) {
      storage_.push_back(std::move(chosen));
    }
    source_ = &storage_;
  }

  BranchView(const BranchView&) = delete;
  BranchView& operator=(const BranchView&) = delete;

  const std::vector<std::unordered_set<Fluent>>& get() const { return *source_; }

private:
  static const std::vector<std::unordered_set<Fluent>>& empty() {
    static const std::vector<std::unordered_set<Fluent>> value;
    return value;
  }
  std::vector<std::unordered_set<Fluent>> storage_;
  const std::vector<std::unordered_set<Fluent>>* source_ = nullptr;
};

// For each probabilistic fluent on the relaxed path to `goal`, return its
// achievers as (action_name, probability, exec_cost, wait_cost). "On the path"
// means the fluent appears in the backward extraction (on_path) of some DNF
// branch; "probabilistic" means it has at least one achiever with probability
// < 1 (forward.has_probabilistic_achiever). This is exactly the set of
// fluents/achievers an action-pruning step should bound per robot.
inline std::unordered_map<Fluent, std::vector<std::tuple<std::string, double, double, double>>>
get_probabilistic_path_achievers(const State &input_state,
                                 const GoalBase *goal,
                                 const std::vector<Action> &all_actions,
                                 bool at_implies_found = true) {

  std::unordered_map<Fluent, std::vector<std::tuple<std::string, double, double, double>>> result;
  if (!goal) return result;

  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) return result;
  State relaxed = relaxed_result[0].first;

  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  auto forward = ff_forward_phase(initial_fluents, all_actions);
  compute_optimistic_costs(forward);

  // Union of on-path fluents across DNF branches (a subset of that union
  // above the enumeration cap; see BranchView).
  std::unordered_set<Fluent> on_path;
  BranchView branch_view(goal, forward);
  for (const auto& branch : branch_view.get()) {
    auto opt = ff_backward_optimistic(forward, branch, at_implies_found);
    if (opt.h_add == std::numeric_limits<double>::infinity()) continue;  // unreachable branch
    on_path.insert(opt.on_path.begin(), opt.on_path.end());
  }

  for (const auto& f : on_path) {
    if (!forward.has_probabilistic_achiever.count(f)) continue;
    auto it = forward.achievers_by_fluent.find(f);
    if (it == forward.achievers_by_fluent.end()) continue;
    auto& tuples = result[f];
    tuples.reserve(it->second.size());
    for (const auto& a : it->second) {
      tuples.emplace_back(a.action->name(), a.probability, a.exec_cost, a.wait_cost);
    }
  }
  return result;
}

// Names of all actions that can contribute to `goal` under relaxed reachability
// (the backward closure following every achiever). Any action whose name is not
// returned cannot help reach the goal and is safe to prune. Union over DNF
// branches.
inline std::vector<std::string> get_goal_relevant_action_names(
    const State &input_state,
    const GoalBase *goal,
    const std::vector<Action> &all_actions,
    bool at_implies_found = true) {

  std::vector<std::string> names;
  if (!goal) return names;

  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) return names;
  State relaxed = relaxed_result[0].first;

  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  auto forward = ff_forward_phase(initial_fluents, all_actions);
  // Only the greedy branch selection reads optimistic_cost, and only above the
  // enumeration cap; below it BranchView walks the cached DNF and this extra
  // fixed point would be wasted work.
  if (goal->dnf_branch_count() > MAX_ENUMERATED_DNF_BRANCHES) {
    compute_optimistic_costs(forward);
  }

  std::unordered_set<const Action*> keep;
  BranchView branch_view(goal, forward);
  for (const auto& branch : branch_view.get()) {
    auto branch_keep = goal_relevant_actions(forward, branch, at_implies_found);
    keep.insert(branch_keep.begin(), branch_keep.end());
  }

  names.reserve(keep.size());
  for (const Action* a : keep) names.push_back(a->name());
  return names;
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

  // The DNF is a product over conjoined disjunctions; walk it only while that
  // is affordable (see BranchView -- the alternative above the cap is not a
  // slow heuristic but an out-of-memory abort).
  BranchView branch_view(goal, forward);
  const auto& branches = branch_view.get();
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
