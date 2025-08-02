#pragma once

#include "mrppddl/core.hpp"
#include "mrppddl/state.hpp"

#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mrppddl {

using HeuristicFn = std::function<double(const State &)>;
using GoalFn = std::function<bool(const std::unordered_set<Fluent> &)>;
using FFMemory = std::unordered_map<std::size_t, double>;

double ff_heuristic(const State &input_state, const GoalFn &is_goal_fn,
                    const std::vector<Action> &all_actions,
                    FFMemory *ff_memory = nullptr) {

  const double t0 = input_state.time();
  // Step 1: Relaxed transition
  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty())
    return std::numeric_limits<double>::infinity();
  State relaxed = relaxed_result[0].first;

  double dtime = relaxed.time() - t0;
  const auto &initial_fluents = relaxed.fluents();
  std::unordered_set<Fluent> known_fluents(initial_fluents.begin(),
                                           initial_fluents.end());

  // Memoization check
  relaxed.set_time(0);
  if (ff_memory && ff_memory->count(relaxed.hash())) {
    return dtime + ff_memory->at(relaxed.hash());
  }

  std::unordered_set<Fluent> newly_added = known_fluents;
  std::unordered_map<Fluent, const Action *> fact_to_action;
  std::unordered_map<const Action *, double> action_to_duration;
  std::unordered_set<const Action *> visited_actions;
  std::unordered_set<const Action *> all_actions_set;
  for (const auto &a : all_actions) {
    all_actions_set.insert(&a);
  }

  // Step 1: Forward relaxed reachability
  while (!newly_added.empty()) {
    std::unordered_set<Fluent> next_new;
    State temp(0.0, known_fluents); // dummy state to test preconditions

    for (const Action *a : all_actions_set) {
      if (visited_actions.count(a))
        continue;
      if (!temp.satisfies_precondition(*a, /*relax=*/true))
        continue;

      auto succs = transition(temp, a, true);
      if (succs.empty())
        continue;

      visited_actions.insert(a);
      double duration = succs[0].first.time() - temp.time();
      action_to_duration[a] = duration;

      for (const auto &f : succs[0].first.fluents()) {
        if (!known_fluents.count(f)) {
          known_fluents.insert(f);
          next_new.insert(f);
          fact_to_action[f] = a;
        }
      }
    }

    newly_added = std::move(next_new);
    for (const Action *a : visited_actions) {
      all_actions_set.erase(a);
    }
  }

  if (!is_goal_fn(known_fluents)) {
    return std::numeric_limits<double>::infinity(); // unreachable goal
  }

  // Step 2: Determine required goal fluents via ablation
  std::unordered_set<Fluent> required_fluents;
  for (const auto &f : known_fluents) {
    std::unordered_set<Fluent> test_set(known_fluents);
    test_set.erase(f);
    if (!is_goal_fn(test_set)) {
      required_fluents.insert(f);
    }
  }

  // Step 3: Backward relaxed plan
  std::unordered_set<Fluent> needed = required_fluents;
  std::unordered_set<const Action *> used_actions;
  double total_duration = 0.0;

  while (!needed.empty()) {
    Fluent f = *needed.begin();
    needed.erase(needed.begin());

    if (initial_fluents.count(f))
      continue;

    auto it = fact_to_action.find(f);
    if (it == fact_to_action.end())
      continue;
    const Action *a = it->second;
    if (used_actions.count(a))
      continue;

    used_actions.insert(a);
    total_duration += action_to_duration[a];
    for (const Fluent &p : a->pos_preconditions()) {
      if (!initial_fluents.count(p)) {
        needed.insert(p);
      }
    }
  }

  if (ff_memory) {
    (*ff_memory)[relaxed.hash()] = total_duration;
  }

  return dtime + total_duration;
}

HeuristicFn make_ff_heuristic(GoalFn is_goal_fn,
                              std::vector<Action> all_actions,
                              FFMemory *memory = nullptr) {
  return [=](const State &s) -> double {
    return ff_heuristic(s, is_goal_fn, all_actions, memory);
  };
}

} // namespace mrppddl
