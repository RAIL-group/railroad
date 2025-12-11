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
using FFMemory = std::unordered_map<std::size_t, double>;

class GoalFn {
private:
    std::unordered_set<Fluent> goal_fluents_;

public:
    // Constructor - replaces make_goal_fn
    GoalFn(const std::unordered_set<Fluent> &goal_fluents)
        : goal_fluents_(goal_fluents) {}

    // Make it callable - checks if all goal fluents are present
    bool operator()(const std::unordered_set<Fluent> &fluents) const {
        for (const auto &gf : goal_fluents_) {
            if (!fluents.count(gf))
                return false;
        }
        return true;
    }

    // Count how many goal fluents are present in the active set
    int goal_count(const std::unordered_set<Fluent> &active_fluents) const {
        int count = 0;
        for (const auto &gf : goal_fluents_) {
            if (active_fluents.count(gf)) {
                count++;
            }
        }
        return count;
    }
};

double ff_heuristic(const State &input_state, const GoalFn *is_goal_fn,
                    const std::vector<Action> &all_actions,
                    FFMemory *ff_memory = nullptr,
                    std::unordered_set<const Action *> *output_visited_actions = nullptr) {

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

  // Memoization check (only if goal function provided)
  relaxed.set_time(0);
  if (is_goal_fn && ff_memory && ff_memory->count(relaxed.hash())) {
    return dtime + ff_memory->at(relaxed.hash());
  }

  std::unordered_set<Fluent> newly_added = known_fluents;
  std::unordered_map<Fluent, const Action *> fact_to_action;
  std::unordered_map<const Action *, double> action_to_duration;
  // NEW: Track the probability of achieving each fluent
  std::unordered_map<Fluent, double> fact_to_probability;
  std::unordered_set<const Action *> visited_actions;
  std::unordered_set<const Action *> all_actions_set;
  for (const auto &a : all_actions) {
    all_actions_set.insert(&a);
  }

  // Step 1: Forward relaxed reachability
  while (!newly_added.empty()) {
    std::unordered_set<Fluent> next_new;
    State state_all_known(0.0, known_fluents); // dummy state to test preconditions

    for (const Action *a : all_actions_set) {
      if (visited_actions.count(a))
        continue;
      if (!state_all_known.satisfies_precondition(*a, /*relax=*/true))
        continue;

      const auto& succs = a->get_relaxed_successors();
      visited_actions.insert(a);
      if (succs.empty())
        continue;

      double duration = 0;

      // In relaxed planning, consider fluents from ALL probabilistic outcomes
      // NEW: But now track the maximum probability of achieving each fluent
      for (const auto &[succ_state, succ_prob] : succs) {
	duration = std::max(succ_state.time(), duration);
	if (succ_prob <= 0.0) {
	  continue;
	}
        for (const auto &f : succ_state.fluents()) {
          if (!known_fluents.count(f)) {  // If 'f' not in known_fluents
            known_fluents.insert(f);
            next_new.insert(f);
            fact_to_action[f] = a;
            // NEW: Record the probability of achieving this fluent
            fact_to_probability[f] = succ_prob;
          } else {  // If we've seen 'f' before
	    // FIXME: I don't think this will actually work...
            fact_to_probability[f] = std::max(fact_to_probability[f], succ_prob);

	    // Count the minimum time to reach a place
	    if (duration < action_to_duration[fact_to_action[f]]) {
	      fact_to_action[f] = a;
	    }
	  }
        }
      } // for over successor states
      action_to_duration[a] = duration;
    }  // for over actions

    newly_added = std::move(next_new);
    for (const Action *a : visited_actions) {
      all_actions_set.erase(a);
    }
  }

  // If output_visited_actions requested, populate it
  if (output_visited_actions) {
    *output_visited_actions = visited_actions;
  }

  // If no goal function provided, skip backward planning and return 0
  if (!is_goal_fn) {
    return 0.0;
  }

  if (!(*is_goal_fn)(known_fluents)) {
    return std::numeric_limits<double>::infinity(); // unreachable goal
  }

  // Step 2: Determine required goal fluents via ablation
  std::unordered_set<Fluent> required_fluents;
  for (const auto &f : known_fluents) {
    std::unordered_set<Fluent> test_set(known_fluents);
    test_set.erase(f);
    if (!(*is_goal_fn)(test_set)) {
      required_fluents.insert(f);
    }
  }

  // Step 3: Backward relaxed plan
  std::unordered_set<Fluent> needed = required_fluents;
  std::unordered_set<const Action *> used_actions;
  // NEW: Track which fluents are needed from each action for probability scaling
  std::unordered_map<const Action *, std::vector<Fluent>> action_to_needed_fluents;
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

    // NEW: Track which fluent(s) this action is being used for
    action_to_needed_fluents[a].push_back(f);

    if (used_actions.count(a)) {
      continue;
    }

    used_actions.insert(a);

    // NEW: Scale duration by probability of achieving needed fluent
    double base_duration = action_to_duration[a];
    double prob = fact_to_probability.count(f) ? fact_to_probability[f] : 1.0;

    // Cap the probability scaling to avoid extreme values
    // If prob < 0.01, treat as if prob = 0.01 (max 100x scaling)
    const double MIN_PROB = 0.01;
    prob = std::max(prob, MIN_PROB);

    // Expected attempts = 1 / probability
    double expected_duration = base_duration / prob;
    // total_duration += expected_duration;
    total_duration += base_duration;

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
    return ff_heuristic(s, &is_goal_fn, all_actions, memory);
  };
}

const std::vector<Action> get_usable_actions(const State &input_state,
					     const std::vector<Action> &all_actions) {
  // Use ff_heuristic with nullptr goal function to get visited actions
  std::unordered_set<const Action *> visited_actions;
  ff_heuristic(input_state, nullptr, all_actions, nullptr, &visited_actions);

  // Convert set of action pointers to vector of actions
  std::vector<Action> feasible_actions;
  feasible_actions.reserve(visited_actions.size());
  for (const Action *action_ptr : visited_actions) {
    feasible_actions.push_back(*action_ptr);
  }

  return feasible_actions;
}

const std::vector<Action> get_usable_actions_fluent_list(const State &input_state,
					     const std::unordered_set<Fluent> &goal_fluents,
					     const std::vector<Action> &all_actions) {
 return get_usable_actions(input_state, all_actions);
}


} // namespace mrppddl
