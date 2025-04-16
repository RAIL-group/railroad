#pragma once

#include "mrppddl/core.hpp"
#include "mrppddl/state.hpp"

#include <iostream>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <functional>
#include <optional>
#include <tuple>

namespace mrppddl {

inline std::vector<Action> get_next_actions(
    const State& state,
    const std::vector<Action>& all_actions
) {
    // Step 1: Extract all `free(...)` fluents
    std::vector<Fluent> free_robot_fluents;
    for (const auto& f : state.fluents()) {
        if (f.name() == "free") {
            free_robot_fluents.push_back(f);
        }
    }

    std::sort(free_robot_fluents.begin(), free_robot_fluents.end(),
              [](const Fluent& a, const Fluent& b) { return a.name() < b.name(); });

    // Step 2: Create negated state (excluding all free)
    std::unordered_set<Fluent> negated;
    for (const auto& f : free_robot_fluents) {
        negated.insert(f.invert());
    }

    State neg_state = State(state.time(), state.fluents());
    neg_state.update_fluents(negated);

    // Step 3: For each free predicate, create temp state with just that one enabled
    for (const auto& free_pred : free_robot_fluents) {
        State temp_state = State(
            state.time(),
            neg_state.fluents()
        );
        temp_state.update_fluents({free_pred});

        std::vector<Action> applicable;
        for (const auto& action : all_actions) {
            if (temp_state.satisfies_precondition(action)) {
                applicable.push_back(action);
            }
        }

        if (!applicable.empty()) {
            return applicable;
        }
    }

    // Step 4: Fall back to any applicable action
    std::vector<Action> fallback;
    for (const auto& action : all_actions) {
        if (state.satisfies_precondition(action)) {
            fallback.push_back(action);
        }
    }

    return fallback;
}


using HeuristicFn = std::function<double(const State&)>;

// For priority queue: (f, counter, state)
using QueueEntry = std::tuple<double, int, State>;

// For backtracking
using CameFromMap = std::unordered_map<State, std::pair<State, Action>>;

inline std::vector<Action> reconstruct_path(const CameFromMap& came_from, State current) {
    std::vector<Action> path;
    auto it = came_from.find(current);
    while (it != came_from.end()) {
	std::cerr << it->second.second.str() << std::endl;
        path.push_back(it->second.second);
        current = it->second.first;
        it = came_from.find(current);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

inline std::function<bool(const std::unordered_set<Fluent>&)>
make_goal_test(const std::unordered_set<Fluent>& goal_fluents) {
    return [goal_fluents](const std::unordered_set<Fluent>& fluents) {
        for (const auto& f : goal_fluents) {
            if (fluents.find(f) == fluents.end()) {
                return false;
            }
        }
        return true;
    };
}

inline std::optional<std::vector<Action>> astar(
    const State& start_state,
    const std::vector<Action>& all_actions,
    const std::function<bool(const std::unordered_set<Fluent>&)>& is_goal_state,
    HeuristicFn heuristic_fn = nullptr
) {
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> open_heap;
    std::unordered_set<State> closed_set;
    CameFromMap came_from;

    int counter = 0;
    open_heap.emplace(0.0, counter++, start_state);

    while (!open_heap.empty()) {
        QueueEntry top = open_heap.top();
	State current = std::get<2>(top);
        open_heap.pop();

        if (is_goal_state(current.fluents())) {
	  std::cerr << current.hash() << std::endl;
	  std::cerr << current.str() << std::endl;
	  return reconstruct_path(came_from, current);
        }

        if (closed_set.count(current)) continue;
        closed_set.insert(current);

        for (const auto& action : get_next_actions(current, all_actions)) {
            for (const auto& [successor, prob] : transition(current, &action)) {
                if (prob == 0.0) continue;

                double g = successor.time();

		std::cerr << "HSH" << std::endl;
		std::cerr << current.hash() << std::endl;
		std::cerr << successor.hash() << std::endl;
                came_from[successor] = std::make_pair(current, action);

                double h = heuristic_fn ? heuristic_fn(successor) : 0.0;
                double f = g + h;

                open_heap.emplace(f, counter++, successor);
            }
        }
    }

    return std::nullopt;  // no path found
}


}  // namespace mrppddl
