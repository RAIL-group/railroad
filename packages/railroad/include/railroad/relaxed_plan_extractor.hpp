#pragma once

#include "railroad/ff_heuristic.hpp"
#include <vector>
#include <string>
#include <tuple>
#include <unordered_set>
#include <deque>
#include <algorithm>

namespace railroad {

inline std::string fluent_to_string(const Fluent& f) {
    std::string s = "(";
    if (f.is_negated()) s += "not ";
    s += f.name();
    for (const auto& arg : f.args()) {
        s += " " + arg;
    }
    s += ")";
    return s;
}

// Extracts the cheapest relaxed plan from the goal back to the initial state
// Returns a vector of tuples representing each step:
// (achieves_fluent_str, action_name, exec_cost, wait_cost, list_of_preconditions)
inline std::vector<std::tuple<std::string, std::string, double, double, std::vector<std::string>>>
extract_cheapest_relaxed_plan(const State &input_state,
                              const GoalBase *goal,
                              const std::vector<Action> &all_actions) {
                                
    std::vector<std::tuple<std::string, std::string, double, double, std::vector<std::string>>> plan;

    if (!goal) return plan;

    // Step 1: Relaxed transition
    auto relaxed_result = transition(input_state, nullptr, true);
    if (relaxed_result.empty()) return plan;
    State relaxed = relaxed_result[0].first;

    std::unordered_set<Fluent> initial_fluents(
        relaxed.fluents().begin(), relaxed.fluents().end());

    // Step 2: Forward phase
    auto forward = ff_forward_phase(initial_fluents, all_actions);
    compute_expected_costs(forward);

    // Step 3: Extract branches
    auto branches = extract_or_branches(goal);
    if (branches.empty()) return plan;

    // We take the first branch for the relaxed plan
    const auto& goal_fluents = branches[0];

    std::deque<Fluent> target_fluents;
    for (const auto& f : goal_fluents) {
        target_fluents.push_back(f);
    }

    std::unordered_set<Fluent> satisfied_fluents = initial_fluents;
    std::unordered_set<Fluent> visited_fluents;

    while (!target_fluents.empty()) {
        Fluent f = target_fluents.front();
        target_fluents.pop_front();

        if (satisfied_fluents.count(f) || visited_fluents.count(f)) continue;
        visited_fluents.insert(f);

        auto achievers_it = forward.achievers_by_fluent.find(f);
        if (achievers_it == forward.achievers_by_fluent.end() || achievers_it->second.empty()) {
            continue;
        }

        // Sort achievers to find the cheapest (wait_cost + exec_cost)
        auto achievers = achievers_it->second; // copy to sort
        std::sort(achievers.begin(), achievers.end(), [](const ProbabilisticAchiever& a, const ProbabilisticAchiever& b) {
            return (a.wait_cost + a.exec_cost) < (b.wait_cost + b.exec_cost);
        });

        const auto& top_achiever = achievers[0];
        const Action* matched_action = top_achiever.action;

        std::vector<std::string> preconds;
        for (const auto& p : matched_action->preconditions()) {
            preconds.push_back(fluent_to_string(p));
            target_fluents.push_back(p);
        }

        plan.emplace_back(fluent_to_string(f), matched_action->name(), top_achiever.exec_cost, top_achiever.wait_cost, preconds);
    }

    return plan;
}

} // namespace railroad
