#pragma once

#ifdef RAILROAD_USE_PYBIND
#include <pybind11/pybind11.h>
#include <Python.h>
#endif

#include "railroad/core.hpp"
#include "railroad/goal.hpp"
#include "railroad/heuristic.hpp"
#include "railroad/planner.hpp"
#include "railroad/state.hpp"

#include <cstddef>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

// ############## A* ###############
//
// Best-first search over the temporal state space, for single-task planning:
// the next-state model is `transition` (advance until a robot is free), so
// durative actions and blocking facts behave exactly as they do in MCTS.
//
// A node's cost is accumulated along its path with a per-step discount:
//   g'     = g + discount * action_cost
//   d'     = discount * (1 - p)        if discount_by_interruption
//          = discount * fixed_discount otherwise
//   f      = g' + d' * (h + task_reward)
// where p is the action's interruption probability. With p = 0, a fixed
// discount of 1 and no task reward this is plain A* on summed action costs.
// Ties are broken by h, then by insertion order. States are identified by
// their fluents and their queued effects at their times relative to now (see
// state_key), as the expanded set, the duplicate-pruning table and the
// heuristic cache all key on them: two states differing only in the clock are
// one state, but not two waiting different times for a queued effect.

struct AStarParams {
  int num_steps = 10000;
  // h = heuristic_multiplier * ff_heuristic(lambda_*), or h_constant when !use_ff
  bool use_ff = true;
  double h_constant = 0.0;
  double heuristic_multiplier = 1.0;
  double lambda_add = 0.0;
  double lambda_max = 0.0;
  double lambda_ff = 1.0;
  bool at_implies_found = true;
  // discounting (see above)
  bool discount_by_interruption = false;
  double fixed_discount = 1.0;
  double task_reward = 0.0;
  // check for Ctrl-C every this many expansions (Python builds only)
  int signal_check_interval = 256;
};

struct AStarResult {
  std::vector<std::size_t> plan; // indices into the actions given to astar()
  double cost = std::numeric_limits<double>::infinity();
  bool success = false;
  int expanded = 0;
  int generated = 0;      // applicable actions whose successor was computed
  int pushed = 0;         // successors that entered the frontier
};

namespace astar_detail {

// A state's identity in the search: the hash of its zero-time copy, i.e. of
// its fluents and queued effects relative to now (State compares by hash), as
// the Python search's state_key has it.
inline std::size_t state_key(const State &state) {
  return state.copy_and_zero_out_time().hash();
}

struct Node {
  State state;
  long parent;          // index into nodes, -1 for the root
  std::size_t action;   // index into the actions given to astar()
  double cost;          // accumulated (discounted) cost g
  double value;         // g + discounted future estimate: the priority
  double h;             // undiscounted heuristic: the first tie-break
  double discount;      // running discount applied to the next step
  std::size_t key;      // state_key(state)
};

struct QueueEntry {
  double value;
  double h;
  long order;
  long node;
  // std::priority_queue is a max-heap: "greater" entries come out last
  bool operator<(const QueueEntry &other) const {
    if (value != other.value) return value > other.value;
    if (h != other.h) return h > other.h;
    return order > other.order;
  }
};

inline std::vector<std::size_t> plan_of(const std::deque<Node> &nodes, long node) {
  std::vector<std::size_t> plan;
  for (long n = node; nodes[n].parent >= 0; n = nodes[n].parent) {
    plan.push_back(nodes[n].action);
  }
  return {plan.rbegin(), plan.rend()};
}

// The actions relaxed-reachable from `state` (as get_usable_actions finds
// them), in their original order, as indices. Others can never become
// applicable, nor appear in a relaxed plan from any reachable state, so
// dropping them changes neither the search nor the heuristic.
inline std::vector<std::size_t> usable_indices(const State &state,
                                               const std::vector<Action> &actions) {
  auto usable = get_usable_actions(state, actions);
  std::unordered_set<std::string> names;
  names.reserve(usable.size());
  for (const auto &a : usable) names.insert(a.name());
  std::vector<std::size_t> indices;
  for (std::size_t i = 0; i < actions.size(); ++i) {
    if (names.count(actions[i].name())) indices.push_back(i);
  }
  return indices;
}

} // namespace astar_detail

// action_costs[i] and interruption_probs[i] belong to actions[i].
inline AStarResult astar(const State &root_state,
                         const std::vector<Action> &actions,
                         const GoalBase *goal,
                         const std::vector<double> &action_costs,
                         const std::vector<double> &interruption_probs,
                         const AStarParams &params) {
  using namespace astar_detail;
  if (action_costs.size() != actions.size() ||
      interruption_probs.size() != actions.size()) {
    throw std::invalid_argument(
        "astar: action_costs and interruption_probs must match actions in length");
  }

  // the search runs over the usable actions only; `index_of` maps back
  const auto usable = usable_indices(root_state, actions);
  std::vector<Action> search_actions;
  search_actions.reserve(usable.size());
  for (std::size_t i : usable) search_actions.push_back(actions[i]);
  std::unordered_map<const Action *, std::size_t> index_of;
  for (std::size_t j = 0; j < search_actions.size(); ++j) {
    index_of[&search_actions[j]] = usable[j];
  }

  auto heuristic = [&](const State &s) -> double {
    if (!params.use_ff) return params.h_constant;
    return params.heuristic_multiplier *
           ff_heuristic(s, goal, search_actions, nullptr, params.lambda_add,
                        params.lambda_max, params.lambda_ff, params.at_implies_found);
  };

  std::deque<Node> nodes;
  std::unordered_set<std::size_t> expanded;
  std::unordered_map<std::size_t, double> best_cost;
  std::unordered_map<std::size_t, double> heuristic_cache;
  std::priority_queue<QueueEntry> frontier;
  long insertion_order = 1;
  AStarResult result;

  nodes.push_back(Node{root_state, -1, 0, 0.0, 0.0, 0.0, 1.0, state_key(root_state)});
  frontier.push(QueueEntry{0.0, -1.0, 0, 0});

  while (result.expanded < params.num_steps && !frontier.empty()) {
#ifdef RAILROAD_USE_PYBIND
    if (result.expanded % params.signal_check_interval == 0 && PyErr_CheckSignals() != 0) {
      throw pybind11::error_already_set();
    }
#endif
    const long current = frontier.top().node;
    frontier.pop();

    if (goal->evaluate(nodes[current].state.fluents())) {
      result.plan = plan_of(nodes, current);
      result.cost = nodes[current].cost;
      result.success = true;
      return result;
    }

    if (expanded.count(nodes[current].key)) continue;
    expanded.insert(nodes[current].key);
    ++result.expanded;

    // a reference stays valid while children are pushed: deque::push_back
    // never moves existing elements
    const State &parent_state = nodes[current].state;
    const double parent_cost = nodes[current].cost;
    const double parent_discount = nodes[current].discount;

    for (const Action *action : get_next_actions(parent_state, search_actions)) {
      auto outcomes = transition(parent_state, action);
      if (outcomes.size() != 1 || outcomes[0].second != 1.0) {
        throw std::runtime_error("astar: needs deterministic transitions (" +
                                 action->name() + ")");
      }
      State next_state = std::move(outcomes[0].first);
      ++result.generated;

      const std::size_t index = index_of.at(action);
      const std::size_t key = state_key(next_state);
      if (expanded.count(key)) continue;

      const double p = interruption_probs[index];
      const double cost = parent_cost + parent_discount * (action_costs[index] + 0.0 * p);
      auto best = best_cost.find(key);
      if (best != best_cost.end() && cost >= best->second) continue;

      nodes.push_back(Node{std::move(next_state), current, index, cost, 0.0, 0.0, 0.0, key});
      Node &child = nodes.back();
      best_cost[key] = cost;

      auto cached = heuristic_cache.find(key);
      if (cached != heuristic_cache.end()) {
        child.h = cached->second;
      } else {
        child.h = heuristic(child.state);
        heuristic_cache.emplace(key, child.h);
      }
      child.discount = params.discount_by_interruption
                           ? parent_discount * (1 - p)
                           : parent_discount * params.fixed_discount;
      child.value = cost + child.discount * (child.h + params.task_reward);

      frontier.push(QueueEntry{child.value, child.h, insertion_order++,
                               static_cast<long>(nodes.size() - 1)});
      ++result.pushed;
    }
  }

  if (frontier.empty()) return result; // goal unreachable: every state expanded
  const long best = frontier.top().node;
  result.plan = plan_of(nodes, best);
  result.cost = nodes[best].cost;
  return result;
}

} // namespace railroad
