#pragma once

#include "mrppddl/core.hpp"
#include "mrppddl/ff_heuristic.hpp"
#include "mrppddl/state.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mrppddl {

inline std::vector<const Action *>
get_next_actions(const State &state, const std::vector<Action> &all_actions) {
  // Step 1: Extract all `free(...)` fluents (sorting as I go)
  auto cmp = [](const Fluent &a, const Fluent &b) {
    return a.name() < b.name();
  };

  std::set<Fluent, decltype(cmp)> free_robot_fluents(cmp);

  for (const auto &f : state.fluents()) {
    if (f.is_free()) {
      free_robot_fluents.insert(f);
    }
  }

  // Step 2: Create negated state (excluding all free)
  std::unordered_set<Fluent> negated;
  for (const auto &f : free_robot_fluents) {
    negated.insert(f.invert());
  }

  // Step 3: For each free predicate, create temp state with just that one
  // enabled
  State temp_state = State(0, state.fluents());
  temp_state.update_fluents(negated);
  for (const auto &free_pred : free_robot_fluents) {
    temp_state.update_fluents({free_pred});

    std::vector<const Action *> applicable;
    for (const auto &action : all_actions) {
      if (temp_state.satisfies_precondition(action)) {
        applicable.push_back(&action);
      }
    }

    if (!applicable.empty()) {
      return applicable;
    }
  }

  // Step 4: Fall back to any applicable action
  std::vector<const Action *> fallback;
  for (const auto &action : all_actions) {
    if (state.satisfies_precondition(action)) {
      fallback.push_back(&action);
    }
  }

  return fallback;
}

using HeuristicFn = std::function<double(const State &)>;

// For priority queue: (f, state)
using QueueEntry = std::tuple<double, State>;

// For backtracking
using CameFromMap = std::unordered_map<State, std::pair<State, Action>>;

inline std::vector<Action> reconstruct_path(const CameFromMap &came_from,
                                            State current) {
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

inline std::function<bool(const std::unordered_set<Fluent> &)>
make_goal_test(const std::unordered_set<Fluent> &goal_fluents) {
  return [&goal_fluents](const std::unordered_set<Fluent> &fluents) {
    for (const auto &f : goal_fluents) {
      if (!fluents.count(f)) {
        return false;
      }
    }
    return true;
  };
}

bool is_goal_dbg(const std::unordered_set<Fluent> &fluents,
                 const std::unordered_set<Fluent> &goal_fluents) {
  for (const auto &gf : goal_fluents) {
    if (!fluents.count(gf)) {
      return false;
    }
  }
  return true;
}

inline GoalFn make_goal_fn(const std::unordered_set<Fluent> &goal_fluents) {
  return [goal_fluents](const std::unordered_set<Fluent> &fluents) -> bool {
    for (const auto &gf : goal_fluents) {
      if (!fluents.count(gf))
        return false;
    }
    return true;
  };
}

inline std::optional<std::vector<Action>>
astar(const State &start_state, const std::vector<Action> &all_actions,
      const std::unordered_set<Fluent> &goal_fluents,
      HeuristicFn heuristic_fn = nullptr) {
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>>
      open_heap;
  std::unordered_set<std::size_t> closed_set;
  std::unordered_map<std::size_t, std::pair<std::size_t, const Action *>>
      came_from;

  auto is_goal_fn = make_goal_fn(goal_fluents);
  FFMemory ff_memory;
  heuristic_fn = make_ff_heuristic(is_goal_fn, all_actions, &ff_memory);

  // std::unordered_set<Fluent> goal_fluents;
  // goal_fluents.emplace(Fluent("at r1 a"));
  // goal_fluents.emplace(Fluent("visited a"));
  // goal_fluents.emplace(Fluent("visited b"));
  // goal_fluents.emplace(Fluent("visited c"));
  // goal_fluents.emplace(Fluent("visited d"));
  // goal_fluents.emplace(Fluent("visited e"));

  int counter = 0;
  open_heap.emplace(0.0, start_state);

  std::cerr << "Starting the main planning loop." << std::endl;
  while (!open_heap.empty()) {
    counter++;
    QueueEntry top = open_heap.top();
    State current = std::get<1>(top);
    open_heap.pop();

    if (closed_set.count(current.hash()))
      continue;
    closed_set.insert(current.hash());

    if (is_goal_fn(current.fluents())) {
      std::cerr << "Goal reached!! count: " << counter << std::endl;
      return std::nullopt; // no path found
                           // return reconstruct_path(came_from, current);
    }

    auto next_actions = get_next_actions(current, all_actions);
    for (const auto action : next_actions) {
      for (const auto &[successor, prob] : transition(current, action)) {
        if (prob == 0.0)
          continue;

        double g = successor.time();

        came_from[successor.hash()] = std::make_pair(current.hash(), action);

        double h = heuristic_fn ? heuristic_fn(successor) : 0.0;
        double f = g + h;

        open_heap.emplace(f, std::move(successor));
      }
    }
  }

  return std::nullopt; // no path found
}

// ############## MCTS ###############

// ---------------------- MCTS data structures ----------------------

struct MCTSChanceNode; // forward

struct MCTSDecisionNode {
  State state;
  MCTSChanceNode *parent = nullptr; // non-owning
  std::unordered_map<const Action *, std::unique_ptr<MCTSChanceNode>> children;
  std::vector<const Action *> untried_actions;

  int visits = 0;
  double value = 0.0;

  explicit MCTSDecisionNode(const State &s, MCTSChanceNode *p = nullptr)
      : state(s), parent(p) {}
};

struct MCTSChanceNode {
  const Action *action = nullptr;     // non-owning
  MCTSDecisionNode *parent = nullptr; // non-owning

  std::vector<std::unique_ptr<MCTSDecisionNode>> children;
  std::vector<double> outcome_weights; // probabilities aligned with children

  int visits = 0;
  double value = 0.0;

  MCTSChanceNode(const Action *a, MCTSDecisionNode *p) : action(a), parent(p) {}
};

// What we return.
struct MCTSResult {
  // Keep it consistent with your A* hash map.
  std::unordered_map<std::size_t, const Action *> policy;
  std::unique_ptr<MCTSDecisionNode> root;
};

// ---------------------- helpers ----------------------

inline double ucb_score(int parent_visits, const MCTSChanceNode &child,
                        double c = std::sqrt(2.0)) {
  if (child.visits == 0)
    return std::numeric_limits<double>::infinity();
  const double exploitation = child.value / static_cast<double>(child.visits);
  const double exploration =
      c * std::sqrt(std::log(static_cast<double>(parent_visits)) /
                    static_cast<double>(child.visits));
  return exploitation + exploration;
}

inline std::size_t sample_index(const std::vector<double> &weights,
                                std::mt19937 &rng) {
  // std::discrete_distribution accepts non-normalized weights.
  std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
  return dist(rng);
}

inline void backpropagate(MCTSDecisionNode *leaf, double reward) {
  MCTSDecisionNode *d = leaf;
  MCTSChanceNode *c = nullptr;

  while (d || c) {
    if (d) {
      d->visits += 1;
      d->value += reward;
      c = d->parent;
      d = nullptr;
    } else {
      c->visits += 1;
      c->value += reward;
      d = c->parent;
      c = nullptr;
    }
  }
}

// ---------------------- MCTS core ----------------------

inline std::string mcts(const State &root_state,
                        const std::vector<Action> &all_actions,
                        const GoalFn &is_goal_fn, FFMemory *ff_memory,
                        int max_iterations = 1000, int max_depth = 20,
                        double c = std::sqrt(2.0)) {
  // RNG (thread_local is convenient if you run this in parallel later)
  static thread_local std::mt19937 rng{std::random_device{}()};

  // Root node
  auto root = std::make_unique<MCTSDecisionNode>(root_state);
  root->untried_actions = get_next_actions(root_state, all_actions);
  auto heuristic_fn = make_ff_heuristic(is_goal_fn, all_actions, ff_memory);

  for (int it = 0; it < max_iterations; ++it) {
    MCTSDecisionNode *node = root.get();
    int depth = 0;

    // ---------------- Selection ----------------
    while (depth < max_depth) {
      if (!node->untried_actions.empty())
        break;
      if (node->children.empty())
        break;
      if (is_goal_fn(node->state.fluents()))
        break;

      // choose action / chance node with best UCB
      MCTSChanceNode *best_chance = nullptr;
      double best_score = -std::numeric_limits<double>::infinity();

      for (auto &kv : node->children) {
        MCTSChanceNode *cn = kv.second.get();
        double score = ucb_score(node->visits, *cn, c);
        if (score > best_score) {
          best_score = score;
          best_chance = cn;
        }
      }

      if (!best_chance || best_chance->children.empty())
        break;

      // sample a successor decision node according to outcome weights
      std::size_t idx = sample_index(best_chance->outcome_weights, rng);
      node = best_chance->children[idx].get();
      ++depth;
    }

    // ---------------- Expansion ----------------
    if (!node->untried_actions.empty()) {
      const Action *action = node->untried_actions.back();
      node->untried_actions.pop_back();

      auto outcomes = transition(node->state, action);
      if (!outcomes.empty()) {
        auto chance_node = std::make_unique<MCTSChanceNode>(action, node);
        auto *chance_raw = chance_node.get();
        node->children.emplace(action, std::move(chance_node));

        chance_raw->children.reserve(outcomes.size());
        chance_raw->outcome_weights.reserve(outcomes.size());

        for (auto &[succ, prob] : outcomes) {
          if (prob <= 0.0)
            continue;
          auto child_decision =
              std::make_unique<MCTSDecisionNode>(succ, chance_raw);
          child_decision->untried_actions =
              get_next_actions(child_decision->state, all_actions);
          chance_raw->outcome_weights.push_back(prob);
          chance_raw->children.push_back(std::move(child_decision));
        }

        // If all outcomes had prob 0 -> skip this iteration
        if (chance_raw->children.empty())
          continue;

        // Move to one sampled child
        std::size_t idx = sample_index(chance_raw->outcome_weights, rng);
        node = chance_raw->children[idx].get();
        ++depth;
      }
    }

    // ---------------- Simulation / Evaluation ----------------
    // Your Python code uses: reward = -time - heuristic (bounded).
    double reward;
    if (is_goal_fn(node->state.fluents())) {
      reward = -node->state.time();
    } else {
      double h = heuristic_fn ? heuristic_fn(node->state) : 0.0;
      // double h = 0;
      if (h > 1e10) {
        h = 10.0;
      }
      reward = -node->state.time() - h;
    }

    // ---------------- Backpropagation ----------------
    backpropagate(node, reward);
  }

  // --------------- Extract a (very) shallow policy ---------------
  MCTSResult result;
  result.root = std::move(root);

  if (!result.root->children.empty()) {
    const Action *best_action = nullptr;
    double best_q = -std::numeric_limits<double>::infinity();

    for (auto &kv : result.root->children) {
      MCTSChanceNode *cn = kv.second.get();
      if (cn->visits == 0)
        continue;
      double q = cn->value / static_cast<double>(cn->visits);
      if (q > best_q) {
        best_q = q;
        best_action = kv.first;
      }
    }

    if (best_action) {
      std::cout << best_q << std::endl;
      return best_action->name();
      result.policy.emplace(result.root->state.hash(), best_action);
    }
  }

  return "NONE";

  // return result.best_action();
  // return best_action;
}

class MCTSPlanner {
public:
  explicit MCTSPlanner(std::vector<Action> all_actions)
      : all_actions_(std::move(all_actions)) {}

  // Call operator: planner(initial_state, goal_fluents) → string

  std::string operator()(const State &root_state,
                         const std::unordered_set<Fluent> &goal_fluents,
                         int max_iterations, int max_depth, double c) {
    auto is_goal_fn = make_goal_fn(goal_fluents);
    return mcts(root_state, all_actions_, is_goal_fn, &ff_memory_,
                max_iterations, max_depth, c);
  }

  std::string operator()(const State &root_state,
                         const std::unordered_set<Fluent> &goal_fluents) {
    auto is_goal_fn = make_goal_fn(goal_fluents);
    return mcts(root_state, all_actions_, is_goal_fn, &ff_memory_,
                max_iterations, max_depth, c);
  }

  void clear_cache() { ff_memory_.clear(); }
  std::size_t cache_size() const { return ff_memory_.size(); }

  // Public configuration parameters
  int max_iterations = 1000;
  int max_depth = 20;
  double c = std::sqrt(2.0);

private:
  std::vector<Action> all_actions_;
  FFMemory ff_memory_;
};

} // namespace mrppddl
