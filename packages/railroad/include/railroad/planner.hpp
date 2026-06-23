#pragma once

#ifdef RAILROAD_USE_PYBIND
#include <pybind11/pybind11.h>
#include <Python.h>
#endif

#include "railroad/core.hpp"
#include "railroad/heuristic.hpp"
#include "railroad/goal.hpp"
#include "railroad/state.hpp"
#include "railroad/constants.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <iomanip>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

inline std::vector<const Action *>
get_next_actions(const State &state, const std::vector<Action> &all_actions) {
  // All actions applicable right now.
  std::vector<const Action *> applicable;
  for (const auto &action : all_actions) {
    if (state.satisfies_precondition(action)) {
      applicable.push_back(&action);
    }
  }
  if (applicable.empty()) return applicable;

  // Robot-serialization filter: when several robots are free, commit to the
  // robots' actions in a fixed order rather than branching over every robot at
  // once. Because the robots act concurrently, deciding what each does in a
  // deterministic order loses no generality -- a later decision node (where the
  // first robot is busy) picks up the next free robot -- but it cuts the
  // branching factor from "actions of all free robots" to "actions of one".
  //
  // We keep only the applicable actions *executed by* the first free robot,
  // i.e. those whose positive preconditions include its `free` fluent. A
  // multi-robot action that also needs another robot is still kept (it lists
  // this robot's `free` too), so coordinated actions are not lost -- the reason
  // the earlier "mask out the other robots' free fluents" attempt failed.
  std::vector<const Fluent *> free_fluents;
  for (const auto &f : state.fluents()) {
    if (f.is_free()) free_fluents.push_back(&f);
  }
  if (free_fluents.size() <= 1) return applicable;  // nothing to serialize

  // Deterministic order by robot id (the `free` fluent's argument).
  std::sort(free_fluents.begin(), free_fluents.end(),
            [](const Fluent *a, const Fluent *b) { return a->args() < b->args(); });

  // Return the actions of the first free robot that has any; only fall through
  // to later robots if an earlier one has no applicable action (it is stuck),
  // so a free-but-idle robot never blocks the others.
  for (const Fluent *free_robot : free_fluents) {
    std::vector<const Action *> subset;
    for (const Action *action : applicable) {
      if (action->pos_preconditions().count(*free_robot)) {
        subset.push_back(action);
      }
    }
    if (!subset.empty()) return subset;
  }

  return applicable;  // no applicable action is executed by a free robot
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

inline std::optional<std::vector<Action>>
astar(const State &start_state, const std::vector<Action> &all_actions,
      const GoalPtr &goal,
      HeuristicFn heuristic_fn = nullptr) {
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>>
      open_heap;
  std::unordered_set<std::size_t> closed_set;
  std::unordered_map<std::size_t, std::pair<std::size_t, const Action *>>
      came_from;

  FFMemory ff_memory;
  heuristic_fn = [&goal, &all_actions, &ff_memory](const State& s) -> double {
    return ff_heuristic(s, goal.get(), all_actions, &ff_memory);
  };

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

    if (goal->evaluate(current.fluents())) {
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


void print_best_path(std::ostream& os, const MCTSDecisionNode* node, HeuristicFn& heuristic_fn, int max_print_depth, int current_depth = 0) {
    if (!node || current_depth > max_print_depth) {
        return;
    }

    // --- Print Info for the Current Node ---
    double q_value = (node->visits > 0) ? node->value / static_cast<double>(node->visits) : 0.0;
    double h_value = heuristic_fn ? heuristic_fn(node->state) : 0.0;
    double time_cost = node->state.time();

    // Indent for readability
    for (int i = 0; i < current_depth; ++i) os << " ";

    os << "D:" << current_depth << "|="
       << "visits=" << node->visits << ", "
       << "Q=" << q_value << ", "
       << "g=" << time_cost << ", "
       << "h=" << h_value << ", "
       << "g+h=" << time_cost + h_value << ", "
       << "#A=" << node->children.size()
       << std::endl;

    if (node->children.empty()) {
        for (int i = 0; i < current_depth; ++i) os << "  ";
        os << "  (Leaf Node)" << std::endl;
        return;
    }

    // --- Find the Best Child (Most Visited) to Traverse Next ---
    const MCTSChanceNode* best_chance_node = nullptr;
    int max_visits = -1;

    for (const auto& [action, chance_node_ptr] : node->children) {
        if (chance_node_ptr->visits > max_visits) {
            max_visits = chance_node_ptr->visits;
            best_chance_node = chance_node_ptr.get();
        }
    }

    if (!best_chance_node) {
        for (int i = 0; i < current_depth; ++i) os << "  ";
        os << "  (No best child found)" << std::endl;
        return;
    }

    // Print the action taken
    for (int i = 0; i < current_depth; ++i) os << " ";
    os << "   └── Action: " << best_chance_node->action->name()
       << " (visits=" << best_chance_node->visits 
       << ")" << std::endl;


    // In a probabilistic environment, a chance node can have multiple outcomes.
    // For this diagnostic, let's just follow the most likely or most visited outcome.
    if (!best_chance_node->children.empty()) {
        const MCTSDecisionNode* next_decision_node = nullptr;
        int max_outcome_visits = -1;
        // Let's find the most visited successor state
        for(const auto& child : best_chance_node->children) {
            if (child->visits > max_outcome_visits) {
                max_outcome_visits = child->visits;
                next_decision_node = child.get();
            }
        }
        print_best_path(os, next_decision_node, heuristic_fn, max_print_depth, current_depth + 1);
    }
}


// ---------------------- MCTS core ----------------------

// mcts that accepts GoalBase directly
inline std::string mcts(const State &root_state,
                        const std::vector<Action> &all_actions_base,
                        const GoalBase* goal, FFMemory *ff_memory,
                        int max_iterations = 1000, int max_depth = 20,
                        double c = std::sqrt(2.0),
                        double heuristic_multiplier = HEURISTIC_MULTIPLIER,
                        std::string* out_tree_trace = nullptr,
                        double lambda_add = 0.5,
                        double lambda_max = 0.0,
                        double lambda_ff  = 0.5) {
  // RNG
  static thread_local std::mt19937 rng{std::random_device{}()};

  auto all_actions = get_usable_actions(root_state, all_actions_base);

  // Root node
  auto root = std::make_unique<MCTSDecisionNode>(root_state.copy_and_zero_out_time());
  root->untried_actions = get_next_actions(root_state, all_actions);

  HeuristicFn heuristic_fn = [goal, all_actions, ff_memory,
                              lambda_add, lambda_max, lambda_ff](const State& s) -> double {
    return ff_heuristic(s, goal, all_actions, ff_memory,
                        lambda_add, lambda_max, lambda_ff);
  };

  for (int it = 0; it < max_iterations; ++it) {
    bool is_node_goal = false;
    bool did_need_relaxed_transition = false;

    #ifdef RAILROAD_USE_PYBIND
    if (PyErr_CheckSignals() != 0) {
      throw pybind11::error_already_set();
    }
    #endif
    MCTSDecisionNode *node = root.get();
    int depth = 0;
    double accumulated_extra_cost = 0.0;

    // ---------------- Selection ----------------
    while (depth < max_depth) {
      if (!node->untried_actions.empty())
        break;
      if (node->children.empty())
        break;
      // Use GoalBase::evaluate for goal check
      if (goal->evaluate(node->state.fluents())) {
        is_node_goal = true;
        break;
      }

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

      accumulated_extra_cost += best_chance->action->extra_cost();
      std::size_t idx = sample_index(best_chance->outcome_weights, rng);
      node = best_chance->children[idx].get();
      ++depth;
    }

    // ---------------- Expansion ----------------
    if (!node->untried_actions.empty() && !is_node_goal) {
      const Action *action = node->untried_actions.back();
      node->untried_actions.pop_back();

      accumulated_extra_cost += action->extra_cost();

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

        if (chance_raw->children.empty())
          continue;

        std::size_t idx = sample_index(chance_raw->outcome_weights, rng);
        node = chance_raw->children[idx].get();
        ++depth;
      }
    }

    // ---------------- Simulation / Evaluation ----------------
    double reward;
    double h = 0.0;
    int goal_count_val = goal->goal_count(node->state.fluents());
    if (goal->evaluate(node->state.fluents())) {
      reward = -node->state.time() + SUCCESS_REWARD + 0 * goal_count_val - accumulated_extra_cost;
    } else {
      h = heuristic_fn ? heuristic_fn(node->state) : 0.0;
      if (h > 1e10) {
        h = HEURISTIC_CANNOT_FIND_GOAL_PENALTY;
      }
      if (did_need_relaxed_transition)
        h += 100;

      reward = -node->state.time() - h * heuristic_multiplier + 0 * goal_count_val - accumulated_extra_cost;
    }

    // ---------------- Backpropagation ----------------
    backpropagate(node, reward);
  }

  // Generate tree trace
  std::ostringstream tree_trace_stream;
  tree_trace_stream << std::fixed << std::setprecision(2);
  print_best_path(tree_trace_stream, root.get(), heuristic_fn, 20);

  if (out_tree_trace) {
    *out_tree_trace = tree_trace_stream.str();
  }

  // Extract policy
  MCTSResult result;
  result.root = std::move(root);

  if (!result.root->children.empty()) {
    const Action *best_action = nullptr;
    int most_visits = 0;

    for (auto &kv : result.root->children) {
      MCTSChanceNode *cn = kv.second.get();
      if (cn->visits == 0)
        continue;
      if (cn->visits > most_visits) {
        most_visits = cn->visits;
        best_action = kv.first;
      }
    }

    if (best_action) {
      return best_action->name();
    }
  }

  return "NONE";
}

class MCTSPlanner {
public:
  explicit MCTSPlanner(std::vector<Action> all_actions,
                       double lambda_add = 0.5,
                       double lambda_max = 0.0,
                       double lambda_ff  = 0.5)
      : all_actions_(std::move(all_actions)),
        lambda_add_(lambda_add),
        lambda_max_(lambda_max),
        lambda_ff_(lambda_ff) {}

  // Call operator: planner(initial_state, goal) → string
  std::string operator()(const State &root_state,
                         const GoalPtr &goal,
                         int max_iterations = 1000,
                         int max_depth = 20,
                         double c = std::sqrt(2.0),
                         double heuristic_multiplier = HEURISTIC_MULTIPLIER) {
    return mcts(root_state, all_actions_, goal.get(), &ff_memory_,
                max_iterations, max_depth, c, heuristic_multiplier,
                &last_mcts_tree_trace_,
                lambda_add_, lambda_max_, lambda_ff_);
  }

  void clear_cache() { ff_memory_.clear(); }
  std::size_t cache_size() const { return ff_memory_.size(); }

  double lambda_add() const { return lambda_add_; }
  double lambda_max() const { return lambda_max_; }
  double lambda_ff()  const { return lambda_ff_; }

  // Get the tree trace from the most recent MCTS planning call
  const std::string& get_trace_from_last_mcts_tree() const {
    return last_mcts_tree_trace_;
  }

private:
  std::vector<Action> all_actions_;
  FFMemory ff_memory_;
  std::string last_mcts_tree_trace_;
  double lambda_add_;
  double lambda_max_;
  double lambda_ff_;
};

} // namespace railroad
