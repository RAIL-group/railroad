#pragma once

#ifdef RAILROAD_USE_PYBIND
#include <pybind11/pybind11.h>
#include <Python.h>
#endif

#include "railroad/core.hpp"
#include "railroad/ff_heuristic.hpp"
#include "railroad/goal.hpp"
#include "railroad/state.hpp"
#include "railroad/constants.hpp"

#include <algorithm>
#include <cmath>
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
  // Step 1: Extract all `free(...)` fluents (sorting as I go)
  auto cmp = [](const Fluent &a, const Fluent &b) {
    return a.name() < b.name();
  };

  // // This filters actions based on the 'next free robot'.
  // // Not only does it not work properly, but we need to comment
  // // it out for multi-robot actions to work.

  // std::set<Fluent, decltype(cmp)> free_robot_fluents(cmp);

  // for (const auto &f : state.fluents()) {
  //   if (f.is_free()) {
  //     free_robot_fluents.insert(f);
  //   }
  // }

  // // Step 2: Create negated state (excluding all free)
  // std::unordered_set<Fluent> negated;
  // for (const auto &f : free_robot_fluents) {
  //   negated.insert(f.invert());
  // }

  // // Step 3: For each free predicate, create temp state with just that one
  // // enabled
  // State temp_state = State(0, state.fluents());
  // temp_state.update_fluents(negated);
  // for (const auto &free_pred : free_robot_fluents) {
  //   temp_state.update_fluents({free_pred});

  //   std::vector<const Action *> applicable;
  //   for (const auto &action : all_actions) {
  //     if (temp_state.satisfies_precondition(action)) {
  //       applicable.push_back(&action);
  //     }
  //   }

  //   if (!applicable.empty()) {
  //     return applicable;
  //   }
  // }

  // Step 4: Fall back to any applicable action
  std::vector<const Action *> fallback;
  for (const auto &action : all_actions) {
    if (state.satisfies_precondition(action)) {
      fallback.push_back(&action);
    }
  }

  return fallback;
}

inline std::unordered_set<Fluent> get_unsatisfied_goal_literals(const State &state,
                                                                const GoalBase *goal) {
  std::unordered_set<Fluent> unsatisfied;
  if (!goal) {
    return unsatisfied;
  }

  auto literal_satisfied = [&state](const Fluent &lit) {
    if (lit.is_negated()) {
      return !state.fluents().count(lit.invert());
    }
    return state.fluents().count(lit) > 0;
  };

  // Pick one active DNF branch so OR goals are guided by the most promising
  // branch instead of the union of all branch literals.
  const auto &branches = goal->get_dnf_branches();
  const std::unordered_set<Fluent> *best_branch = nullptr;
  int best_satisfied = -1;
  int best_unsatisfied = std::numeric_limits<int>::max();
  std::size_t best_size = std::numeric_limits<std::size_t>::max();

  for (const auto &branch : branches) {
    int satisfied = 0;
    int branch_unsatisfied = 0;
    for (const auto &lit : branch) {
      if (literal_satisfied(lit)) {
        ++satisfied;
      } else {
        ++branch_unsatisfied;
      }
    }

    if (!best_branch || satisfied > best_satisfied ||
        (satisfied == best_satisfied && branch_unsatisfied < best_unsatisfied) ||
        (satisfied == best_satisfied && branch_unsatisfied == best_unsatisfied &&
         branch.size() < best_size)) {
      best_branch = &branch;
      best_satisfied = satisfied;
      best_unsatisfied = branch_unsatisfied;
      best_size = branch.size();
    }
  }

  if (!best_branch) {
    return unsatisfied;
  }

  for (const auto &lit : *best_branch) {
    if (!literal_satisfied(lit)) {
      unsatisfied.insert(lit);
    }
  }
  return unsatisfied;
}

inline std::vector<const Action *> get_goal_relevant_next_actions(
    const State &state,
    const std::vector<Action> &all_actions,
    const GoalBase *goal,
    const std::unordered_map<const Action *, std::unordered_set<Fluent>> &action_adds_map,
    int relevance_depth = 2) {
  auto applicable = get_next_actions(state, all_actions);
  if (applicable.empty() || !goal) {
    return applicable;
  }

  std::unordered_set<Fluent> relevant_fluents = get_unsatisfied_goal_literals(state, goal);
  if (relevant_fluents.empty()) {
    return applicable;
  }
  std::unordered_map<Fluent, int> fluent_relevance_level;
  fluent_relevance_level.reserve(relevant_fluents.size() * 2 + 8);
  for (const auto &f : relevant_fluents) {
    fluent_relevance_level[f] = 0;
  }

  std::unordered_set<const Action *> relevant_actions;
  for (int depth = 0; depth < relevance_depth; ++depth) {
    std::unordered_set<Fluent> next_relevant_fluents;
    bool any_added = false;

    for (const auto &action : all_actions) {
      const Action *a = &action;
      auto add_it = action_adds_map.find(a);
      if (add_it == action_adds_map.end()) {
        continue;
      }

      bool adds_relevant = false;
      for (const auto &f : add_it->second) {
        if (relevant_fluents.count(f)) {
          adds_relevant = true;
          break;
        }
      }
      if (!adds_relevant) {
        continue;
      }

      relevant_actions.insert(a);
      for (const auto &p : a->pos_preconditions()) {
        if (!relevant_fluents.count(p)) {
          next_relevant_fluents.insert(p);
          fluent_relevance_level.emplace(p, depth + 1);
          any_added = true;
        }
      }
    }

    if (!any_added) {
      break;
    }
    relevant_fluents.insert(next_relevant_fluents.begin(), next_relevant_fluents.end());
  }

  if (relevant_actions.empty()) {
    return applicable;
  }

  std::vector<const Action *> filtered;
  filtered.reserve(applicable.size());
  for (const Action *a : applicable) {
    if (relevant_actions.count(a)) {
      filtered.push_back(a);
    }
  }

  if (filtered.empty()) {
    filtered = applicable;
  }

  // Prefer actions that directly add currently-unsatisfied goal literals.
  auto unsatisfied_goals = get_unsatisfied_goal_literals(state, goal);
  std::unordered_map<const Action *, int> action_score;
  action_score.reserve(filtered.size());
  for (const Action *a : filtered) {
    int adds_unsatisfied = 0;
    int adds_relevant = 0;
    auto add_it = action_adds_map.find(a);
    if (add_it != action_adds_map.end()) {
      for (const auto &f : add_it->second) {
        auto rel_it = fluent_relevance_level.find(f);
        if (rel_it != fluent_relevance_level.end()) {
          int level = rel_it->second;
          int weight = std::max(1, relevance_depth + 1 - level);
          adds_relevant += weight;
        }
        if (unsatisfied_goals.count(f)) {
          ++adds_unsatisfied;
        }
      }
    }
    action_score[a] = 100 * adds_unsatisfied + adds_relevant;
  }

  std::sort(filtered.begin(), filtered.end(),
            [&action_score](const Action *lhs, const Action *rhs) {
              auto ls = action_score.find(lhs);
              auto rs = action_score.find(rhs);
              int lval = (ls == action_score.end()) ? 0 : ls->second;
              int rval = (rs == action_score.end()) ? 0 : rs->second;
              if (lval != rval) {
                return lval < rval;
              }
              return lhs->name() > rhs->name();
            });
  return filtered;
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
  int goal_count = 0;
  int path_best_goal_count = 0;

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

inline void set_goal_progress_fields(MCTSDecisionNode &node, const GoalBase *goal,
                                     int parent_path_best_goal_count = 0) {
  if (!goal) {
    node.goal_count = 0;
    node.path_best_goal_count = parent_path_best_goal_count;
    return;
  }
  node.goal_count = goal->goal_count(node.state.fluents());
  node.path_best_goal_count = std::max(parent_path_best_goal_count, node.goal_count);
}

inline std::size_t progressive_widening_limit(
    int visits, double k = PROGRESSIVE_WIDENING_K,
    double alpha = PROGRESSIVE_WIDENING_ALPHA) {
  int safe_visits = std::max(1, visits);
  double raw = k * std::pow(static_cast<double>(safe_visits), alpha);
  return std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(raw)));
}

inline bool should_expand_node(const MCTSDecisionNode &node) {
  if (node.untried_actions.empty()) {
    return false;
  }
  if (!USE_PROGRESSIVE_WIDENING) {
    return true;
  }
  std::size_t limit = progressive_widening_limit(node.visits + 1);
  return node.children.size() < limit;
}

inline const Action *pop_next_action_to_expand(MCTSDecisionNode &node) {
  if (node.untried_actions.empty()) {
    return nullptr;
  }
  const Action *chosen = node.untried_actions.back();
  node.untried_actions.pop_back();
  return chosen;
}

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
                        bool use_det_heuristic = false) {
  // RNG
  static thread_local std::mt19937 rng{std::random_device{}()};

  auto all_actions = get_usable_actions(root_state, all_actions_base);

  std::unordered_map<const Action *, std::unordered_set<Fluent>> action_adds_map;
  action_adds_map.reserve(all_actions.size());
  for (const auto &action : all_actions) {
    const Action *a = &action;
    std::unordered_set<Fluent> adds;
    const auto &succs = a->get_relaxed_successors();
    for (const auto &[succ_state, succ_prob] : succs) {
      if (succ_prob <= 0.0) {
        continue;
      }
      for (const auto &f : succ_state.fluents()) {
        adds.insert(f);
      }
    }
    action_adds_map[a] = std::move(adds);
  }

  // Root node
  auto root = std::make_unique<MCTSDecisionNode>(root_state.copy_and_zero_out_time());
  set_goal_progress_fields(*root, goal, 0);
  root->untried_actions =
      get_goal_relevant_next_actions(root_state, all_actions, goal, action_adds_map);

  // Select heuristic: deterministic (classic FF) or probabilistic
  HeuristicFn heuristic_fn;
  if (use_det_heuristic) {
    heuristic_fn = [goal, all_actions, ff_memory](const State& s) -> double {
      return det_ff_heuristic(s, goal, all_actions, ff_memory);
    };
  } else {
    heuristic_fn = [goal, all_actions, ff_memory](const State& s) -> double {
      return ff_heuristic(s, goal, all_actions, ff_memory);
    };
  }
  std::bernoulli_distribution do_extra_exploration(PROB_EXTRA_EXPLORE);

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
      // Use GoalBase::evaluate for goal check
      if (goal->evaluate(node->state.fluents())) {
        is_node_goal = true;
        break;
      }
      if (should_expand_node(*node))
        break;
      if (node->children.empty())
        break;

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
    if (should_expand_node(*node) && !is_node_goal) {
      const Action *action = pop_next_action_to_expand(*node);
      if (!action) {
        continue;
      }

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
          set_goal_progress_fields(*child_decision, goal, node->path_best_goal_count);
          child_decision->untried_actions =
              get_goal_relevant_next_actions(child_decision->state, all_actions, goal, action_adds_map);
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
    int goal_count_val = node->goal_count;
    int regression = std::max(0, node->path_best_goal_count - node->goal_count);
    double progress_bonus = LANDMARK_PROGRESS_REWARD * static_cast<double>(goal_count_val);
    double regression_penalty = GOAL_REGRESSION_PENALTY * static_cast<double>(regression);
    if (goal->evaluate(node->state.fluents())) {
      reward = -node->state.time() + SUCCESS_REWARD + progress_bonus - regression_penalty -
               accumulated_extra_cost;
    } else {
      h = heuristic_fn ? heuristic_fn(node->state) : 0.0;
      if (h > 1e10) {
        h = HEURISTIC_CANNOT_FIND_GOAL_PENALTY;
      }
      if (did_need_relaxed_transition)
        h += 100;

      reward = -node->state.time() - h * heuristic_multiplier + progress_bonus -
               regression_penalty - accumulated_extra_cost;
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
                       bool use_det_heuristic = false)
      : all_actions_(std::move(all_actions)),
        use_det_heuristic_(use_det_heuristic) {}

  // Call operator: planner(initial_state, goal) → string
  std::string operator()(const State &root_state,
                         const GoalPtr &goal,
                         int max_iterations = 1000,
                         int max_depth = 20,
                         double c = std::sqrt(2.0),
                         double heuristic_multiplier = HEURISTIC_MULTIPLIER) {
    return mcts(root_state, all_actions_, goal.get(), &ff_memory_,
                max_iterations, max_depth, c, heuristic_multiplier,
                &last_mcts_tree_trace_, use_det_heuristic_);
  }

  void clear_cache() { ff_memory_.clear(); }
  std::size_t cache_size() const { return ff_memory_.size(); }

  // Get the tree trace from the most recent MCTS planning call
  const std::string& get_trace_from_last_mcts_tree() const {
    return last_mcts_tree_trace_;
  }

private:
  std::vector<Action> all_actions_;
  bool use_det_heuristic_;
  FFMemory ff_memory_;
  std::string last_mcts_tree_trace_;
};

} // namespace railroad
