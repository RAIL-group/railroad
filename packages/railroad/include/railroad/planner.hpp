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

  // Ban no_op actions unless they are the only applicable option. A free robot
  // can always no_op, which means every non-terminal node has a trivial child
  // that just advances time; without filtering, MCTS burns rollouts on these
  // do-nothing actions instead of exploring real progress.
  std::vector<const Action *> non_no_op;
  non_no_op.reserve(fallback.size());
  for (const Action *a : fallback) {
    if (!a->is_no_op()) {
      non_no_op.push_back(a);
    }
  }
  if (!non_no_op.empty()) {
    return non_no_op;
  }

  return fallback;
}

// Pick the active DNF branch index for a state using the same ordering as
// `get_unsatisfied_goal_literals` (most-satisfied, then fewest-unsatisfied,
// then smallest). Returns -1 if the goal has no branches.
inline int select_best_branch_index(const State &state, const GoalBase *goal) {
  if (!goal) return -1;
  const auto &branches = goal->get_dnf_branches();
  if (branches.empty()) return -1;

  auto literal_satisfied = [&state](const Fluent &lit) {
    if (lit.is_negated()) return !state.fluents().count(lit.invert());
    return state.fluents().count(lit) > 0;
  };

  int best_idx = -1;
  int best_satisfied = -1;
  int best_unsatisfied = std::numeric_limits<int>::max();
  std::size_t best_size = std::numeric_limits<std::size_t>::max();
  for (std::size_t i = 0; i < branches.size(); ++i) {
    const auto &branch = branches[i];
    int satisfied = 0;
    int branch_unsatisfied = 0;
    for (const auto &lit : branch) {
      if (literal_satisfied(lit)) ++satisfied;
      else ++branch_unsatisfied;
    }
    if (best_idx < 0 || satisfied > best_satisfied ||
        (satisfied == best_satisfied && branch_unsatisfied < best_unsatisfied) ||
        (satisfied == best_satisfied && branch_unsatisfied == best_unsatisfied &&
         branch.size() < best_size)) {
      best_idx = static_cast<int>(i);
      best_satisfied = satisfied;
      best_unsatisfied = branch_unsatisfied;
      best_size = branch.size();
    }
  }
  return best_idx;
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

inline std::unordered_set<Fluent> unsatisfied_literals_from_branch(
    const State &state, const std::unordered_set<Fluent> &branch) {
  auto literal_satisfied = [&state](const Fluent &lit) {
    if (lit.is_negated()) return !state.fluents().count(lit.invert());
    return state.fluents().count(lit) > 0;
  };
  std::unordered_set<Fluent> unsatisfied;
  for (const auto &lit : branch) {
    if (!literal_satisfied(lit)) unsatisfied.insert(lit);
  }
  return unsatisfied;
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
  std::unordered_set<const Action *> helpful_actions; // FF helpful set for this state
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
                        double c = std::sqrt(2.0), bool is_helpful = false) {
  if (child.visits == 0)
    return std::numeric_limits<double>::infinity();
  const double exploitation = child.value / static_cast<double>(child.visits);
  const double exploration =
      c * std::sqrt(std::log(static_cast<double>(parent_visits)) /
                    static_cast<double>(child.visits));
  // Preferred-actions prior: small bonus for FF-helpful actions that decays
  // as the child accumulates visits. Steers early exploration without ever
  // hard-pruning the unhelpful set.
  const double prior = is_helpful
                           ? HELPFUL_ACTION_PRIOR /
                                 (1.0 + static_cast<double>(child.visits))
                           : 0.0;
  return exploitation + exploration + prior;
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

    // Count how many helpful actions remain unexpanded in untried.
    int helpful_untried = 0;
    for (const Action *a : node->untried_actions) {
      if (node->helpful_actions.count(a)) ++helpful_untried;
    }
    const int total_applicable =
        static_cast<int>(node->children.size() + node->untried_actions.size());

    os << "D:" << current_depth << "|="
       << "visits=" << node->visits << ", "
       << "Q=" << q_value << ", "
       << "g=" << time_cost << ", "
       << "h=" << h_value << ", "
       << "g+h=" << time_cost + h_value << ", "
       << "#A=" << node->children.size() << "/" << total_applicable
       << " (helpful " << helpful_untried << "/"
       << node->untried_actions.size() << ")"
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
                        std::string* out_tree_trace = nullptr) {
  // RNG
  static thread_local std::mt19937 rng{std::random_device{}()};

  auto all_actions = get_usable_actions(root_state, all_actions_base);

  // Build per-decision-node action lists: keep all applicable actions in
  // `untried_actions`, but record which are FF-helpful and order the vector so
  // helpful actions are at the back (popped first). UCB selection reads
  // `helpful_actions` to apply a decaying prior bonus.
  auto populate_actions = [&all_actions, goal, ff_memory](MCTSDecisionNode &node) {
    auto applicable = get_next_actions(node.state, all_actions);
    auto helpful_vec =
        get_helpful_actions(node.state, applicable, goal, all_actions, ff_memory);
    node.helpful_actions = std::unordered_set<const Action *>(
        helpful_vec.begin(), helpful_vec.end());
    node.untried_actions.clear();
    node.untried_actions.reserve(applicable.size());
    // Non-helpful first, helpful last so pop_back yields helpful first.
    for (const Action *a : applicable) {
      if (!node.helpful_actions.count(a)) node.untried_actions.push_back(a);
    }
    for (const Action *a : applicable) {
      if (node.helpful_actions.count(a)) node.untried_actions.push_back(a);
    }
  };

  // Root node
  auto root = std::make_unique<MCTSDecisionNode>(root_state.copy_and_zero_out_time());
  set_goal_progress_fields(*root, goal, 0);
  populate_actions(*root);

  HeuristicFn heuristic_fn = [goal, all_actions, ff_memory](const State& s) -> double {
    return ff_heuristic(s, goal, all_actions, ff_memory);
  };
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
        bool is_helpful = node->helpful_actions.count(kv.first) > 0;
        double score = ucb_score(node->visits, *cn, c, is_helpful);
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
          populate_actions(*child_decision);
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
  explicit MCTSPlanner(std::vector<Action> all_actions)
      : all_actions_(std::move(all_actions)) {}

  // Call operator: planner(initial_state, goal) → string
  std::string operator()(const State &root_state,
                         const GoalPtr &goal,
                         int max_iterations = 1000,
                         int max_depth = 20,
                         double c = std::sqrt(2.0),
                         double heuristic_multiplier = HEURISTIC_MULTIPLIER) {
    return mcts(root_state, all_actions_, goal.get(), &ff_memory_,
                max_iterations, max_depth, c, heuristic_multiplier,
                &last_mcts_tree_trace_);
  }

  void clear_cache() { ff_memory_.clear(); }
  std::size_t cache_size() const { return ff_memory_.size(); }

  // Get the tree trace from the most recent MCTS planning call
  const std::string& get_trace_from_last_mcts_tree() const {
    return last_mcts_tree_trace_;
  }

  std::string debug_heuristic(const State &state, const GoalPtr &goal) {
    return ff_heuristic_debug_report(state, goal.get(), all_actions_, &ff_memory_);
  }

private:
  std::vector<Action> all_actions_;
  FFMemory ff_memory_;
  std::string last_mcts_tree_trace_;
};

} // namespace railroad
