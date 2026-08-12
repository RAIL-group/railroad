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

  std::vector<const Action *> result = applicable;

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
  if (free_fluents.size() > 1) {
    // Deterministic order by robot id (the `free` fluent's argument).
    std::sort(free_fluents.begin(), free_fluents.end(),
              [](const Fluent *a, const Fluent *b) { return a->args() < b->args(); });

    // Take the actions of the first free robot that has any; only fall through
    // to later robots if an earlier one has no applicable action (it is stuck),
    // so a free-but-idle robot never blocks the others.
    for (const Fluent *free_robot : free_fluents) {
      std::vector<const Action *> subset;
      for (const Action *action : applicable) {
        if (action->pos_preconditions().count(*free_robot)) {
          subset.push_back(action);
        }
      }
      if (!subset.empty()) { result = std::move(subset); break; }
    }
  }

  // Don't let the planner wait by choice: drop no_op (wait) actions unless they
  // are the only option, i.e. a free robot genuinely has nothing else to do.
  std::vector<const Action *> non_wait;
  for (const Action *action : result) {
    if (action->name().rfind("no_op", 0) != 0) non_wait.push_back(action);
  }
  if (!non_wait.empty()) return non_wait;
  return result;
}

// ############## A* ###############

using QueueEntry = std::tuple<double, State>;

// For backtracking. Keyed on the state hash, holding the parent's hash and the action that got
// here by value -- get_next_actions hands back pointers into the caller's vector, and those are
// only good for the length of this search. The action is optional because letting the clock run is
// also a step between states, and it belongs in the chain without appearing in the plan.
using CameFromMap =
    std::unordered_map<std::size_t, std::pair<std::size_t, std::optional<Action>>>;

inline std::vector<Action> reconstruct_path(const CameFromMap &came_from,
                                            std::size_t current) {
  std::vector<Action> path;
  auto it = came_from.find(current);
  while (it != came_from.end()) {
    if (it->second.second.has_value())
      path.push_back(*it->second.second);
    current = it->second.first;
    it = came_from.find(current);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

// A* over the concurrent state model. g is the state's own clock, so what this minimises is the
// makespan of the plan, and a heuristic handed in here has to be in the same units to keep the
// result optimal.
inline std::optional<std::vector<Action>>
astar(const State &start_state, const std::vector<Action> &all_actions,
      const GoalPtr &goal,
      HeuristicFn heuristic_fn = nullptr) {
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>>
      open_heap;
  CameFromMap came_from;
  // cheapest g seen per state, which is what stops a worse route to somewhere we have already been
  // from being followed at all.
  std::unordered_map<std::size_t, double> best_g;
  // states already expanded. The same state reaches the heap by several routes, and without this
  // every one of those entries gets expanded again; on the reduced multi-robot problem that is the
  // difference between a search that closes and one that walks the same ground repeatedly.
  std::unordered_set<std::size_t> visited;

  FFMemory ff_memory;
  if (!heuristic_fn) {
    heuristic_fn = [&goal, &all_actions, &ff_memory](const State &s) -> double {
      return ff_heuristic(s, goal.get(), all_actions, &ff_memory);
    };
  }

  best_g[start_state.hash()] = start_state.time();
  open_heap.emplace(heuristic_fn(start_state), start_state);

  while (!open_heap.empty()) {
    QueueEntry top = open_heap.top();
    State current = std::get<1>(top);
    open_heap.pop();

    if (goal->evaluate(current.fluents())) {
      return reconstruct_path(came_from, current.hash());
    }

    // a stale heap entry for a state we have since reached more cheaply
    auto seen = best_g.find(current.hash());
    if (seen != best_g.end() && current.time() > seen->second)
      continue;

    // expanded already, by this or a cheaper route: do not walk it again
    if (!visited.insert(current.hash()).second)
      continue;

    auto push_successor = [&](State successor, const Action *via) {
      double g = successor.time();
      auto known = best_g.find(successor.hash());
      if (known != best_g.end() && g >= known->second)
        return;

      best_g[successor.hash()] = g;
      came_from[successor.hash()] = std::make_pair(
          current.hash(), via ? std::optional<Action>(*via) : std::nullopt);

      open_heap.emplace(g + heuristic_fn(successor), std::move(successor));
    };

    auto next_actions = get_next_actions(current, all_actions);
    for (const auto action : next_actions) {
      for (const auto &[successor, prob] : transition(current, action)) {
        if (prob == 0.0)
          continue;
        push_successor(successor, action);
      }
    }

    // Nothing applies but something is still in flight: let the clock run to the next scheduled
    // effect and resolve it, which is what SymbolicEnvironment::act does between decisions.
    // Without this a state whose last robot is still crossing has no successor at all, so a goal
    // that only becomes true on arrival is unreachable and the search reports no plan. Jumping the
    // clock to the due time first is what gets past advance_to_terminal's rule of handing control
    // back while anyone is free -- here nobody free has anything to do.
    if (next_actions.empty() && !current.upcoming_effects().empty()) {
      State waited = current;
      double due = waited.upcoming_effects().front().first;
      if (due > waited.time())
        waited.set_time(due);
      for (const auto &[successor, prob] : transition(waited, nullptr)) {
        if (prob == 0.0)
          continue;
        push_successor(successor, nullptr);
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

// The RNG behind MCTS outcome sampling. Thread-local, seeded from
// std::random_device by default; seed via seed_mcts_rng() for reproducible
// planning (the seed applies only to the calling thread).
inline std::mt19937 &mcts_rng() {
  static thread_local std::mt19937 rng{std::random_device{}()};
  return rng;
}

inline void seed_mcts_rng(unsigned int seed) { mcts_rng().seed(seed); }

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
                        double lambda_ff  = 0.5,
                        std::optional<double> dead_end_penalty = std::nullopt,
                        HeuristicFn external_heuristic = nullptr,
                        double unreachable_penalty = HEURISTIC_CANNOT_FIND_GOAL_PENALTY) {
  // RNG
  std::mt19937 &rng = mcts_rng();

  auto all_actions = get_usable_actions(root_state, all_actions_base);

  // Root node
  auto root = std::make_unique<MCTSDecisionNode>(root_state.copy_and_zero_out_time());
  root->untried_actions = get_next_actions(root_state, all_actions);

  // A caller-supplied leaf value (e.g. a risk-aware value function) replaces FF entirely; the
  // lambda_* weights then have nothing to mix, since they only ever fed ff_heuristic.
  HeuristicFn heuristic_fn = external_heuristic
      ? external_heuristic
      : HeuristicFn([goal, all_actions, ff_memory,
                     lambda_add, lambda_max, lambda_ff](const State& s) -> double {
          return ff_heuristic(s, goal, all_actions, ff_memory,
                              lambda_add, lambda_max, lambda_ff);
        });

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
      if (h > 1e10 && dead_end_penalty) {
        // The relaxation proved the goal unreachable from here. Charge a
        // *flat* cost: what this branch spent getting here is irrelevant,
        // because no continuation of it can reach the goal. Folding the
        // elapsed time and accumulated cost in would rank a slow failure
        // below a fast one and push the search toward failing quickly,
        // which is not a preference we want to express.
        reward = -*dead_end_penalty;
      } else {
        if (h > 1e10) {
          // The heuristic says the goal is out of reach and no flat dead-end cost was set, so
          // the caller decides what that is worth. Defaults to HEURISTIC_CANNOT_FIND_GOAL_PENALTY,
          // which is the behaviour every existing caller already gets.
          h = unreachable_penalty;
        }
        if (did_need_relaxed_transition)
          h += 100;

        reward = -node->state.time() - h * heuristic_multiplier + 0 * goal_count_val - accumulated_extra_cost;
      }
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
                       double lambda_ff  = 0.5,
                       std::optional<double> dead_end_penalty = std::nullopt)
      : all_actions_(std::move(all_actions)),
        lambda_add_(lambda_add),
        lambda_max_(lambda_max),
        lambda_ff_(lambda_ff),
        dead_end_penalty_(dead_end_penalty) {}

  // Call operator: planner(initial_state, goal) → string
  std::string operator()(const State &root_state,
                         const GoalPtr &goal,
                         int max_iterations = 1000,
                         int max_depth = 20,
                         double c = std::sqrt(2.0),
                         double heuristic_multiplier = HEURISTIC_MULTIPLIER,
                         HeuristicFn heuristic_fn = nullptr,
                         double unreachable_penalty = HEURISTIC_CANNOT_FIND_GOAL_PENALTY) {
    return mcts(root_state, all_actions_, goal.get(), &ff_memory_,
                max_iterations, max_depth, c, heuristic_multiplier,
                &last_mcts_tree_trace_,
                lambda_add_, lambda_max_, lambda_ff_, dead_end_penalty_,
                heuristic_fn, unreachable_penalty);
  }

  void clear_cache() { ff_memory_.clear(); }
  std::size_t cache_size() const { return ff_memory_.size(); }

  double lambda_add() const { return lambda_add_; }
  double lambda_max() const { return lambda_max_; }
  double lambda_ff()  const { return lambda_ff_; }
  std::optional<double> dead_end_penalty() const { return dead_end_penalty_; }

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
  std::optional<double> dead_end_penalty_;
};

} // namespace railroad
