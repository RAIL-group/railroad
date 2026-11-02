#pragma once

// ############## Timed A* ###############
//
// Best-first search over the temporal state space for objectives that depend on
// the clock: deadlines, request rates that change over time, a price on
// waiting. The search mechanics are here; the objective comes from a Model
// (costs, heuristic, the price of waiting), so a planner written in Python can
// keep its costs in Python while successors, transitions, goal tests, state
// keys and duplicate detection run in C++ (see `timed_astar_search` in the
// bindings). It is the search of the interruption-aware planners (the iaap
// package) and reproduces its Python reference step for step.
//
//   g' = g + step_cost(s, a, s')  [+ interval_cost(s, a, s')]  [+ terminal_cost(s') if s' is a goal]
//   f  = g' + heuristic(s');   ties break on h, then on insertion order
//
// States are keyed on their fluents and their pending effects at absolute times
// (rounded to 1/key_per_second): a state's cost-to-go depends on the clock, so
// unlike the A* in astar.hpp two copies at different times are not merged.
// With time dominance a copy is pruned only if a copy with the same key was
// reached no later and at no greater cost. When waiting costs something
// (idle_slack), the earlier copy must also pay for the wait: it prunes the
// later one only if cost + idle_cost(s, t_earlier, t_later) is no greater.
// Without time dominance, a key is expanded once and a successor is kept only
// if it is cheaper than every earlier copy.
//
// Out of budget, the result is the best node on the frontier (success false).
// Past the deadline, the search stops with timed_out set.
//
// A Model provides:
//   using Handle = ...;                     // what a node keeps (e.g. a Python State object)
//   Handle make(State &&s);                 // wrap a successor
//   const State &state(const Handle &h);    // the state in a handle (must stay valid while h lives)
//   double step_cost(const Handle &s, std::size_t a, const Handle &s2);
//   double interval_cost(const Handle &s, std::size_t a, const Handle &s2);  // if params.interval
//   double terminal_cost(const Handle &s2);                                  // if params.terminal
//   double heuristic(const Handle &s2);
//   double idle_cost(const Handle &s, double t0, double t1);                 // if params.idle_slack
//   void check_interrupt();                 // every check_interval expansions; throws to stop (Ctrl-C)
// where a is an index into the actions given to timed_astar.

#include "railroad/core.hpp"
#include "railroad/goal.hpp"
#include "railroad/planner.hpp"
#include "railroad/state.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace railroad {

struct TimedAStarParams {
  int budget = 20000;          // node expansions
  bool time_dominance = true;  // see above; false: plain duplicate detection
  bool interval = false;       // the model has an interval cost
  bool terminal = false;       // the model has a terminal cost
  bool idle_slack = false;     // the model prices waiting: idle-slack dominance
  double key_per_second = 1000.0;  // pending effects' times in keys are rounded to 1/this
  double seconds_left = -1.0;  // wall-clock budget; < 0: none
  int check_interval = 256;    // expansions between deadline and interrupt checks
};

struct TimedAStarResult {
  std::vector<std::size_t> plan;  // indices into the actions given to timed_astar
  double cost = std::numeric_limits<double>::infinity();
  bool success = false;
  bool timed_out = false;
  int expanded = 0;
  int generated = 0;  // successors computed
  int pushed = 0;     // successors that entered the frontier
};

namespace timed_astar_detail {

// A state's identity: its fluents (compared through the state) and its pending
// effects as (rounded absolute time, effect hash), sorted.
struct Key {
  const State *state;
  std::vector<std::pair<std::int64_t, std::size_t>> pending;
  std::size_t hash;
};

inline Key make_key(const State &s, double per_second) {
  Key k{&s, {}, 0};
  k.pending.reserve(s.upcoming_effects().size());
  for (const auto &[t, e] : s.upcoming_effects()) {
    k.pending.emplace_back(static_cast<std::int64_t>(std::nearbyint(t * per_second)), e->hash());
  }
  std::sort(k.pending.begin(), k.pending.end());
  std::size_t h_fluents = 0;  // order-free, as State::hash
  for (const auto &f : s.fluents()) {
    std::size_t x = f.hash();
    hash_combine(x, 0);
    h_fluents ^= x;
  }
  std::size_t h = 0;
  hash_combine(h, h_fluents);
  for (const auto &[t, eh] : k.pending) {
    hash_combine(h, std::hash<std::int64_t>{}(t));
    hash_combine(h, eh);
  }
  k.hash = h;
  return k;
}

struct KeyHash {
  std::size_t operator()(const Key &k) const { return k.hash; }
};

struct KeyEq {
  bool operator()(const Key &a, const Key &b) const {
    return a.hash == b.hash && a.pending == b.pending && a.state->fluents() == b.state->fluents();
  }
};

// (cost, time) of the copies of a key seen so far
using Table = std::unordered_map<Key, std::vector<std::pair<double, double>>, KeyHash, KeyEq>;

struct QueueEntry {
  double f;
  double h;
  long order;
  long node;
  // std::priority_queue is a max-heap: "greater" entries come out last
  bool operator<(const QueueEntry &o) const {
    if (f != o.f) return f > o.f;
    if (h != o.h) return h > o.h;
    return order > o.order;
  }
};

template <class Handle>
struct Node {
  Handle handle;
  const State *state;
  long parent;          // -1 for the root
  std::size_t action;   // index into the actions
  double cost;          // g
};

template <class Handle>
std::vector<std::size_t> plan_of(const std::deque<Node<Handle>> &nodes, long n) {
  std::vector<std::size_t> plan;
  for (; nodes[n].parent >= 0; n = nodes[n].parent) plan.push_back(nodes[n].action);
  return {plan.rbegin(), plan.rend()};
}

}  // namespace timed_astar_detail

template <class Model>
TimedAStarResult timed_astar(typename Model::Handle root, const std::vector<Action> &actions,
                             const GoalBase *goal, Model &model, const TimedAStarParams &params) {
  using namespace timed_astar_detail;
  using Handle = typename Model::Handle;
  using Clock = std::chrono::steady_clock;
  const bool has_deadline = params.seconds_left >= 0;
  const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                           std::chrono::duration<double>(std::max(0.0, params.seconds_left)));

  // pointer -> index, for the actions get_next_actions returns
  std::unordered_map<const Action *, std::size_t> index_of;
  index_of.reserve(actions.size());
  for (std::size_t i = 0; i < actions.size(); ++i) index_of[&actions[i]] = i;

  std::deque<Node<Handle>> nodes;  // never moves its elements: states and keys can point into it
  Table closed, reached;           // expanded copies; pushed copies
  std::unordered_set<Key, KeyHash, KeyEq> expanded_keys;              // without time dominance
  std::unordered_map<Key, double, KeyHash, KeyEq> best_cost;          // without time dominance
  std::priority_queue<QueueEntry> frontier;
  long order = 0;
  TimedAStarResult result;

  auto dominated = [&](const Table &table, const Key &k, double cost, double t, const Handle &s) {
    auto it = table.find(k);
    if (it == table.end()) return false;
    for (const auto &[c, u] : it->second) {
      if (!params.idle_slack) {
        if (c <= cost && u <= t) return true;
      } else if (u <= t && c + (t > u ? model.idle_cost(s, u, t) : 0.0) <= cost) {
        return true;
      }
    }
    return false;
  };

  {
    const State *s0 = &model.state(root);
    nodes.push_back(Node<Handle>{std::move(root), s0, -1, 0, 0.0});
  }
  frontier.push(QueueEntry{0.0, -1.0, order++, 0});

  while (result.expanded < params.budget) {
    if (frontier.empty()) return result;  // unreachable: plan empty, cost infinite
    const long current = frontier.top().node;
    frontier.pop();
    const State &s = *nodes[current].state;
    if (goal->evaluate(s.fluents())) {
      result.plan = plan_of(nodes, current);
      result.cost = nodes[current].cost;
      result.success = true;
      return result;
    }
    Key k = make_key(s, params.key_per_second);
    const double node_cost = nodes[current].cost;
    if (params.time_dominance) {
      if (dominated(closed, k, node_cost, s.time(), nodes[current].handle)) continue;
      closed[k].emplace_back(node_cost, s.time());
    } else {
      if (expanded_keys.count(k)) continue;
      expanded_keys.insert(k);
    }
    ++result.expanded;
    if (result.expanded % params.check_interval == 0) {
      model.check_interrupt();
      if (has_deadline && Clock::now() > deadline) {
        result.timed_out = true;
        return result;
      }
    }

    for (const Action *a : get_next_actions(s, actions)) {
      auto outcomes = transition(s, a);
      if (outcomes.size() != 1 || outcomes[0].second != 1.0) {
        throw std::runtime_error("timed_astar: needs deterministic transitions (" + a->name() + ")");
      }
      ++result.generated;
      const std::size_t ai = index_of.at(a);
      Handle h2 = model.make(std::move(outcomes[0].first));
      const State *s2 = &model.state(h2);
      const Handle &h1 = nodes[current].handle;  // deque::push_back leaves existing elements in place
      double g = node_cost + model.step_cost(h1, ai, h2);
      if (params.interval) g += model.interval_cost(h1, ai, h2);
      if (params.terminal && goal->evaluate(s2->fluents())) g += model.terminal_cost(h2);
      Key k2 = make_key(*s2, params.key_per_second);
      const double t = s2->time();
      if (params.time_dominance) {
        if (dominated(closed, k2, g, t, h2) || dominated(reached, k2, g, t, h2)) continue;
        auto it = reached.find(k2);
        if (it == reached.end()) {
          reached.emplace(k2, std::vector<std::pair<double, double>>{{g, t}});
        } else {
          auto &copies = it->second;
          std::vector<std::pair<double, double>> kept;
          kept.reserve(copies.size() + 1);
          for (const auto &[c, u] : copies) {
            const double wait = (params.idle_slack && u > t) ? model.idle_cost(h2, t, u) : 0.0;
            if (!(t <= u && g + wait <= c)) kept.emplace_back(c, u);
          }
          kept.emplace_back(g, t);
          copies = std::move(kept);
        }
      } else {
        if (expanded_keys.count(k2)) continue;
        auto best = best_cost.find(k2);
        if (best != best_cost.end() && g >= best->second) continue;
        if (best == best_cost.end()) {
          best_cost.emplace(k2, g);
        } else {
          best->second = g;
        }
      }
      const double h = model.heuristic(h2);
      nodes.push_back(Node<Handle>{std::move(h2), s2, current, ai, g});
      // the keys stored above point at s2, which now lives in the node (the handle owns it)
      frontier.push(QueueEntry{g + h, h, order++, static_cast<long>(nodes.size() - 1)});
      ++result.pushed;
    }
  }

  if (frontier.empty()) return result;
  const long best = frontier.top().node;
  result.plan = plan_of(nodes, best);
  result.cost = nodes[best].cost;
  return result;
}

}  // namespace railroad
