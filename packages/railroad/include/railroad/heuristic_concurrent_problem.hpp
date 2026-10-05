#pragma once

// The concurrent heuristic's compiled problem: everything that depends only on
// the action set and the goal, compiled once into integer-indexed arrays so
// that a per-state evaluation touches flat arrays only. Also the heuristic's
// public option and breakdown types. See heuristic_concurrent.hpp for the
// overview; the bracketed tags below name the part of the heuristic that
// reads each block.

#include "railroad/core.hpp"
#include "railroad/goal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace railroad {

// Defaults are the configuration to use; the switches exist for ablations.
struct ConcurrentHeuristicOptions {
  // Value = lambda_add * sum of goal completion times + lambda_ms * makespan.
  double lambda_add = 0.5;
  double lambda_ms = 0.5;
  // Object-search convention: a goal `at X L` also needs `found X`.
  bool at_implies_found = true;
  // Schedule goals onto agents (true) or onto one serial "team" agent (false).
  bool agent_aware = true;
  // Make in-flight effects available when they are scheduled (true) or at
  // time 0, as the FF heuristic's relaxed transition does (false).
  bool timed_init = true;
  // Plan a goal `at X L` together with the `found X` it implies, as one task.
  bool joint_found = true;
  // Re-cost an agent's moves as a route through the locations its task
  // needs (false: the relaxed plan's own move durations).
  bool route_chaining = true;
  // Treat achievers that consume the same precondition as one attempt.
  bool group_attempts = true;
  // On one agent, do a task before any task whose plan would destroy a fact
  // (true now) that it relies on -- e.g. deliver the object in hand before
  // fetching another, which needs the hand free.
  bool order_conflicts = true;
  // Cost a task's uncertain search as an expected route over its candidate
  // places (and the rest of the task from wherever the object turns up),
  // instead of a plan through one place plus a retry delta.
  bool expected_search = true;
  // The lambda_add term: the scheduled goals' summed completion times (true)
  // or the contention-blind h_add + delta of the unrestricted relaxation.
  bool sum_completion = true;
  // MCTS charges what the value estimates: per unit of time, lambda_ms plus
  // lambda_add per open goal task (makespan plus flowtime). False: elapsed
  // time only, under which the value falls faster than time passes whenever
  // several goal tasks are open.
  bool flowtime_objective = false;
};

// Per-branch components of the most recent evaluation (for introspection).
struct ConcurrentHeuristicBreakdown {
  double value = 0.0;
  double h_add = 0.0;
  double delta = 0.0;
  double makespan = 0.0;
  double completion_sum = 0.0;
  double h_ff = 0.0;
  std::vector<std::pair<std::string, double>> goal_finish;  // fluent, finish
  std::vector<std::pair<std::string, std::string>> assignment;  // fluent, agent
  std::vector<std::pair<std::string, double>> deltas;  // fluent, retry delta
  std::vector<std::pair<std::string, double>> plan;    // action, duration
  // task, agent index, serial duration, critical-path time, retry delta
  std::vector<std::tuple<std::string, int, double, double, double>> loads;
};

namespace concurrent {

inline constexpr double INF = std::numeric_limits<double>::infinity();

struct Add {
  int fluent;
  double prob;       // P(action adds fluent), summed over branches
  bool conditional;  // added in some but not all branches
  bool certain() const { return prob >= 1.0 - 1e-9; }
  bool uncertain() const { return !certain() || conditional; }
};

struct CompiledAction {
  const Action *src;
  double dur;                 // max relaxed successor time + extra_cost
  std::vector<int> pre;       // positive preconditions
  std::vector<Add> adds;
  std::vector<int> consumes;  // preconditions it deterministically deletes
  int agent;                  // -1 agent-free, -2 several agents, else id
};

struct AchieverRef {
  int action;
  int add;    // index into CompiledAction::adds
  int group;  // achievers sharing a group are one attempt
};

struct Problem {
  Problem(const std::vector<Action> &actions, const GoalBase *goal,
          const ConcurrentHeuristicOptions &opts) {
    compile_actions(actions, goal);
    compile_attempt_groups(opts);
    compile_routes();
    compile_found();
    compile_goal(goal);
  }
  Problem(const Problem &) = delete;
  Problem &operator=(const Problem &) = delete;

  // -- Fluents ---------------------------------------------------------
  std::size_t num_fluents() const { return fluents_.size(); }
  const Fluent &fluent(int f) const { return fluents_[f]; }
  int lookup(const Fluent &f) const {
    auto it = fid_.find(f);
    return it == fid_.end() ? -1 : it->second;
  }
  std::string str(int f) const {
    const Fluent &fl = fluents_[f];
    std::string s = fl.is_negated() ? "not " : "";
    s += fl.name();
    for (const auto &a : fl.args()) s += " " + a;
    return s;
  }

  // -- Actions [relaxation] ----------------------------------------------
  std::vector<CompiledAction> acts;
  std::vector<std::vector<int>> consumers;          // fluent -> actions needing it
  std::vector<std::vector<AchieverRef>> achievers;  // fluent -> achievers
  bool adds(int a, int f) const {
    if (f < 0) return false;
    for (const auto &ad : acts[a].adds) {
      if (ad.fluent == f) return true;
    }
    return false;
  }

  // -- Agents [schedule]: the arguments of `free` preconditions ----------
  std::vector<int> agent_free;  // agent -> fid of `free <agent>`
  std::vector<std::string> agent_names;
  std::vector<std::vector<int>> agent_acts;  // agent -> its actions
  std::size_t num_agents() const { return agent_free.size(); }

  // -- Attempt groups [search]: achievers of a fluent that consume the
  //    same precondition are one attempt (whichever runs first uses it up).
  //    fluent -> (consumed fluent, group), for grouping pending outcomes.
  std::vector<std::vector<std::pair<int, int>>> group_by_consumed;

  // -- Routes [tasks]: each agent's location fluents -- the mutex group its
  //    moves switch between -- and the places they name.
  std::vector<int> loc_agent;                // location fluent -> its agent (or -1)
  std::vector<std::vector<int>> loc_fluents;  // agent -> its location fluents
  std::vector<int> loc_place;                // location fluent -> place (or -1)
  std::vector<std::unordered_map<int, int>> place_loc;  // agent -> place -> fluent

  // Agent r's location fluent for the place of location fluent `loc`.
  int as_agent_loc(int r, int loc) const {
    if (loc < 0 || r < 0 || loc_agent[loc] == r) return loc;
    int pl = loc_place[loc];
    if (pl < 0) return -1;
    auto it = place_loc[r].find(pl);
    return it == place_loc[r].end() ? -1 : it->second;
  }
  // The location fluent of agent r that action a moves it to (-1: a is not
  // one of r's moves).
  int move_dest(int a, int r) const {
    for (const auto &ad : acts[a].adds) {
      if (loc_agent[ad.fluent] == r && ad.certain() &&
          std::find(acts[a].pre.begin(), acts[a].pre.end(), ad.fluent) == acts[a].pre.end()) {
        return ad.fluent;
      }
    }
    return -1;
  }
  // Duration of agent r's cheapest deterministic action that moves it from
  // location fluent `from` to `to` (INF if none). Cached.
  double move_dur(int r, int from, int to) const {
    if (from == to) return 0.0;
    uint64_t key = (static_cast<uint64_t>(r) << 42) ^
                   (static_cast<uint64_t>(from) << 21) ^ static_cast<uint64_t>(to);
    auto it = move_cache_.find(key);
    if (it != move_cache_.end()) return it->second;
    double best = INF;
    for (const auto &ach : achievers[to]) {
      const CompiledAction &a = acts[ach.action];
      if (a.agent != r || !a.adds[ach.add].certain()) continue;
      if (std::find(a.pre.begin(), a.pre.end(), from) != a.pre.end()) best = std::min(best, a.dur);
    }
    move_cache_.emplace(key, best);
    return best;
  }

  // -- Object-search convention [tasks]: `at X L` -> `found X` (or -1) ----
  std::vector<int> found_of;

  // -- Goal: DNF branches (fluent ids, -1 = in no action or state) --------
  std::vector<std::vector<int>> branches;

 private:
  std::unordered_map<Fluent, int> fid_;
  std::vector<Fluent> fluents_;
  mutable std::unordered_map<uint64_t, double> move_cache_;

  int intern(const Fluent &f) {
    auto it = fid_.find(f);
    if (it != fid_.end()) return it->second;
    int id = static_cast<int>(fluents_.size());
    fid_.emplace(f, id);
    fluents_.push_back(f);
    return id;
  }

  void compile_actions(const std::vector<Action> &actions, const GoalBase *goal) {
    acts.reserve(actions.size());
    std::unordered_map<int, int> agent_of_free;  // free fid -> agent id
    for (const auto &a : actions) {
      CompiledAction ca;
      ca.src = &a;
      ca.agent = -1;
      for (const auto &p : a.pos_preconditions()) {
        int id = intern(p);
        ca.pre.push_back(id);
        if (p.is_free()) {
          auto it = agent_of_free.find(id);
          int ag;
          if (it == agent_of_free.end()) {
            ag = static_cast<int>(agent_free.size());
            agent_of_free.emplace(id, ag);
            agent_free.push_back(id);
            agent_names.push_back(p.args().empty() ? p.name() : p.args()[0]);
          } else {
            ag = it->second;
          }
          ca.agent = (ca.agent == -1) ? ag : -2;
        }
      }

      // Aggregate per-fluent achievement probability across the action's
      // mutually exclusive outcome branches (as ff_forward_phase does).
      const auto &succs = a.get_relaxed_successors();
      std::unordered_map<Fluent, double> prob;
      std::unordered_map<Fluent, std::size_t> membership;
      double duration = 0.0;
      for (const auto &[succ_state, succ_prob] : succs) {
        duration = std::max(duration, succ_state.time());
        for (const auto &f : succ_state.fluents()) {
          membership[f] += 1;
          if (succ_prob > 0.0) prob[f] += succ_prob;
        }
      }
      ca.dur = duration + a.extra_cost();
      // Branch probabilities may come from single-precision estimates; a
      // fluent every branch adds is certain whenever they sum to one.
      double total = 0.0;
      for (const auto &sp : succs) total += std::max(sp.second, 0.0);
      const bool complete = std::abs(total - 1.0) < 1e-6;
      for (const auto &[f, p] : prob) {
        if (f.is_negated()) continue;
        bool conditional = succs.size() > 1 && membership[f] < succs.size();
        double pf = (!conditional && complete) ? 1.0 : std::min(p, 1.0);
        ca.adds.push_back({intern(f), pf, conditional});
      }

      // Deterministic consumption: a precondition deleted by a top-level
      // effect and not re-added at or after that time by another one.
      std::unordered_map<int, double> del_time, readd_time;
      for (const auto &e : a.effects()) {
        for (const auto &f : e->flipped_neg_fluents()) {
          auto it = fid_.find(f);
          if (it == fid_.end()) continue;
          auto [dit, inserted] = del_time.emplace(it->second, e->time());
          if (!inserted) dit->second = std::max(dit->second, e->time());
        }
      }
      for (const auto &e : a.effects()) {
        for (const auto &f : e->pos_fluents()) {
          auto it = fid_.find(f);
          if (it == fid_.end()) continue;
          auto [rit, inserted] = readd_time.emplace(it->second, e->time());
          if (!inserted) rit->second = std::max(rit->second, e->time());
        }
      }
      for (int p : ca.pre) {
        auto dit = del_time.find(p);
        if (dit == del_time.end()) continue;
        auto rit = readd_time.find(p);
        if (rit != readd_time.end() && rit->second >= dit->second) continue;
        ca.consumes.push_back(p);
      }
      acts.push_back(std::move(ca));
    }

    agent_acts.assign(agent_free.size(), {});
    for (int ai = 0; ai < static_cast<int>(acts.size()); ++ai) {
      if (acts[ai].agent >= 0) agent_acts[acts[ai].agent].push_back(ai);
    }

    // Goal fluents may be absent from every action; intern them too so that
    // they can at least be satisfied by the state.
    if (goal) {
      for (const auto &f : goal->get_all_literals()) intern(f);
    }

    std::size_t nf = fluents_.size();
    consumers.assign(nf, {});
    achievers.assign(nf, {});
    for (int ai = 0; ai < static_cast<int>(acts.size()); ++ai) {
      for (int p : acts[ai].pre) consumers[p].push_back(ai);
      for (int k = 0; k < static_cast<int>(acts[ai].adds.size()); ++k) {
        achievers[acts[ai].adds[k].fluent].push_back({ai, k, 0});
      }
    }
  }

  // Group each fluent's achievers by shared consumed preconditions.
  void compile_attempt_groups(const ConcurrentHeuristicOptions &opts) {
    std::size_t nf = fluents_.size();
    group_by_consumed.assign(nf, {});
    for (std::size_t f = 0; f < nf; ++f) {
      auto &achs = achievers[f];
      std::vector<int> parent(achs.size());
      std::iota(parent.begin(), parent.end(), 0);
      auto find = [&parent](int x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
      };
      std::unordered_map<int, int> first_with;
      for (int i = 0; i < static_cast<int>(achs.size()); ++i) {
        for (int c : acts[achs[i].action].consumes) {
          auto [it, inserted] = first_with.emplace(c, i);
          if (!inserted) parent[find(i)] = find(it->second);
        }
      }
      for (int i = 0; i < static_cast<int>(achs.size()); ++i) {
        achs[i].group = opts.group_attempts ? find(i) : i;
      }
      bool any_prob = false;
      for (const auto &r : achs) {
        if (acts[r.action].adds[r.add].uncertain()) any_prob = true;
      }
      if (any_prob && opts.group_attempts) {
        for (const auto &[c, i] : first_with) group_by_consumed[f].push_back({c, find(i)});
      }
    }
  }

  // Agent location fluents: an action run by agent r that, through its
  // top-level effects, adds P(r, y, ...) and deletes P(r, x, ...) moves r
  // between two values of one mutex group (e.g. `at r x` -> `at r y`).
  // Places: a location fluent with its agent argument removed, so that
  // `at robot1 L` and `at robot2 L` name the same place.
  void compile_routes() {
    std::size_t nf = fluents_.size();
    loc_agent.assign(nf, -1);
    loc_fluents.assign(agent_free.size(), {});
    for (const auto &ca : acts) {
      if (ca.agent < 0) continue;
      const std::string &name = agent_names[ca.agent];
      std::vector<const Fluent *> adds, dels;
      for (const auto &e : ca.src->effects()) {
        for (const auto &f : e->pos_fluents()) adds.push_back(&f);
        for (const auto &f : e->flipped_neg_fluents()) dels.push_back(&f);
      }
      for (const Fluent *x : adds) {
        if (x->args().size() < 2 || x->args()[0] != name) continue;
        for (const Fluent *y : dels) {
          if (y->name() != x->name() || y->args().size() != x->args().size() ||
              y->args()[0] != name || *y == *x) continue;
          for (const Fluent *z : {x, y}) {
            int id = lookup(*z);
            if (id >= 0 && loc_agent[id] < 0) {
              loc_agent[id] = ca.agent;
              loc_fluents[ca.agent].push_back(id);
            }
          }
        }
      }
    }

    loc_place.assign(nf, -1);
    place_loc.assign(agent_free.size(), {});
    std::unordered_map<std::string, int> place_id;
    for (std::size_t ag = 0; ag < agent_free.size(); ++ag) {
      for (int f : loc_fluents[ag]) {
        const Fluent &fl = fluents_[f];
        std::string key = fl.name();
        for (std::size_t i = 1; i < fl.args().size(); ++i) key += " " + fl.args()[i];
        auto [it, inserted] = place_id.emplace(key, static_cast<int>(place_id.size()));
        loc_place[f] = it->second;
        place_loc[ag].emplace(it->second, f);
      }
    }
  }

  void compile_found() {
    std::size_t nf = fluents_.size();
    found_of.assign(nf, -1);
    for (std::size_t f = 0; f < nf; ++f) {
      const Fluent &fl = fluents_[f];
      if (fl.is_negated() || fl.name() != "at" || fl.args().empty()) continue;
      found_of[f] = lookup(Fluent("found", {fl.args()[0]}));
    }
  }

  void compile_goal(const GoalBase *goal) {
    if (!goal) return;
    GoalType type = goal->get_type();
    if (type == GoalType::TRUE_GOAL || type == GoalType::FALSE_GOAL) return;
    // Large DNFs are rare; cap enumeration like ff_heuristic does and fall
    // back to the literal union (an over-constrained but finite estimate).
    constexpr std::size_t kMaxBranches = 1024;
    if (goal->dnf_branch_count() == 0) return;
    if (goal->dnf_branch_count() > kMaxBranches) {
      std::vector<int> b;
      for (const auto &f : goal->get_all_literals()) b.push_back(lookup(f));
      branches.push_back(std::move(b));
      return;
    }
    for (const auto &br : goal->get_dnf_branches()) {
      std::vector<int> b;
      for (const auto &f : br) b.push_back(lookup(f));
      branches.push_back(std::move(b));
    }
  }
};

}  // namespace concurrent
}  // namespace railroad
