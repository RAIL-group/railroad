#pragma once

// Concurrency-aware relaxed-plan heuristic ("h_conc").
//
// The FF heuristic (heuristic.hpp) estimates the cost of a *sequential* plan:
// h_ff sums the durations of every relaxed-plan action whichever robot runs
// it, and its relaxed initial state treats every in-flight effect as already
// done. Both are harmless with one robot. With several, the objective (time
// until the goal holds) is a makespan, MCTS leaves are reached while other
// robots are mid-action, and the estimate degrades (it undercounts remaining
// time by 30-50% on the ProcTHOR search benchmark with 2-3 robots). This
// header estimates the remaining time of the *team*:
//
//   1. Timed relaxed initial state. Fluents an in-flight action will add are
//      available when they are scheduled, not at time 0, so committed but
//      unfinished work still counts (cf. the temporal relaxed planning graph
//      of CRIKEY/POPF). The outcomes of an in-flight probabilistic effect
//      become *pending achievers* carrying their branch probability, rather
//      than one branch picked arbitrarily.
//
//   2. Probability-aware relaxed costs. Each fluent carries the probability
//      rho that its relaxed support succeeds (the product along the chosen
//      achievers); achievers are ranked by cost / rho^prob_exponent. The
//      ranking is monotone along supports, so all costs come from one
//      Dijkstra-style pass. A large exponent routes a relaxed plan through
//      where an object most likely is, not the nearest place it might be.
//
//   3. Per-agent relaxations. Agents are the arguments of `free` fluents (the
//      core's existing convention). The relaxation is re-run per agent with
//      only that agent's actions (plus agent-free actions and pending
//      effects): the cost of each goal if that agent does it alone.
//
//   4. Route chaining. A delete relaxation lets an agent be in several places
//      at once -- a fetch costs start->object + start->target, and an agent
//      standing at the target never has to come back. Each agent's location
//      fluents are recognised as a mutex group (actions that add P(r, y) and
//      delete P(r, x)); a task's moves are re-costed as one route through the
//      locations the plan needs, in the order it needs them.
//
//   5. List scheduling. Goals are tasks (a goal `at X L` together with the
//      `found X` it implies); longest first, each goes to the agent that
//      would finish it earliest (LPT/EFT), from each agent's ready time.
//
//   6. Retry deltas without phantom independence. Probabilistic achievers
//      that consume the same precondition (two robots searching one place
//      both delete `not-searched place obj`) are one attempt, not two. The
//      expected retry overhead (best of several attempt orderings) is charged
//      to the task of the agent that makes the attempts.
//
// The value of a goal branch is
//     lambda_add * sum_g C_g + lambda_ms * max_g C_g
// over the scheduled completion times C_g -- makespan plus a sum that gives
// every task, not only the critical one, a gradient -- minimised over DNF
// branches. With one agent and no in-flight effects it is a sequential,
// route-aware h_ff.
//
// Everything that depends only on the action set and goal is compiled once
// (integer fluent ids, precondition/achiever adjacency, agent partition), so a
// per-state evaluation touches flat arrays only; it is several times faster
// than ff_heuristic on large grounded problems.

#include "railroad/core.hpp"
#include "railroad/goal.hpp"
#include "railroad/state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace railroad {

struct ConcurrentHeuristicOptions {
  double lambda_add = 0.5;
  double lambda_ms = 0.5;
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
  // Cost a task's uncertain search as an expected route over its candidate
  // places (and the rest of the task from wherever the object turns up),
  // instead of a plan through one place plus a retry delta.
  bool expected_search = false;
  // With expected_search: agents left without a task join the slowest
  // uncertain search (a parallel list schedule of its attempts).
  bool parallel_search = false;
  // Achievers are ranked by cost / rho^prob_exponent. 1 is the expected cost
  // of independent retries; larger values let probability dominate, so a
  // relaxed plan routes through where an object most likely is rather than
  // the nearest place it might be.
  double prob_exponent = 8.0;
  // The lambda_add term: the scheduled goals' summed completion times (true)
  // or the contention-blind h_add + delta of the unrestricted relaxation.
  bool sum_completion = true;
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

class ConcurrentHeuristic {
public:
  static constexpr double INF = std::numeric_limits<double>::infinity();

  ConcurrentHeuristic(const std::vector<Action> &actions, const GoalBase *goal,
                      ConcurrentHeuristicOptions opts = {})
      : opts_(opts), goal_(goal) {
    compile_actions(actions);
    compile_goal();
    // One unrestricted pass, plus one per agent when there is a choice of
    // agents to schedule onto.
    const bool per_agent = opts_.agent_aware && agents_.size() > 1;
    passes_.resize(1 + (per_agent ? agents_.size() : 0));
    for (std::size_t p = 0; p < passes_.size(); ++p) {
      passes_[p].agent = (p == 0) ? -1 : static_cast<int>(p - 1);
    }
  }

  ConcurrentHeuristic(const ConcurrentHeuristic &) = delete;
  ConcurrentHeuristic &operator=(const ConcurrentHeuristic &) = delete;

  // Memoized evaluation (keyed on fluents + upcoming effects relative to now).
  double operator()(const State &s) {
    std::size_t key = state_key(s);
    auto it = memo_.find(key);
    if (it != memo_.end()) return it->second;
    double v = evaluate(s, nullptr);
    memo_.emplace(key, v);
    return v;
  }

  double evaluate(const State &s, ConcurrentHeuristicBreakdown *out) {
    if (!goal_) return 0.0;
    if (goal_->get_type() == GoalType::TRUE_GOAL) return 0.0;
    if (goal_->get_type() == GoalType::FALSE_GOAL || branches_.empty()) return INF;

    load_state(s);
    // The unrestricted pass is always needed (h_add, reachability, fallback);
    // the per-agent passes feed the schedule.
    for (auto &P : passes_) run_pass(P);

    double best = INF;
    for (const auto &branch : branches_) {
      ConcurrentHeuristicBreakdown bd;
      double v = evaluate_branch(branch, out ? &bd : nullptr);
      if (v < best) {
        best = v;
        if (out) *out = std::move(bd);
      }
    }
    if (out) out->value = best;
    return best;
  }

  std::size_t num_agents() const { return agents_.size(); }
  std::size_t memo_size() const { return memo_.size(); }

private:
  // ------------------------------------------------------------------
  //  Compiled problem
  // ------------------------------------------------------------------
  struct CAdd {
    int fluent;
    double prob;       // P(action adds fluent), summed over branches
    bool conditional;  // added in some but not all branches
  };
  struct CAction {
    const Action *src;
    double dur;                 // max relaxed successor time + extra_cost
    std::vector<int> pre;       // positive preconditions
    std::vector<CAdd> adds;
    std::vector<int> consumes;  // preconditions it deterministically deletes
    int agent;                  // -1 agent-free, -2 several agents, else id
  };
  struct AchRef {
    int action;
    int add;    // index into CAction::adds
    int group;  // achievers sharing a group are one attempt
  };

  ConcurrentHeuristicOptions opts_;
  const GoalBase *goal_;

  std::unordered_map<Fluent, int> fid_;
  std::vector<Fluent> fluents_;
  std::vector<CAction> acts_;
  std::vector<std::vector<int>> consumers_;      // fluent -> actions needing it
  std::vector<std::vector<AchRef>> achievers_;   // fluent -> achievers
  // fluent -> (consumed fluent, group) for pending-achiever grouping
  std::vector<std::vector<std::pair<int, int>>> group_by_consumed_;
  std::vector<int> found_of_;  // `at X ...` -> `found X` (or -1)
  std::vector<int> loc_agent_;  // location fluent -> its agent (or -1)
  std::vector<std::vector<int>> loc_fluents_;  // agent -> its location fluents
  std::vector<int> loc_place_;  // location fluent -> place id (or -1)
  std::vector<std::unordered_map<int, int>> place_loc_;  // agent -> place -> fluent

  // Agent r's location fluent for the place of location fluent `loc`.
  int as_agent_loc(int r, int loc) const {
    if (loc < 0 || r < 0 || loc_agent_[loc] == r) return loc;
    int pl = loc_place_[loc];
    if (pl < 0) return -1;
    auto it = place_loc_[r].find(pl);
    return it == place_loc_[r].end() ? -1 : it->second;
  }
  std::vector<int> agents_;    // agent id -> fid of `free <agent>`
  std::vector<std::string> agent_names_;
  std::vector<std::vector<int>> branches_;  // DNF branches (fids, -1 = unknown)

  int intern(const Fluent &f) {
    auto it = fid_.find(f);
    if (it != fid_.end()) return it->second;
    int id = static_cast<int>(fluents_.size());
    fid_.emplace(f, id);
    fluents_.push_back(f);
    return id;
  }
  int lookup(const Fluent &f) const {
    auto it = fid_.find(f);
    return it == fid_.end() ? -1 : it->second;
  }

  void compile_actions(const std::vector<Action> &actions) {
    acts_.reserve(actions.size());
    std::unordered_map<int, int> agent_of_free;  // free fid -> agent id
    for (const auto &a : actions) {
      CAction ca;
      ca.src = &a;
      ca.agent = -1;
      for (const auto &p : a.pos_preconditions()) {
        int id = intern(p);
        ca.pre.push_back(id);
        if (p.is_free()) {
          auto it = agent_of_free.find(id);
          int ag;
          if (it == agent_of_free.end()) {
            ag = static_cast<int>(agents_.size());
            agent_of_free.emplace(id, ag);
            agents_.push_back(id);
            agent_names_.push_back(p.args().empty() ? p.name() : p.args()[0]);
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
      for (const auto &[f, p] : prob) {
        if (f.is_negated()) continue;
        bool conditional = succs.size() > 1 && membership[f] < succs.size();
        ca.adds.push_back({intern(f), std::min(p, 1.0), conditional});
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
      acts_.push_back(std::move(ca));
    }

    // Goal fluents may be absent from every action; intern them too so that
    // they can at least be satisfied by the state.
    if (goal_) {
      for (const auto &f : goal_->get_all_literals()) intern(f);
    }

    std::size_t nf = fluents_.size();
    consumers_.assign(nf, {});
    achievers_.assign(nf, {});
    for (int ai = 0; ai < static_cast<int>(acts_.size()); ++ai) {
      for (int p : acts_[ai].pre) consumers_[p].push_back(ai);
      for (int k = 0; k < static_cast<int>(acts_[ai].adds.size()); ++k) {
        achievers_[acts_[ai].adds[k].fluent].push_back({ai, k, 0});
      }
    }

    // Group achievers of each fluent by shared consumed preconditions.
    group_by_consumed_.assign(nf, {});
    for (std::size_t f = 0; f < nf; ++f) {
      auto &achs = achievers_[f];
      std::vector<int> parent(achs.size());
      std::iota(parent.begin(), parent.end(), 0);
      auto find = [&parent](int x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
      };
      std::unordered_map<int, int> first_with;
      for (int i = 0; i < static_cast<int>(achs.size()); ++i) {
        for (int c : acts_[achs[i].action].consumes) {
          auto [it, inserted] = first_with.emplace(c, i);
          if (!inserted) parent[find(i)] = find(it->second);
        }
      }
      for (int i = 0; i < static_cast<int>(achs.size()); ++i) {
        achs[i].group = opts_.group_attempts ? find(i) : i;
      }
      bool any_prob = false;
      for (const auto &r : achs) {
        const auto &ad = acts_[r.action].adds[r.add];
        if (ad.prob < 1.0 - 1e-9 || ad.conditional) any_prob = true;
      }
      if (any_prob && opts_.group_attempts) {
        for (const auto &[c, i] : first_with) group_by_consumed_[f].push_back({c, find(i)});
      }
    }

    // Agent location fluents: an action run by agent r that, through its
    // top-level effects, adds P(r, y, ...) and deletes P(r, x, ...) moves r
    // between two values of one mutex group (e.g. `at r x` -> `at r y`).
    loc_agent_.assign(nf, -1);
    loc_fluents_.assign(agents_.size(), {});
    for (const auto &ca : acts_) {
      if (ca.agent < 0) continue;
      const std::string &name = agent_names_[ca.agent];
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
            if (id >= 0 && loc_agent_[id] < 0) {
              loc_agent_[id] = ca.agent;
              loc_fluents_[ca.agent].push_back(id);
            }
          }
        }
      }
    }

    // Places: a location fluent with its agent argument removed, so that
    // `at robot1 L` and `at robot2 L` name the same place.
    loc_place_.assign(nf, -1);
    place_loc_.assign(agents_.size(), {});
    {
      std::unordered_map<std::string, int> place_id;
      for (std::size_t ag = 0; ag < agents_.size(); ++ag) {
        for (int f : loc_fluents_[ag]) {
          const Fluent &fl = fluents_[f];
          std::string key = fl.name();
          for (std::size_t i = 1; i < fl.args().size(); ++i) key += " " + fl.args()[i];
          auto [it, inserted] = place_id.emplace(key, static_cast<int>(place_id.size()));
          loc_place_[f] = it->second;
          place_loc_[ag].emplace(it->second, f);
        }
      }
    }

    // `at X L` -> `found X` for the at-implies-found augmentation.
    found_of_.assign(nf, -1);
    for (std::size_t f = 0; f < nf; ++f) {
      const Fluent &fl = fluents_[f];
      if (fl.is_negated() || fl.name() != "at" || fl.args().empty()) continue;
      int g = lookup(Fluent("found", {fl.args()[0]}));
      found_of_[f] = g;
    }
  }

  void compile_goal() {
    if (!goal_) return;
    GoalType type = goal_->get_type();
    if (type == GoalType::TRUE_GOAL || type == GoalType::FALSE_GOAL) return;
    // Large DNFs are rare; cap enumeration like ff_heuristic does and fall
    // back to the literal union (an over-constrained but finite estimate).
    constexpr std::size_t kMaxBranches = 1024;
    if (goal_->dnf_branch_count() == 0) return;
    if (goal_->dnf_branch_count() > kMaxBranches) {
      std::vector<int> b;
      for (const auto &f : goal_->get_all_literals()) b.push_back(lookup(f));
      branches_.push_back(std::move(b));
      return;
    }
    for (const auto &br : goal_->get_dnf_branches()) {
      std::vector<int> b;
      for (const auto &f : br) b.push_back(lookup(f));
      branches_.push_back(std::move(b));
    }
  }

  // ------------------------------------------------------------------
  //  Per-state data
  // ------------------------------------------------------------------
  struct Pending {
    int fluent;
    double time;
    double prob;
    int group;  // achiever group of `fluent` it belongs to (-1: its own)
  };

  std::vector<std::pair<int, double>> avail_;  // deterministic (fid, time)
  std::vector<Pending> pending_;

  struct Pass {
    int agent = -1;
    std::vector<double> cost, rho, score, wait, act_rho, delta;
    std::vector<int> best;  // >=0 action; -1 none; -2 available; <=-3 pending
    std::vector<int> unmet;
    std::vector<uint8_t> done;
  };
  std::vector<Pass> passes_;

  static constexpr int BEST_NONE = -1;
  static constexpr int BEST_AVAIL = -2;
  static int pending_code(int k) { return -3 - k; }

  bool allowed(const Pass &P, int a) const {
    if (P.agent < 0) return true;
    int ag = acts_[a].agent;
    return ag == -1 || ag == P.agent;
  }

  static std::size_t state_key(const State &s) {
    std::size_t h_fluents = 0;
    for (const auto &f : s.fluents()) {
      std::size_t h = f.hash();
      hash_combine(h, 0);
      h_fluents ^= h;
    }
    std::size_t h_up = 0;
    for (const auto &[t, e] : s.upcoming_effects()) {
      std::size_t h = e->hash();
      hash_combine(h, std::hash<double>{}(t - s.time()));
      h_up ^= h;
    }
    hash_combine(h_fluents, h_up);
    return h_fluents;
  }

  void walk_effect(const GroundedEffect &e, double trel, double prob,
                   std::vector<std::pair<int, double>> &prob_adds,
                   std::vector<double> &prob_times) {
    for (const auto &f : e.pos_fluents()) {
      int id = lookup(f);
      if (id < 0) continue;
      if (prob >= 1.0 - 1e-9) {
        avail_.push_back({id, trel});
      } else {
        // Accumulate across mutually exclusive branches of this root effect.
        bool merged = false;
        for (std::size_t i = 0; i < prob_adds.size(); ++i) {
          if (prob_adds[i].first == id) {
            prob_adds[i].second += prob;
            prob_times[i] = std::min(prob_times[i], trel);
            merged = true;
            break;
          }
        }
        if (!merged) {
          prob_adds.push_back({id, prob});
          prob_times.push_back(trel);
        }
      }
    }
    // Relaxed: conditional branches are assumed to fire.
    for (const auto &cb : e.cond_effects()) {
      for (const auto &sub : cb.effects()) {
        walk_effect(*sub, trel + sub->time(), prob, prob_adds, prob_times);
      }
    }
    for (const auto &pb : e.prob_effects()) {
      if (pb.prob() <= 0.0) continue;
      for (const auto &sub : pb.effects()) {
        walk_effect(*sub, trel + sub->time(), prob * pb.prob(), prob_adds, prob_times);
      }
    }
  }

  void load_state(const State &s) {
    avail_.clear();
    pending_.clear();
    for (const auto &f : s.fluents()) {
      int id = lookup(f);
      if (id >= 0) avail_.push_back({id, 0.0});
    }
    const double t0 = s.time();
    std::vector<std::pair<int, double>> prob_adds;
    std::vector<double> prob_times;
    for (const auto &[t_abs, e] : s.upcoming_effects()) {
      double trel = opts_.timed_init ? std::max(0.0, t_abs - t0) : 0.0;
      prob_adds.clear();
      prob_times.clear();
      walk_effect(*e, trel, 1.0, prob_adds, prob_times);
      for (std::size_t i = 0; i < prob_adds.size(); ++i) {
        int f = prob_adds[i].first;
        double p = std::min(prob_adds[i].second, 1.0);
        double t = opts_.timed_init ? prob_times[i] : 0.0;
        if (p >= 1.0 - 1e-9) {
          avail_.push_back({f, t});
          continue;
        }
        // Which achiever group of f does this pending outcome belong to?
        int group = -1;
        for (const auto &df : e->flipped_neg_fluents()) {
          int c = lookup(df);
          if (c < 0) continue;
          for (const auto &[cf, g] : group_by_consumed_[f]) {
            if (cf == c) { group = g; break; }
          }
          if (group >= 0) break;
        }
        pending_.push_back({f, t, p, group});
      }
    }

    // `waiting a b`: a becomes free when b does (transition() resolves it).
    for (const auto &f : s.fluents()) {
      if (!f.is_waiting() || f.args().size() < 2) continue;
      int fa = lookup(Fluent("free", {f.args().front()}));
      int fb = lookup(Fluent("free", {f.args().back()}));
      if (fa < 0 || fb < 0) continue;
      double tb = INF;
      for (const auto &[id, t] : avail_) {
        if (id == fb) tb = std::min(tb, t);
      }
      if (std::isfinite(tb)) avail_.push_back({fa, tb});
    }
  }

  // ------------------------------------------------------------------
  //  Relaxed cost propagation (Dijkstra on cost / rho)
  // ------------------------------------------------------------------
  using QItem = std::pair<double, int>;  // (score, fluent)

  void offer(Pass &P, int f, double cost, double rho, int best,
             std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> &pq) {
    if (rho <= 1e-12) return;
    // Most fluents sit on deterministic supports (rho == 1): skip the pow.
    double score = rho >= 1.0 ? cost : cost / std::pow(rho, opts_.prob_exponent);
    const double eps = 1e-9;
    if (score < P.score[f] - eps ||
        (score <= P.score[f] + eps && cost < P.cost[f] - eps)) {
      P.score[f] = score;
      P.cost[f] = cost;
      P.rho[f] = rho;
      P.best[f] = best;
      pq.push({score, f});
    }
  }

  void fire(Pass &P, int a,
            std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> &pq) {
    const CAction &ca = acts_[a];
    double w = P.wait[a];
    for (const auto &ad : ca.adds) {
      if (ad.prob <= 1e-9) continue;
      offer(P, ad.fluent, w + ca.dur, P.act_rho[a] * ad.prob, a, pq);
    }
  }

  void run_pass(Pass &P) {
    const std::size_t nf = fluents_.size();
    const std::size_t na = acts_.size();
    P.cost.assign(nf, INF);
    P.rho.assign(nf, 0.0);
    P.score.assign(nf, INF);
    P.best.assign(nf, BEST_NONE);
    P.done.assign(nf, 0);
    P.delta.assign(nf, -1.0);
    P.wait.assign(na, 0.0);
    P.act_rho.assign(na, 1.0);
    P.unmet.assign(na, 0);

    std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;
    for (const auto &[f, t] : avail_) offer(P, f, t, 1.0, BEST_AVAIL, pq);
    for (int k = 0; k < static_cast<int>(pending_.size()); ++k) {
      const auto &pd = pending_[k];
      offer(P, pd.fluent, pd.time, pd.prob, pending_code(k), pq);
    }
    for (int a = 0; a < static_cast<int>(na); ++a) {
      if (!allowed(P, a)) {
        P.unmet[a] = -1;
        continue;
      }
      P.unmet[a] = static_cast<int>(acts_[a].pre.size());
      if (P.unmet[a] == 0) fire(P, a, pq);
    }

    while (!pq.empty()) {
      auto [score, f] = pq.top();
      pq.pop();
      if (P.done[f] || score > P.score[f] + 1e-9) continue;
      P.done[f] = 1;
      for (int a : consumers_[f]) {
        if (P.unmet[a] <= 0) continue;  // disallowed or already fired
        P.wait[a] = std::max(P.wait[a], P.cost[f]);
        P.act_rho[a] *= P.rho[f];
        if (--P.unmet[a] == 0) fire(P, a, pq);
      }
    }
  }

  bool reachable(const Pass &P, int f) const { return f >= 0 && P.best[f] != BEST_NONE; }

  bool is_prob_choice(const Pass &P, int f) const {
    int b = P.best[f];
    if (b <= -3) return true;
    if (b < 0) return false;
    for (const auto &ad : acts_[b].adds) {
      if (ad.fluent == f) return ad.prob < 1.0 - 1e-9 || ad.conditional;
    }
    return false;
  }

  // Expected extra time over the optimistic cost of f from trying its
  // achievers (one per group) in the best of three orderings.
  double delta(Pass &P, int f) {
    if (P.delta[f] >= 0.0) return P.delta[f];
    double d = 0.0;
    if (is_prob_choice(P, f)) d = compute_delta(P, f);
    P.delta[f] = d;
    return d;
  }

  struct Attempt {
    double wait, exec, prob;
    double attempt() const { return wait + exec; }
    double efficiency() const { return exec > 1e-9 ? prob / exec : prob * 1e9; }
  };

  double compute_delta(const Pass &P, int f) const {
    const auto &achs = achievers_[f];
    // Representative per group: the cheapest attempt.
    std::vector<Attempt> reps;
    std::vector<int> rep_group;
    auto add_rep = [&](int group, const Attempt &at) {
      if (group >= 0) {
        for (std::size_t i = 0; i < rep_group.size(); ++i) {
          if (rep_group[i] == group) {
            if (at.attempt() < reps[i].attempt()) reps[i] = at;
            return;
          }
        }
      }
      reps.push_back(at);
      rep_group.push_back(group);
    };
    // Pending outcomes first: once in flight, an attempt is not repeatable.
    for (const auto &pd : pending_) {
      if (pd.fluent != f) continue;
      add_rep(pd.group, {0.0, pd.time, pd.prob});
    }
    for (const auto &r : achs) {
      if (!allowed(P, r.action) || P.unmet[r.action] != 0) continue;
      const auto &ad = acts_[r.action].adds[r.add];
      if (ad.prob <= 1e-9) continue;
      Attempt at{P.wait[r.action], acts_[r.action].dur, ad.prob};
      // A group already holding a pending attempt is spent.
      bool spent = false;
      for (const auto &pd : pending_) {
        if (pd.fluent == f && pd.group >= 0 && pd.group == r.group) { spent = true; break; }
      }
      if (spent) continue;
      add_rep(r.group, at);
    }
    if (reps.empty()) return 0.0;

    auto expected = [](const std::vector<Attempt> &ordered) {
      double total = 0.0, fail = 1.0, time = 0.0;
      for (const auto &a : ordered) {
        double dwait = std::max(a.wait - time, 0.0);
        total += fail * (dwait + a.exec);
        fail *= (1.0 - a.prob);
        time = std::max(time, a.wait);
      }
      return total;
    };
    double best_E = INF;
    std::sort(reps.begin(), reps.end(),
              [](const Attempt &a, const Attempt &b) { return a.efficiency() > b.efficiency(); });
    best_E = std::min(best_E, expected(reps));
    std::sort(reps.begin(), reps.end(),
              [](const Attempt &a, const Attempt &b) { return a.prob > b.prob; });
    best_E = std::min(best_E, expected(reps));
    std::sort(reps.begin(), reps.end(),
              [](const Attempt &a, const Attempt &b) { return a.attempt() < b.attempt(); });
    best_E = std::min(best_E, expected(reps));
    double d = best_E - P.cost[f];
    return d > 1e-9 ? d : 0.0;
  }

  // ------------------------------------------------------------------
  //  Relaxed-plan extraction
  // ------------------------------------------------------------------
  std::vector<uint32_t> fl_stamp_, ach_stamp_, act_stamp_;
  uint32_t stamp_ = 0;

  struct Extraction {
    double dur = 0.0;     // sum of action durations
    double delta = 0.0;   // sum of retry deltas over probabilistic fluents
    std::vector<int> fluents;  // fluents visited (for coverage)
    std::vector<int> actions;  // actions on the plan
  };

  void next_stamp() {
    if (fl_stamp_.size() != fluents_.size()) fl_stamp_.assign(fluents_.size(), 0);
    if (ach_stamp_.size() != fluents_.size()) ach_stamp_.assign(fluents_.size(), 0);
    if (act_stamp_.size() != acts_.size()) act_stamp_.assign(acts_.size(), 0);
    if (++stamp_ == 0) {
      std::fill(fl_stamp_.begin(), fl_stamp_.end(), 0);
      std::fill(ach_stamp_.begin(), ach_stamp_.end(), 0);
      std::fill(act_stamp_.begin(), act_stamp_.end(), 0);
      stamp_ = 1;
    }
  }

  // Walk back from `roots` via the pass's chosen achievers, costliest subgoal
  // first (as FF does): every fluent an action on the plan adds counts as
  // achieved, so a later, cheaper subgoal it covers as a side effect (e.g. the
  // `hand-full` that a pick also adds) does not pull in an achiever of its
  // own. Probabilistic subgoals still pay their retry delta.
  void extract(Pass &P, const std::vector<int> &roots, Extraction &ex) {
    // fl_stamp_: required subgoal already processed; ach_stamp_: achieved by
    // an action already on the plan.
    if (ach_stamp_.size() != fluents_.size()) ach_stamp_.assign(fluents_.size(), 0);
    std::priority_queue<std::pair<double, int>> heap;  // (cost, fluent), max first
    for (int f : roots) {
      if (f >= 0) heap.push({P.cost[f], f});
    }
    while (!heap.empty()) {
      int f = heap.top().second;
      heap.pop();
      if (fl_stamp_[f] == stamp_) continue;
      fl_stamp_[f] = stamp_;
      int b = P.best[f];
      if (b == BEST_AVAIL || b == BEST_NONE) continue;
      ex.fluents.push_back(f);
      if (is_prob_choice(P, f)) ex.delta += delta(P, f);
      if (b <= -3 || ach_stamp_[f] == stamp_) continue;
      if (act_stamp_[b] == stamp_) continue;
      act_stamp_[b] = stamp_;
      ex.dur += acts_[b].dur;
      ex.actions.push_back(b);
      for (const auto &ad : acts_[b].adds) {
        if (ad.prob > 1e-9) ach_stamp_[ad.fluent] = stamp_;
      }
      for (int p : acts_[b].pre) {
        heap.push({P.cost[p], p});
        if (opts_.at_implies_found && found_of_[p] >= 0) {
          heap.push({P.cost[found_of_[p]], found_of_[p]});
        }
      }
    }
  }

  // ------------------------------------------------------------------
  //  Branch evaluation
  // ------------------------------------------------------------------
  double evaluate_branch(const std::vector<int> &branch, ConcurrentHeuristicBreakdown *bd) {
    Pass &U = passes_[0];
    std::vector<int> goals;
    goals.reserve(branch.size() * 2);
    for (int g : branch) {
      if (g < 0 || !reachable(U, g)) return INF;
      goals.push_back(g);
    }
    if (opts_.at_implies_found) {
      std::size_t n = goals.size();
      for (std::size_t i = 0; i < n; ++i) {
        int fg = found_of_[goals[i]];
        if (fg >= 0 && reachable(U, fg) &&
            std::find(goals.begin(), goals.end(), fg) == goals.end()) {
          goals.push_back(fg);
        }
      }
    }

    // h_add and the shared (unrestricted) relaxed plan.
    double h_add = 0.0;
    for (int g : goals) h_add += U.cost[g];
    next_stamp();
    Extraction ux;
    extract(U, goals, ux);

    if (bd) {
      for (int f : ux.fluents) {
        if (is_prob_choice(U, f)) bd->deltas.push_back({fluent_str(f), delta(U, f)});
      }
      for (int a = 0; a < static_cast<int>(acts_.size()); ++a) {
        if (act_stamp_[a] == stamp_) bd->plan.push_back({acts_[a].src->name(), acts_[a].dur});
      }
      bd->h_add = h_add;
      bd->delta = ux.delta;
      bd->h_ff = ux.dur;
    }
    double completion_sum = 0.0;
    double makespan = schedule(goals, bd, completion_sum);
    if (bd) {
      bd->makespan = makespan;
      bd->completion_sum = completion_sum;
    }
    double additive = opts_.sum_completion ? completion_sum : h_add + ux.delta;
    return opts_.lambda_add * additive + opts_.lambda_ms * makespan;
  }

  // Duration of agent r's cheapest deterministic action that moves it from
  // location fluent `from` to `to` (INF if none). State-independent; cached.
  double move_dur(int r, int from, int to) {
    if (from == to) return 0.0;
    uint64_t key = (static_cast<uint64_t>(r) << 42) ^
                   (static_cast<uint64_t>(from) << 21) ^ static_cast<uint64_t>(to);
    auto it = move_cache_.find(key);
    if (it != move_cache_.end()) return it->second;
    double best = INF;
    for (const auto &ach : achievers_[to]) {
      const CAction &a = acts_[ach.action];
      if (a.agent != r || a.adds[ach.add].prob < 1.0 - 1e-9) continue;
      if (std::find(a.pre.begin(), a.pre.end(), from) != a.pre.end()) best = std::min(best, a.dur);
    }
    move_cache_.emplace(key, best);
    return best;
  }

  // Where agent r is (or will be once its current action ends): its
  // location fluent available at the latest time. -1 if it has none.
  int agent_location(const Pass &P, int r) const {
    int loc = -1;
    double t = -1.0;
    for (int f : loc_fluents_[r]) {
      if (P.best[f] == BEST_AVAIL && P.cost[f] > t) { t = P.cost[f]; loc = f; }
    }
    return loc;
  }

  // One agent's relaxed plan for one task, with its moves re-costed as a
  // route. The delete relaxation lets an agent be in several places at once,
  // so a relaxed plan reaches every location from wherever the agent is
  // *first* (fetching an object costs start->object + start->target instead
  // of start->object->target). Instead, the locations the plan's moves reach
  // are visited in the order they are needed, each leg costed from the
  // previous one.
  struct TaskPlan {
    bool ok = false;
    double other = 0.0;     // durations of the plan's non-move actions
    double critical = 0.0;  // relaxed time to the goal after the agent is ready
    double delta = 0.0;
    std::vector<int> locs;  // locations to visit, in order
    std::vector<double> leg_fallback;  // relaxed-plan cost of each leg
    std::vector<int> covers;
    // Expected-search costing (expected_search): the uncertain subgoal, the
    // places needed before and after it, the non-move durations other than
    // the search itself, and the retry deltas of everything else.
    int search_f = -1;
    std::vector<int> pre_locs, post_locs;
    std::vector<double> pre_fb, post_fb;
    double other_ns = 0.0;
    double delta_rest = 0.0;
  };

  double route(int r, int start, const TaskPlan &tp) {
    double total = 0.0;
    int prev = start;
    for (std::size_t i = 0; i < tp.locs.size(); ++i) {
      int y = tp.locs[i];
      if (y == prev) continue;
      double d = (prev >= 0) ? move_dur(r, prev, y) : INF;
      total += std::isfinite(d) ? d : tp.leg_fallback[i];
      prev = y;
    }
    return total;
  }

  TaskPlan plan_task(Pass &P, int g, int companion, int r, double ready) {
    TaskPlan tp;
    if (!reachable(P, g)) return tp;
    if (companion >= 0 && !reachable(P, companion)) companion = -1;
    next_stamp();
    Extraction ex;
    std::vector<int> roots{g};
    if (companion >= 0) roots.push_back(companion);
    extract(P, roots, ex);
    tp.ok = true;
    tp.delta = ex.delta;
    double goal_cost = P.cost[g];
    if (companion >= 0) goal_cost = std::max(goal_cost, P.cost[companion]);
    tp.critical = std::max(0.0, goal_cost - ready);
    tp.covers = std::move(ex.fluents);
    if (r < 0 || !opts_.route_chaining) {
      tp.other = ex.dur;  // no single agent to route, or routing disabled
      return tp;
    }
    // Locations the plan needs the agent at: the location preconditions of
    // its non-move actions (including where the agent already is -- the
    // relaxation never makes it walk back), plus move destinations nothing
    // else on the plan consumes (waypoints, or location goals). Each is
    // needed when the earliest plan action requiring it could start.
    auto dest_of = [&](int a) {
      for (const auto &ad : acts_[a].adds) {
        if (loc_agent_[ad.fluent] == r && ad.prob >= 1.0 - 1e-9 &&
            std::find(acts_[a].pre.begin(), acts_[a].pre.end(), ad.fluent) == acts_[a].pre.end()) {
          return ad.fluent;
        }
      }
      return -1;
    };
    std::vector<std::pair<double, int>> visits;  // (need time, location)
    std::vector<int> move_dests;
    for (int a : ex.actions) {
      int dest = dest_of(a);
      if (dest >= 0) {
        move_dests.push_back(dest);
        continue;
      }
      tp.other += acts_[a].dur;
      for (int p : acts_[a].pre) {
        if (loc_agent_[p] == r) visits.push_back({P.wait[a], p});
      }
    }
    for (int d : move_dests) {
      bool consumed = false;
      for (const auto &v : visits) consumed = consumed || v.second == d;
      if (!consumed) visits.push_back({P.cost[d], d});
    }
    if (loc_agent_[g] == r) visits.push_back({P.cost[g], g});
    // Keep the earliest need per location.
    std::sort(visits.begin(), visits.end());
    std::vector<std::pair<double, int>> uniq;
    for (const auto &v : visits) {
      bool seen = false;
      for (const auto &u : uniq) seen = seen || u.second == v.second;
      if (!seen) uniq.push_back(v);
    }
    for (const auto &v : uniq) {
      tp.locs.push_back(v.second);
      // If no single move reaches it from the previous stop, fall back to
      // the relaxed plan's own cost of getting there.
      int b = P.best[v.second];
      tp.leg_fallback.push_back(b >= 0 ? acts_[b].dur : 0.0);
    }
    if (opts_.expected_search) split_search(P, r, companion, uniq, tp);
    return tp;
  }

  // Probabilistic attempts at f open to agent r in pass P, one per attempt
  // group (the nearest in the relaxation): (group, location, exec, prob,
  // relaxed start time). `spent` groups (already in flight) are skipped.
  struct SearchAttempt {
    int group, loc;
    double exec, prob, fallback;
  };
  std::vector<SearchAttempt> attempts_for(const Pass &P, int r, int f) const {
    std::vector<SearchAttempt> out;
    for (const auto &ach : achievers_[f]) {
      if (!allowed(P, ach.action) || P.unmet[ach.action] != 0) continue;
      const CAction &a = acts_[ach.action];
      double p = a.adds[ach.add].prob;
      if (p <= 1e-9) continue;
      bool spent = false;
      for (const auto &pd : pending_) {
        if (pd.fluent == f && pd.group >= 0 && pd.group == ach.group) { spent = true; break; }
      }
      if (spent) continue;
      int loc = -1;
      for (int q : a.pre) {
        if (loc_agent_[q] == r) { loc = q; break; }
      }
      SearchAttempt at{ach.group, loc, a.dur, p, P.wait[ach.action]};
      bool merged = false;
      for (auto &e : out) {
        if (e.group == at.group) {
          if (at.fallback < e.fallback) e = at;
          merged = true;
          break;
        }
      }
      if (!merged) out.push_back(at);
    }
    return out;
  }

  // Identify the task's uncertain search -- the implied `found X` when the
  // task has one, else the probabilistic subgoal with the most attempt groups
  // -- and split the plan's places into those needed before and after it.
  void split_search(Pass &P, int r, int companion,
                    const std::vector<std::pair<double, int>> &visits, TaskPlan &tp) {
    int f_star = -1;
    std::size_t best_n = 1;
    for (int f : tp.covers) {
      if (!is_prob_choice(P, f)) continue;
      std::size_t n = attempts_for(P, r, f).size();
      if (f == companion && n >= 2) { f_star = f; break; }
      if (n > best_n) { best_n = n; f_star = f; }
    }
    if (f_star < 0) return;
    int b = P.best[f_star];
    // Where the plan searches, and when.
    int search_loc = -1;
    double search_need = 0.0;
    if (b >= 0) {
      for (int q : acts_[b].pre) {
        if (loc_agent_[q] == r) { search_loc = q; break; }
      }
      search_need = P.wait[b];
    }
    tp.search_f = f_star;
    for (std::size_t i = 0; i < visits.size(); ++i) {
      int loc = visits[i].second;
      double fb = tp.leg_fallback[i];
      if (loc == search_loc) continue;
      if (visits[i].first < search_need) {
        tp.pre_locs.push_back(loc);
        tp.pre_fb.push_back(fb);
      } else {
        tp.post_locs.push_back(loc);
        tp.post_fb.push_back(fb);
      }
    }
    // The search's own duration is inside the expected search time; the
    // retry deltas of the search outcome (and of anything else the same
    // action achieves, e.g. `at X place`) are superseded by it.
    tp.other_ns = tp.other - (b >= 0 ? acts_[b].dur : 0.0);
    tp.delta_rest = 0.0;
    for (int f : tp.covers) {
      if (!is_prob_choice(P, f)) continue;
      if (f == f_star || (b >= 0 && P.best[f] == b)) continue;
      tp.delta_rest += delta(P, f);
    }
  }

  double route_through(int r, int start, const std::vector<int> &locs,
                       const std::vector<double> &fb, int &end) {
    double total = 0.0;
    int prev = start;
    for (std::size_t i = 0; i < locs.size(); ++i) {
      int y = locs[i];
      if (y == prev) continue;
      double d = (prev >= 0) ? move_dur(r, prev, y) : INF;
      total += std::isfinite(d) ? d : fb[i];
      prev = y;
    }
    end = prev;
    return total;
  }

  // Expected time, from t_start, until f holds when `searchers` (agent,
  // place, ready time; the first is the task's own agent) work through its
  // attempt groups: whenever a searcher frees up it walks to the unused
  // attempt with the most probability per unit of travel + execution -- a
  // list schedule with travel as setup time (with one searcher, a greedy
  // route). Pending outcomes complete at their scheduled times regardless.
  // With all events in time order, E = sum_i (T_i - T_{i-1}) P(no success
  // before T_i) from t_start. `where` receives (place as the first searcher's
  // location fluent, probability the search ends there).
  struct Searcher {
    int agent, loc;
    double t;
  };
  double expected_search(int f, std::vector<Searcher> searchers, double t_start,
                         std::vector<std::pair<int, double>> &where) {
    where.clear();
    const int owner = searchers.front().agent;
    struct Event { double t, p; int loc; };
    std::vector<Event> events;
    std::vector<int> spent;
    for (const auto &pd : pending_) {
      if (pd.fluent != f) continue;
      int loc = -1;  // where the pending attempt happens, as owner's fluent
      for (const auto &ach : achievers_[f]) {
        if (ach.group != pd.group || pd.group < 0) continue;
        for (int q : acts_[ach.action].pre) {
          if (loc_agent_[q] >= 0) { loc = as_agent_loc(owner, q); break; }
        }
        if (loc >= 0) break;
      }
      events.push_back({pd.time, pd.prob, loc});
    }
    std::vector<std::vector<SearchAttempt>> cands(searchers.size());
    for (std::size_t k = 0; k < searchers.size(); ++k) {
      int a = searchers[k].agent;
      const Pass &P = (passes_.size() > 1 && a >= 0) ? passes_[a + 1] : passes_[0];
      cands[k] = attempts_for(P, a, f);
    }
    std::vector<int> used;
    double fail = 1.0;
    for (const auto &e : events) fail *= 1.0 - e.p;
    searchers.front().t = t_start;
    while (fail > 1e-3) {
      // The searcher that frees up first takes its best remaining attempt.
      int si = -1;
      for (int k = 0; k < static_cast<int>(searchers.size()); ++k) {
        bool any = false;
        for (const auto &c : cands[k]) {
          if (std::find(used.begin(), used.end(), c.group) == used.end()) { any = true; break; }
        }
        if (any && (si < 0 || searchers[k].t < searchers[si].t)) si = k;
      }
      if (si < 0) break;
      Searcher &sr = searchers[si];
      int best = -1;
      double best_ratio = -1.0, best_dt = 0.0;
      for (int i = 0; i < static_cast<int>(cands[si].size()); ++i) {
        const SearchAttempt &c = cands[si][i];
        if (std::find(used.begin(), used.end(), c.group) != used.end()) continue;
        double travel = 0.0;
        if (c.loc >= 0 && c.loc != sr.loc) {
          travel = (sr.loc >= 0) ? move_dur(sr.agent, sr.loc, c.loc) : INF;
          if (!std::isfinite(travel)) travel = std::max(0.0, c.fallback - sr.t);
        }
        double dt = travel + c.exec;
        double ratio = c.prob / std::max(dt, 1e-9);
        if (ratio > best_ratio) { best_ratio = ratio; best = i; best_dt = dt; }
      }
      const SearchAttempt &c = cands[si][best];
      used.push_back(c.group);
      sr.t += best_dt;
      if (c.loc >= 0) sr.loc = c.loc;
      events.push_back({sr.t, c.prob, as_agent_loc(owner, c.loc)});
      fail *= 1.0 - c.prob;
    }
    if (events.empty()) return INF;
    std::sort(events.begin(), events.end(),
              [](const Event &a, const Event &b) { return a.t < b.t; });
    double expected = 0.0, prev = t_start, still = 1.0, mass = 0.0;
    for (const auto &e : events) {
      double tt = std::max(e.t, prev);
      expected += (tt - prev) * still;
      prev = tt;
      double here = still * e.p;
      if (here > 0.0) where.push_back({e.loc, here});
      mass += here;
      still *= 1.0 - e.p;
    }
    // Condition the places on the search succeeding at all.
    if (mass > 0.0) {
      for (auto &w : where) w.second /= mass;
    }
    return expected;
  }

  // Serial duration of task tp for agent r starting at `start` at time
  // t_start, and the retry delta to add on top.
  double task_serial(const Pass &P, int r, int start, double t_start,
                     const TaskPlan &tp, double &delta_out,
                     const std::vector<Searcher> &helpers = {}) {
    if (r < 0) {
      delta_out = tp.delta;
      return tp.other;
    }
    if (tp.search_f < 0) {
      delta_out = tp.delta;
      return tp.other + route(r, start, tp);
    }
    int pos = start;
    double pre = route_through(r, start, tp.pre_locs, tp.pre_fb, pos);
    (void)P;
    std::vector<std::pair<int, double>> where;
    std::vector<Searcher> searchers{{r, pos, t_start + pre}};
    searchers.insert(searchers.end(), helpers.begin(), helpers.end());
    double search = expected_search(tp.search_f, searchers, t_start + pre, where);
    if (!std::isfinite(search)) {
      delta_out = tp.delta;
      return tp.other + route(r, start, tp);
    }
    double post = 0.0;
    for (const auto &[loc, w] : where) {
      int end;
      post += w * route_through(r, loc >= 0 ? loc : pos, tp.post_locs, tp.post_fb, end);
    }
    delta_out = tp.delta_rest;
    return pre + search + post + tp.other_ns;
  }

  // List-schedule the goal fluents onto agents (or one serial agent).
  double schedule(const std::vector<int> &goals, ConcurrentHeuristicBreakdown *bd,
                  double &completion_sum) {
    completion_sum = 0.0;
    Pass &U = passes_[0];
    const bool per_agent = opts_.agent_aware && agents_.size() > 1;
    const std::size_t n_agents = per_agent ? agents_.size() : 1;
    // With a single agent its route can still be chained.
    const int solo = (!per_agent && agents_.size() == 1) ? 0 : -1;

    std::vector<double> ready(n_agents, 0.0);
    std::vector<int> start_loc(n_agents, -1);
    for (std::size_t r = 0; r < n_agents; ++r) {
      Pass &P = per_agent ? passes_[r + 1] : U;
      int agent = per_agent ? static_cast<int>(r) : solo;
      ready[r] = per_agent ? P.cost[agents_[r]] : agent_ready_team();
      if (agent >= 0) start_loc[r] = agent_location(P, agent);
    }

    // Tasks: goal fluents that are not already true now.
    struct Task {
      int fluent;
      double finish_fixed;  // >= 0: needs no agent (in-flight)
      std::vector<TaskPlan> plans;  // per agent
      double key;
    };
    // `at X L` and the `found X` it implies are one job -- whoever brings X
    // to L must find it first -- so they are planned as one task. Planned
    // apart, `at X L` can be "achieved" by searching L in the hope that X is
    // already there, while finding X becomes a second task for another agent.
    auto companion_of = [&](int g) {
      if (!opts_.joint_found) return -1;
      int fx = found_of_[g];
      if (fx < 0 || std::find(goals.begin(), goals.end(), fx) == goals.end()) return -1;
      if (U.best[fx] == BEST_AVAIL && U.cost[fx] <= 1e-9) return -1;
      return fx;
    };
    std::vector<int> paired;
    for (int g : goals) {
      int fx = companion_of(g);
      if (fx >= 0) paired.push_back(fx);
    }
    std::vector<Task> tasks;
    for (int g : goals) {
      if (U.best[g] == BEST_AVAIL && U.cost[g] <= 1e-9) continue;
      if (std::find(paired.begin(), paired.end(), g) != paired.end()) continue;
      Task t;
      t.fluent = g;
      t.finish_fixed = (U.best[g] == BEST_AVAIL) ? U.cost[g] : -1.0;
      t.key = -1.0;
      if (t.finish_fixed < 0.0) {
        t.plans.resize(n_agents);
        double m = INF;
        for (std::size_t r = 0; r < n_agents; ++r) {
          if (!std::isfinite(ready[r])) continue;
          Pass &P = per_agent ? passes_[r + 1] : U;
          int agent = per_agent ? static_cast<int>(r) : solo;
          t.plans[r] = plan_task(P, g, companion_of(g), agent, ready[r]);
          if (!t.plans[r].ok) continue;
          const TaskPlan &tp = t.plans[r];
          double dl = 0.0;
          double serial = task_serial(P, agent, start_loc[r], ready[r], tp, dl);
          double load = std::max(serial, tp.critical) + dl;
          m = std::min(m, load);
          if (bd) bd->loads.push_back({fluent_str(g), static_cast<int>(r), serial, tp.critical, dl});
        }
        t.key = m;
      }
      tasks.push_back(std::move(t));
    }
    // Longest processing time first.
    std::stable_sort(tasks.begin(), tasks.end(),
                     [](const Task &a, const Task &b) { return a.key > b.key; });

    std::vector<double> finish = ready;
    std::vector<int> end_loc = start_loc;
    std::vector<int> n_assigned(n_agents, 0);

    // Fluents achieved along an assigned task's plan (e.g. `found X` on the
    // way to `at X L`) need no task of their own.
    if (cover_.size() != fluents_.size()) cover_.assign(fluents_.size(), 0);
    if (++cover_gen_ == 0) {
      std::fill(cover_.begin(), cover_.end(), 0);
      cover_gen_ = 1;
    }
    struct Done {
      int task;    // index into tasks
      int agent;   // -1: fixed / fallback
      double at;
      double start_t = 0.0;  // when the agent began this task, and where
      int start_loc = -1;
      bool first = false;    // the agent's first task (critical path applies)
    };
    std::vector<Done> done;
    for (std::size_t ti = 0; ti < tasks.size(); ++ti) {
      Task &t = tasks[ti];
      double done_at;
      int done_agent = -1;
      double dn_start_t = 0.0;
      int dn_start_loc = -1;
      bool dn_first = false;
      if (t.finish_fixed >= 0.0) {
        done_at = t.finish_fixed;
      } else if (cover_[t.fluent] == cover_gen_) {
        continue;  // achieved along another task's plan
      } else {
        int best_r = -1;
        double best_f = INF;
        for (std::size_t r = 0; r < n_agents; ++r) {
          if (!t.plans[r].ok || !std::isfinite(finish[r])) continue;
          const TaskPlan &tp = t.plans[r];
          int agent = per_agent ? static_cast<int>(r) : solo;
          Pass &P = per_agent ? passes_[r + 1] : U;
          double dl = 0.0;
          double serial = task_serial(P, agent, end_loc[r], finish[r], tp, dl);
          // The relaxed critical path only bounds an agent's first task.
          double load = (n_assigned[r] == 0 ? std::max(serial, tp.critical) : serial) + dl;
          double f = finish[r] + load;
          if (f < best_f) { best_f = f; best_r = static_cast<int>(r); }
        }
        if (best_r < 0) {
          // No single agent can do it: fall back to the team relaxation.
          next_stamp();
          Extraction ex;
          extract(U, {t.fluent}, ex);
          done_at = U.cost[t.fluent] + ex.delta;
        } else {
          const TaskPlan &tp = t.plans[best_r];
          dn_start_t = finish[best_r];
          dn_start_loc = end_loc[best_r];
          dn_first = n_assigned[best_r] == 0;
          finish[best_r] = best_f;
          n_assigned[best_r] += 1;
          if (!tp.locs.empty()) end_loc[best_r] = tp.locs.back();
          done_at = best_f;
          done_agent = best_r;
          for (int f : tp.covers) cover_[f] = cover_gen_;
          if (bd) {
            bd->assignment.push_back({fluent_str(t.fluent),
                                      per_agent ? agent_names_[best_r] : "team"});
          }
        }
      }
      done.push_back({static_cast<int>(ti), done_agent, done_at, dn_start_t, dn_start_loc, dn_first});
    }

    // Agents left without a task join the uncertain searches, slowest first.
    if (opts_.parallel_search && per_agent) {
      std::vector<Searcher> spare;
      for (std::size_t r = 0; r < n_agents; ++r) {
        if (n_assigned[r] == 0 && std::isfinite(ready[r])) {
          spare.push_back({static_cast<int>(r), start_loc[r], ready[r]});
        }
      }
      std::vector<std::size_t> order(done.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&done](std::size_t a, std::size_t b) { return done[a].at > done[b].at; });
      for (std::size_t k : order) {
        if (spare.empty()) break;
        Done &dn = done[k];
        if (dn.agent < 0) continue;
        const TaskPlan &tp = tasks[dn.task].plans[dn.agent];
        if (tp.search_f < 0) continue;
        Pass &P = passes_[dn.agent + 1];
        double dl = 0.0;
        double serial = task_serial(P, dn.agent, dn.start_loc, dn.start_t, tp, dl, spare);
        double load = (dn.first ? std::max(serial, tp.critical) : serial) + dl;
        if (dn.start_t + load < dn.at - 1e-9) {
          dn.at = dn.start_t + load;
          spare.clear();  // committed to this search
        }
      }
    }

    double makespan = 0.0;
    for (const auto &dn : done) {
      makespan = std::max(makespan, dn.at);
      completion_sum += dn.at;
      if (bd) bd->goal_finish.push_back({fluent_str(tasks[dn.task].fluent), dn.at});
    }
    return makespan;
  }

  double agent_ready_team() const {
    // Serial "team" agent: ready when the first agent is.
    const Pass &U = passes_[0];
    double r = INF;
    for (int f : agents_) r = std::min(r, U.cost[f]);
    return std::isfinite(r) ? r : 0.0;
  }

  std::string fluent_str(int f) const {
    const Fluent &fl = fluents_[f];
    std::string s = fl.is_negated() ? "not " : "";
    s += fl.name();
    for (const auto &a : fl.args()) s += " " + a;
    return s;
  }

  std::vector<uint32_t> cover_;
  uint32_t cover_gen_ = 0;
  std::unordered_map<uint64_t, double> move_cache_;
  std::unordered_map<std::size_t, double> memo_;
};

}  // namespace railroad
