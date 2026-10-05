#pragma once

// Core of the concurrent heuristic: the timed relaxed state and the
// probability-aware relaxation over it.
//
//   TimedState  The state as the relaxation sees it. Fluents true now are
//               available at time 0; an in-flight effect's fluents become
//               available when it fires. An in-flight probabilistic effect is
//               an *attempt* already under way: its uncertain outcomes are
//               pending achievers that keep their probability, and the
//               attempt is recorded so that expected search treats it like
//               one an agent could still start. A free agent must act at
//               once, so its own pending effects count only after the
//               shortest action it can start now.
//   Pass        One relaxation over the timed state, restricted to one
//               agent's actions (plus agent-free ones) or unrestricted. Each
//               fluent gets a cost and the probability rho that its relaxed
//               support succeeds; achievers are ranked by cost / rho, the
//               expected cost of retrying an independent attempt until it
//               succeeds, which is monotone along supports, so one
//               Dijkstra-style pass computes all of them. Retry deltas price
//               a probabilistic subgoal's expected extra time over its
//               optimistic cost.
//   Extractor   Relaxed-plan extraction from a pass, FF style.

#include "railroad/heuristic_concurrent_problem.hpp"
#include "railroad/state.hpp"

#include <queue>

namespace railroad {
namespace concurrent {

// An uncertain outcome of an in-flight probabilistic effect: a pending
// achiever of `fluent`.
struct Pending {
  int fluent;
  double time;
  double prob;
  int group;    // achiever group of `fluent` it belongs to (-1: its own)
  int attempt;  // the in-flight attempt it is an outcome of
};

// An in-flight probabilistic effect: an attempt already under way.
struct InFlight {
  std::vector<int> outcomes;  // its uncertain outcomes (indices into pending)
  int loc;  // a location fluent of where it happens (any agent's; -1: unknown)
};

class TimedState {
 public:
  std::vector<std::pair<int, double>> avail;  // deterministic (fluent, time)
  std::vector<Pending> pending;
  std::vector<InFlight> in_flight;

  // Is q an uncertain outcome of in-flight attempt k?
  bool reveals(int k, int q) const {
    for (int i : in_flight[k].outcomes) {
      if (pending[i].fluent == q) return true;
    }
    return false;
  }

  void load(const Problem &pb, const ConcurrentHeuristicOptions &opts, const State &s) {
    avail.clear();
    pending.clear();
    in_flight.clear();
    for (const auto &f : s.fluents()) {
      int id = pb.lookup(f);
      if (id >= 0) avail.push_back({id, 0.0});
    }
    const double t0 = s.time();
    std::vector<std::pair<int, double>> prob_adds;
    std::vector<double> prob_times;
    for (const auto &[t_abs, e] : s.upcoming_effects()) {
      double trel = opts.timed_init ? std::max(0.0, t_abs - t0) : 0.0;
      prob_adds.clear();
      prob_times.clear();
      walk_effect(pb, *e, trel, 1.0, prob_adds, prob_times);
      const int attempt = static_cast<int>(in_flight.size());
      InFlight fl{{}, -1};
      for (std::size_t i = 0; i < prob_adds.size(); ++i) {
        int f = prob_adds[i].first;
        double p = std::min(prob_adds[i].second, 1.0);
        double t = opts.timed_init ? prob_times[i] : 0.0;
        if (p >= 1.0 - 1e-9) {
          avail.push_back({f, t});
          continue;
        }
        // Which achiever group of f does this pending outcome belong to?
        int group = -1;
        for (const auto &df : e->flipped_neg_fluents()) {
          int c = pb.lookup(df);
          if (c < 0) continue;
          for (const auto &[cf, g] : pb.group_by_consumed[f]) {
            if (cf == c) { group = g; break; }
          }
          if (group >= 0) break;
        }
        // Where the attempt happens: the location precondition of an
        // achiever in the same group (the action that was started).
        if (fl.loc < 0 && group >= 0) {
          for (const auto &ach : pb.achievers[f]) {
            if (ach.group != group) continue;
            for (int q : pb.acts[ach.action].pre) {
              if (pb.loc_agent[q] >= 0) { fl.loc = q; break; }
            }
            if (fl.loc >= 0) break;
          }
        }
        fl.outcomes.push_back(static_cast<int>(pending.size()));
        pending.push_back({f, t, p, group, attempt});
      }
      if (!fl.outcomes.empty()) in_flight.push_back(std::move(fl));
    }

    // `waiting a b`: a becomes free when b does (transition() resolves it).
    for (const auto &f : s.fluents()) {
      if (!f.is_waiting() || f.args().size() < 2) continue;
      int fa = pb.lookup(Fluent("free", {f.args().front()}));
      int fb = pb.lookup(Fluent("free", {f.args().back()}));
      if (fa < 0 || fb < 0) continue;
      double tb = INF;
      for (const auto &[id, t] : avail) {
        if (id == fb) tb = std::min(tb, t);
      }
      if (std::isfinite(tb)) avail.push_back({fa, tb});
    }
    if (opts.timed_init) defer_own_effects(pb);
  }

 private:
  std::vector<uint8_t> now_;  // scratch: fluent available now

  void walk_effect(const Problem &pb, const GroundedEffect &e, double trel, double prob,
                   std::vector<std::pair<int, double>> &prob_adds,
                   std::vector<double> &prob_times) {
    for (const auto &f : e.pos_fluents()) {
      int id = pb.lookup(f);
      if (id < 0) continue;
      if (prob >= 1.0 - 1e-9) {
        avail.push_back({id, trel});
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
        walk_effect(pb, *sub, trel + sub->time(), prob, prob_adds, prob_times);
      }
    }
    for (const auto &pb_ : e.prob_effects()) {
      if (pb_.prob() <= 0.0) continue;
      for (const auto &sub : pb_.effects()) {
        walk_effect(pb, *sub, trel + sub->time(), prob * pb_.prob(), prob_adds, prob_times);
      }
    }
  }

  // A free agent acts now; it cannot idle until one of its own pending
  // effects fires (e.g. a "just picked" flag that clears 0.1 s after the
  // pick, which forbids putting the object straight back). Such effects are
  // usable by the agent only after the shortest action it can start now.
  void defer_own_effects(const Problem &pb) {
    if (now_.size() != pb.num_fluents()) now_.assign(pb.num_fluents(), 0);
    std::fill(now_.begin(), now_.end(), 0);
    for (const auto &[f, t] : avail) {
      if (t <= 1e-9) now_[f] = 1;
    }
    for (std::size_t r = 0; r < pb.num_agents(); ++r) {
      if (!now_[pb.agent_free[r]]) continue;  // busy: its effects are part of its action
      double d_min = -1.0;
      for (auto &[f, t] : avail) {
        if (t <= 1e-9 || static_cast<int>(f) == pb.agent_free[r]) continue;
        const Fluent &fl = pb.fluent(f);
        if (fl.args().empty() || fl.args()[0] != pb.agent_names[r]) continue;
        if (d_min < 0.0) {
          d_min = INF;
          for (int a : pb.agent_acts[r]) {
            bool ok = true;
            for (int q : pb.acts[a].pre) ok = ok && now_[q];
            if (ok) d_min = std::min(d_min, pb.acts[a].dur);
          }
          if (!std::isfinite(d_min)) d_min = 0.0;
        }
        t = std::max(t, d_min);
      }
    }
  }
};

// One relaxation over the timed state. best[f] records f's chosen support:
// an action (>= 0), nothing (NONE), the state (AVAIL), or pending outcome k
// (pending_code(k)).
struct Pass {
  static constexpr int NONE = -1;
  static constexpr int AVAIL = -2;
  static int pending_code(int k) { return -3 - k; }
  static int pending_index(int code) { return -3 - code; }
  static bool is_pending(int code) { return code <= -3; }

  int agent = -1;  // -1: unrestricted
  std::vector<double> cost, rho, score, wait, act_rho, delta_memo;
  std::vector<int> best;
  std::vector<int> unmet;
  std::vector<uint8_t> done;

  bool allowed(const Problem &pb, int a) const {
    if (agent < 0) return true;
    int ag = pb.acts[a].agent;
    return ag == -1 || ag == agent;
  }
  bool reachable(int f) const { return f >= 0 && best[f] != NONE; }
  // Is f's chosen support uncertain (a probabilistic achiever or outcome)?
  bool uncertain(const Problem &pb, int f) const {
    int b = best[f];
    if (is_pending(b)) return true;
    if (b < 0) return false;
    for (const auto &ad : pb.acts[b].adds) {
      if (ad.fluent == f) return ad.uncertain();
    }
    return false;
  }

  void run(const Problem &pb, const TimedState &ts) {
    const std::size_t nf = pb.num_fluents();
    const std::size_t na = pb.acts.size();
    cost.assign(nf, INF);
    rho.assign(nf, 0.0);
    score.assign(nf, INF);
    best.assign(nf, NONE);
    done.assign(nf, 0);
    delta_memo.assign(nf, -1.0);
    wait.assign(na, 0.0);
    act_rho.assign(na, 1.0);
    unmet.assign(na, 0);

    Queue pq;
    for (const auto &[f, t] : ts.avail) offer(f, t, 1.0, AVAIL, pq);
    for (int k = 0; k < static_cast<int>(ts.pending.size()); ++k) {
      const auto &pd = ts.pending[k];
      offer(pd.fluent, pd.time, pd.prob, pending_code(k), pq);
    }
    for (int a = 0; a < static_cast<int>(na); ++a) {
      if (!allowed(pb, a)) {
        unmet[a] = -1;
        continue;
      }
      unmet[a] = static_cast<int>(pb.acts[a].pre.size());
      if (unmet[a] == 0) fire(pb, a, pq);
    }

    while (!pq.empty()) {
      auto [s, f] = pq.top();
      pq.pop();
      if (done[f] || s > score[f] + 1e-9) continue;
      done[f] = 1;
      for (int a : pb.consumers[f]) {
        if (unmet[a] <= 0) continue;  // disallowed or already fired
        wait[a] = std::max(wait[a], cost[f]);
        act_rho[a] *= rho[f];
        if (--unmet[a] == 0) fire(pb, a, pq);
      }
    }
  }

  // Expected extra time over the optimistic cost of f from trying its
  // achievers (one per attempt group) in the best of three orderings.
  double delta(const Problem &pb, const TimedState &ts, int f) {
    if (delta_memo[f] >= 0.0) return delta_memo[f];
    double d = uncertain(pb, f) ? compute_delta(pb, ts, f) : 0.0;
    delta_memo[f] = d;
    return d;
  }

 private:
  using QItem = std::pair<double, int>;  // (score, fluent)
  using Queue = std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>>;

  void offer(int f, double c, double r, int b, Queue &pq) {
    if (r <= 1e-12) return;
    // Expected cost of retrying an independent attempt until it succeeds.
    double s = c / r;
    const double eps = 1e-9;
    if (s < score[f] - eps || (s <= score[f] + eps && c < cost[f] - eps)) {
      score[f] = s;
      cost[f] = c;
      rho[f] = r;
      best[f] = b;
      pq.push({s, f});
    }
  }

  void fire(const Problem &pb, int a, Queue &pq) {
    const CompiledAction &ca = pb.acts[a];
    double w = wait[a];
    for (const auto &ad : ca.adds) {
      if (ad.prob <= 1e-9) continue;
      offer(ad.fluent, w + ca.dur, act_rho[a] * ad.prob, a, pq);
    }
  }

  struct Try {
    double wait, exec, prob;
    double attempt() const { return wait + exec; }
    double efficiency() const { return exec > 1e-9 ? prob / exec : prob * 1e9; }
  };

  double compute_delta(const Problem &pb, const TimedState &ts, int f) const {
    // Representative per group: the cheapest attempt.
    std::vector<Try> reps;
    std::vector<int> rep_group;
    auto add_rep = [&](int group, const Try &at) {
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
    for (const auto &pd : ts.pending) {
      if (pd.fluent != f) continue;
      add_rep(pd.group, {0.0, pd.time, pd.prob});
    }
    for (const auto &r : pb.achievers[f]) {
      if (!allowed(pb, r.action) || unmet[r.action] != 0) continue;
      const auto &ad = pb.acts[r.action].adds[r.add];
      if (ad.prob <= 1e-9) continue;
      // A group already holding a pending attempt is spent.
      bool spent = false;
      for (const auto &pd : ts.pending) {
        if (pd.fluent == f && pd.group >= 0 && pd.group == r.group) { spent = true; break; }
      }
      if (spent) continue;
      add_rep(r.group, {wait[r.action], pb.acts[r.action].dur, ad.prob});
    }
    if (reps.empty()) return 0.0;

    auto expected = [](const std::vector<Try> &ordered) {
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
              [](const Try &a, const Try &b) { return a.efficiency() > b.efficiency(); });
    best_E = std::min(best_E, expected(reps));
    std::sort(reps.begin(), reps.end(), [](const Try &a, const Try &b) { return a.prob > b.prob; });
    best_E = std::min(best_E, expected(reps));
    std::sort(reps.begin(), reps.end(),
              [](const Try &a, const Try &b) { return a.attempt() < b.attempt(); });
    best_E = std::min(best_E, expected(reps));
    double d = best_E - cost[f];
    return d > 1e-9 ? d : 0.0;
  }
};

// A relaxed plan read off a pass.
struct Extraction {
  double dur = 0.0;     // sum of action durations
  double delta = 0.0;   // sum of retry deltas over probabilistic fluents
  std::vector<int> fluents;  // fluents visited (for coverage)
  std::vector<int> actions;  // actions on the plan
  std::vector<int> avail;    // subgoals already available in the state
};

class Extractor {
 public:
  // Start a new plan; extract() calls until the next one add to it.
  void next_stamp(const Problem &pb) {
    if (fl_stamp_.size() != pb.num_fluents()) fl_stamp_.assign(pb.num_fluents(), 0);
    if (ach_stamp_.size() != pb.num_fluents()) ach_stamp_.assign(pb.num_fluents(), 0);
    if (act_stamp_.size() != pb.acts.size()) act_stamp_.assign(pb.acts.size(), 0);
    if (++stamp_ == 0) {
      std::fill(fl_stamp_.begin(), fl_stamp_.end(), 0);
      std::fill(ach_stamp_.begin(), ach_stamp_.end(), 0);
      std::fill(act_stamp_.begin(), act_stamp_.end(), 0);
      stamp_ = 1;
    }
  }

  // Is action a on the current plan?
  bool on_plan(int a) const { return act_stamp_[a] == stamp_; }

  // Walk back from `roots` via the pass's chosen achievers, costliest subgoal
  // first (as FF does): every fluent an action on the plan adds counts as
  // achieved, so a later, cheaper subgoal it covers as a side effect (e.g. the
  // `hand-full` that a pick also adds) does not pull in an achiever of its
  // own. Probabilistic subgoals still pay their retry delta.
  void extract(const Problem &pb, const ConcurrentHeuristicOptions &opts, const TimedState &ts,
               Pass &P, const std::vector<int> &roots, Extraction &ex) {
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
      if (b == Pass::AVAIL) ex.avail.push_back(f);
      if (b == Pass::AVAIL || b == Pass::NONE) continue;
      ex.fluents.push_back(f);
      if (P.uncertain(pb, f)) ex.delta += P.delta(pb, ts, f);
      if (Pass::is_pending(b) || ach_stamp_[f] == stamp_) continue;
      if (act_stamp_[b] == stamp_) continue;
      act_stamp_[b] = stamp_;
      ex.dur += pb.acts[b].dur;
      ex.actions.push_back(b);
      for (const auto &ad : pb.acts[b].adds) {
        if (ad.prob > 1e-9) ach_stamp_[ad.fluent] = stamp_;
      }
      for (int p : pb.acts[b].pre) {
        heap.push({P.cost[p], p});
        if (opts.at_implies_found && pb.found_of[p] >= 0) {
          heap.push({P.cost[pb.found_of[p]], pb.found_of[p]});
        }
      }
    }
  }

 private:
  // fl_stamp_: required subgoal already processed; ach_stamp_: achieved by an
  // action already on the plan; act_stamp_: action on the plan.
  std::vector<uint32_t> fl_stamp_, ach_stamp_, act_stamp_;
  uint32_t stamp_ = 0;
};

}  // namespace concurrent
}  // namespace railroad
