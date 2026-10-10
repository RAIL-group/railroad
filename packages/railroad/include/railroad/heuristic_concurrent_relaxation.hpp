#pragma once

// Core of the concurrent heuristic: the timed relaxed state and the
// probability-aware relaxation over it.
//
//   TimedState  The state as the relaxation sees it. Fluents true now are
//               available at time 0, an in-flight effect's fluents when it
//               fires. An in-flight probabilistic effect is an *attempt*
//               under way: its uncertain outcomes are pending achievers that
//               keep their probability, and it uses up the planned attempts
//               that consume what it deletes (e.g. another robot's search of
//               the same place), so it is counted once.
//   Pass        One relaxation, restricted to one agent's actions (plus
//               agent-free ones) or unrestricted. Each fluent gets a cost and
//               the probability rho that its support succeeds; achievers are
//               ranked by cost / rho, the expected cost of retrying an
//               independent attempt until it succeeds. The ranking is
//               monotone along supports, so one Dijkstra-style pass computes
//               it.
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
  int attempt;  // the in-flight attempt it is an outcome of
};

// An in-flight probabilistic effect: an attempt already under way.
struct InFlight {
  std::vector<int> outcomes;  // its uncertain outcomes (indices into pending)
  std::vector<int> dels;      // fluents it deletes
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

  // Is planned action a used up by an attempt in flight (it consumes a fluent
  // that attempt deletes)?
  bool spent(const Problem &pb, int a) const {
    for (const auto &fl : in_flight) {
      if (uses_up(pb, fl, a)) return true;
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
    std::vector<ProbAdd> prob_adds;
    for (const auto &[t_abs, e] : s.upcoming_effects()) {
      prob_adds.clear();
      walk_effect(pb, *e, std::max(0.0, t_abs - t0), 1.0, prob_adds);
      const int attempt = static_cast<int>(in_flight.size());
      InFlight fl{{}, {}, -1};
      for (const auto &pa : prob_adds) {
        double p = std::min(pa.prob, 1.0);
        if (p >= 1.0 - 1e-9) {
          avail.push_back({pa.fluent, pa.time});
          continue;
        }
        fl.outcomes.push_back(static_cast<int>(pending.size()));
        pending.push_back({pa.fluent, pa.time, p, attempt});
      }
      if (fl.outcomes.empty()) continue;
      for (const auto &df : e->flipped_neg_fluents()) {
        int c = pb.lookup(df);
        if (c >= 0) fl.dels.push_back(c);
      }
      // Where it happens: the location precondition of a planned attempt it
      // uses up.
      for (int i : fl.outcomes) {
        for (const auto &ach : pb.achievers[pending[i].fluent]) {
          if (fl.loc >= 0 || !uses_up(pb, fl, ach.action)) continue;
          for (int q : pb.acts[ach.action].pre) {
            if (pb.loc_agent[q] >= 0) { fl.loc = q; break; }
          }
        }
      }
      in_flight.push_back(std::move(fl));
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
    if (opts.timed_init) {
      defer_own_effects(pb);
    } else {
      for (auto &a : avail) a.second = 0.0;
      for (auto &pd : pending) pd.time = 0.0;
    }
  }

 private:
  std::vector<uint8_t> now_;  // scratch: fluent available now

  // An uncertain add of one in-flight effect, summed over its branches.
  struct ProbAdd {
    int fluent;
    double prob, time;
  };

  static bool uses_up(const Problem &pb, const InFlight &fl, int a) {
    for (int c : pb.acts[a].consumes) {
      if (std::find(fl.dels.begin(), fl.dels.end(), c) != fl.dels.end()) return true;
    }
    return false;
  }

  void walk_effect(const Problem &pb, const GroundedEffect &e, double trel, double prob,
                   std::vector<ProbAdd> &prob_adds) {
    for (const auto &f : e.pos_fluents()) {
      int id = pb.lookup(f);
      if (id < 0) continue;
      if (prob >= 1.0 - 1e-9) {
        avail.push_back({id, trel});
        continue;
      }
      // Accumulate across mutually exclusive branches of this root effect.
      auto it = std::find_if(prob_adds.begin(), prob_adds.end(),
                             [id](const ProbAdd &pa) { return pa.fluent == id; });
      if (it == prob_adds.end()) {
        prob_adds.push_back({id, prob, trel});
      } else {
        it->prob += prob;
        it->time = std::min(it->time, trel);
      }
    }
    // Relaxed: conditional branches are assumed to fire.
    for (const auto &cb : e.cond_effects()) {
      for (const auto &sub : cb.effects()) {
        walk_effect(pb, *sub, trel + sub->time(), prob, prob_adds);
      }
    }
    for (const auto &pb_ : e.prob_effects()) {
      if (pb_.prob() <= 0.0) continue;
      for (const auto &sub : pb_.effects()) {
        walk_effect(pb, *sub, trel + sub->time(), prob * pb_.prob(), prob_adds);
      }
    }
  }

  // A free agent must act now: it cannot idle until one of its own pending
  // effects fires (e.g. a "just picked" flag that clears 0.1 s after a pick,
  // so the object cannot go straight back down). Such effects count for the
  // agent only after the shortest action it can start now.
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
  std::vector<double> cost, rho, score, wait, act_rho;
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
    const Add *ad = pb.add(b, f);
    return ad && ad->uncertain();
  }

  void run(const Problem &pb, const TimedState &ts) {
    const std::size_t nf = pb.num_fluents();
    const std::size_t na = pb.acts.size();
    cost.assign(nf, INF);
    rho.assign(nf, 0.0);
    score.assign(nf, INF);
    best.assign(nf, NONE);
    done.assign(nf, 0);
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
};

// A relaxed plan read off a pass.
struct Extraction {
  double dur = 0.0;     // sum of action durations
  std::vector<int> fluents;  // fluents visited (for coverage)
  std::vector<int> actions;  // actions on the plan
};

class Extractor {
 public:
  // Start a new plan; extract() calls until the next one add to it.
  void next_stamp(const Problem &pb) {
    if (fl_stamp_.size() != pb.num_fluents()) fl_stamp_.assign(pb.num_fluents(), 0);
    if (ach_stamp_.size() != pb.num_fluents()) ach_stamp_.assign(pb.num_fluents(), 0);
    if (++stamp_ == 0) {
      std::fill(fl_stamp_.begin(), fl_stamp_.end(), 0);
      std::fill(ach_stamp_.begin(), ach_stamp_.end(), 0);
      stamp_ = 1;
    }
  }

  // Walk back from `roots` via the pass's chosen achievers, costliest subgoal
  // first (as FF does). A fluent added by an action already on the plan
  // counts as achieved, so a cheaper subgoal covered as a side effect (the
  // `hand-full` a pick adds) pulls in no achiever of its own. `via`, if
  // given, achieves roots[0] in place of the pass's choice.
  void extract(const Problem &pb, const Pass &P, const std::vector<int> &roots, Extraction &ex,
               int via = Pass::NONE) {
    std::priority_queue<std::pair<double, int>> heap;  // (cost, fluent), max first
    for (int f : roots) {
      if (f >= 0) heap.push({P.cost[f], f});
    }
    while (!heap.empty()) {
      auto [key, f] = heap.top();
      heap.pop();
      if (fl_stamp_[f] == stamp_) continue;
      if (!heap.empty() && heap.top().first >= key - 1e-9) f = take_tied(pb, P, f, key, heap);
      fl_stamp_[f] = stamp_;
      int b = (via != Pass::NONE && f == roots[0]) ? via : P.best[f];
      if (b == Pass::AVAIL || b == Pass::NONE) continue;
      ex.fluents.push_back(f);
      if (Pass::is_pending(b) || ach_stamp_[f] == stamp_) continue;
      ex.dur += pb.acts[b].dur;
      ex.actions.push_back(b);
      for (const auto &ad : pb.acts[b].adds) {
        if (ad.prob > 1e-9) ach_stamp_[ad.fluent] = stamp_;
      }
      for (int p : pb.acts[b].pre) heap.push({P.cost[p], p});
    }
  }

 private:
  // fl_stamp_: required subgoal already processed; ach_stamp_: added by an
  // action already on the plan. A fluent's support is an action that adds it,
  // so this also keeps each action on the plan once.
  std::vector<uint32_t> fl_stamp_, ach_stamp_;
  uint32_t stamp_ = 0;
  std::vector<int> tied_;

  // Of the subgoals tied with f at cost `key`, the one to extract first: the
  // one whose support adds the most of the others, so `holding r X` comes
  // before the `hand-full r` its pick also adds. Fluent ids (hash order)
  // would otherwise decide, and when `hand-full r` came first it pulled in a
  // pick of whichever object fills the hand best, making the value depend on
  // which robot is called what. Ids still break what remains. The rest go
  // back.
  int take_tied(const Problem &pb, const Pass &P, int f, double key,
                std::priority_queue<std::pair<double, int>> &heap) {
    tied_.assign(1, f);
    while (!heap.empty() && heap.top().first >= key - 1e-9) {
      int g = heap.top().second;
      heap.pop();
      if (fl_stamp_[g] != stamp_ && std::find(tied_.begin(), tied_.end(), g) == tied_.end()) {
        tied_.push_back(g);
      }
    }
    int pick = f, most = -1;
    for (int g : tied_) {
      int n = 0, b = P.best[g];
      if (b >= 0 && ach_stamp_[g] != stamp_) {
        for (const auto &ad : pb.acts[b].adds) {
          if (ad.prob > 1e-9 && ad.fluent != g &&
              std::find(tied_.begin(), tied_.end(), ad.fluent) != tied_.end()) {
            ++n;
          }
        }
      }
      if (n > most) {
        most = n;
        pick = g;
      }
    }
    for (int g : tied_) {
      if (g != pick) heap.push({P.cost[g], g});
    }
    return pick;
  }
};

}  // namespace concurrent
}  // namespace railroad
