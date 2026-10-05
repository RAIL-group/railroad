#pragma once

// Expected search: costing an uncertain subgoal -- one with several
// probabilistic attempts, such as the places an object may be found -- as an
// expected route over its attempts instead of a plan through one of them plus
// a retry delta.
//
// From where the agent is, it walks to the unused attempt with the most
// probability per unit of travel + execution (a greedy route). Attempts
// already in flight complete at their scheduled times regardless. With all
// events in time order, the expected time to the first success is
// E = sum_i (T_i - T_{i-1}) P(no success before T_i), and the search reports
// where it may succeed, weighted by the *unconditional* probability of
// succeeding there: independent attempts may all fail, and conditioning on
// success would re-weight the remaining places after every failure, so the
// estimate would jump whenever a search fails.

#include "railroad/heuristic_concurrent_relaxation.hpp"

namespace railroad {
namespace concurrent {

// An attempt at uncertain fluent f that agent r can still start: one per
// attempt group (the nearest in the relaxation).
struct SearchAttempt {
  int group, loc;   // loc: r's location fluent where it happens (-1: none)
  double exec, prob;
  double fallback;  // relaxed time the attempt could start
  bool achieves;    // succeeding also achieves the task goal (in place)
};

// Attempts at f open to agent r in pass P. Groups with an attempt already in
// flight are spent.
inline std::vector<SearchAttempt> attempts_for(const Problem &pb, const TimedState &ts,
                                               const Pass &P, int r, int f, int goal = -1) {
  std::vector<SearchAttempt> out;
  for (const auto &ach : pb.achievers[f]) {
    if (!P.allowed(pb, ach.action) || P.unmet[ach.action] != 0) continue;
    const CompiledAction &a = pb.acts[ach.action];
    double p = a.adds[ach.add].prob;
    if (p <= 1e-9) continue;
    bool spent = false;
    for (const auto &pd : ts.pending) {
      if (pd.fluent == f && pd.group >= 0 && pd.group == ach.group) { spent = true; break; }
    }
    if (spent) continue;
    int loc = -1;
    for (int q : a.pre) {
      if (pb.loc_agent[q] == r) { loc = q; break; }
    }
    SearchAttempt at{ach.group, loc, a.dur, p, P.wait[ach.action], pb.adds(ach.action, goal)};
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

// Where a search may succeed (as r's location fluent), with the
// unconditional probability that it succeeds there, and whether that
// success already achieves the task goal.
struct Found {
  int loc;
  double prob;
  bool achieves;
};

// Expected time, from t_start, until f holds when agent r, starting at
// location fluent `pos`, works through f's attempts in pass P.
inline double expected_search(const Problem &pb, const TimedState &ts, const Pass &P, int f,
                              int goal, int r, int pos, double t_start,
                              std::vector<Found> &where) {
  where.clear();
  struct Event { double t, p; int loc; bool achieves; };
  std::vector<Event> events;
  for (const auto &pd : ts.pending) {
    if (pd.fluent != f) continue;
    int loc = -1;  // where the pending attempt happens, as r's fluent
    bool achieves = false;
    for (const auto &ach : pb.achievers[f]) {
      if (ach.group != pd.group || pd.group < 0) continue;
      for (int q : pb.acts[ach.action].pre) {
        if (pb.loc_agent[q] >= 0) { loc = pb.as_agent_loc(r, q); break; }
      }
      achieves = pb.adds(ach.action, goal);
      if (loc >= 0) break;
    }
    events.push_back({pd.time, pd.prob, loc, achieves});
  }
  const std::vector<SearchAttempt> cands = attempts_for(pb, ts, P, r, f, goal);
  std::vector<char> used(cands.size(), 0);
  double fail = 1.0;
  for (const auto &e : events) fail *= 1.0 - e.p;
  double t = t_start;
  while (fail > 1e-3) {
    int best = -1;
    double best_ratio = -1.0, best_dt = 0.0;
    for (int i = 0; i < static_cast<int>(cands.size()); ++i) {
      if (used[i]) continue;
      const SearchAttempt &c = cands[i];
      double travel = 0.0;
      if (c.loc >= 0 && c.loc != pos) {
        travel = (pos >= 0) ? pb.move_dur(r, pos, c.loc) : INF;
        if (!std::isfinite(travel)) travel = std::max(0.0, c.fallback - t);
      }
      double dt = travel + c.exec;
      double ratio = c.prob / std::max(dt, 1e-9);
      if (ratio > best_ratio) { best_ratio = ratio; best = i; best_dt = dt; }
    }
    if (best < 0) break;
    const SearchAttempt &c = cands[best];
    used[best] = 1;
    t += best_dt;
    if (c.loc >= 0) pos = c.loc;
    events.push_back({t, c.prob, pb.as_agent_loc(r, c.loc), c.achieves});
    fail *= 1.0 - c.prob;
  }
  if (events.empty()) return INF;
  std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) { return a.t < b.t; });
  double expected = 0.0, prev = t_start, still = 1.0;
  for (const auto &e : events) {
    double tt = std::max(e.t, prev);
    expected += (tt - prev) * still;
    prev = tt;
    double here = still * e.p;
    if (here > 0.0) where.push_back({e.loc, here, e.achieves});
    still *= 1.0 - e.p;
  }
  return expected;
}

}  // namespace concurrent
}  // namespace railroad
