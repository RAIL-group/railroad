#pragma once

// Expected search: cost an uncertain subgoal -- one with several
// probabilistic attempts, such as the places an object may be found -- as an
// expected route over its attempts.
//
// The agent walks greedily to the unused attempt with the most probability
// per unit of travel + execution; attempts already in flight complete at
// their scheduled times. With all events in time order, the expected time to
// the first success is E = sum_i (T_i - T_{i-1}) P(no success before T_i).
// Where the search may succeed is weighted by the *unconditional* probability
// of succeeding there: conditioning on success would re-weight the remaining
// places after every failure, so the estimate would jump whenever a search
// fails.

#include "railroad/heuristic_concurrent_relaxation.hpp"

namespace railroad {
namespace concurrent {

// An attempt at uncertain fluent f, as agent r sees it: one r can still
// start, or one already in flight, whoever started it. The two are treated
// alike, so starting an attempt does not change how the search is costed.
struct SearchAttempt {
  int loc;          // r's location fluent where it happens (-1: none)
  double exec, prob;
  double fallback;  // relaxed time it could start (in flight: when it ends)
  bool achieves;    // succeeding also achieves the goal (in place)
  int in_flight;    // >= 0: the in-flight attempt (TimedState::in_flight)
};

inline std::vector<SearchAttempt> attempts_for(const Problem &pb, const TimedState &ts,
                                               const Pass &P, int r, int f, int goal = -1) {
  std::vector<SearchAttempt> out;
  for (const auto &pd : ts.pending) {
    if (pd.fluent != f) continue;
    int loc = pb.as_agent_loc(r, ts.in_flight[pd.attempt].loc);
    out.push_back({loc, 0.0, pd.prob, pd.time, ts.reveals(pd.attempt, goal), pd.attempt});
  }
  for (const auto &ach : pb.achievers[f]) {
    if (!P.allowed(pb, ach.action) || P.unmet[ach.action] != 0 || ts.spent(pb, ach.action)) continue;
    const CompiledAction &a = pb.acts[ach.action];
    double p = a.adds[ach.add].prob;
    if (p <= 1e-9) continue;
    int loc = -1;
    for (int q : a.pre) {
      if (pb.loc_agent[q] == r) { loc = q; break; }
    }
    out.push_back({loc, a.dur, p, P.wait[ach.action], pb.adds(ach.action, goal), -1});
  }
  return out;
}

// The attempt behind f's chosen support in P, as an id comparable across
// fluents: an action (>= 0), an in-flight attempt (-2 - index), or none (-1).
inline int support_attempt(const TimedState &ts, const Pass &P, int f) {
  int b = P.best[f];
  if (b >= 0) return b;
  if (Pass::is_pending(b)) return -2 - ts.pending[Pass::pending_index(b)].attempt;
  return -1;
}

// Is q an uncertain outcome of attempt id `att` (see support_attempt)?
inline bool attempt_reveals(const Problem &pb, const TimedState &ts, int att, int q) {
  if (att >= 0) {
    for (const auto &ad : pb.acts[att].adds) {
      if (ad.uncertain() && ad.fluent == q) return true;
    }
    return false;
  }
  return att <= -2 && ts.reveals(-2 - att, q);
}

// A place a search may succeed, with the unconditional probability that it
// succeeds there and the time left from there (`rest`).
struct Found {
  double prob;
  double rest;
};

// Expected time, from t_start, until f holds when agent r, starting at
// location fluent `pos`, works through f's attempts in pass P. rest(loc,
// achieves) is the time left after succeeding at `loc`.
template <class Rest>
inline double expected_search(const Problem &pb, const TimedState &ts, const Pass &P, int f,
                              int goal, int r, int pos, double t_start, const Rest &rest,
                              std::vector<Found> &where) {
  where.clear();
  struct Event { double t, p, rest; };
  std::vector<Event> events;
  std::vector<SearchAttempt> cands;
  double fail = 1.0;
  for (const auto &at : attempts_for(pb, ts, P, r, f, goal)) {
    if (at.in_flight >= 0) {
      events.push_back({at.fallback, at.prob, rest(at.loc, at.achieves)});
      fail *= 1.0 - at.prob;
    } else {
      cands.push_back(at);
    }
  }
  std::vector<char> used(cands.size(), 0);
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
    events.push_back({t, c.prob, rest(c.loc, c.achieves)});
    fail *= 1.0 - c.prob;
  }
  if (events.empty()) return INF;
  // Outcomes at the same time are credited cheapest-rest first: if several
  // succeed, the agent goes on from the best of them.
  std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
    return a.t < b.t || (a.t == b.t && a.rest < b.rest);
  });
  double expected = 0.0, prev = t_start, still = 1.0;
  for (const auto &e : events) {
    double tt = std::max(e.t, prev);
    expected += (tt - prev) * still;
    prev = tt;
    double here = still * e.p;
    if (here > 0.0) where.push_back({here, e.rest});
    still *= 1.0 - e.p;
  }
  return expected;
}

}  // namespace concurrent
}  // namespace railroad
