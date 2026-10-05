#pragma once

// Task plans: one agent's relaxed plan for one goal task, and its duration.
//
// Core: the plan is read off the agent's relaxation pass, and its duration is
// the sum of its actions (plus retry deltas), bounded below by the relaxed
// critical path.
//
// Route chaining (when the agent has location fluents): a delete relaxation
// lets an agent be in several places at once -- a fetch costs start->object +
// start->target, and an agent standing at the target never has to come back.
// Instead the locations the plan needs are visited in the order they are
// needed, each leg costed from the previous one.
//
// Expected search (when the task has an uncertain subgoal; see
// heuristic_concurrent_search.hpp): the plan is split around the search, the
// search is costed as an expected route, and the rest of the task is costed
// from wherever the search succeeds.

#include "railroad/heuristic_concurrent_search.hpp"

namespace railroad {
namespace concurrent {

// What task planning reads: the compiled problem, the options, the timed
// state, and the shared relaxed-plan extractor.
struct Context {
  const Problem &pb;
  const ConcurrentHeuristicOptions &opts;
  const TimedState &ts;
  Extractor &ex;
};

struct TaskPlan {
  bool ok = false;
  double other = 0.0;     // durations of the plan's non-move actions
  double critical = 0.0;  // relaxed time to the goal after the agent is ready
  double delta = 0.0;
  std::vector<int> locs;  // locations to visit, in order
  std::vector<double> leg_fallback;  // relaxed-plan cost of each leg
  std::vector<int> covers;
  // Expected-search costing: the uncertain subgoal, the places needed before
  // and after it (after it, from wherever it succeeds), the non-move
  // durations before and after it (the search's own is inside the expected
  // search time), and the retry deltas of everything else.
  int search_f = -1;
  int goal = -1;  // the task's goal fluent
  std::vector<int> pre_locs, post_locs;
  std::vector<double> pre_fb, post_fb;
  double other_pre = 0.0, other_post = 0.0;
  double delta_rest = 0.0;
  // For ordering an agent's tasks (order_conflicts), excluding the agent's
  // location (route chaining handles it): facts the plan relies on from the
  // state, facts its actions delete for good, and facts they add.
  std::vector<int> relies, destroys, adds;
};

// Where agent r is (or will be once its current action ends): its location
// fluent available at the latest time. -1 if it has none.
inline int agent_location(const Problem &pb, const Pass &P, int r) {
  int loc = -1;
  double t = -1.0;
  for (int f : pb.loc_fluents[r]) {
    if (P.best[f] == Pass::AVAIL && P.cost[f] > t) { t = P.cost[f]; loc = f; }
  }
  return loc;
}

// Travel time for agent r from `start` through `locs` in order; a leg no
// single move covers costs its relaxed-plan fallback. `end` receives the
// last location.
inline double route_through(const Problem &pb, int r, int start, const std::vector<int> &locs,
                            const std::vector<double> &fb, int &end) {
  double total = 0.0;
  int prev = start;
  for (std::size_t i = 0; i < locs.size(); ++i) {
    int y = locs[i];
    if (y == prev) continue;
    double d = (prev >= 0) ? pb.move_dur(r, prev, y) : INF;
    total += std::isfinite(d) ? d : fb[i];
    prev = y;
  }
  end = prev;
  return total;
}

// g's cheapest deterministic achiever in P (-1 if none).
inline int deterministic_achiever(const Problem &pb, const Pass &P, int g) {
  int best = -1;
  double best_t = INF;
  for (const auto &ach : pb.achievers[g]) {
    const CompiledAction &a = pb.acts[ach.action];
    if (!P.allowed(pb, ach.action) || P.unmet[ach.action] != 0) continue;
    if (a.adds[ach.add].uncertain()) continue;
    double t = P.wait[ach.action] + a.dur;
    if (t < best_t) { best_t = t; best = ach.action; }
  }
  return best;
}

// Identify the task's uncertain search -- the implied `found X` when the
// task has one (even with a single place left, so that the estimate does not
// switch costing schemes when the second-last place fails), else the
// probabilistic subgoal with the most attempt groups -- and split the plan
// into what comes before it and what comes after. After it, an action that
// needs something the search reveals (picking the object up where it is)
// happens wherever the search succeeds, so only the other actions' places
// (where to bring it) are on the route from there.
inline void split_search(Context &cx, Pass &P, int r, int g, int companion,
                         const std::vector<int> &plan, const std::vector<int> &move_dests,
                         TaskPlan &tp) {
  const Problem &pb = cx.pb;
  int f_star = -1;
  std::size_t best_n = 1;
  for (int f : tp.covers) {
    if (!P.uncertain(pb, f)) continue;
    std::size_t n = attempts_for(pb, cx.ts, P, r, f, g).size();
    if (f == companion && n >= 1) { f_star = f; break; }
    if (n > best_n) { best_n = n; f_star = f; }
  }
  if (f_star < 0) return;
  int b = P.best[f_star];
  double search_need = (b >= 0) ? P.wait[b] : 0.0;
  auto anchored = [&](int a) {
    if (b < 0) return false;
    for (const auto &ad : pb.acts[b].adds) {
      if (!ad.uncertain()) continue;
      const auto &pre = pb.acts[a].pre;
      if (std::find(pre.begin(), pre.end(), ad.fluent) != pre.end()) return true;
    }
    return false;
  };
  std::vector<std::pair<double, int>> before, after;  // (need time, location)
  for (int a : plan) {
    if (a == b || pb.move_dest(a, r) >= 0) continue;
    bool post = P.wait[a] >= search_need;
    (post ? tp.other_post : tp.other_pre) += pb.acts[a].dur;
    if (post && anchored(a)) continue;
    for (int q : pb.acts[a].pre) {
      if (pb.loc_agent[q] == r) (post ? after : before).push_back({P.wait[a], q});
    }
  }
  // Waypoints and location goals, as in plan_task.
  for (int d : move_dests) {
    bool consumed = false;
    for (int a : plan) {
      if (pb.move_dest(a, r) >= 0) continue;
      const auto &pre = pb.acts[a].pre;
      consumed = consumed || std::find(pre.begin(), pre.end(), d) != pre.end();
    }
    if (!consumed) (P.cost[d] >= search_need ? after : before).push_back({P.cost[d], d});
  }
  if (pb.loc_agent[g] == r) (P.cost[g] >= search_need ? after : before).push_back({P.cost[g], g});
  auto order = [&](std::vector<std::pair<double, int>> &v, std::vector<int> &locs,
                   std::vector<double> &fb) {
    std::sort(v.begin(), v.end());
    for (const auto &[t, loc] : v) {
      if (std::find(locs.begin(), locs.end(), loc) != locs.end()) continue;
      locs.push_back(loc);
      int bb = P.best[loc];
      fb.push_back(bb >= 0 ? pb.acts[bb].dur : 0.0);
    }
  };
  order(before, tp.pre_locs, tp.pre_fb);
  order(after, tp.post_locs, tp.post_fb);
  tp.search_f = f_star;
  tp.goal = g;
  // The retry deltas of the search outcome (and of anything else the same
  // action achieves, e.g. `at X place`) are superseded by the expected
  // search time.
  tp.delta_rest = 0.0;
  for (int f : tp.covers) {
    if (!P.uncertain(pb, f)) continue;
    if (f == f_star || (b >= 0 && P.best[f] == b)) continue;
    tp.delta_rest += P.delta(pb, cx.ts, f);
  }
}

// Agent r's plan for goal g (with its companion `found X`, if any), ready to
// start at `ready`. r < 0: no single agent (the team relaxation).
inline TaskPlan plan_task(Context &cx, Pass &P, int g, int companion, int r, double ready) {
  const Problem &pb = cx.pb;
  TaskPlan tp;
  if (!P.reachable(g)) return tp;
  if (companion >= 0 && !P.reachable(companion)) companion = -1;
  // Under expected search, a goal whose cheapest support is itself an
  // uncertain search (searching the target place, which would reveal the
  // object already there) is planned through its deterministic achiever:
  // wherever else the object turns up it still has to be brought over. The
  // search outcomes that achieve the goal in place cost no delivery.
  int forced_from = Pass::NONE;
  if (cx.opts.expected_search && cx.opts.route_chaining && r >= 0 && P.uncertain(pb, g)) {
    int d = deterministic_achiever(pb, P, g);
    if (d >= 0) {
      forced_from = P.best[g];
      P.best[g] = d;
    }
  }
  struct Restore {
    Pass &P;
    int g, from;
    ~Restore() { if (from != Pass::NONE) P.best[g] = from; }
  } restore{P, g, forced_from};
  cx.ex.next_stamp(pb);
  Extraction ex;
  if (forced_from != Pass::NONE) {
    // The goal first, so that the search on its forced plan does not count
    // as achieving it in passing.
    cx.ex.extract(pb, cx.opts, cx.ts, P, {g}, ex);
    if (companion >= 0) cx.ex.extract(pb, cx.opts, cx.ts, P, {companion}, ex);
  } else {
    std::vector<int> roots{g};
    if (companion >= 0) roots.push_back(companion);
    cx.ex.extract(pb, cx.opts, cx.ts, P, roots, ex);
  }
  tp.ok = true;
  tp.delta = ex.delta;
  double goal_cost = P.cost[g];
  if (companion >= 0) goal_cost = std::max(goal_cost, P.cost[companion]);
  tp.critical = std::max(0.0, goal_cost - ready);
  tp.covers = std::move(ex.fluents);
  if (r >= 0 && cx.opts.order_conflicts) {
    for (int f : ex.avail) {
      if (pb.loc_agent[f] < 0) tp.relies.push_back(f);
    }
    for (int a : ex.actions) {
      for (int c : pb.acts[a].consumes) {
        if (pb.loc_agent[c] < 0) tp.destroys.push_back(c);
      }
      for (const auto &ad : pb.acts[a].adds) {
        if (pb.loc_agent[ad.fluent] < 0 && ad.prob > 1e-9) tp.adds.push_back(ad.fluent);
      }
    }
  }
  if (r < 0 || !cx.opts.route_chaining) {
    tp.other = ex.dur;  // no single agent to route, or routing disabled
    return tp;
  }
  // Locations the plan needs the agent at: the location preconditions of its
  // non-move actions (including where the agent already is -- the relaxation
  // never makes it walk back), plus move destinations nothing else on the
  // plan consumes (waypoints, or location goals). Each is needed when the
  // earliest plan action requiring it could start.
  std::vector<std::pair<double, int>> visits;  // (need time, location)
  std::vector<int> move_dests;
  for (int a : ex.actions) {
    int dest = pb.move_dest(a, r);
    if (dest >= 0) {
      move_dests.push_back(dest);
      continue;
    }
    tp.other += pb.acts[a].dur;
    for (int p : pb.acts[a].pre) {
      if (pb.loc_agent[p] == r) visits.push_back({P.wait[a], p});
    }
  }
  for (int d : move_dests) {
    bool consumed = false;
    for (const auto &v : visits) consumed = consumed || v.second == d;
    if (!consumed) visits.push_back({P.cost[d], d});
  }
  if (pb.loc_agent[g] == r) visits.push_back({P.cost[g], g});
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
    // If no single move reaches it from the previous stop, fall back to the
    // relaxed plan's own cost of getting there.
    int b = P.best[v.second];
    tp.leg_fallback.push_back(b >= 0 ? pb.acts[b].dur : 0.0);
  }
  if (cx.opts.expected_search) split_search(cx, P, r, g, companion, ex.actions, move_dests, tp);
  return tp;
}

// Serial duration of task tp for agent r starting at `start` at time
// t_start, and the retry delta to add on top.
inline double task_serial(Context &cx, const Pass &P, int r, int start, double t_start,
                          const TaskPlan &tp, double &delta_out) {
  const Problem &pb = cx.pb;
  int end;
  if (r < 0) {
    delta_out = tp.delta;
    return tp.other;
  }
  if (tp.search_f < 0) {
    delta_out = tp.delta;
    return tp.other + route_through(pb, r, start, tp.locs, tp.leg_fallback, end);
  }
  int pos = start;
  double pre = route_through(pb, r, start, tp.pre_locs, tp.pre_fb, pos);
  std::vector<Found> where;
  double search = expected_search(pb, cx.ts, P, tp.search_f, tp.goal, r, pos, t_start + pre, where);
  if (!std::isfinite(search)) {
    delta_out = tp.delta;
    return tp.other + route_through(pb, r, start, tp.locs, tp.leg_fallback, end);
  }
  // The rest of the task from wherever the object turns up; nothing if
  // finding it there already achieves the goal.
  double post = 0.0;
  for (const auto &w : where) {
    if (w.achieves) continue;
    post += w.prob * (route_through(pb, r, w.loc >= 0 ? w.loc : pos, tp.post_locs, tp.post_fb, end) +
                      tp.other_post);
  }
  delta_out = tp.delta_rest;
  return pre + search + post + tp.other_pre;
}

// Lower bound on a task's duration from the relaxed critical path. Not for an
// expected search: that is an expectation over outcomes, some of which end
// sooner than the relaxed time to success.
inline double bound(const TaskPlan &tp) { return tp.search_f >= 0 ? 0.0 : tp.critical; }

}  // namespace concurrent
}  // namespace railroad
