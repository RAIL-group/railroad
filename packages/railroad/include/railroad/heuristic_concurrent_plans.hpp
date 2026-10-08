#pragma once

// Goal plans: one agent's relaxed plan for one goal, and its duration.
//
// Core: the plan is read off the agent's pass; its duration is the sum of its
// actions plus retry deltas, bounded below by the relaxed critical path.
//
// Route chaining (when the agent has location fluents): the delete relaxation
// lets an agent be in several places at once -- a fetch costs start->object +
// start->target, and an agent standing at the target never has to come back.
// Instead the places the plan needs are visited in the order they are needed,
// each leg costed from the previous one.
//
// Expected search (when the goal has an uncertain subgoal): the plan is split
// around the search, the search is costed as an expected route, and the rest
// of the goal from wherever the search succeeds.

#include "railroad/heuristic_concurrent_search.hpp"

namespace railroad {
namespace concurrent {

// What goal planning reads: the compiled problem, the options, the timed
// state, and the shared relaxed-plan extractor.
struct Context {
  const Problem &pb;
  const ConcurrentHeuristicOptions &opts;
  const TimedState &ts;
  Extractor &ex;
};

struct GoalPlan {
  bool ok = false;
  double other = 0.0;     // durations of the plan's non-move actions
  double critical = 0.0;  // relaxed time to the goal after the agent is ready
  double delta = 0.0;
  std::vector<int> locs;  // locations to visit, in order
  std::vector<double> leg_fallback;  // relaxed-plan cost of each leg
  std::vector<int> covers;
  // Expected search: the uncertain subgoal, and the places and non-move
  // durations before and after it (the search's own is in its expected time).
  int search_f = -1;
  int goal = -1;  // the goal fluent
  std::vector<int> pre_locs, post_locs;
  std::vector<double> pre_fb, post_fb;
  double other_pre = 0.0, other_post = 0.0;
  double delta_rest = 0.0;  // retry deltas the expected search does not cover
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

// Find the goal's uncertain search and split the plan around it. The search
// is the implied `found X` when the goal has one -- even with one attempt
// left, so the costing does not switch schemes when the second-last place
// fails -- else the probabilistic subgoal with the most attempts. After the
// search, an action that needs what the search reveals (picking the object
// up) happens wherever it succeeds, so only the other actions' places (where
// to bring it) are on the route from there.
inline void split_search(Context &cx, Pass &P, int r, int g, int companion,
                         const std::vector<int> &plan, const std::vector<int> &move_dests,
                         GoalPlan &tp) {
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
  // The plan's search attempt (planned or in flight) and when it must start.
  const int b = P.best[f_star];
  const int att = support_attempt(cx.ts, P, f_star);
  double search_need = (b >= 0) ? P.wait[b] : 0.0;
  auto anchored = [&](int a) {
    for (int q : pb.acts[a].pre) {
      if (attempt_reveals(pb, cx.ts, att, q)) return true;
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
  // Waypoints and location goals, as in plan_goal.
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
  // The expected search time supersedes the retry deltas of what the search
  // attempt reveals (e.g. `found X` and `at X place`).
  tp.delta_rest = 0.0;
  for (int f : tp.covers) {
    if (!P.uncertain(pb, f)) continue;
    if (f == f_star || (att != -1 && support_attempt(cx.ts, P, f) == att)) continue;
    tp.delta_rest += P.delta(pb, cx.ts, f);
  }
}

// Agent r's plan for goal g (with its companion `found X`, if any), ready to
// start at `ready`. r < 0: no single agent (the team relaxation).
inline GoalPlan plan_goal(Context &cx, Pass &P, int g, int companion, int r, double ready) {
  const Problem &pb = cx.pb;
  GoalPlan tp;
  if (!P.reachable(g)) return tp;
  if (companion >= 0 && !P.reachable(companion)) companion = -1;
  // Under expected search, a goal whose cheapest support is an uncertain
  // search (searching the target place, hoping the object is already there)
  // is planned through its deterministic achiever: wherever else the object
  // turns up, it must be brought over. Outcomes that achieve the goal in
  // place cost no delivery (goal_duration).
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
    cx.ex.extract(pb, cx.ts, P, {g}, ex);
    if (companion >= 0) cx.ex.extract(pb, cx.ts, P, {companion}, ex);
  } else {
    std::vector<int> roots{g};
    if (companion >= 0) roots.push_back(companion);
    cx.ex.extract(pb, cx.ts, P, roots, ex);
  }
  tp.ok = true;
  tp.delta = ex.delta;
  double goal_cost = P.cost[g];
  if (companion >= 0) goal_cost = std::max(goal_cost, P.cost[companion]);
  tp.critical = std::max(0.0, goal_cost - ready);
  tp.covers = std::move(ex.fluents);
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
    int b = P.best[v.second];
    tp.leg_fallback.push_back(b >= 0 ? pb.acts[b].dur : 0.0);
  }
  if (cx.opts.expected_search) split_search(cx, P, r, g, companion, ex.actions, move_dests, tp);
  return tp;
}

// Serial duration of goal tp for agent r starting at `start` at time
// t_start, and the retry delta to add on top.
inline double goal_duration(Context &cx, const Pass &P, int r, int start, double t_start,
                          const GoalPlan &tp, double &delta_out) {
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
  // The rest of the goal from wherever the object turns up; nothing if
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

// Lower bound on a goal's duration from the relaxed critical path. Not for an
// expected search: that is an expectation over outcomes, some of which end
// sooner than the relaxed time to success.
inline double bound(const GoalPlan &tp) { return tp.search_f >= 0 ? 0.0 : tp.critical; }

}  // namespace concurrent
}  // namespace railroad
