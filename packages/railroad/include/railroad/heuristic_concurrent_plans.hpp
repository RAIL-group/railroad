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

// Locations to visit in order, and the relaxed-plan cost of each leg (its
// fallback where no single move covers it).
struct Route {
  std::vector<int> locs;
  std::vector<double> fallback;
};

struct GoalPlan {
  bool ok = false;
  double other = 0.0;     // durations of the plan's non-move actions (of all
                          // its actions when it is not routed)
  double critical = 0.0;  // relaxed time to the goal after the agent is ready
  double delta = 0.0;
  Route route;
  std::vector<int> covers;
  // Expected search: the uncertain subgoal, and the routes and non-move
  // durations before and after it (the search's own is in its expected time).
  int search_f = -1;
  int goal = -1;  // the goal fluent
  Route pre, post;
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

// Travel time for agent r from `start` along `route`; a leg no single move
// covers costs its fallback. `end`, if given, receives the last location.
inline double route_through(const Problem &pb, int r, int start, const Route &route,
                            int *end = nullptr) {
  double total = 0.0;
  int prev = start;
  for (std::size_t i = 0; i < route.locs.size(); ++i) {
    int y = route.locs[i];
    if (y == prev) continue;
    double d = (prev >= 0) ? pb.move_dur(r, prev, y) : INF;
    total += std::isfinite(d) ? d : route.fallback[i];
    prev = y;
  }
  if (end) *end = prev;
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

// When agent r needs to do each of the plan's actions: when it could start
// in the relaxation, but after any other action that needs what it uses up --
// the relaxation has no deletes to order them (putting an egg into a bowl, no
// longer holding it, comes after boiling it).
inline std::vector<double> need_times(const Problem &pb, const Pass &P, int r,
                                      const std::vector<int> &plan) {
  std::vector<double> need(plan.size());
  for (std::size_t i = 0; i < plan.size(); ++i) need[i] = P.wait[plan[i]];
  for (std::size_t round = 0; round < plan.size(); ++round) {
    bool changed = false;
    for (std::size_t i = 0; i < plan.size(); ++i) {
      if (pb.move_dest(plan[i], r) >= 0) continue;
      for (int q : pb.acts[plan[i]].consumes) {
        for (std::size_t k = 0; k < plan.size(); ++k) {
          if (k == i || need[i] > need[k] || pb.move_dest(plan[k], r) >= 0) continue;
          const std::vector<int> &pre = pb.acts[plan[k]].pre;
          if (std::find(pre.begin(), pre.end(), q) == pre.end()) continue;
          need[i] = need[k] + 1e-6;
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  return need;
}

// A location the plan needs agent r at: when, and the plan action that needs
// it (-1: a waypoint or the goal itself).
struct Visit {
  double t;
  int loc;
  int action;
};

// The locations of `visits`, each once, in the order they are first needed.
inline Route ordered_route(const Problem &pb, const Pass &P, std::vector<Visit> visits) {
  std::sort(visits.begin(), visits.end(), [](const Visit &a, const Visit &b) {
    return a.t < b.t || (a.t == b.t && a.loc < b.loc);
  });
  Route route;
  for (const Visit &v : visits) {
    if (std::find(route.locs.begin(), route.locs.end(), v.loc) != route.locs.end()) continue;
    route.locs.push_back(v.loc);
    int b = P.best[v.loc];
    route.fallback.push_back(b >= 0 ? pb.acts[b].dur : 0.0);
  }
  return route;
}

// Find the goal's uncertain search and split the plan around it. The search
// is the implied `found X` when the goal has one -- even with one attempt
// left, so the costing does not switch schemes when the second-last place
// fails -- else the probabilistic subgoal with the most attempts. After the
// search, an action that needs what the search reveals (picking the object
// up) happens wherever it succeeds, so only the other actions' places (where
// to bring it) are on the route from there.
inline void split_search(Context &cx, Pass &P, int r, int g, int companion,
                         const std::vector<int> &plan, const std::vector<double> &need,
                         const std::vector<Visit> &visits, GoalPlan &tp) {
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
  const Attempt att = support_attempt(cx.ts, P, f_star);
  double search_need = (b >= 0) ? P.wait[b] : 0.0;
  for (std::size_t i = 0; i < plan.size(); ++i) {
    const int a = plan[i];
    if (a == b || pb.move_dest(a, r) >= 0) continue;
    (need[i] >= search_need ? tp.other_post : tp.other_pre) += pb.acts[a].dur;
  }
  auto anchored = [&](int a) {
    for (int q : pb.acts[a].pre) {
      if (attempt_reveals(pb, cx.ts, att, q)) return true;
    }
    return false;
  };
  std::vector<Visit> before, after;
  for (const Visit &v : visits) {
    const bool post = v.t >= search_need;
    // Off the route: the search's own place, and after it the places of
    // actions done wherever the search succeeds.
    if (v.action >= 0 && (v.action == b || (post && anchored(v.action)))) continue;
    (post ? after : before).push_back(v);
  }
  tp.pre = ordered_route(pb, P, std::move(before));
  tp.post = ordered_route(pb, P, std::move(after));
  tp.search_f = f_star;
  tp.goal = g;
  // The expected search time supersedes the retry deltas of what the search
  // attempt reveals (e.g. `found X` and `at X place`).
  tp.delta_rest = 0.0;
  for (int f : tp.covers) {
    if (!P.uncertain(pb, f)) continue;
    if (f == f_star || (att.any() && support_attempt(cx.ts, P, f) == att)) continue;
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
  std::vector<Visit> visits;
  std::vector<int> move_dests;
  const std::vector<double> need = need_times(pb, P, r, ex.actions);
  for (std::size_t i = 0; i < ex.actions.size(); ++i) {
    const int a = ex.actions[i];
    int dest = pb.move_dest(a, r);
    if (dest >= 0) {
      move_dests.push_back(dest);
      continue;
    }
    tp.other += pb.acts[a].dur;
    for (int p : pb.acts[a].pre) {
      if (pb.loc_agent[p] == r) visits.push_back({need[i], p, a});
    }
  }
  for (int d : move_dests) {
    bool consumed = false;
    for (const auto &v : visits) consumed = consumed || v.loc == d;
    if (!consumed) visits.push_back({P.cost[d], d, -1});
  }
  if (pb.loc_agent[g] == r) visits.push_back({P.cost[g], g, -1});
  tp.route = ordered_route(pb, P, visits);
  if (cx.opts.expected_search) split_search(cx, P, r, g, companion, ex.actions, need, visits, tp);
  return tp;
}

// Serial duration of goal tp for agent r starting at `start` at time
// t_start, and the retry delta to add on top.
inline double goal_duration(Context &cx, const Pass &P, int r, int start, double t_start,
                          const GoalPlan &tp, double &delta_out) {
  const Problem &pb = cx.pb;
  if (tp.search_f >= 0) {
    int pos = start;
    double pre = route_through(pb, r, start, tp.pre, &pos);
    // The rest of the goal from wherever the object turns up; nothing if
    // finding it there already achieves the goal.
    auto rest = [&](int loc, bool achieves) {
      if (achieves) return 0.0;
      return route_through(pb, r, loc >= 0 ? loc : pos, tp.post) + tp.other_post;
    };
    std::vector<Found> where;
    double search =
        expected_search(pb, cx.ts, P, tp.search_f, tp.goal, r, pos, t_start + pre, rest, where);
    if (std::isfinite(search)) {
      double post = 0.0;
      for (const auto &w : where) post += w.prob * w.rest;
      delta_out = tp.delta_rest;
      return pre + search + post + tp.other_pre;
    }
  }
  // An unrouted plan has an empty route.
  delta_out = tp.delta;
  return tp.other + route_through(pb, r, start, tp.route);
}

// Lower bound on a goal's duration from the relaxed critical path. Not for an
// expected search: that is an expectation over outcomes, some of which end
// sooner than the relaxed time to success.
inline double bound(const GoalPlan &tp) { return tp.search_f >= 0 ? 0.0 : tp.critical; }

}  // namespace concurrent
}  // namespace railroad
