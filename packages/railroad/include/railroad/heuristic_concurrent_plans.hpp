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
//
// Provisions (jobs, heuristic_concurrent_schedule.hpp): a plan may leave some
// of its subgoals to other jobs -- a fluent (a door opened), or everything
// about an object it only uses where it is (the pot it boils in). Its relaxed
// plan stops there, and the action that needs one waits for the provision:
// until the providing job is done, and for an object, wherever it turns out
// to be.

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

// When a provided subgoal holds, and for an object where: a distribution over
// places (one if delivered) with the time each holds from. No places: it
// holds from time[0] wherever the agent is.
struct Provision {
  std::vector<int> place;  // place ids
  std::vector<double> prob;
  std::vector<double> time;
  int mode() const {
    int k = 0;
    for (std::size_t i = 1; i < prob.size(); ++i) {
      if (prob[i] > prob[k]) k = static_cast<int>(i);
    }
    return place.empty() ? -1 : place[k];
  }
};

// A route stop below 0 is provision -1 - k. Where an agent is between stops:
// a location fluent, or a provision (any of its places).
struct Spot {
  int loc = -1;
  const Provision *pv = nullptr;
};

// Agent r's travel time from `from` to location fluent y: its move, else the
// relaxed plan's cost `fb` for that leg; from a provision, expected over its
// places.
inline double leg(const Problem &pb, int r, const Spot &from, int y, double fb) {
  auto one = [&](int x, double fallback) {
    if (x == y) return 0.0;
    double d = x >= 0 ? pb.move_dur(r, x, y) : INF;
    return std::isfinite(d) ? d : fallback;
  };
  if (!from.pv) return one(from.loc, fb);
  double e = 0.0;
  for (std::size_t i = 0; i < from.pv->place.size(); ++i) {
    e += from.pv->prob[i] * one(pb.place_loc_of(r, from.pv->place[i]), 0.0);
  }
  return e;
}

// Time for agent r, at `start` at time t0, to go through `stops` in order,
// doing `work[i]` at stop i and waiting at a provision until it holds. Legs
// are expected over where provisions turn out to be; `end` receives the last
// stop.
inline double route(const Problem &pb, int r, Spot start, double t0, const std::vector<int> &stops,
                    const std::vector<double> &fb, const std::vector<double> &work,
                    const std::vector<const Provision *> &prov, Spot &end) {
  double t = t0;
  Spot pos = start;
  for (std::size_t i = 0; i < stops.size(); ++i) {
    int code = stops[i];
    if (code >= 0) {
      if (pos.pv || pos.loc != code) {
        t += leg(pb, r, pos, code, fb[i]);
        pos = Spot{code, nullptr};
      }
    } else if (const Provision *pv = prov.empty() ? nullptr : prov[-1 - code]) {
      if (pv->place.empty()) {
        if (!pv->time.empty()) t = std::max(t, pv->time[0]);
      } else if (pv != pos.pv) {
        double tn = 0.0;
        for (std::size_t j = 0; j < pv->place.size(); ++j) {
          double arrive = t + leg(pb, r, pos, pb.place_loc_of(r, pv->place[j]), 0.0);
          tn += pv->prob[j] * std::max(arrive, pv->time[j]);
        }
        t = tn;
        pos = Spot{-1, pv};
      }
    }
    t += work[i];
  }
  end = pos;
  return t - t0;
}

// A stop on a route: when it is needed, where (a location fluent or provision
// -1 - k), and the work done there.
struct Visit {
  double need;
  int code;
  double work;
};

// Order visits by need, one per stop (its earliest need, all its work).
inline void order_visits(const Problem &pb, const Pass &P, std::vector<Visit> &v,
                         std::vector<int> &stops, std::vector<double> &fb,
                         std::vector<double> &work) {
  std::sort(v.begin(), v.end(), [](const Visit &a, const Visit &b) {
    return a.need < b.need || (a.need == b.need && a.code < b.code);
  });
  for (const Visit &x : v) {
    auto it = std::find(stops.begin(), stops.end(), x.code);
    if (it != stops.end()) {
      work[it - stops.begin()] += x.work;
      continue;
    }
    stops.push_back(x.code);
    int b = x.code >= 0 ? P.best[x.code] : -1;
    fb.push_back(b >= 0 ? pb.acts[b].dur : 0.0);
    work.push_back(x.work);
  }
}

struct GoalPlan {
  bool ok = false;
  double other = 0.0;     // durations of the plan's non-move actions
  double critical = 0.0;  // relaxed time to the goal after the agent is ready
  double delta = 0.0;
  // Stops to visit in order, the relaxed-plan cost of each leg, and the work
  // done at each (what is done at no stop, wherever the agent is).
  std::vector<int> locs;
  std::vector<double> leg_fallback, work;
  std::vector<int> covers;
  // Expected search: the uncertain subgoal, and the stops and non-move
  // durations before and after it (the search's own is in its expected time).
  int search_f = -1;
  int goal = -1;  // the goal fluent (the first root)
  std::vector<int> pre_locs, post_locs;
  std::vector<double> pre_fb, post_fb, pre_work, post_work;
  double other_pre = 0.0, other_post = 0.0;
  double delta_rest = 0.0;  // retry deltas the expected search does not cover
  // Subgoals other jobs provide (fluents or objects; see item_index); a stop
  // below 0 is provision -1 - k for provided[k].
  std::vector<int> provided;
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

// The stops at which agent r does non-move action a, needed at `need`: its
// location, then a wait for each provision it needs -- unless it uses a
// provided object where the object is, which then decides where. The action
// is done at the last of them.
inline void action_stops(const Problem &pb, int a, int r, double need,
                         const std::vector<int> &provided, std::vector<Visit> &out) {
  const CompiledAction &ca = pb.acts[a];
  std::vector<int> waits;
  bool placed = false;
  for (int q : ca.pre) {
    int k = item_index(pb, provided, q);
    if (k < 0 || std::find(waits.begin(), waits.end(), -1 - k) != waits.end()) continue;
    waits.push_back(-1 - k);
    placed = placed || (item_object(provided[k]) >= 0 && pb.obj_place[q] >= 0);
  }
  std::size_t first = out.size();
  if (!placed) {
    for (int q : ca.pre) {
      if (pb.loc_agent[q] == r) out.push_back({need, q, 0.0});
    }
  }
  // Just after the location, so the agent goes there and then waits.
  for (int w : waits) out.push_back({placed ? need : std::nextafter(need, INF), w, 0.0});
  if (out.size() > first) out.back().work = ca.dur;
}

// Does any non-move action on the plan need agent r at location fluent d?
inline bool needed_at(const Problem &pb, const std::vector<int> &plan, int r, int d) {
  for (int a : plan) {
    if (pb.move_dest(a, r) >= 0) continue;
    const auto &pre = pb.acts[a].pre;
    if (std::find(pre.begin(), pre.end(), d) != pre.end()) return true;
  }
  return false;
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

// Find the goal's uncertain search and split the plan around it. The search
// is the implied `found X` when the goal has one -- even with one attempt
// left, so the costing does not switch schemes when the second-last place
// fails -- else the probabilistic subgoal with the most attempts. After the
// search, an action that needs what the search reveals (picking the object
// up) happens wherever it succeeds, so only the other actions' stops (where
// to bring it) are on the route from there.
inline void split_search(Context &cx, Pass &P, int r, int g, int companion,
                         const std::vector<int> &plan, const std::vector<double> &need,
                         const std::vector<int> &move_dests, GoalPlan &tp) {
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
  std::vector<Visit> before, after;
  for (std::size_t i = 0; i < plan.size(); ++i) {
    const int a = plan[i];
    if (a == b || pb.move_dest(a, r) >= 0) continue;
    bool post = need[i] >= search_need;
    (post ? tp.other_post : tp.other_pre) += pb.acts[a].dur;
    if (post && anchored(a)) continue;
    action_stops(pb, a, r, need[i], tp.provided, post ? after : before);
  }
  // Waypoints and location goals, as in plan_goal.
  for (int d : move_dests) {
    if (!needed_at(pb, plan, r, d)) (P.cost[d] >= search_need ? after : before).push_back({P.cost[d], d, 0.0});
  }
  if (pb.loc_agent[g] == r) (P.cost[g] >= search_need ? after : before).push_back({P.cost[g], g, 0.0});
  order_visits(pb, P, before, tp.pre_locs, tp.pre_fb, tp.pre_work);
  order_visits(pb, P, after, tp.post_locs, tp.post_fb, tp.post_work);
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

// Agent r's plan for the goal facts `roots` (with the companion `found X` of
// the object it carries, if any), ready to start at `ready`, leaving what is
// in `provided` to other jobs. r < 0: no single agent (the team relaxation).
inline GoalPlan plan_goal(Context &cx, Pass &P, const std::vector<int> &roots, int companion,
                          int r, double ready, const std::vector<int> &provided = {}) {
  const Problem &pb = cx.pb;
  GoalPlan tp;
  for (int g : roots) {
    if (!P.reachable(g)) return tp;
  }
  if (companion >= 0 && !P.reachable(companion)) companion = -1;
  const int g = roots.front();
  if (r >= 0) tp.provided = provided;
  // Under expected search, a goal whose cheapest support is an uncertain
  // search (searching the target place, hoping the object is already there)
  // is planned through its deterministic achiever: wherever else the object
  // turns up, it must be brought over. Outcomes that achieve the goal in
  // place cost no delivery (goal_duration).
  std::vector<std::pair<int, int>> forced;  // (goal, its own support)
  if (cx.opts.expected_search && cx.opts.route_chaining && r >= 0) {
    for (int root : roots) {
      if (!P.uncertain(pb, root)) continue;
      int d = deterministic_achiever(pb, P, root);
      if (d >= 0) {
        forced.push_back({root, P.best[root]});
        P.best[root] = d;
      }
    }
  }
  struct Restore {
    Pass &P;
    std::vector<std::pair<int, int>> &forced;
    ~Restore() {
      for (const auto &[root, from] : forced) P.best[root] = from;
    }
  } restore{P, forced};
  cx.ex.next_stamp(pb);
  Extraction ex;
  if (!forced.empty()) {
    // The goals first, so that the search on a forced plan does not count as
    // achieving one in passing.
    for (int root : roots) cx.ex.extract(pb, cx.ts, P, {root}, ex, tp.provided);
    if (companion >= 0) cx.ex.extract(pb, cx.ts, P, {companion}, ex, tp.provided);
  } else {
    std::vector<int> all(roots);
    if (companion >= 0) all.push_back(companion);
    cx.ex.extract(pb, cx.ts, P, all, ex, tp.provided);
  }
  tp.ok = true;
  tp.goal = g;
  double goal_cost = 0.0;
  for (int root : roots) goal_cost = std::max(goal_cost, P.cost[root]);
  if (companion >= 0) goal_cost = std::max(goal_cost, P.cost[companion]);
  tp.critical = std::max(0.0, goal_cost - ready);
  const std::vector<int> &plan = ex.actions;
  tp.delta = ex.delta;
  tp.covers = ex.fluents;
  if (r < 0 || !cx.opts.route_chaining) {
    tp.other = ex.dur;  // no single agent to route, or routing disabled
    return tp;
  }
  // Stops the plan needs the agent at: the location preconditions of its
  // non-move actions (including where the agent already is -- the relaxation
  // never makes it walk back) or the provisions they wait for, plus move
  // destinations nothing else on the plan needs (waypoints, or location
  // goals). Each is needed when the earliest plan action requiring it could
  // start.
  std::vector<Visit> visits;
  std::vector<int> move_dests;
  const std::vector<double> need = need_times(pb, P, r, plan);
  for (std::size_t i = 0; i < plan.size(); ++i) {
    const int a = plan[i];
    int dest = pb.move_dest(a, r);
    if (dest >= 0) {
      move_dests.push_back(dest);
      continue;
    }
    tp.other += pb.acts[a].dur;
    action_stops(pb, a, r, need[i], tp.provided, visits);
  }
  for (int d : move_dests) {
    if (!needed_at(pb, plan, r, d)) visits.push_back({P.cost[d], d, 0.0});
  }
  for (int root : roots) {
    if (pb.loc_agent[root] == r) visits.push_back({P.cost[root], root, 0.0});
  }
  order_visits(pb, P, visits, tp.locs, tp.leg_fallback, tp.work);
  if (cx.opts.expected_search) split_search(cx, P, r, g, companion, plan, need, move_dests, tp);
  return tp;
}

// Work done at no stop.
inline double unattached(double total, const std::vector<double> &work) {
  for (double w : work) total -= w;
  return std::max(0.0, total);
}

// The last fixed stop of a route (where the agent ends up), or the most
// likely place of the last provision.
inline int last_location(const Problem &pb, int r, const std::vector<int> &stops,
                         const std::vector<const Provision *> &prov) {
  for (auto it = stops.rbegin(); it != stops.rend(); ++it) {
    if (*it >= 0) return *it;
    const Provision *pv = prov.empty() ? nullptr : prov[-1 - *it];
    if (pv && !pv->place.empty()) return pb.place_loc_of(r, pv->mode());
  }
  return -1;
}

// One goal's (or job's) duration on agent r from `start` at t_start, its
// retry deltas, and -- when it is an expected search -- the outcomes it is
// an expectation over: when the goal would be done (from t_start) for each
// place the search may first succeed, and the probability left over.
struct Duration {
  double serial = 0.0;
  double delta = 0.0;
  std::vector<Found> where;  // .t: when done (from t_start) if found there
  double residual = 0.0;
  double residual_done = 0.0;
  int end = -1;  // where the agent ends up (location fluent; -1: unknown)
};

// Duration of goal tp for agent r starting at `start` at time t_start, and
// the retry delta to add on top. `prov[k]` is where and when provided[k]
// holds (null: never, so not waited for).
inline Duration goal_duration(Context &cx, const Pass &P, int r, int start, double t_start,
                              const GoalPlan &tp,
                              const std::vector<const Provision *> &prov = {}) {
  const Problem &pb = cx.pb;
  Duration out;
  if (r < 0) {
    out.delta = tp.delta;
    out.serial = tp.other;
    return out;
  }
  out.end = last_location(pb, r, tp.locs, prov);
  auto whole = [&]() {
    Spot end;
    out.delta = tp.delta;
    out.serial = route(pb, r, Spot{start, nullptr}, t_start, tp.locs, tp.leg_fallback, tp.work,
                       prov, end) +
                 unattached(tp.other, tp.work);
    return out;
  };
  if (tp.search_f < 0) return whole();
  Spot pre_end;
  const double pre = route(pb, r, Spot{start, nullptr}, t_start, tp.pre_locs, tp.pre_fb,
                           tp.pre_work, prov, pre_end);
  const int pos = pre_end.pv ? pb.place_loc_of(r, pre_end.pv->mode()) : pre_end.loc;
  const double extra_pre = unattached(tp.other_pre, tp.pre_work);
  const double extra_post = unattached(tp.other_post, tp.post_work);
  // The rest of the goal from wherever the object turns up -- what needs the
  // object (picking it up) first, there; nothing if finding it there already
  // achieves the goal.
  auto rest = [&](int loc, bool achieves, double t) {
    if (achieves) return 0.0;
    Spot e;
    return extra_post + route(pb, r, Spot{loc >= 0 ? loc : pos, nullptr}, t + extra_post,
                              tp.post_locs, tp.post_fb, tp.post_work, prov, e);
  };
  double t_last = 0.0;
  double search = expected_search(pb, cx.ts, P, tp.search_f, tp.goal, r, pos, t_start + pre, rest,
                                  out.where, &t_last);
  if (!std::isfinite(search)) {
    out.where.clear();
    return whole();
  }
  double post = 0.0, found = 0.0;
  for (auto &w : out.where) {
    post += w.prob * w.rest;
    found += w.prob;
    w.t = w.t - t_start + w.rest + extra_pre;  // done, from t_start, if found here
  }
  out.delta = tp.delta_rest;
  out.serial = pre + search + post + extra_pre;
  out.residual = std::max(0.0, 1.0 - found);
  out.residual_done = t_last - t_start + extra_pre;
  return out;
}

// Lower bound on a goal's duration from the relaxed critical path. Not for an
// expected search, an expectation over outcomes some of which end sooner than
// the relaxed time to success; nor with provisions, whose times it ignores.
inline double bound(const GoalPlan &tp) {
  return (tp.search_f >= 0 || !tp.provided.empty()) ? 0.0 : tp.critical;
}

}  // namespace concurrent
}  // namespace railroad
