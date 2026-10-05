#pragma once

// Core of the concurrent heuristic: list-scheduling goal tasks onto agents.
//
// Each goal fluent not yet true is a task (a goal `at X L` together with the
// `found X` it implies, under the object-search convention). In priority
// order, each task goes to the agent that would finish it earliest from that
// agent's ready time and position (earliest finish time), and the schedule
// yields every task's completion time. Longest-first (LPT) suits the
// makespan, shortest-first (SPT) the sum of completion times; both orders are
// tried and the one the value prefers is kept.
//
// Task ordering (order_conflicts): tasks are planned independently, so one
// task's relaxed plan may destroy a fact another relies on -- fetching a
// second object "frees" the hand by setting down the one being delivered,
// whose own delivery then still finds it in hand. On one agent such a task
// waits for the tasks it would spoil: a threat between causal links from the
// state, resolved by ordering.

#include "railroad/heuristic_concurrent_tasks.hpp"

namespace railroad {
namespace concurrent {

class Scheduler {
 public:
  // passes[0] is the unrestricted pass; passes[r + 1] agent r's, when the
  // schedule is per agent. Returns the makespan; completion_sum receives the
  // sum of completion times.
  double schedule(Context &cx, std::vector<Pass> &passes, const std::vector<int> &goals,
                  ConcurrentHeuristicBreakdown *bd, double &completion_sum) {
    const Problem &pb = cx.pb;
    const ConcurrentHeuristicOptions &opts = cx.opts;
    completion_sum = 0.0;
    Pass &U = passes[0];
    const bool per_agent = opts.agent_aware && pb.num_agents() > 1;
    const std::size_t n_agents = per_agent ? pb.num_agents() : 1;
    // With a single agent its route can still be chained.
    const int solo = (!per_agent && pb.num_agents() == 1) ? 0 : -1;
    auto pass_of = [&](std::size_t r) -> Pass & { return per_agent ? passes[r + 1] : U; };
    auto agent_of = [&](std::size_t r) { return per_agent ? static_cast<int>(r) : solo; };

    std::vector<double> ready(n_agents, 0.0);
    std::vector<int> start_loc(n_agents, -1);
    for (std::size_t r = 0; r < n_agents; ++r) {
      Pass &P = pass_of(r);
      ready[r] = per_agent ? P.cost[pb.agent_free[r]] : team_ready(pb, U);
      if (agent_of(r) >= 0) start_loc[r] = agent_location(pb, P, agent_of(r));
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
      if (!opts.joint_found) return -1;
      int fx = pb.found_of[g];
      if (fx < 0 || std::find(goals.begin(), goals.end(), fx) == goals.end()) return -1;
      if (U.best[fx] == Pass::AVAIL && U.cost[fx] <= 1e-9) return -1;
      return fx;
    };
    std::vector<int> paired;
    for (int g : goals) {
      int fx = companion_of(g);
      if (fx >= 0) paired.push_back(fx);
    }
    std::vector<Task> tasks;
    for (int g : goals) {
      if (U.best[g] == Pass::AVAIL && U.cost[g] <= 1e-9) continue;
      if (std::find(paired.begin(), paired.end(), g) != paired.end()) continue;
      Task t;
      t.fluent = g;
      t.finish_fixed = (U.best[g] == Pass::AVAIL) ? U.cost[g] : -1.0;
      t.key = -1.0;
      if (t.finish_fixed < 0.0) {
        t.plans.resize(n_agents);
        double m = INF;
        for (std::size_t r = 0; r < n_agents; ++r) {
          if (!std::isfinite(ready[r])) continue;
          Pass &P = pass_of(r);
          t.plans[r] = plan_task(cx, P, g, companion_of(g), agent_of(r), ready[r]);
          if (!t.plans[r].ok) continue;
          const TaskPlan &tp = t.plans[r];
          double dl = 0.0;
          double serial = task_serial(cx, P, agent_of(r), start_loc[r], ready[r], tp, dl);
          double load = std::max(serial, bound(tp)) + dl;
          m = std::min(m, load);
          if (bd) bd->loads.push_back({pb.str(g), static_cast<int>(r), serial, tp.critical, dl});
        }
        t.key = m;
      }
      tasks.push_back(std::move(t));
    }

    std::vector<double> finish;
    std::vector<int> end_loc;
    std::vector<int> n_assigned;
    // The agent that would finish task t earliest (earliest finish time).
    auto earliest = [&](const Task &t, int &best_r) {
      best_r = -1;
      double best_f = INF;
      for (std::size_t r = 0; r < n_agents; ++r) {
        if (!t.plans[r].ok || !std::isfinite(finish[r])) continue;
        const TaskPlan &tp = t.plans[r];
        double dl = 0.0;
        double serial = task_serial(cx, pass_of(r), agent_of(r), end_loc[r], finish[r], tp, dl);
        // The relaxed critical path only bounds an agent's first task.
        double load = (n_assigned[r] == 0 ? std::max(serial, bound(tp)) : serial) + dl;
        double f = finish[r] + load;
        if (f < best_f) { best_f = f; best_r = static_cast<int>(r); }
      }
      return best_f;
    };
    // Plan a destroys a fact plan b relies on from the state, and adds
    // nothing b still needs (if it did, as delivering the object in hand
    // frees the hand for the next fetch, the fact only fed a step that a
    // would make unnecessary).
    auto has = [](const std::vector<int> &v, int x) {
      return std::find(v.begin(), v.end(), x) != v.end();
    };
    auto clobbers = [&has](const TaskPlan &a, const TaskPlan &b) {
      bool hit = false;
      for (int c : a.destroys) hit = hit || has(b.relies, c);
      if (!hit) return false;
      for (int f : a.adds) {
        if (has(b.covers, f)) return false;
      }
      return true;
    };
    std::vector<char> taken;
    auto needs_agent = [&](const Task &t) {
      return t.finish_fixed < 0.0 && cover_[t.fluent] != cover_gen_;
    };
    // Would doing task ti on agent r now spoil a task still to be scheduled?
    auto spoils = [&](std::size_t ti, int r) {
      for (std::size_t tj = 0; tj < tasks.size(); ++tj) {
        if (tj == ti || taken[tj] || !needs_agent(tasks[tj])) continue;
        if (tasks[tj].plans[r].ok && clobbers(tasks[ti].plans[r], tasks[tj].plans[r])) return true;
      }
      return false;
    };

    // List-schedule the tasks in the given priority order: each goes to the
    // agent that would finish it earliest, deferring a task whose best agent
    // would spoil another (unless every remaining task would).
    struct Done {
      int task;  // index into tasks
      double at;
    };
    std::vector<Done> done;
    auto run = [&](const std::vector<std::size_t> &order, bool record) {
      finish = ready;
      end_loc = start_loc;
      n_assigned.assign(n_agents, 0);
      taken.assign(tasks.size(), 0);
      done.clear();
      // Fluents achieved along an assigned task's plan (e.g. `found X` on the
      // way to `at X L`) need no task of their own.
      if (cover_.size() != pb.num_fluents()) cover_.assign(pb.num_fluents(), 0);
      if (++cover_gen_ == 0) {
        std::fill(cover_.begin(), cover_.end(), 0);
        cover_gen_ = 1;
      }
      for (std::size_t step = 0; step < tasks.size(); ++step) {
        std::size_t ti = tasks.size(), first = tasks.size();
        for (std::size_t k : order) {
          if (taken[k]) continue;
          if (first == tasks.size()) first = k;
          if (!opts.order_conflicts || !needs_agent(tasks[k])) { ti = k; break; }
          int r;
          earliest(tasks[k], r);
          if (r < 0 || !spoils(k, r)) { ti = k; break; }
        }
        if (ti == tasks.size()) ti = first;
        taken[ti] = 1;
        Task &t = tasks[ti];
        double done_at;
        if (t.finish_fixed >= 0.0) {
          done_at = t.finish_fixed;
        } else if (cover_[t.fluent] == cover_gen_) {
          continue;  // achieved along another task's plan
        } else {
          int best_r = -1;
          double best_f = earliest(t, best_r);
          if (best_r < 0) {
            // No single agent can do it: fall back to the team relaxation.
            cx.ex.next_stamp(pb);
            Extraction ex;
            cx.ex.extract(pb, opts, cx.ts, U, {t.fluent}, ex);
            done_at = U.cost[t.fluent] + ex.delta;
          } else {
            const TaskPlan &tp = t.plans[best_r];
            finish[best_r] = best_f;
            n_assigned[best_r] += 1;
            if (!tp.locs.empty()) end_loc[best_r] = tp.locs.back();
            done_at = best_f;
            for (int f : tp.covers) cover_[f] = cover_gen_;
            if (record) {
              bd->assignment.push_back({pb.str(t.fluent),
                                        per_agent ? pb.agent_names[best_r] : "team"});
            }
          }
        }
        done.push_back({static_cast<int>(ti), done_at});
      }
      double ms = 0.0, sum = 0.0;
      for (const auto &dn : done) {
        ms = std::max(ms, dn.at);
        sum += dn.at;
        if (record) bd->goal_finish.push_back({pb.str(tasks[dn.task].fluent), dn.at});
      }
      return std::make_pair(ms, sum);
    };

    std::vector<std::size_t> lpt(tasks.size());
    std::iota(lpt.begin(), lpt.end(), 0);
    std::stable_sort(lpt.begin(), lpt.end(),
                     [&tasks](std::size_t a, std::size_t b) { return tasks[a].key > tasks[b].key; });
    std::vector<std::size_t> best_order = lpt;
    auto [makespan, sum] = run(lpt, false);
    if (tasks.size() > 1) {
      std::vector<std::size_t> spt = lpt;
      std::stable_sort(spt.begin(), spt.end(),
                       [&tasks](std::size_t a, std::size_t b) { return tasks[a].key < tasks[b].key; });
      if (spt != lpt) {
        auto [ms2, sum2] = run(spt, false);
        if (opts.lambda_add * sum2 + opts.lambda_ms * ms2 <
            opts.lambda_add * sum + opts.lambda_ms * makespan - 1e-9) {
          makespan = ms2;
          sum = sum2;
          best_order = spt;
        }
      }
    }
    if (bd) run(best_order, true);
    completion_sum = sum;
    return makespan;
  }

 private:
  std::vector<uint32_t> cover_;
  uint32_t cover_gen_ = 0;

  // Serial "team" agent: ready when the first agent is.
  static double team_ready(const Problem &pb, const Pass &U) {
    double r = INF;
    for (int f : pb.agent_free) r = std::min(r, U.cost[f]);
    return std::isfinite(r) ? r : 0.0;
  }
};

}  // namespace concurrent
}  // namespace railroad
