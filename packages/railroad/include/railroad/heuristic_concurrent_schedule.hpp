#pragma once

// Core of the concurrent heuristic: list-scheduling goals onto agents.
//
// Each goal fact not yet true is one unit (with the `found X` an `at X L`
// implies). In priority order, each goes to the agent that would finish it
// earliest from that agent's ready time and position. Longest-first (LPT)
// suits the makespan and shortest-first (SPT) the sum of completion times;
// both are tried and the one the value prefers is kept.

#include "railroad/heuristic_concurrent_plans.hpp"

#include <numeric>

namespace railroad {
namespace concurrent {

class Scheduler {
 public:
  // passes[0] is the unrestricted pass; passes[r + 1] agent r's, when the
  // schedule is per agent. Returns the makespan; completion_sum receives the
  // sum of completion times.
  double schedule(Context &cx, const std::vector<Pass> &passes, const std::vector<int> &goals,
                  ConcurrentHeuristicBreakdown *bd, double &completion_sum) {
    const Problem &pb = cx.pb;
    const ConcurrentHeuristicOptions &opts = cx.opts;
    completion_sum = 0.0;
    const Pass &U = passes[0];
    const bool per_agent = opts.agent_aware && pb.num_agents() > 1;
    const std::size_t n_agents = per_agent ? pb.num_agents() : 1;
    // With a single agent its route can still be chained.
    const int solo = (!per_agent && pb.num_agents() == 1) ? 0 : -1;
    auto pass_of = [&](std::size_t r) -> const Pass & { return per_agent ? passes[r + 1] : U; };
    auto agent_of = [&](std::size_t r) { return per_agent ? static_cast<int>(r) : solo; };
    // When agent r, at location `at` from time t, would finish goal plan tp.
    auto finish_at = [&](std::size_t r, const GoalPlan &tp, int at, double t) {
      return goal_finish(cx, pass_of(r), agent_of(r), at, t, tp);
    };

    std::vector<double> ready(n_agents, 0.0);
    std::vector<int> start_loc(n_agents, -1);
    for (std::size_t r = 0; r < n_agents; ++r) {
      const Pass &P = pass_of(r);
      ready[r] = per_agent ? P.cost[pb.agent_free[r]] : team_ready(pb, U);
      if (agent_of(r) >= 0) start_loc[r] = agent_location(pb, P, agent_of(r));
    }

    // Open goals: goal facts not already true now.
    struct OpenGoal {
      int fluent;
      double finish_fixed;  // >= 0: needs no agent (in-flight)
      std::vector<GoalPlan> plans;  // per agent
      double key;  // priority: its shortest time on any agent
    };
    // `at X L` and the `found X` it implies are one job: whoever brings X to
    // L must find it first. Planned apart, `at X L` could be "achieved" by
    // searching L in the hope X is there while another agent finds X.
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
    std::vector<OpenGoal> open;
    for (int g : goals) {
      if (U.best[g] == Pass::AVAIL && U.cost[g] <= 1e-9) continue;
      if (std::find(paired.begin(), paired.end(), g) != paired.end()) continue;
      OpenGoal t;
      t.fluent = g;
      t.finish_fixed = (U.best[g] == Pass::AVAIL) ? U.cost[g] : -1.0;
      t.key = -1.0;
      if (t.finish_fixed < 0.0) {
        t.plans.resize(n_agents);
        double m = INF;
        for (std::size_t r = 0; r < n_agents; ++r) {
          if (!std::isfinite(ready[r])) continue;
          t.plans[r] = plan_goal(cx, pass_of(r), g, companion_of(g), agent_of(r));
          if (!t.plans[r].ok) continue;
          m = std::min(m, finish_at(r, t.plans[r], start_loc[r], ready[r]) - ready[r]);
        }
        t.key = m;
      }
      open.push_back(std::move(t));
    }

    std::vector<double> finish;
    std::vector<int> end_loc;
    // The agent that would finish goal t earliest.
    auto best_agent = [&](const OpenGoal &t, int &best_r) {
      best_r = -1;
      double best_f = INF;
      for (std::size_t r = 0; r < n_agents; ++r) {
        if (!t.plans[r].ok || !std::isfinite(finish[r])) continue;
        double f = finish_at(r, t.plans[r], end_loc[r], finish[r]);
        if (f < best_f) { best_f = f; best_r = static_cast<int>(r); }
      }
      return best_f;
    };
    // One list-scheduling pass in the given priority order: the makespan and
    // the sum of completion times.
    auto run = [&](const std::vector<std::size_t> &order, bool record) {
      finish = ready;
      end_loc = start_loc;
      double ms = 0.0, sum = 0.0;
      // Fluents achieved along an assigned goal's plan (e.g. `found X` on the
      // way to `at X L`) need no goal of their own.
      if (cover_.size() != pb.num_fluents()) cover_.assign(pb.num_fluents(), 0);
      if (++cover_gen_ == 0) {
        std::fill(cover_.begin(), cover_.end(), 0);
        cover_gen_ = 1;
      }
      for (std::size_t ti : order) {
        const OpenGoal &t = open[ti];
        double done_at;
        if (t.finish_fixed >= 0.0) {
          done_at = t.finish_fixed;
        } else if (cover_[t.fluent] == cover_gen_) {
          continue;  // achieved along another goal's plan
        } else {
          int best_r = -1;
          double best_f = best_agent(t, best_r);
          if (best_r < 0) {
            // No single agent can do it: the team relaxation's time.
            done_at = U.cost[t.fluent];
          } else {
            const GoalPlan &tp = t.plans[best_r];
            finish[best_r] = best_f;
            if (!tp.route.locs.empty()) end_loc[best_r] = tp.route.locs.back();
            done_at = best_f;
            for (int f : tp.covers) cover_[f] = cover_gen_;
            if (record) {
              bd->assignment.push_back({pb.str(t.fluent),
                                        per_agent ? pb.agent_names[best_r] : "team"});
            }
          }
        }
        ms = std::max(ms, done_at);
        sum += done_at;
        if (record) bd->goal_finish.push_back({pb.str(t.fluent), done_at});
      }
      return std::make_pair(ms, sum);
    };

    std::vector<std::size_t> lpt(open.size());
    std::iota(lpt.begin(), lpt.end(), 0);
    std::stable_sort(lpt.begin(), lpt.end(),
                     [&open](std::size_t a, std::size_t b) { return open[a].key > open[b].key; });
    std::vector<std::size_t> best_order = lpt;
    auto [makespan, sum] = run(lpt, false);
    if (open.size() > 1) {
      std::vector<std::size_t> spt = lpt;
      std::stable_sort(spt.begin(), spt.end(),
                       [&open](std::size_t a, std::size_t b) { return open[a].key < open[b].key; });
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
