#pragma once

// Concurrency-aware heuristic ("h_conc"): the team's expected remaining time.
//
// The FF heuristic (heuristic.hpp) estimates a *sequential* plan: it sums
// every relaxed-plan action whichever robot runs it, and treats in-flight
// effects as already done. With several robots the objective is a makespan,
// and MCTS leaves are reached while other robots are mid-action. This
// heuristic instead estimates how long the *team* needs to finish, in
// expectation over the outcomes of uncertain actions.
//
// Core (any domain whose agents are the arguments of `free`):
//
//   1. Timed relaxed state: in-flight effects become available when they
//      fire; in-flight uncertain outcomes keep their probability.
//                                          [heuristic_concurrent_relaxation.hpp]
//   2. Probability-aware relaxation: per agent, each fluent's cost and the
//      probability rho its support succeeds; achievers ranked by cost / rho.
//                                          [heuristic_concurrent_relaxation.hpp]
//   3. Jobs: the team's relaxed plan for the goal's facts (of a DNF branch)
//      is cut into jobs. Actions linked through a fluent naming an agent and
//      something else (`holding r X`) are one agent's work; any other link
//      (`at pot L`, a door opened) makes one job wait for another. Each goal
//      fact (`at mug L`, `boiled egg`) is achieved by one job; work no goal
//      needs directly is a provider job.  [heuristic_concurrent_schedule.hpp]
//   4. Job plans: each job gets a relaxed plan and a duration on each agent,
//      stopping at what other jobs provide and waiting for it. Jobs are
//      planned independently, so one job's plan may use what another's
//      relies on.                         [heuristic_concurrent_plans.hpp]
//   5. List schedule: once its providers are, each job goes to the agent
//      that would finish it earliest, giving the goal facts' completion
//      times C_g and the value
//          lambda_add * sum_g C_g + lambda_ms * max_g C_g
//      of the best DNF branch. The objective is the makespan; the sum is a
//      shaping term that gives an agent off the critical path a gradient.
//      The price: with n goals open the value falls at
//      lambda_ms + lambda_add * n per unit of time while MCTS charges 1.
//                                          [heuristic_concurrent_schedule.hpp]
//   6. Hedging: agents done with their part of the best branch before it is
//      finished go on to the second best, and the goal is done when either
//      is. The value is lowered by lambda_ms * (E[MS_1] - E[min(MS_1,
//      MS_2)]), from the two schedules' completion-time distributions.
//
// Refinements, where the domain has the structure they need:
//
//   - Route chaining: an agent with location fluents visits the places its
//     goal needs in order, each leg costed from the previous one.
//                                          [heuristic_concurrent_plans.hpp]
//   - Expected search: a subgoal with several probabilistic attempts is
//     costed as an expected route over them, in-flight attempts included,
//     and the rest of the goal from wherever it succeeds. A job that only
//     searches for an object leaves it wherever it turns up; an object a goal
//     delivers is used where it is delivered.
//                                          [heuristic_concurrent_search.hpp]
//
// Object-search convention: a goal `at X L` also needs `found X`, and the two
// are planned together.
//
// The estimate aims to agree with its own one-step lookahead: for the action
// its plan starts with, h(s) = dt + sum_o p_o h(o), so starting an action or
// an outcome arriving does not move the value MCTS sees.
//
// With one agent, no in-flight effects and nothing to share out, it reduces
// to a sequential, route-aware h_ff. Everything that depends only on the actions and the goal
// is compiled once per search (heuristic_concurrent_problem.hpp).

#include "railroad/heuristic_concurrent_schedule.hpp"
#include "railroad/state.hpp"

#include <memory>
#include <numeric>
#include <unordered_map>

namespace railroad {

class ConcurrentHeuristic {
 public:
  static constexpr double INF = concurrent::INF;

  ConcurrentHeuristic(const std::vector<Action> &actions, const GoalBase *goal,
                      ConcurrentHeuristicOptions opts = {})
      : opts_(opts), goal_(goal), pb_(actions, goal) {
    // One unrestricted pass, plus one per agent when there is a choice of
    // agents to schedule onto.
    const bool per_agent = opts_.agent_aware && pb_.num_agents() > 1;
    passes_.resize(1 + (per_agent ? pb_.num_agents() : 0));
    for (std::size_t p = 0; p < passes_.size(); ++p) {
      passes_[p].agent = (p == 0) ? -1 : static_cast<int>(p - 1);
    }
  }

  ConcurrentHeuristic(const ConcurrentHeuristic &) = delete;
  ConcurrentHeuristic &operator=(const ConcurrentHeuristic &) = delete;

  // Memoized evaluation (keyed on fluents + upcoming effects relative to now).
  double operator()(const State &s) {
    std::size_t key = state_key(s);
    auto it = memo_.find(key);
    if (it != memo_.end()) return it->second;
    double v = evaluate(s, nullptr);
    memo_.emplace(key, v);
    return v;
  }

  double evaluate(const State &s, ConcurrentHeuristicBreakdown *out) {
    if (!goal_) return 0.0;
    if (goal_->get_type() == GoalType::TRUE_GOAL) return 0.0;
    if (goal_->get_type() == GoalType::FALSE_GOAL || pb_.branches.empty()) return INF;

    ts_.load(pb_, opts_, s);
    // The unrestricted pass gives reachability and the team fallback; the
    // per-agent passes feed the schedule.
    for (auto &P : passes_) P.run(pb_, ts_);

    concurrent::Context cx{pb_, opts_, ts_, extractor_};
    std::vector<Option> options;
    for (const auto &branch : pb_.branches) {
      std::vector<int> goals;
      if (!branch_goals(branch, goals)) continue;
      Option o;
      o.schedule.build(cx, passes_, goals);
      o.run = o.schedule.best_run();
      o.value = o.run.value(opts_);
      if (o.schedule.threatened()) {
        // An object both used in place and delivered: also use it first.
        Option alt;
        alt.schedule.build(cx, passes_, goals, /*in_place=*/true);
        alt.run = alt.schedule.best_run();
        alt.value = alt.run.value(opts_);
        if (alt.value < o.value) o = std::move(alt);
      }
      options.push_back(std::move(o));
    }
    if (options.empty()) {
      if (out) out->value = INF;
      return INF;
    }
    std::size_t best_i = 0;
    for (std::size_t i = 1; i < options.size(); ++i) {
      if (options[i].value < options[best_i].value) best_i = i;
    }
    double best = options[best_i].value;
    const concurrent::BranchSchedule::Run *primary = &options[best_i].run;
    concurrent::BranchSchedule::Run second;
    double gain = 0.0;
    if (options.size() > 1 && options[best_i].schedule.per_agent()) {
      gain = hedge(options, best_i, second);
      best -= opts_.lambda_ms * gain;
    }
    if (out) {
      *out = ConcurrentHeuristicBreakdown{};
      out->value = best;
      out->makespan = primary->makespan;
      out->completion_sum = primary->sum;
      record(*primary, out->goal_finish, out->assignment);
      for (const auto &[fy, r] : primary->searches) {
        out->searches.push_back({pb_.str(fy), agent_name(r)});
      }
      if (gain > 0.0) {
        std::vector<std::pair<std::string, double>> unused;
        record(second, unused, out->hedge);
        out->hedge_gain = gain;
      }
    }
    return best;
  }

  std::size_t num_agents() const { return pb_.num_agents(); }
  std::size_t memo_size() const { return memo_.size(); }

 private:
  ConcurrentHeuristicOptions opts_;
  const GoalBase *goal_;
  concurrent::Problem pb_;
  concurrent::TimedState ts_;
  std::vector<concurrent::Pass> passes_;
  concurrent::Extractor extractor_;
  std::unordered_map<std::size_t, double> memo_;

  // One way to reach the goal (a DNF branch), scheduled.
  struct Option {
    concurrent::BranchSchedule schedule;
    concurrent::BranchSchedule::Run run;
    double value = INF;
  };

  static std::size_t state_key(const State &s) {
    std::size_t h_fluents = 0;
    for (const auto &f : s.fluents()) {
      std::size_t h = f.hash();
      hash_combine(h, 0);
      h_fluents ^= h;
    }
    std::size_t h_up = 0;
    for (const auto &[t, e] : s.upcoming_effects()) {
      std::size_t h = e->hash();
      hash_combine(h, std::hash<double>{}(t - s.time()));
      h_up ^= h;
    }
    hash_combine(h_fluents, h_up);
    return h_fluents;
  }

  // The branch's goals, plus the `found X` each `at X L` implies (false if
  // one is unreachable).
  bool branch_goals(const std::vector<int> &branch, std::vector<int> &goals) const {
    const concurrent::Pass &U = passes_[0];
    goals.reserve(branch.size() * 2);
    for (int g : branch) {
      if (g < 0 || !U.reachable(g)) return false;
      goals.push_back(g);
    }
    for (std::size_t i = 0, n = goals.size(); i < n; ++i) {
      int fg = pb_.found_of[goals[i]];
      if (fg >= 0 && U.reachable(fg) && std::find(goals.begin(), goals.end(), fg) == goals.end()) {
        goals.push_back(fg);
      }
    }
    return true;
  }

  // Agents done with their part of the best way before it is finished go on
  // to a second way. Returns how much sooner, in expectation, the goal is then
  // done (the best second way's), and that way's schedule.
  double hedge(std::vector<Option> &options, std::size_t best_i,
               concurrent::BranchSchedule::Run &second) const {
    const concurrent::BranchSchedule::Run &first = options[best_i].run;
    std::vector<char> skip(first.free_at.size(), 0);
    bool spare = false;
    for (std::size_t r = 0; r < skip.size(); ++r) {
      skip[r] = !std::isfinite(first.free_at[r]) || first.free_at[r] >= first.makespan - 1e-9;
      spare = spare || !skip[r];
    }
    auto da = first.dists();
    if (!spare || da.empty() || !first.complete) return 0.0;
    const double alone = concurrent::expected_max(da);
    double gain = 0.0;
    for (std::size_t k = 0; k < options.size(); ++k) {
      if (k == best_i || !std::isfinite(options[k].value)) continue;
      auto b = options[k].schedule.best_run(&first, skip);
      auto db = b.dists();
      if (!b.complete || db.empty()) continue;
      double g = alone - concurrent::expected_min_of_max(da, db);
      if (g > gain + 1e-9) {
        gain = g;
        second = std::move(b);
      }
    }
    return gain;
  }

  std::string agent_name(int r) const {
    if (r < 0) return "team";
    const bool per_agent = opts_.agent_aware && pb_.num_agents() > 1;
    return per_agent ? pb_.agent_names[r] : "team";
  }

  void record(const concurrent::BranchSchedule::Run &run,
              std::vector<std::pair<std::string, double>> &finish,
              std::vector<std::pair<std::string, std::string>> &assignment) const {
    for (std::size_t i = 0; i < run.done_goal.size(); ++i) {
      std::string g = pb_.str(run.done_goal[i]);
      finish.push_back({g, run.done_at[i]});
      if (run.done_agent[i] >= 0) assignment.push_back({g, agent_name(run.done_agent[i])});
    }
  }
};

}  // namespace railroad
