#pragma once

// Concurrency-aware heuristic ("h_conc"): the team's expected remaining time.
//
// The FF heuristic (heuristic.hpp) estimates the cost of a *sequential* plan:
// it sums the durations of every relaxed-plan action whichever robot runs it,
// and its relaxed initial state treats every in-flight effect as already done.
// With several robots the objective (time until the goal holds) is a makespan,
// and MCTS leaves are reached while other robots are mid-action. This
// heuristic instead estimates how long the *team* needs to finish, in
// expectation over the outcomes of uncertain actions.
//
// Core (general to any railroad domain whose agents are the arguments of
// `free`):
//
//   1. Timed relaxed state -- in-flight effects become available when they
//      fire; in-flight uncertain outcomes are pending achievers that keep
//      their probability.                  [heuristic_concurrent_relaxation.hpp]
//   2. Probability-aware relaxation -- per agent, each fluent's cost and the
//      probability rho its support succeeds; achievers ranked by cost / rho.
//                                          [heuristic_concurrent_relaxation.hpp]
//   3. Task plans -- each goal task's relaxed plan on each agent and its
//      duration.                           [heuristic_concurrent_tasks.hpp]
//   4. List schedule -- tasks go to the agent that would finish them
//      earliest; the value is computed from the completion times C_g:
//          lambda_add * sum_g C_g + lambda_ms * max_g C_g
//      minimised over goal DNF branches. The makespan alone leaves an agent
//      whose task is off the critical path without a gradient (in ProcTHOR,
//      plans 10-20% longer with 2-3 robots); the sum gives every task one.
//      With n tasks open the value falls at lambda_ms + lambda_add * n per
//      unit of time, which is what MCTS charges under flowtime_objective
//      (makespan plus flowtime); charged elapsed time only, as by default,
//      the value is pessimistic by that difference, as if multiplied by
//      (n + 1) / 2 at the default weights.
//                                          [heuristic_concurrent_schedule.hpp]
//
// Refinements that apply only where the domain has the structure they need:
//
//   - Route chaining: an agent with location fluents (a mutex group its moves
//     switch between) visits the places its task needs in order, each leg
//     costed from the previous one.        [tasks; compiled in problem]
//   - Expected search: a subgoal with several probabilistic attempts is
//     costed as an expected route over them, attempts already in flight
//     included, and the rest of the task from wherever it succeeds.
//                                          [heuristic_concurrent_search.hpp]
//   - Attempt grouping: achievers that consume the same precondition are one
//     attempt, not independent retries.   [problem; relaxation, search]
//   - Task ordering: on one agent, a task whose plan destroys a fact (true
//     now) that another task relies on waits for it.   [schedule]
//
// Object-search convention (the core's name-keyed `at`/`found` machinery): a
// goal `at X L` also needs `found X`, and the two are planned as one task.
//
// The estimate aims to be consistent with its own one-step lookahead: for an
// action the estimate's plan starts with, h(s) = dt + sum_o p_o h(o) over its
// outcomes, so starting an action, or an outcome arriving, does not move the
// value MCTS sees. railroad.consistency measures this on any problem.
//
// With one agent and no in-flight effects it is a sequential, route-aware
// h_ff. Problem-dependent structure is compiled once per search
// (heuristic_concurrent_problem.hpp), so an evaluation touches flat arrays
// only.

#include "railroad/heuristic_concurrent_schedule.hpp"
#include "railroad/state.hpp"

#include <limits>
#include <memory>
#include <unordered_map>

namespace railroad {

class ConcurrentHeuristic {
 public:
  static constexpr double INF = concurrent::INF;

  ConcurrentHeuristic(const std::vector<Action> &actions, const GoalBase *goal,
                      ConcurrentHeuristicOptions opts = {})
      : opts_(opts), goal_(goal), pb_(actions, goal, opts_) {
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
    // The unrestricted pass is always needed (h_add, reachability, fallback);
    // the per-agent passes feed the schedule.
    for (auto &P : passes_) P.run(pb_, ts_);

    double best = INF;
    for (const auto &branch : pb_.branches) {
      ConcurrentHeuristicBreakdown bd;
      double v = evaluate_branch(branch, out ? &bd : nullptr);
      if (v < best) {
        best = v;
        if (out) *out = std::move(bd);
      }
    }
    if (out) out->value = best;
    return best;
  }

  // Goal tasks still open in s, counted as the value's sum term counts them:
  // the unsatisfied literals of the goal branch with the fewest, with a
  // `found X` that goes with an open `at X L` (one joint task) not counted
  // again.
  int open_tasks(const State &s) const {
    if (!goal_) return 0;
    const auto &fl = s.fluents();
    auto holds = [&fl](const Fluent &f) {
      return f.is_negated() ? fl.count(f.invert()) == 0 : fl.count(f) > 0;
    };
    auto count = [&](const auto &literals) {
      int n = 0;
      for (const auto &f : literals) {
        if (holds(f)) continue;
        bool joint = false;
        if (opts_.joint_found && !f.is_negated() && f.name() == "found" && f.args().size() == 1) {
          for (const auto &g : literals) {
            joint = joint || (!g.is_negated() && g.name() == "at" && !g.args().empty() &&
                              g.args()[0] == f.args()[0] && !holds(g));
          }
        }
        n += !joint;
      }
      return n;
    };
    constexpr std::size_t kMaxBranches = 1024;
    if (goal_->dnf_branch_count() == 0 || goal_->dnf_branch_count() > kMaxBranches) {
      return count(goal_->get_all_literals());
    }
    int best = std::numeric_limits<int>::max();
    for (const auto &br : goal_->get_dnf_branches()) best = std::min(best, count(br));
    return best;
  }

  // What MCTS charges, beyond elapsed time, for a step from `from` to `to`
  // (the flowtime objective; 0 when it is off).
  double step_cost(const State &from, const State &to) const {
    if (!opts_.flowtime_objective) return 0.0;
    const double dt = to.time() - from.time();
    if (dt <= 0.0) return 0.0;
    return (opts_.lambda_ms - 1.0 + opts_.lambda_add * open_tasks(from)) * dt;
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
  concurrent::Scheduler scheduler_;
  std::unordered_map<std::size_t, double> memo_;

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

  double evaluate_branch(const std::vector<int> &branch, ConcurrentHeuristicBreakdown *bd) {
    using concurrent::Pass;
    Pass &U = passes_[0];
    std::vector<int> goals;
    goals.reserve(branch.size() * 2);
    for (int g : branch) {
      if (g < 0 || !U.reachable(g)) return INF;
      goals.push_back(g);
    }
    if (opts_.at_implies_found) {
      std::size_t n = goals.size();
      for (std::size_t i = 0; i < n; ++i) {
        int fg = pb_.found_of[goals[i]];
        if (fg >= 0 && U.reachable(fg) && std::find(goals.begin(), goals.end(), fg) == goals.end()) {
          goals.push_back(fg);
        }
      }
    }

    // h_add and the shared (unrestricted) relaxed plan.
    double h_add = 0.0;
    for (int g : goals) h_add += U.cost[g];
    extractor_.next_stamp(pb_);
    concurrent::Extraction ux;
    extractor_.extract(pb_, opts_, ts_, U, goals, ux);

    if (bd) {
      for (int f : ux.fluents) {
        if (U.uncertain(pb_, f)) bd->deltas.push_back({pb_.str(f), U.delta(pb_, ts_, f)});
      }
      for (int a = 0; a < static_cast<int>(pb_.acts.size()); ++a) {
        if (extractor_.on_plan(a)) bd->plan.push_back({pb_.acts[a].src->name(), pb_.acts[a].dur});
      }
      bd->h_add = h_add;
      bd->delta = ux.delta;
      bd->h_ff = ux.dur;
    }
    concurrent::Context cx{pb_, opts_, ts_, extractor_};
    double completion_sum = 0.0;
    double makespan = scheduler_.schedule(cx, passes_, goals, bd, completion_sum);
    if (bd) {
      bd->makespan = makespan;
      bd->completion_sum = completion_sum;
    }
    double additive = opts_.sum_completion ? completion_sum : h_add + ux.delta;
    return opts_.lambda_add * additive + opts_.lambda_ms * makespan;
  }
};

}  // namespace railroad
