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
//   3. Goal plans: a "goal" is one fact of the goal (of its best DNF branch),
//      e.g. `at mug L`. Each gets a relaxed plan on each agent, which the
//      agent finishes after the plan's actions but not before the relaxation
//      could achieve the goal -- the one place waiting for in-flight effects
//      is charged. Goals are planned independently, so one goal's plan may
//      use what another's relies on.      [heuristic_concurrent_plans.hpp]
//   4. List schedule: each goal goes to the agent that would finish it
//      earliest, giving completion times C_g and the value
//          lambda_add * sum_g C_g + lambda_ms * max_g C_g
//      minimised over DNF branches. The objective is the makespan; the sum
//      is a shaping term that gives an agent off the critical path a
//      gradient. The price: with n goals open the value falls at
//      lambda_ms + lambda_add * n per unit of time while MCTS charges 1.
//                                          [heuristic_concurrent_schedule.hpp]
//
// Refinements, where the domain has the structure they need:
//
//   - Route chaining: an agent with location fluents visits the places its
//     goal needs in order, each leg costed from the previous one.
//                                          [heuristic_concurrent_plans.hpp]
//   - Expected search: a subgoal with several probabilistic attempts is
//     costed as an expected route over them, in-flight attempts included,
//     and the rest of the goal from wherever it succeeds. A goal whose
//     cheapest support is itself such an attempt (searching its target place
//     in the hope the object is there) is planned through its deterministic
//     achiever, so that bringing the object from elsewhere is costed.
//                                          [heuristic_concurrent_search.hpp]
//
// Probability thus enters twice: the ranking decides which attempt a plan
// goes through, and the expected search what the attempts cost.
//
// Object-search convention: a goal `at X L` also needs `found X`, and the two
// are planned together.
//
// The estimate aims to agree with its own one-step lookahead: for the action
// its plan starts with, h(s) = dt + sum_o p_o h(o), so starting an action or
// an outcome arriving does not move the value MCTS sees.
//
// With one agent and no in-flight effects it reduces to a sequential,
// route-aware h_ff. Everything that depends only on the actions and the goal
// is compiled once per search (heuristic_concurrent_problem.hpp).

#include "railroad/heuristic_concurrent_schedule.hpp"
#include "railroad/state.hpp"

#include <memory>
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
    const concurrent::Pass &U = passes_[0];
    // The branch's goals, plus the `found X` each `at X L` implies.
    std::vector<int> goals;
    goals.reserve(branch.size() * 2);
    for (int g : branch) {
      if (g < 0 || !U.reachable(g)) return INF;
      goals.push_back(g);
    }
    for (std::size_t i = 0, n = goals.size(); i < n; ++i) {
      int fg = pb_.found_of[goals[i]];
      if (fg >= 0 && U.reachable(fg) && std::find(goals.begin(), goals.end(), fg) == goals.end()) {
        goals.push_back(fg);
      }
    }
    concurrent::Context cx{pb_, opts_, ts_, extractor_};
    double completion_sum = 0.0;
    double makespan = scheduler_.schedule(cx, passes_, goals, bd, completion_sum);
    if (bd) {
      bd->makespan = makespan;
      bd->completion_sum = completion_sum;
    }
    return opts_.lambda_add * completion_sum + opts_.lambda_ms * makespan;
  }
};

}  // namespace railroad
