#pragma once

namespace railroad {

// Heuristic value substituted when the relaxed heuristic reports the goal
// unreachable (h = inf). NOTE: at 0, dead-end states look goal-adjacent to
// MCTS (h below every reachable state), so the planner does not avoid --
// and can even seek out -- irreversible mistakes. Raising it perturbs
// multi-robot search-ordering ties (see test_mcts_search_picks_more_likely_
// location), so it stays 0 for now; dead-end-aware planning is future work.
const double HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 0.0;
const double HEURISTIC_MULTIPLIER = 5;
const double SUCCESS_REWARD = 0.0;
const double ALL_ROBOTS_WAITING_PENALTY = 10.0;
const int NUM_EXTRA_VISITS_PROB = 0;

}
