#pragma once

namespace railroad {

const double HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 0.0;
const double HEURISTIC_MULTIPLIER = 5;
const double SUCCESS_REWARD = 0.0;
const double ALL_ROBOTS_WAITING_PENALTY = 10.0;
const int NUM_EXTRA_VISITS_PROB = 0;

// Prioritize exploration during planning
const double SEARCH_PHASE_RATIO = 0.00;  // [0,1] Initial search phase as % of total iterations.
const double PROB_EXTRA_EXPLORE = 0.0;  // [0,1] Prob randomly decide to prioritize search

// Progressive widening for high branching-factor decision states in MCTS.
const bool USE_PROGRESSIVE_WIDENING = false;
const double PROGRESSIVE_WIDENING_K = 1.0;
const double PROGRESSIVE_WIDENING_ALPHA = 0.5;

// Goal-shaping terms for MCTS reward.
// Reward states that satisfy more goal structure and penalize regressions from
// the best progress reached along the current path.
const double LANDMARK_PROGRESS_REWARD = 250.0;
const double GOAL_REGRESSION_PENALTY = 500.0;

// Multiplier for probabilistic backward-extraction delta in FF heuristic.
const double PROBABILISTIC_DELTA_MULTIPLIER = 4.0;

}
