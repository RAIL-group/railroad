#pragma once

namespace mrppddl {

const double HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 0.0;
const double HEURISTIC_MULTIPLIER = 2;
const double SUCCESS_REWARD = 0.0;
const double ALL_ROBOTS_WAITING_PENALTY = 10.0;
const int NUM_EXTRA_VISITS_PROB = 0;

// Prioritize exploration during planning
const double SEARCH_PHASE_RATIO = 0.05;  // [0,1] Initial search phase as % of total iterations.
const double PROB_EXTRA_EXPLORE = 0.0;  // [0,1] Prob randomly decide to prioritize search

}
