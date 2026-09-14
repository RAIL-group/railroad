"""
A module for keeping track of experiment related user-defined constants.
"""
from .utilities import calibrate_beta_parameter

## railroad heuristic related constants
LAMBDA_ADD = 0
LAMBDA_MAX = 0
LAMBDA_FF = 1

## debug related constants
AP_DEBUG = False
ACTION_PROB_DEBUG = False
SEARCH_DEBUG = False

## learned function for expected value of interrupting task distribution
MODEL_NAME = "best_model_two_room_model_linux.pt"

## benchmark/experiment settings
# benchmark run settings
EXPERIMENT_REPEATS = 5
TIMEOUT = 1200

# number of tasks in the task distribution
# current: 2-room -> 16; 1-room -> 11
NUM_TASKS = 16

# seeds
# current: 2-room -> 64; 1-room -> 201
PROCTHOR_SEED = 64
# current: 2-room -> 2; 1-room -> 19
OBJ_PLACEMENT_SEED = 2
# filter out non-task-relevant objects
FILTER_OBJECTS = True

AUGMENT_TASK = True
RETRY_WITH_SUBGOALS = True

EXPECTED_TIME_NEXT_ARRIVAL = [
    calibrate_beta_parameter(0, 5), # No interruptions
    # 5% of tasks from the training dataset take longer to complete
    calibrate_beta_parameter(0.5, 295.368),
    # 10% of tasks from the training dataset take longer to complete
    calibrate_beta_parameter(0.5, 257.341),
    # 25% of tasks from the training dataset take longer to complete
    calibrate_beta_parameter(0.5, 217.495),
    # 50% of tasks from the training dataset take longer to complete
    calibrate_beta_parameter(0.5, 180.088),
    # 75% of tasks from the training dataset take longer to complete
    calibrate_beta_parameter(0.5, 136.660),
    # 95% of tasks from the training dataset take longer to complete
    calibrate_beta_parameter(0.5, 76.998),
]

## interuption heuristic related constants
# interruption heuristic weights (ff-heuristic_weight, EV_weight)
# NOTE: keep EV_weight fixed at 1
# INT_H_WEIGHTS = (0.35, 1)
INT_H_WEIGHTS = (1, 1)

# discount factor for augment experiment heuristic function
# AUGMENT_DISCOUNT_FACTOR = 0.99
AUGMENT_DISCOUNT_FACTOR = 0.99

# heuristic multiplier (larger -> more greedy search)
H_MULTIPLIER = 2

## task failure cost
# current: 2-room (seed = 64) -> 375; 1-room (seed = 201) -> 500
PLANNER_FAILURE_COST = 500
