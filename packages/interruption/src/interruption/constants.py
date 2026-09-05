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
MODEL_NAME = "best_model_experiment12_val.pt"

## benchmark settings
EXPERIMENT_REPEATS = 3
AUGMENT_TASK = True
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
INT_H_WEIGHTS = (0.9, 1)

# discount factor for augment experiment heuristic function
AUGMENT_DISCOUNT_FACTOR = 0.99

# heuristic multiplier (larger -> more greedy search)
H_MULTIPLIER = 2
