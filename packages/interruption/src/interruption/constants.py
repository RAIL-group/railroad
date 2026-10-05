"""
A module for keeping track of experiment related user-defined constants.
"""

## railroad heuristic related constants
LAMBDA_ADD = 0
LAMBDA_MAX = 0
LAMBDA_FF = 1

## debug related constants
AP_DEBUG = False
ACTION_PROB_DEBUG = False
SEARCH_DEBUG = False

## learned function for expected value of interrupting task distribution
MODEL_NAME = "best_model_one_room_multi_scene_lr=0.001.pt"

## benchmark/experiment settings
# benchmark run settings
EXPERIMENT_REPEATS = 30
TIMEOUT = 3600
# filter out non-task-relevant objects
FILTER_OBJECTS = True
# current task gets augmented by arriving task
AUGMENT_TASK = True
# retry upon solver failing to find a plan
RETRY_WITH_SUBGOALS = True

# number of tasks in the task distribution
# current: 2-room -> 16; 1-room -> 11
NUM_TASKS = 11

# scene filters for multi-scene data generation (see filter_procthor_scenes;
# "num_scenes" only bounds how many matching scenes get remapped, it is not a
# filter criterion). Shared by data generation and the experiments that must
# rebuild the same task distribution (get_task_distribution_from_remap_dir).
NUM_ROOMS_FILTER = {1}
ONE_ROOM_FILTER = {
    "num_scenes": 22, "num_pickupable_objects": 11, "num_valid_locations": 6,
    "room_types": {"Kitchen"},
}
TWO_ROOM_FILTER = {"num_scenes": 10, "num_pickupable_objects": 20, "num_valid_locations": 10}

# evaluate on the scenes (and task distribution) of a multi-scene data-generation
# run: the name of its remapped_scenes/<hash>/ directory, which datagen prints
# as "Remapped scenes for this run". None -> the plain PROCTHOR_SEED scene.
# When set, PROCTHOR_SEED must be one of that directory's scenes.
REMAPPED_SCENES_HASH: str | None = "767821370226"

# seeds
# current: 2-room -> 64; 1-room -> 201
PROCTHOR_SEED = 113
# current: 2-room -> 2; 1-room -> 19
OBJ_PLACEMENT_SEED = None

# task-arrival rates, each given as the fraction of the training tasks that take
# longer to complete than the median time to the next arrival (0 -> no
# interruptions). Turned into the average time between arrivals by
# utilities.get_expected_time_next_arrival, from the task-cost quantiles that
# scripts/generate_task_cost_distribution.py --write-quantiles writes for
# REMAPPED_SCENES_HASH.
ARRIVAL_EXCEEDANCE_FRACTIONS = [0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95]
# experiment seed for each arrival rate above (paired by position)
INTERRUPTION_SEEDS = [140, 42, 240, 57, 1096, 4065, 720]

## interuption heuristic related constants
# interruption heuristic weights (ff-heuristic_weight, EV_weight)
# NOTE: keep EV_weight fixed at 1
# INT_H_WEIGHTS = (0.35, 1)
INT_H_WEIGHTS = (1, 1)

# discount factor for augment experiment heuristic function
# AUGMENT_DISCOUNT_FACTOR = 0.99
AUGMENT_DISCOUNT_FACTOR = 1

# heuristic multiplier (larger -> more greedy search)
H_MULTIPLIER = 1

## task failure cost
# current: 2-room (seed = 64) -> 375; 1-room (seed = 201) -> 500
PLANNER_FAILURE_COST = 500

# A.P. related constants
NUM_AUGMENTED_TASK_SAMPLES = 200 #8
AP_SEED = 12
