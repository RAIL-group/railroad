import pytest
import environments

def get_args():
    args = lambda: None
    args.current_seed = 1024
    args.resolution = 0.05
    return args

def construct_random_goal_fluents(env):
    all_objects = env.all_objects


def test_procthor_environment_initialization():
    args = get_args()
    env = environments.procthor.ProcTHOREnvironment(args)
    print(env.objects_at_locations)

def test_goal_fluents():
    args = get_args()
    env = environments.procthor.ProcTHOREnvironment(args)
