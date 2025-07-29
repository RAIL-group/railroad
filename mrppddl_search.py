from time import time
from mrppddl.core import Fluent
from mrppddl.core import State
# from mrppddl.planner import mcts
from mrppddl.planner import MCTSPlanner
from mrppddl.helper import construct_move_operator, construct_search_operator


# Define objects
objects_by_type = {
    "robot": ["r1", "r2"],
    "location": ["start", "roomA", "roomB"],
    "object": ["cup", "bowl"]
}

def object_search_prob(robot, search_loc, obj):
    if search_loc == 'roomA':
        return 0.6
    else:
        return 0.4

# Ground actions
search_actions = construct_search_operator(object_search_prob, 5.0, 3).instantiate(objects_by_type)
# Initial state
initial_state = State(
    time=0,
    fluents={
        Fluent("at r1 start"),
        Fluent("at r2 start"),
        Fluent("free r1"),
        Fluent("free r2"),
    }
)

goal_fluents = {Fluent("found cup"), Fluent("found bowl")}
all_actions = search_actions
mcts = MCTSPlanner(all_actions)
for _ in range(100):
    stime = time()
    action = mcts(initial_state, goal_fluents, max_iterations=10000, c=10)
    print(f"MCTS: {{Initial State: Best Action}}, time={time()-stime:.3f}")
    print(action)
