import re
import time
import numpy as np

from railroad import operators
from railroad.core import Fluent as F, get_action_by_name
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner

from railroad.experimental.llm_heuristic import HeuristicContextBuilder, LLMActionReranker

# Define locations with coordinates (for move cost calculation)
LOCATIONS = {
    "start_loc": np.array([-5, -5]),
    "table": np.array([0, 0]),
    "pantry": np.array([10, 0]),
    "bed": np.array([0, 12]),
    "cabinet": np.array([10, 12]),
}

# Define where objects actually are (ground truth)
OBJECTS_AT_LOCATIONS = {
    "start_loc": set(),
    "table": {"Notebook", "Clock"},
    "pantry": {"Cereal"},
    "bed": {"Pillow"},
    "cabinet": {"Mug"},
}

# Fixed operator times for symbolic planning
ROBOT_VELOCITY = 1.0
PICK_TIME = 5.0
PLACE_TIME = 5.0
OPEN_TIME = 3.0


def main():
    objects_of_interest = ["Cereal", "Notebook", "Clock", "Mug", "Pillow"]

    # Define initial fluents
    initial_fluents = {
        F("at", "robot1", "start_loc"),
        F("free", "robot1"),
        F("at Notebook table"),
        F("at Clock table"),
        F("at Cereal pantry"),
        F("at Pillow bed"),
        F("at Mug cabinet"),
        F("has-door cabinet cabinet_door"),
        F("closed cabinet_door"),
        F("blocking-access cabinet"),
    }

    goal = F("at Mug table") # & F("at Cereal cabinet")

    objects_by_type = {
        "robot": {"robot1"},
        "location": set(LOCATIONS.keys()),
        "object": set(objects_of_interest),
        "container-door": {"cabinet_door"}
    }

    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        distance = float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))
        return distance / ROBOT_VELOCITY

    move_op = operators.construct_move_operator_blocking(move_time)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
    open_op = operators.construct_open_container_door_operator_blocking(OPEN_TIME)

    initial_state = initial_state = type('State', (), {})() # Dummy for import
    from railroad._bindings import State
    initial_state = State(0.0, initial_fluents, [])

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, pick_op, place_op, open_op],
        true_object_locations=OBJECTS_AT_LOCATIONS,
    )

    max_iterations = 60
    total_planning_time = 0.0
    total_llm_time = 0.0
    total_iterations = 0

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "closed", "blocking"])

    context_builder = HeuristicContextBuilder(fluent_filter=fluent_filter)
    llm_reranker = LLMActionReranker(context_builder)

    print("Starting planning loop with LLM Action Reranker...")
    for iteration in range(max_iterations):
        if goal.evaluate(env.state.fluents):
            print("\n[SUCCESS] Goal achieved!")
            break

        all_actions = env.get_actions()

        # Step 1: LLM Reranking (Currently mocked)
        llm_start_time = time.perf_counter()
        reranked_actions = llm_reranker.rerank(env.state, goal, all_actions, all_actions)
        
        # If we have a real LLM ranking them, restrict MCTS branching by taking the top 15
        if llm_reranker.client:
            reduced_actions = reranked_actions[:15]
        else:
            reduced_actions = reranked_actions
        
        llm_time = time.perf_counter() - llm_start_time
        total_llm_time += llm_time

        # Step 2: MCTS Planning
        mcts = MCTSPlanner(reduced_actions)

        mcts_start_time = time.perf_counter()
        max_mcts_iters = 100

        action_name = mcts(env.state, goal, max_iterations=max_mcts_iters, c=300, max_depth=20)
        
        mcts_time = time.perf_counter() - mcts_start_time
        total_planning_time += mcts_time
        
        try:
            tree_trace = mcts.get_trace_from_last_mcts_tree()
            iterations = max_mcts_iters 
            if isinstance(tree_trace, str):
                match = re.search(r"D:0\|=visits=(\d+)", tree_trace)
                if match:
                    iterations = int(match.group(1))
            total_iterations += iterations
        except Exception:
            pass

        print(f"Step {iteration}: Selected '{action_name}' | MCTS time: {mcts_time:.3f}s | LLM query time: {llm_time:.3f}s")

        if action_name == "NONE":
            print("\n[FAILED] No more actions available.")
            break

        action = get_action_by_name(all_actions, action_name)
        env.act(action)

    print("\n" + "="*40)
    print("LLM HEURISTIC BENCHMARK METRICS:")
    print("="*40)
    print(f"Total iterations to first solution: {iteration + 1}")
    print(f"Total time taken (MCTS):            {total_planning_time:.3f} seconds")
    print(f"Total time taken (LLM/Context):     {total_llm_time:.3f} seconds")
    print(f"Total expanded nodes (MCTS visits): {total_iterations}")
    print("="*40)

if __name__ == "__main__":
    main()
