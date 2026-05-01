import re
import time
import numpy as np

from railroad import operators
from railroad.core import Fluent as F, get_action_by_name
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad._bindings import State


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
# SEARCH_TIME = 5.0
PICK_TIME = 5.0
PLACE_TIME = 5.0


def main(
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    objects_of_interest = ["Cereal", "Notebook", "Clock", "Mug", "Pillow"]
    # print(objects_of_interest)

    # Define initial fluents
    initial_fluents = {
        F("at", "robot1", "start_loc"),
        F("free", "robot1"),
        F("at Notebook table"),
        F("at Clock table"),
        F("at Cereal pantry"),
        F("at Pillow bed"),
        F("at Mug cabinet"),
    }

    # Define goal: all items at their proper locations
    goal = (
        F("at Mug table")
    )

    # Objects by type
    objects_by_type = {
        "robot": {"robot1", "robot2"},
        "location": set(LOCATIONS.keys()),
        "object": set(objects_of_interest),
    }

    # Distance-based move time function
    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        distance = float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))
        return distance / ROBOT_VELOCITY

    # Create operators - move uses distance-based time
    move_op = operators.construct_move_operator_blocking(move_time)
    # search_op = operators.construct_search_operator(object_find_prob, SEARCH_TIME)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initialize symbolic environment with initial state
    initial_state = State(0.0, initial_fluents, [])
    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, pick_op, place_op],
        true_object_locations=OBJECTS_AT_LOCATIONS,
    )

    # Planning loop
    max_iterations = 60
    total_planning_time = 0.0
    total_iterations = 0

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found", "searched"])
    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal achieved![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)

            start_time = time.perf_counter()
            max_mcts_iters = 100

            action_name = mcts(env.state, goal, max_iterations=max_mcts_iters, c=300, max_depth=20)

            ################ Record timing ####################
            step_time = time.perf_counter() - start_time
            total_planning_time += step_time
            
            # Attempt to extract iterations and expanded nodes
            try:
                tree_trace = mcts.get_trace_from_last_mcts_tree()
                
                expanded_nodes = 0
                iterations = max_mcts_iters 
                
                # The repository's MCTS implementation returns a string trace
                # where the root node is formatted like: D:0|=visits=1000, 
                if isinstance(tree_trace, str):
                    match = re.search(r"D:0\|=visits=(\d+)", tree_trace)
                    if match:
                        iterations = int(match.group(1))
                        
                total_iterations += iterations
            except Exception:
                # Silently catch if parsing fails
                pass
            #####################################################

            if action_name == "NONE":
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            action = get_action_by_name(all_actions, action_name)
            env.act(action)
            dashboard.update(mcts, action_name)

    location_coords = {name: (float(c[0]), float(c[1])) for name, c in LOCATIONS.items()}
    dashboard.show_plots(
        save_plot=save_plot, show_plot=show_plot, save_video=save_video,
        video_fps=video_fps, video_dpi=video_dpi,
        location_coords=location_coords,
    )

        # Print baseline metrics
    print("\n" + "="*40)
    print("BASELINE METRICS:")
    print("="*40)
    print(f"Total time taken (planning): {total_planning_time:.3f} seconds")
    try:
        print(f"Total iterations executed:   {total_iterations}")
    except NameError:
        pass
    print("="*40)


if __name__ == "__main__":
    main()
