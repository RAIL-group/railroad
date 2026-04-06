"""ProcTHOR multi-object location swap task.

Demonstrates using ProcTHOR environment with MCTS planning
for a single robot multi-object location swap goal.
"""
import time
import random
from pathlib import Path

from railroad import operators
from railroad.core import Fluent as F, get_action_by_name
from railroad.dashboard import PlannerDashboard
from railroad.planner import MCTSPlanner
from railroad._bindings import State


def sample_swap_objects_and_locations(scene, seed: int | None = None):
    if seed is not None:
        random.seed(seed)
    
    # We need two different locations that have at least one object each
    valid_locations = [loc for loc, objs in scene.object_locations.items() if objs]
    if len(valid_locations) < 2:
        raise ValueError("Scene does not have enough objects in different locations to swap.")
        
    loc1, loc2 = random.sample(valid_locations, 2)
    obj1 = random.choice(list(scene.object_locations[loc1]))
    obj2 = random.choice(list(scene.object_locations[loc2]))
    print(valid_locations)
    for loc in valid_locations:
        print(f"Location: {loc}, Objects: {scene.object_locations[loc]}")
    return (obj1, loc1), (obj2, loc2)


def main(
    seed: int | None = None,
    estimate_object_find_prob: bool = False,    # leaving as provision for future expansion with learning
    nn_model_path: str | None = None,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    try:
        from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
        from railroad.environment.procthor.learning.utils import get_default_fcnn_model_path
    except ImportError as e:
        print(f"Error: {e}\nInstall ProcTHOR dependencies with: pip install railroad[procthor]")
        return

    # Configuration
    robot_names = ["robot1"]
    
    # We fix the seed just to ensure reproducibility if none is provided
    scene_seed = seed if seed is not None else 5050
    print(f"Loading ProcTHOR scene (seed={scene_seed})...")
    scene = ProcTHORScene(seed=scene_seed)

    (obj1, loc1), (obj2, loc2) = sample_swap_objects_and_locations(scene, seed=scene_seed)
    target_objects = [obj1, obj2]
    print(f"Swap Task: Move {obj1} from {loc1} to {loc2}, and {obj2} from {loc2} to {loc1}")

    # Build operators
    move_cost_fn = scene.get_move_cost_fn()

    if estimate_object_find_prob:
        model_path = Path(nn_model_path) if nn_model_path is not None else get_default_fcnn_model_path()
        if not model_path.exists():
            raise FileNotFoundError("Trained neural network model not found.")
        object_find_prob_fn = scene.get_object_find_prob_fn(
            nn_model_path=str(model_path), objects_to_find=target_objects
        )
    else:
        def object_find_prob_fn(robot: str, location: str, obj: str) -> float:
            for loc, objs in scene.object_locations.items():
                if obj in objs:
                    return 0.8 if loc == location else 0.1
            return 0.1

    move_op = operators.construct_move_operator_blocking(move_cost_fn)
    search_op = operators.construct_search_operator(object_find_prob_fn, 10.0)
    pick_op = operators.construct_pick_operator_blocking(10.0)
    place_op = operators.construct_place_operator_blocking(10.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initial state
    initial_fluents = {F("revealed start_loc")}
    for robot in robot_names:
        initial_fluents.add(F(f"at {robot} start_loc"))
        initial_fluents.add(F(f"free {robot}"))
    initial_state = State(0.0, initial_fluents, [])

    # Goal: swap locations and both must be found
    goal = (
        F(f"at {obj1} {loc2}") & F(f"at {obj2} {loc1}")
        # & F(f"found {obj1}")
        # & F(f"found {obj2}")
    )

    # Create environment
    env = ProcTHOREnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": set(robot_names),
            "location": set(scene.locations.keys()),
            "object": set(target_objects),
        },
        operators=[no_op, pick_op, place_op, move_op, search_op],
    )

    # Planning loop metrics
    max_steps = 100
    total_planning_time = 0.0
    total_expanded_nodes = 0
    total_iterations = 0

    print("Starting planner on swap task...")
    
    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found", "searched"])
        
    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for step in range(max_steps):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Swap Goal reached![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            
            start_time = time.perf_counter()
            max_mcts_iters = 1000
            
            action_name = mcts(
                env.state,
                goal,
                max_iterations=max_mcts_iters,
                c=300,
                max_depth=20,
                heuristic_multiplier=2,
            )
            
            # Record timing
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
                    import re
                    match = re.search(r"D:0\|=visits=(\d+)", tree_trace)
                    if match:
                        iterations = int(match.group(1))
                        # In MCTS, one iteration typically expands one node
                        expanded_nodes = iterations
                        
                total_expanded_nodes += expanded_nodes
                total_iterations += iterations
            except Exception:
                # Silently catch if parsing fails
                pass

            if action_name == "NONE":
                dashboard.console.print("No more actions available.")
                break

            action = get_action_by_name(all_actions, action_name)
            env.act(action)
            dashboard.update(mcts, action_name)

    dashboard.show_plots(
        save_plot=save_plot, show_plot=show_plot, save_video=save_video,
        video_fps=video_fps, video_dpi=video_dpi,
    )

    # Print baseline metrics
    print("\n" + "="*40)
    print("BASELINE METRICS:")
    print("="*40)
    print(f"Total time taken (planning): {total_planning_time:.3f} seconds")
    try:
        print(f"Total iterations executed:   {total_iterations}")
        print(f"Total nodes expanded:        {total_expanded_nodes}")
    except NameError:
        pass
    print("="*40)


if __name__ == "__main__":
    random.seed(24)
    main()
