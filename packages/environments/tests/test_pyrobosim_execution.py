from railroad.environment.pyrobosim import PyRoboSimEnvironment, PyRoboSimScene, get_default_pyrobosim_world_file_path
from railroad._bindings import Fluent, GroundedEffect, State
import railroad.operators as operators
from railroad.planner import MCTSPlanner

F = Fluent

def test_pyrobosim_execution():
    world_file = get_default_pyrobosim_world_file_path()
    scene = PyRoboSimScene(world_file, show_plot=False)

    move_cost_fn = scene.get_move_cost_fn()

    ops = [
        operators.construct_move_operator_blocking(move_cost_fn),
        operators.construct_pick_operator_blocking(1.0),
        operators.construct_place_operator_blocking(1.0),
    ]

    # Create the real PyRoboSim environment with only one robot in state
    robot_names = {"robot1"}
    initial_fluents = {
        F("at", "robot1", "robot1_loc"),
        F("free", "robot1"),
        F("revealed", "robot1_loc"),
    }
    state = State(0.0, initial_fluents)
    env = PyRoboSimEnvironment(
        scene=scene,
        state=state,
        objects_by_type={
            "robot": robot_names,
            "location": set(scene.locations.keys()),
            "object": scene.objects,
        },
        operators=ops,
    )

    try:
        # We'll use banana0 which is at table0, and move it to my_desk
        planner = MCTSPlanner(actions=env.get_actions())
        goal = F("at banana0 my_desk")

        print("Initial state:", env.state)

        # In test_world.yaml, banana0 is at table0. Let's make sure it's known.
        env.apply_effect(GroundedEffect(0, {F("found banana0"), F("at banana0 table0")}))

        for i in range(15):
            if env.is_goal_reached([goal]):
                print("Goal reached!")
                break

            print(f"Iteration {i}, State: {env.state}")
            current_actions = env.get_actions()
            action_name = planner(env.state, goal, max_iterations=10000, c=300, max_depth=20, heuristic_multiplier=3)
            if not action_name or action_name == "NONE":
                print("No plan found!")
                break

            # Find the Action object for the selected action name
            from railroad.core import get_action_by_name
            action = get_action_by_name(current_actions, action_name)

            print(f"Executing: {action.name}")
            env.act(action)

        assert env.is_goal_reached([goal])
    finally:
        scene.close()

if __name__ == "__main__":
    test_pyrobosim_execution()
