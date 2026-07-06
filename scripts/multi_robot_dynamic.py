import argparse
from railroad.core import Fluent as F, State, get_action_by_name, Operator, Effect
from railroad.planner import MCTSPlanner
from railroad.environment.symbolic import SymbolicEnvironment
from railroad import operators

def main():
    objects_by_type = {
        "robot": {"robot1", "robot2", "robot3"},
        "location": {"base1", "base2", "base3", "pickup", "dropoff"},
        "object": {"box1", "box2", "box3"}
    }

    initial_fluents = {
        F("at", "robot1", "base1"),
        F("free", "robot1"),
        F("at", "robot2", "base2"),
        F("free", "robot2"),
        F("at", "robot3", "base3"),
        F("free", "robot3"),
        
        F("at", "box1", "pickup"),
        F("at", "box2", "pickup"),
        F("at", "box3", "pickup"),
    }
    
    for loc in objects_by_type["location"]:
        initial_fluents.add(F("revealed", loc))

    initial_state = State(0.0, initial_fluents)

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        return 5.0
        
    move_op = operators.construct_move_operator_blocking(move_time=move_time_fn)
    pick_op = operators.construct_pick_operator_blocking(pick_time=1.0)
    place_op = operators.construct_place_operator_blocking(place_time=1.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=50.0)

    my_operators = [move_op, pick_op, place_op, no_op]

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=my_operators
    )

    # Individual goals per robot
    goals = {
        "robot1": F("at box1 dropoff"),
        "robot2": None, # No task initially
        "robot3": None  # No task initially
    }

    max_iterations = 200
    print("Starting decentralized planners...")
    
    for i in range(max_iterations):
        # Inject task for robot2 at t >= 20.0
        if env.state.time >= 20.0 and goals["robot2"] is None:
            print(f"\n>>> [SYSTEM] Time is {env.state.time:05.1f} >= 20.0! Assigning Task to robot2: Move box2 to dropoff! <<<")
            goals["robot2"] = F("at box2 dropoff")
            
        # Inject task for robot3 at t >= 25.0
        if env.state.time >= 25.0 and goals["robot3"] is None:
            print(f"\n>>> [SYSTEM] Time is {env.state.time:05.1f} >= 25.0! Assigning Task to robot3: Move box3 to dropoff! <<<")
            goals["robot3"] = F("at box3 dropoff")

        # Check if all goals are met
        all_met = True
        for r, g in goals.items():
            if g is not None and not g.evaluate(env.state.fluents):
                all_met = False
        if all_met and goals["robot2"] is not None and goals["robot3"] is not None:
            print("\n[SUCCESS] All Robot Goals Reached!")
            break

        all_actions = env.get_actions()
        
        # Find which robots are currently free
        free_robots = [r for r in objects_by_type["robot"] if F("free", r) in env.state.fluents]
        
        if not free_robots:
            print("\n[ERROR] No free robots. Deadlock?")
            break

        for robot in free_robots:
            # Double check if the robot is STILL free (in case an earlier env.act() advanced time)
            if not F("free", robot) in env.state.fluents:
                continue

            robot_goal = goals[robot]
            
            # If robot has no task or finished its task, it idles
            if robot_goal is None or robot_goal.evaluate(env.state.fluents):
                action_name = f"no_op {robot}"
                action = get_action_by_name(all_actions, action_name)
                print(f"Time {env.state.time:05.1f} | Executing: {action_name} (Idling)")
                env.act(action)
                continue
                
            # Filter actions so this robot's planner ONLY sees its own actions!
            robot_actions = [a for a in all_actions if a.name.split()[1] == robot]
            
            # Run MCTS specifically for this robot
            mcts = MCTSPlanner(robot_actions)
            action_name = mcts(
                env.state, 
                robot_goal, 
                max_iterations=5000, 
                c=300, 
                max_depth=20, 
                heuristic_multiplier=3
            )

            if action_name == 'NONE':
                # If planner fails, gracefully wait
                action_name = f"no_op {robot}"
                print(f"Time {env.state.time:05.1f} | Executing: {action_name} (Waiting/No Plan)")
            else:
                print(f"Time {env.state.time:05.1f} | Executing: {action_name}")

            action = get_action_by_name(all_actions, action_name)
            env.act(action)

if __name__ == '__main__':
    main()
