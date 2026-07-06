import argparse
from railroad.core import Fluent as F, State, get_action_by_name, Operator, Effect
from railroad.planner import MCTSPlanner
from railroad.environment.symbolic import SymbolicEnvironment
from railroad import operators

def main():
    objects_by_type = {
        "robot": {"robot1", "robot2"},
        "location": {"base1", "base2", "table", "counter", "fridge", "shelf", "toaster"},
        "object": {"bread1", "bread2", "knife1", "butter1", "butter2"}
    }

    initial_fluents = {
        F("at", "robot1", "base1"),
        F("free", "robot1"),
        F("at", "robot2", "base2"),
        F("free", "robot2"),
        
        F("at", "bread1", "shelf"),
        F("at", "bread2", "shelf"),
        F("at", "knife1", "counter"),
        F("at", "butter1", "fridge"),
        F("at", "butter2", "fridge"),
        
        F("available", "toaster"),
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

    # Restrict pick operation to items explicitly reserved for this robot's task
    pick_op.preconditions.append(F("reserved_by ?r ?obj"))

    reserve_op = Operator(
        name="reserve",
        parameters=[("?r", "robot"), ("?obj", "object")],
        preconditions=[
            F("free ?r"),
            ~F("reserved ?obj")
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=0.1, resulting_fluents={
                F("free ?r"), 
                F("reserved ?obj"), 
                F("reserved_by ?r ?obj")
            })
        ],
        extra_cost=0.0
    )

    unreserve_op = Operator(
        name="unreserve",
        parameters=[("?r", "robot"), ("?obj", "object")],
        preconditions=[
            F("free ?r"),
            F("reserved_by ?r ?obj")
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=0.1, resulting_fluents={
                F("free ?r"), 
                ~F("reserved ?obj"), 
                ~F("reserved_by ?r ?obj")
            })
        ],
        extra_cost=0.0
    )

    make_toast_op = Operator(
        name="make_toast",
        parameters=[("?robot", "robot"), ("?bread", "object")],
        preconditions=[
            F("at ?robot toaster"), 
            F("free ?robot"), 
            F("holding ?robot ?bread"),
            F("available toaster")
        ],
        effects=[
            Effect(time=0, resulting_fluents={
                ~F("free ?robot"), 
                ~F("available toaster")
            }),
            Effect(time=10.0, resulting_fluents={
                F("free ?robot"), 
                F("available toaster"), 
                F("toasted ?bread")
            }),
        ],
    )

    my_operators = [move_op, pick_op, place_op, no_op, make_toast_op, reserve_op, unreserve_op]

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=my_operators
    )

    # Individual goals per robot
    goals = {
        "robot1": F("toasted bread1"),
        "robot2": None # No task initially
    }

    max_iterations = 100
    print("Starting decentralized planners...")
    
    for i in range(max_iterations):
        # Inject task for robot2 at t >= 3.0
        if env.state.time >= 10.0 and goals["robot2"] is None:
            print(f"\n>>> [SYSTEM] Time is {env.state.time:05.1f} >= 10.0! Assigning Task to robot2: Toast bread2! <<<")
            goals["robot2"] = F("toasted bread2")
            
            # (Robot2 will now dynamically reserve the items it needs via reserve_op)

        # Check if all goals are met
        all_met = True
        for r, g in goals.items():
            if g is not None and not g.evaluate(env.state.fluents):
                all_met = False
        if all_met and goals["robot2"] is not None:
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
                # If planner fails (e.g. waiting for toaster), gracefully wait
                action_name = f"no_op {robot}"
                print(f"Time {env.state.time:05.1f} | Executing: {action_name} (Waiting for resource)")
            else:
                print(f"Time {env.state.time:05.1f} | Executing: {action_name}")

            action = get_action_by_name(all_actions, action_name)
            env.act(action)

if __name__ == '__main__':
    main()
