"""Find and Move Couch Task.

This example demonstrates:
1. OR goals - satisfying one of several possible conditions
2. Multi-robot coordination with search and transport
3. Complex goal expressions using the fluent operator syntax

The goal is to move either a Remote OR Plate to the den, AND either a Cookie
OR Couch to the den. This shows how OR goals allow flexibility in achieving
objectives.
"""

import numpy as np

from railroad.core import Fluent as F, State, get_action_by_name, ff_heuristic
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad import operators
from railroad.environment import EnvironmentInterface, SimpleEnvironment


# Define locations with coordinates
LOCATIONS = {
    "living_room": np.array([0, 0]),
    "kitchen": np.array([10, 0]),
    "bedroom": np.array([0, 12]),
    "office": np.array([10, 12]),
    "den": np.array([15, 5]),
}

# Define where objects actually are (ground truth)
OBJECTS_AT_LOCATIONS = {
    "living_room": {"object": {"Remote"}},
    "kitchen": {"object": {"Cookie", "Plate"}},
    "bedroom": {"object": set()},
    "office": {"object": {"Couch"}},
    "den": {"object": set()},
}


def main() -> None:
    """Run the find-and-move-couch example."""
    # Initialize environment
    robot_locations = {"robot1": "living_room", "robot2": "living_room"}
    env = SimpleEnvironment(LOCATIONS, OBJECTS_AT_LOCATIONS, robot_locations=robot_locations)

    # Define the objects we're looking for
    objects_of_interest = ["Remote", "Cookie", "Plate", "Couch"]

    # Define initial state - robots start with some knowledge
    initial_state = State(
        time=0,
        fluents={
            F("free robot1"),
            F("free robot2"),
            F("at robot1 living_room"),
            F("at robot2 living_room"),
            # Living room is revealed (searched), so we know Remote is there
            F("revealed living_room"),
            F("at Remote living_room"),
            F("found Remote"),
            # Den is revealed but empty
            F("revealed den"),
        },
    )

    # Define goal using OR expressions:
    # (Remote at den OR Plate at den) AND (Cookie at den OR Couch at den)
    # This allows flexibility - the planner can choose which objects to move
    goal = (F("at Remote den") | F("at Plate den")) & (F("at Cookie den") | F("at Couch den"))

    # Objects by type
    objects_by_type = {
        "robot": ["robot1", "robot2"],
        "location": list(LOCATIONS.keys()),
        "object": objects_of_interest,
    }

    # Create operators
    move_time_fn = env.get_skills_time_fn(skill_name="move")
    search_time = env.get_skills_time_fn(skill_name="search")
    pick_time = env.get_skills_time_fn(skill_name="pick")
    place_time = env.get_skills_time_fn(skill_name="place")

    # Higher find probability if object is actually at the location
    def object_find_prob(robot: str, loc: str, obj: str) -> float:
        objects_here = OBJECTS_AT_LOCATIONS.get(loc, {}).get("object", set())
        return 0.8 if obj in objects_here else 0.2

    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time)
    pick_op = operators.construct_pick_operator_blocking(pick_time)
    place_op = operators.construct_place_operator_blocking(place_time)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create simulator
    sim = EnvironmentInterface(
        initial_state, objects_by_type, [no_op, pick_op, place_op, move_op, search_op], env
    )

    # Planning loop
    actions_taken = []
    max_iterations = 60

    h_value = ff_heuristic(initial_state, goal, sim.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        dashboard.update(sim_state=sim.state)

        for iteration in range(max_iterations):
            if goal.evaluate(sim.state.fluents):
                dashboard.console.print("[green]Goal achieved![/green]")
                break

            all_actions = sim.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(sim.state, goal, max_iterations=4000, c=300, max_depth=20)

            if action_name == "NONE":
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            action = get_action_by_name(all_actions, action_name)
            sim.advance(action, do_interrupt=False)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic(sim.state, goal, sim.get_actions())
            relevant_fluents = {
                f
                for f in sim.state.fluents
                if any(kw in f.name for kw in ["at", "holding", "found", "searched"])
            }
            dashboard.update(
                sim_state=sim.state,
                relevant_fluents=relevant_fluents,
                tree_trace=tree_trace,
                step_index=iteration,
                last_action_name=action_name,
                heuristic_value=h_value,
            )

    dashboard.print_history(sim.state, actions_taken)


if __name__ == "__main__":
    main()
