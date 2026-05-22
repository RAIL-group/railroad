import os
import json
import numpy as np

from railroad import operators
from railroad.core import Fluent as F
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State

from railroad.experimental.llm_heuristic import HeuristicContextBuilder

# Define locations with coordinates
LOCATIONS = {
    "start_loc": np.array([-5, -5]),
    "table": np.array([0, 0]),
    "pantry": np.array([10, 0]),
    "bed": np.array([0, 12]),
    "cabinet": np.array([10, 12]),
    "fridge": np.array([15, 12]),
}

# Define where objects actually are
OBJECTS_AT_LOCATIONS = {
    "start_loc": set(),
    "table": {"Notebook", "Clock", "Mug"},
    "pantry": set(),
    "bed": {"Pillow", "Waterbottle"},
    "cabinet": {"Plate"},
    "fridge": {"Apple"},
}

ROBOT_VELOCITY = 1.0
PICK_TIME = 5.0
PLACE_TIME = 5.0
OPEN_TIME = 3.0

def main():
    objects_of_interest = ["Notebook", "Clock", "Mug", "Pillow", "Waterbottle", "Plate", "Apple"]

    # Define initial fluents
    initial_fluents = {
        F("at", "robot1", "start_loc"),
        F("free", "robot1"),
        F("at Notebook table"),
        F("at Clock table"),
        F("at Cereal pantry"),
        F("at Pillow bed"),
        F("at Waterbottle bed"),
        F("at Mug table"),
        F("at Plate cabinet"),
        F("has-door cabinet cabinet_door"),
        F("closed cabinet_door"),
        F("blocking-access cabinet"),
        F("has-door fridge fridge_door"),
        F("closed fridge_door"),
        F("blocking-access fridge"),
    }

    goal = F("at Mug cabinet") & F("at Waterbottle fridge")

    objects_by_type = {
        "robot": {"robot1"},
        "location": set(LOCATIONS.keys()),
        "object": set(objects_of_interest),
        "container-door": {"cabinet_door", "fridge_door"}
    }

    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        distance = float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))
        return distance / ROBOT_VELOCITY

    move_op = operators.construct_move_operator_blocking(move_time)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
    open_op = operators.construct_open_container_door_operator_blocking(OPEN_TIME)

    initial_state = State(0.0, initial_fluents, [])

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, pick_op, place_op, open_op],
        true_object_locations=OBJECTS_AT_LOCATIONS,
    )

    all_actions = env.get_actions()

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "closed", "blocking"])

    context_builder = HeuristicContextBuilder(fluent_filter=fluent_filter)
    
    print("Building context...")
    context_json = context_builder.to_json(env.state, goal, all_actions)
    
    output_path = "context.json"
    with open(output_path, "w") as f:
        f.write(context_json)
        
    print(f"Context successfully written to {output_path}")

if __name__ == "__main__":
    main()
