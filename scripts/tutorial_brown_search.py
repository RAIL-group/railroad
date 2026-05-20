import numpy as np
import os
import sys
import time

from railroad import operators
from railroad.core import Fluent as F, State, get_action_by_name, Operator, Effect
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner
from railroad.operators import OptNumeric, _to_numeric

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def fluent_filter(f):
    return any(kw in f.name for kw in ["at", "holding", "found", "searched", "free"])

from tutorial_brown_base import BenchmarkCase, LABEL_ENV_VAR, tutorial


@tutorial(description="Single rover: move, pick, place, search",
          repeat=8,
          timeout=120.0)
def tutorial_main(bcase: BenchmarkCase) -> dict:
    locations = {
        "start": np.array([0, 0]),
        "A": np.array([10, 9]),
        "B": np.array([9, 0]),
        "C": np.array([1, 2]),
        "D": np.array([1, 10]),
    }
    # Ground truth: the supplies are actually at 'B'.
    objects_at_locations = {loc: set() for loc in locations}
    objects_at_locations["B"] = {"supplies"}

    robots = ["rover"]

    initial_fluents = {
        F("revealed start"),
        F("at rover start"),
        F("free rover"),
    }
    goal = F("at supplies A")

    objects_by_type = {
        "robot": set(robots),
        "location": set(locations.keys()),
        "object": {"supplies"},
    }

    # Define the operators
    def move_time(robot, loc_from, loc_to):
        if loc_from not in locations or loc_to not in locations:
            raise ValueError(f"One of {loc_from}/{loc_to} not found.")
        return float(np.linalg.norm(locations[loc_from] - locations[loc_to]))
    move_op = operators.construct_move_operator(move_time)

    pick_op = operators.construct_pick_operator(10.0)
    place_op = operators.construct_place_operator(10.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    def find_prob(robot, loc, obj):
        return 0.9 if obj in objects_at_locations.get(loc, set()) else 0.1
    search_op = operators.construct_search_operator(find_prob, 10.0)

    object_find_prob_fn = _to_numeric(find_prob)
    search_op = Operator(
        name="search",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[
            F("at ?r ?loc"), F("free ?r"),
            F("not revealed ?loc"), F("not found ?obj"),
            F("not searched ?loc ?obj"), F("not lock-search ?loc"),
        ],
        effects=[Effect(time=0,
                        resulting_fluents={
                            F("not free ?r"),
                            F("lock-search ?loc")}),
                 Effect(time=10.0,
                        resulting_fluents={
                            F("free ?r"),
                            F("searched ?loc ?obj"),
                            F("not lock-search ?loc"),},
                        prob_effects=[(
                            (object_find_prob_fn, ["?r", "?loc", "?obj"]),
                            [Effect(time=0, resulting_fluents={F("found ?obj"), F("at ?obj ?loc")})],
                        ), (
                            (1 - object_find_prob_fn, ["?r", "?loc", "?obj"]),
                            [],
                        ),],),],)

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, search_op, pick_op, place_op],
        true_object_locations=objects_at_locations,
    )


    # For plotting
    location_coords = {
        name: (float(c[0]), float(c[1])) for name, c in locations.items()
    }

    start_time = time.perf_counter()
    with bcase.make_dashboard(goal, env, fluent_filter=fluent_filter, location_coords=location_coords) as dashboard:
        for _ in range(60):
            if goal.evaluate(env.state.fluents):
                break
            all_actions = env.get_actions()
            planner = MCTSPlanner(all_actions)
            action_name = planner(
                env.state, goal,
                max_iterations=bcase.mcts.iterations,
                c=bcase.mcts.c, max_depth=20,
            )
            if action_name == "NONE":
                break
            env.act(get_action_by_name(all_actions, action_name))
            dashboard.update(planner, action_name)

    return {"success": goal.evaluate(env.state.fluents),
            "wall_time": time.perf_counter() - start_time,
            "plan_cost": float(env.state.time),
            "actions": [name for name, _ in dashboard.actions_taken],}


# Parameter sweep for --bench mode (single mode uses index --case, default 0).
tutorial_main.add_cases(
    [
        {"mcts.iterations": 30000, "mcts.c": 300},
        {"mcts.iterations": 10000, "mcts.c": 300},
        {"mcts.iterations": 3000, "mcts.c": 300},
        {"mcts.iterations": 1000, "mcts.c": 300},
        {"mcts.iterations": 300, "mcts.c": 300},
        {"mcts.iterations": 100, "mcts.c": 300},
        {"mcts.iterations": 300000, "mcts.c": 300},
    ]
)


# Needed for the tutorial to make sure the name is static for the dashboard
if os.environ.get(LABEL_ENV_VAR):
    tutorial_main._register(os.environ[LABEL_ENV_VAR])


if __name__ == "__main__":
    tutorial_main.run_cli()
