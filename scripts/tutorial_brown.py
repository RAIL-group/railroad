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

from tutorial_brown_base import BenchmarkCase, LABEL_ENV_VAR, tutorial


def fluent_filter(f):
    return any(kw in f.name for kw in ["at", "holding", "found", "searched", "free"])


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
    objects_by_type = {
        "robot": set(["rover"]),
        "location": set(locations.keys()),
        "object": {"supplies", "crate"},
    }

    initial_fluents = {
        F("at rover start"), F("free rover"),
        F("at supplies B"),
    }
    goal = F("at supplies A")


    # Define the operators
    def move_time(robot, loc_from, loc_to):
        if loc_from not in locations or loc_to not in locations:
            raise ValueError(f"One of {loc_from}/{loc_to} not found.")
        return float(np.linalg.norm(locations[loc_from] - locations[loc_to]))
    move_time_fn = _to_numeric(move_time)
    move_op = Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
            Effect(time=(move_time_fn, ["?r", "?from", "?to"]),
                   resulting_fluents={F("free ?r"), F("at ?r ?to")},
            ),
        ],
    )

    pick_op = operators.construct_pick_operator(10.0)
    place_op = operators.construct_place_operator(10.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, pick_op, place_op],
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
            planner = MCTSPlanner(all_actions,
                                  lambda_add=0.0,
                                  lambda_ff=1.0)
            action_name = planner(
                env.state, goal,
                max_iterations=bcase.mcts.iterations,
                c=bcase.mcts.c, max_depth=20,)
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
