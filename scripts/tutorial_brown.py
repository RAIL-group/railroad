import os
import sys
import time

# Ensure the sibling base module is importable both when run as a script and
# when re-imported by benchmark worker processes (which use
# spec_from_file_location and do not auto-add this directory to sys.path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tutorial_brown_base import BenchmarkCase, LABEL_ENV_VAR, tutorial

@tutorial(
    description="Heterogeneous robots search for supplies and deliver to base.",
    repeat=8,
    timeout=120.0,
)
def tutorial_main(case: BenchmarkCase) -> dict:
    """
    Heterogeneous multi-robot search-and-deliver.

    Three robots with different capabilities (rover/crawler can pick & place,
    drone can only search but moves twice as fast) must locate `supplies`
    somewhere in the map and bring them back to `start`. This runs many
    planning steps (search -> pick -> move -> place), so the live dashboard
    has plenty to show.

    Parameters come from `case` (see add_cases below):
      * case.mcts.iterations -- MCTS search budget per planning step
      * case.mcts.c          -- UCT exploration constant
    """
    import numpy as np

    from railroad import operators
    from railroad.core import Fluent as F, State, get_action_by_name
    from railroad.environment import SymbolicEnvironment
    from railroad.planner import MCTSPlanner

    # --- Map: location -> (x, y) --------------------------------------
    locations = {
        "start": np.array([0, 0]),
        "location1": np.array([10, 9]),
        "location2": np.array([9, 0]),
        "location3": np.array([1, 2]),
        "location4": np.array([1, 10]),
    }
    # Ground truth: the supplies are actually at location2.
    objects_at_locations = {loc: set() for loc in locations}
    objects_at_locations["location2"] = {"supplies"}

    # --- Heterogeneous robot capabilities -----------------------------
    skills_time = {
        "rover": {"pick": 10.0, "place": 10.0, "search": 10.0},
        "crawler": {"pick": 10.0, "place": 10.0, "search": 10.0},
        "drone": {"search": 10.0},  # drone can only search (plus move)
    }
    speed_multiplier = {"rover": 1.0, "crawler": 1.0, "drone": 2.0}
    robots = ["rover", "drone", "crawler"]

    def skill_time(skill):
        return lambda robot, *a, **k: skills_time.get(robot, {}).get(
            skill, float("inf")
        )

    def move_time(robot, loc_from, loc_to):
        if loc_from not in locations or loc_to not in locations:
            return float("inf")
        dist = float(np.linalg.norm(locations[loc_from] - locations[loc_to]))
        return dist / speed_multiplier.get(robot, 1.0)

    def find_prob(robot, loc, obj):
        return 0.9 if obj in objects_at_locations.get(loc, set()) else 0.1

    # --- Initial state & goal -----------------------------------------
    initial_fluents = {F("revealed", "start")}
    for robot in robots:
        initial_fluents.add(F("at", robot, "start"))
        initial_fluents.add(F("free", robot))
    goal = F("at supplies start")

    objects_by_type = {
        "robot": set(robots),
        "location": set(locations.keys()),
        "object": {"supplies"},
    }

    # --- Operators & environment --------------------------------------
    move_op = operators.construct_move_operator_blocking(move_time)
    search_op = operators.construct_search_operator(
        find_prob, skill_time("search")
    )
    pick_op = operators.construct_pick_operator_blocking(skill_time("pick"))
    place_op = operators.construct_place_operator_blocking(skill_time("place"))
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    env = SymbolicEnvironment(
        state=State(0.0, initial_fluents, []),
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, search_op, pick_op, place_op],
        true_object_locations=objects_at_locations,
    )

    # --- Plan, with the dashboard the harness picked for this mode -----
    def fluent_filter(f):
        return any(
            kw in f.name
            for kw in ["at", "holding", "found", "searched", "free"]
        )

    # SymbolicEnvironment has no coords to give the plotter, so pass them
    # explicitly or the trajectory plot/video is mislocated.
    location_coords = {
        name: (float(c[0]), float(c[1])) for name, c in locations.items()
    }

    start_time = time.perf_counter()
    with case.make_dashboard(goal, env, fluent_filter=fluent_filter, location_coords=location_coords) as dashboard:
        for _ in range(60):
            if goal.evaluate(env.state.fluents):
                break
            all_actions = env.get_actions()
            planner = MCTSPlanner(all_actions)
            action_name = planner(
                env.state,
                goal,
                max_iterations=case.mcts.iterations,
                c=case.mcts.c,
                max_depth=20,
            )
            if action_name == "NONE":
                break
            env.act(get_action_by_name(all_actions, action_name))
            dashboard.update(planner, action_name)

    wall_time = time.perf_counter() - start_time
    actions_taken = [name for name, _ in dashboard.actions_taken]
    return {
        "success": goal.evaluate(env.state.fluents),
        "wall_time": wall_time,
        "plan_cost": float(env.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,
    }


# Parameter sweep for --bench mode (single mode uses index --case, default 0).
tutorial_main.add_cases(
    [
        {"mcts.iterations": 10000, "mcts.c": 300},
        {"mcts.iterations": 4000, "mcts.c": 300},
        {"mcts.iterations": 1000, "mcts.c": 300},
    ]
)


# Needed for the tutorial to make sure the name is static for the dashboard
if os.environ.get(LABEL_ENV_VAR):
    tutorial_main._register(os.environ[LABEL_ENV_VAR])


if __name__ == "__main__":
    tutorial_main.run_cli()
