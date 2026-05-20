import os
import sys
import time
import random
from functools import reduce
from operator import and_

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tutorial_brown_base import BenchmarkCase, LABEL_ENV_VAR, tutorial


def fluent_filter(f):
    return any(kw in f.name for kw in ["at", "holding", "found", "searched"])


def _sample_objects_and_location(scene, num_objects: int, seed: int | None):
    rng = random.Random(seed)
    all_objects = sorted({
        obj
        for objs in scene.object_locations.values()
        for obj in objs
    })
    all_locations = sorted(scene.object_locations.keys())
    return (
        rng.sample(all_objects, k=min(num_objects, len(all_objects))),
        rng.choice(all_locations),
    )


@tutorial(description="Multi-robot search in a ProcTHOR-generated household scene",
          repeat=8,
          timeout=600.0)
def tutorial_main(bcase: BenchmarkCase) -> dict:
    from railroad import operators
    from railroad.core import Fluent as F, Operator, State, get_action_by_name
    from railroad.environment.procthor import ProcTHOREnvironment
    from railroad.planner import MCTSPlanner

    num_robots = bcase.num_robots
    num_objects = bcase.num_objects
    scene_seed = bcase.scene_seed

    class SearchProcTHOREnvironment(ProcTHOREnvironment):
        """Tutorial-local ProcTHOR env with internal operator construction."""

        def set_target_objects(self, target_objects: list[str]) -> None:
            self._target_objects_for_search = list(target_objects)
            self.objects_by_type["object"] = set(target_objects)
            self._operators = self.define_operators()

        def define_operators(self) -> list[Operator]:
            def object_find_prob_fn(robot: str, location: str, obj: str) -> float:
                del robot
                for loc, objs in self.scene.object_locations.items():
                    if obj in objs:
                        return 0.8 if loc == location else 0.1
                return 0.1

            move_op = operators.construct_move_operator_blocking(self.estimate_move_time)
            search_op = operators.construct_search_operator(object_find_prob_fn, 10.0)
            pick_op = operators.construct_pick_operator_blocking(10.0)
            place_op = operators.construct_place_operator_blocking(10.0)
            no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
            return [no_op, pick_op, place_op, move_op, search_op]

    robot_names = [f"robot{i + 1}" for i in range(num_robots)]

    initial_fluents = {F("revealed start_loc")}
    for robot in robot_names:
        initial_fluents.add(F(f"at {robot} start_loc"))
        initial_fluents.add(F(f"free {robot}"))
    initial_state = State(0.0, initial_fluents, [])

    env = SearchProcTHOREnvironment(
        seed=scene_seed,
        state=initial_state,
        objects_by_type={
            "robot": set(robot_names),
            "location": {"start_loc"},
        },
    )
    env.objects_by_type["location"] = set(env.scene.locations.keys())

    target_objects, target_location = _sample_objects_and_location(
        env.scene,
        num_objects=num_objects,
        seed=scene_seed,
    )
    env.set_target_objects(target_objects)

    goal = reduce(and_, [
        F(f"at {obj} {target_location}")
        for obj in target_objects
    ])

    location_coords = {
        name: (float(coord[0]), float(coord[1]))
        for name, coord in env.scene.locations.items()
    }

    start_time = time.perf_counter()
    with bcase.make_dashboard(goal, env, fluent_filter=fluent_filter,
                             location_coords=location_coords) as dashboard:
        for _ in range(60):
            if goal.evaluate(env.state.fluents):
                break
            all_actions = env.get_actions()
            planner = MCTSPlanner(all_actions)
            action_name = planner(
                env.state, goal,
                max_iterations=bcase.mcts.iterations,
                c=bcase.mcts.c, max_depth=20,
                heuristic_multiplier=bcase.mcts.h_mult,
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
tutorial_main.add_cases([
    {"mcts.iterations": 10000, "mcts.c": 300, "mcts.h_mult": 3, "scene_seed": 8616, "num_robots": 2, "num_objects": 4},
])


# Needed for the tutorial to make sure the name is static for the dashboard
if os.environ.get(LABEL_ENV_VAR):
    tutorial_main._register(os.environ[LABEL_ENV_VAR])


if __name__ == "__main__":
    tutorial_main.run_cli()
