"""ProcTHOR multi-robot search example.

Demonstrates using ProcTHOR environment with MCTS planning
for multi-robot object search and retrieval.
"""

from __future__ import annotations

from pathlib import Path


def sample_objects_and_location(scene, num_objects: int, seed: int | None = None):
    import random

    if seed:
        random.seed(seed)
    all_objects = sorted({
        obj
        for objs in scene.object_locations.values()
        for obj in objs
    })

    all_locations = sorted(scene.object_locations.keys())

    return (
        random.sample(list(all_objects), k=min(num_objects, len(all_objects))),
        random.choice(all_locations),
    )


def main(
    seed: int | None = None,
    num_objects: int = 2,
    num_robots: int = 2,
    save_plot: str | None = None,
    show_plot: bool = False,
    save_video: str | None = None,
    estimate_object_find_prob: bool = False,
    nn_model_path: str | None = None,
    video_fps: int = 60,
    video_dpi: int = 150,
) -> None:
    """Run ProcTHOR multi-robot search example."""
    try:
        from railroad.environment.procthor import ProcTHOREnvironment
        from railroad.environment.procthor.learning.utils import get_default_fcnn_model_path
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall ProcTHOR dependencies with: pip install railroad[procthor]")
        return

    from railroad import operators
    from railroad.core import Fluent as F, Operator, get_action_by_name
    from railroad.dashboard import PlannerDashboard
    from railroad.planner import MCTSPlanner
    from railroad._bindings import State

    class SearchProcTHOREnvironment(ProcTHOREnvironment):
        """Example-local ProcTHOR env with internal operator construction."""

        def __init__(
            self,
            *,
            seed: int,
            state: State,
            objects_by_type: dict[str, set[str]],
            estimate_object_find_prob: bool,
            nn_model_path: str | None,
            target_objects: list[str],
        ) -> None:
            self._estimate_object_find_prob = estimate_object_find_prob
            self._nn_model_path = nn_model_path
            self._target_objects_for_search = list(target_objects)
            super().__init__(
                seed=seed,
                state=state,
                objects_by_type=objects_by_type,
            )

        def set_target_objects(self, target_objects: list[str]) -> None:
            """Update target objects and rebuild operators."""
            self._target_objects_for_search = list(target_objects)
            self.objects_by_type["object"] = set(target_objects)
            self._operators = self.define_operators()

        def define_operators(self) -> list[Operator]:
            move_op = operators.construct_move_operator_blocking(
                self.estimate_move_time
            )

            if self._estimate_object_find_prob and self._target_objects_for_search:
                model_path = (
                    Path(self._nn_model_path)
                    if self._nn_model_path is not None
                    else get_default_fcnn_model_path()
                )
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Trained neural network model not found at {model_path} "
                        "to estimate object find probabilities. Please provide a "
                        "valid path or omit the --estimate-object-find-prob flag."
                    )
                object_find_prob_fn = self.scene.get_object_find_prob_fn(
                    nn_model_path=str(model_path),
                    objects_to_find=self._target_objects_for_search,
                )
            else:
                # Ground-truth-backed fallback for example usage.
                def object_find_prob_fn(robot: str, location: str, obj: str) -> float:
                    del robot
                    for loc, objs in self.scene.object_locations.items():
                        if obj in objs:
                            return 0.8 if loc == location else 0.1
                    return 0.1

            search_op = operators.construct_search_operator(object_find_prob_fn, 10.0)
            pick_op = operators.construct_pick_operator_blocking(10.0)
            place_op = operators.construct_place_operator_blocking(10.0)
            no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
            return [no_op, pick_op, place_op, move_op, search_op]

    robot_names = [f"robot{i + 1}" for i in range(num_robots)]
    scene_seed = seed if seed is not None else 4001
    print(f"Loading ProcTHOR scene (seed={scene_seed})...")

    # Bootstrap target set for initialization; real sampled targets are set after
    # env construction for variable-seed runs.
    bootstrap_targets = (
        ["teddybear_6", "pencil_17"] if seed is None else []
    )

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
            "object": set(bootstrap_targets),
        },
        estimate_object_find_prob=estimate_object_find_prob,
        nn_model_path=nn_model_path,
        target_objects=list(bootstrap_targets),
    )

    # Populate full location domain now that scene is available internally.
    env.objects_by_type["location"] = set(env.scene.locations.keys())

    if seed is None:
        target_objects = ["teddybear_6", "pencil_17"]
        target_location = "garbagecan_5"
        # Guard against stale hardcoded defaults in case assets change.
        for obj in list(target_objects):
            if obj not in env.scene.objects:
                target_objects, target_location = sample_objects_and_location(
                    env.scene, num_objects=num_objects, seed=scene_seed
                )
                break
    else:
        target_objects, target_location = sample_objects_and_location(
            env.scene,
            num_objects=num_objects,
            seed=seed,
        )

    env.set_target_objects(target_objects)

    from functools import reduce
    from operator import and_

    goal = reduce(and_, [
        F(f"at {obj} {target_location}") & F(f"found {obj}")
        for obj in target_objects
    ])

    max_iterations = 60

    def fluent_filter(f):
        return any(kw in f.name for kw in ["at", "holding", "found", "searched"])

    with PlannerDashboard(goal, env, fluent_filter=fluent_filter) as dashboard:
        for _iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal reached![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(
                env.state,
                goal,
                max_iterations=10000,
                c=300,
                max_depth=20,
                heuristic_multiplier=2,
            )

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


if __name__ == "__main__":
    main()
