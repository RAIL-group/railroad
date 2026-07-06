#!/usr/bin/env python3
"""
Full pipeline: railroad plans a "move one item" task, MolmoSpaces executes it
automatically (no hand-tuned coordinates anywhere).

Task given: move a single object to a destination receptacle.

  1. SAMPLE   Ask MolmoSpaces' PickAndPlaceTaskSampler for a real, valid
              (object, receptacle) pair in a real ProcTHOR house — this is
              the only place any object/location identifier comes from.
  2. PLAN     Hand that (object, receptacle) pair to railroad as a symbolic
              PDDL problem (robot0, the object, two location tokens standing
              in for "near the object" / "near the receptacle") and run
              railroad.planner.MCTSPlanner to produce the move/pick/move/place
              action sequence.
  3. EXECUTE  Run MolmoSpaces' PickAndPlacePlannerPolicy for that exact
              (object, receptacle) pair to completion — it auto-computes the
              robot standing pose, grasp pose, and place pose; we never
              specify a single (x, y, theta) or 3D position ourselves.

If step 1's sample turns out to be geometrically infeasible (IK failure —
a normal, expected occurrence for this stochastic sampler, not a bug), the
whole thing (sample -> plan -> execute) is retried with a fresh sample.

Output: full_pipeline_demo.mp4
"""

import warnings
warnings.filterwarnings("ignore")

import imageio

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickAndPlaceDroidDataGenConfig,
)
from molmo_spaces.tasks.task_sampler_errors import RetriableError, HouseInvalidForTask

OUTPUT_PATH = "full_pipeline_demo.mp4"
FPS = 15  # matches policy_dt_ms ~66ms per task.step()
CAMERA_NAME = "exo_camera_1"  # base-mounted, kept pointed at the task object

ROBOT = "robot0"
START_LOC = "start"
PICKUP_SITE = "pickup_site"       # stands in for "wherever near the object is"
DESTINATION_SITE = "destination_site"  # stands in for "wherever near the receptacle is"

SAMPLING_FAILURES = (ValueError, RetriableError, HouseInvalidForTask)


def plan_move_one_item(obj: str) -> list[str]:
    """Railroad symbolic plan for moving a single named object to the destination.

    No real coordinates are involved: railroad only reasons about *order*
    (move, then pick, then move, then place) over abstract location tokens.
    The physical realization of each of those steps is entirely delegated to
    MolmoSpaces (step EXECUTE), which is the one that knows real geometry.
    """
    from railroad.core import Fluent as F, State, get_action_by_name
    from railroad.environment.symbolic import SymbolicEnvironment
    from railroad.planner import MCTSPlanner
    from railroad import operators

    move_op = operators.construct_move_operator_blocking(move_time=5.0)
    pick_op = operators.construct_pick_operator_blocking(pick_time=5.0)
    place_op = operators.construct_place_operator_blocking(place_time=5.0)

    initial_fluents = {
        F(f"at {ROBOT} {START_LOC}"),
        F(f"free {ROBOT}"),
        F(f"at {obj} {PICKUP_SITE}"),
    }
    state = State(0.0, initial_fluents, [])
    env = SymbolicEnvironment(
        state=state,
        objects_by_type={
            "robot": {ROBOT},
            "location": {START_LOC, PICKUP_SITE, DESTINATION_SITE},
            "object": {obj},
        },
        operators=[move_op, pick_op, place_op],
    )
    goal = F(f"at {obj} {DESTINATION_SITE}")

    plan: list[str] = []
    for _ in range(10):
        if goal.evaluate(env.fluents):
            return plan
        all_actions = env.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(env.state, goal, max_iterations=5000, max_depth=15)
        if action_name == "NONE":
            raise RuntimeError("Planner could not find a next action toward the goal.")
        action = get_action_by_name(all_actions, action_name)
        env.act(action)
        plan.append(action_name)

    raise RuntimeError("Plan did not reach the goal in time.")


def main():
    config = FrankaPickAndPlaceDroidDataGenConfig()
    config.use_passive_viewer = False
    config.profile = False
    config.use_wandb = False
    config.seed = 2
    config.task_horizon = 300

    config.task_sampler_config.house_inds = list(range(5))
    config.task_sampler_config.samples_per_house = 1
    config.task_sampler_config.max_tasks = 20
    config.task_sampler_config.pickup_types = None
    config.task_sampler_config.enable_texture_randomization = False

    task_sampler_config = config.task_sampler_config
    task_sampler = task_sampler_config.task_sampler_class(config)
    task_sampler.reset()
    policy_config = config.policy_config

    max_attempts = 8
    frames = None
    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Attempt {attempt} ===")
        try:
            # --- 1. SAMPLE: get a real (object, receptacle) pair ---
            print("[SAMPLE] Asking MolmoSpaces for a valid object/receptacle pair...")
            task = task_sampler.sample_task()
            obj = task.config.task_config.pickup_obj_name
            receptacle = task.config.task_config.place_receptacle_name
            print(f"[SAMPLE] object={obj!r}  destination={receptacle!r}")

            # --- 2. PLAN: railroad decides the move/pick/move/place order ---
            print("[PLAN] Running railroad MCTSPlanner...")
            plan = plan_move_one_item(obj)
            for i, action_name in enumerate(plan, 1):
                print(f"[PLAN]   {i}. {action_name}")

            # --- 3. EXECUTE: MolmoSpaces auto-computes poses and runs them ---
            print("[EXECUTE] Running PickAndPlacePlannerPolicy "
                  "(auto grasp pose, auto standing pose, auto place pose)...")
            policy = policy_config.policy_factory(config, task)
            task.register_policy(policy)

            observation, _info = task.reset()
            attempt_frames = [observation[0][CAMERA_NAME]]
            for i in range(config.task_horizon):
                action_cmd = policy.get_action(observation)
                observation, reward, terminated, truncated, infos = task.step(action_cmd)
                attempt_frames.append(observation[0][CAMERA_NAME])
                if task.is_done():
                    print(f"[EXECUTE] Policy signaled done at step {i}.")
                    break
            else:
                print(f"[EXECUTE] Reached task_horizon ({config.task_horizon}) "
                      f"without policy signaling done.")
            frames = attempt_frames
            break
        except SAMPLING_FAILURES as e:
            print(f"  Attempt {attempt} failed ({e!r}); resampling and re-planning...")

    if frames is None:
        raise RuntimeError(f"All {max_attempts} sampled tasks failed to produce a feasible plan.")

    print(f"\nRendered {len(frames)} frames. Writing {OUTPUT_PATH}...")
    with imageio.get_writer(OUTPUT_PATH, fps=FPS, codec="libx264",
                            quality=7, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"Done! {OUTPUT_PATH}  {len(frames)} frames  {FPS}fps  {len(frames) / FPS:.1f}s")


if __name__ == "__main__":
    main()
