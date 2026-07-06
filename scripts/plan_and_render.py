#!/usr/bin/env python3
"""
End-to-end pipeline (no arguments): sample a real pick-and-place task from a
randomly chosen ProcTHOR house, plan it with railroad's MCTSPlanner, execute
the plan step by step in MolmoSpaces, and render the run to an MP4.

Pipeline:
  1. SAMPLE   Randomly pick a ProcTHOR house (procthor-10k train split) and ask
              MolmoSpaces' PickAndPlaceTaskSampler for a real, feasible
              (object, receptacle) pair in it. Retries with a new object/target
              in the same house, then a different house, on failure.
  2. PLAN     Hand (object, receptacle) to railroad as a symbolic PDDL problem
              and run railroad.planner.MCTSPlanner to get the move/pick/move/
              place action sequence. Retries (new object/target, then new
              house) if planning fails.
  3. VALIDATE Assert the picked-up object actually exists, at the pose the
              task assumes, in the loaded MolmoSpaces scene.
  4. EXECUTE  Run MolmoSpaces' PickAndPlacePlannerPolicy, printing progress as
              railroad's plan steps (translation layer in bridge.py). Any
              mid-execution failure is logged, the video recorded so far is
              saved, and the script exits non-zero.

Output: demo_house<idx>_<Object>_to_<Receptacle>.mp4 (in the working directory)

Run with: python scripts/plan_and_render.py
"""

import random
import sys
import warnings

warnings.filterwarnings("ignore")

import imageio

from bridge import (
    PLAN_STEP_PHASES,
    assert_pickup_pose_matches_sim,
    category_from_object_name,
    plan_move_one_item,
    plan_step_for_phase,
)

CAMERA_NAME = "exo_camera_1"  # base-mounted, kept pointed at the task object
FPS = 15  # matches policy_dt_ms ~66ms per task.step()
MAX_HOUSE_ATTEMPTS = 4
MAX_TASK_ATTEMPTS_PER_HOUSE = 3
# Random houses are drawn from this many lowest indices of the split. The full
# procthor-10k train split has 10k houses, all lazily downloaded on first use;
# capping the pool keeps a single run's worst-case download time bounded while
# still giving genuine per-run randomness.
HOUSE_INDEX_POOL = 50


def build_task_sampler(seed: int):
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaPickAndPlaceDroidDataGenConfig,
    )

    config = FrankaPickAndPlaceDroidDataGenConfig()
    config.use_passive_viewer = False
    config.profile = False
    config.use_wandb = False
    config.seed = seed
    config.task_horizon = 300
    config.task_sampler_config.pickup_types = None  # any pickup-able object
    config.task_sampler_config.enable_texture_randomization = False
    config.task_sampler_config.samples_per_house = MAX_TASK_ATTEMPTS_PER_HOUSE

    task_sampler = config.task_sampler_config.task_sampler_class(config)
    task_sampler.reset()
    return config, task_sampler


def main() -> int:
    seed = random.SystemRandom().randint(0, 2**31 - 1)
    print(f"[SEED] {seed}")
    house_rng = random.Random(seed)

    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask, RetriableError

    sampling_failures = (ValueError, RetriableError, HouseInvalidForTask)

    config, task_sampler = build_task_sampler(seed)

    # Errors discovered only once a policy is built for the sampled pair (e.g.
    # IK infeasible for the grasp/lift/place poses) are geometric-infeasibility
    # failures of the *sample*, not of plan execution -- MolmoSpaces' own demos
    # treat these as routine and retry with a fresh sample. They must be caught
    # here, alongside pure sampling errors, so they trigger a resample instead
    # of being mistaken for an in-progress-execution failure.
    reset_failures = (*sampling_failures, ValueError)

    tried_houses: set[int] = set()
    task = policy = observation = None
    obj = receptacle = None
    plan: list[str] = []
    house_idx = None

    for house_attempt in range(1, MAX_HOUSE_ATTEMPTS + 1):
        house_idx = house_rng.randrange(HOUSE_INDEX_POOL)
        while house_idx in tried_houses and len(tried_houses) < HOUSE_INDEX_POOL:
            house_idx = house_rng.randrange(HOUSE_INDEX_POOL)
        tried_houses.add(house_idx)
        print(f"\n=== House attempt {house_attempt}/{MAX_HOUSE_ATTEMPTS}: house index {house_idx} ===")

        got_task = False
        for task_attempt in range(1, MAX_TASK_ATTEMPTS_PER_HOUSE + 1):
            try:
                task = task_sampler.sample_task(
                    house_index=house_idx if task_attempt == 1 else None
                )
                obj = task.config.task_config.pickup_obj_name
                receptacle = task.config.task_config.place_receptacle_name
                print(f"  [SAMPLE] object={obj!r}  destination={receptacle!r}")

                plan = plan_move_one_item(obj, receptacle, task.env)

                policy = config.policy_config.policy_factory(config, task)
                task.register_policy(policy)
                observation, _info = task.reset()
                assert_pickup_pose_matches_sim(observation[0], task.env, obj)

                got_task = True
                break
            except reset_failures as e:
                print(f"  [SAMPLE] task attempt {task_attempt} failed ({e!r}); resampling...")
                task = policy = observation = None
            except RuntimeError as e:
                print(
                    f"  [PLAN] planning failed for {obj!r} -> {receptacle!r} ({e!r}); "
                    f"resampling a new object/target..."
                )
                task = policy = observation = None

        if got_task:
            break

    if task is None:
        print(
            f"\nFAILED: could not find a plannable, geometrically feasible pick-and-place "
            f"task after {MAX_HOUSE_ATTEMPTS} houses x {MAX_TASK_ATTEMPTS_PER_HOUSE} samples each."
        )
        return 1

    print("\n[PLAN] railroad MCTSPlanner produced:")
    for i, action_name in enumerate(plan, 1):
        print(f"  {i}. {action_name}")

    print("  [VALIDATE] pickup object pose in simulator matches the task's assumed start pose OK")

    frames = [observation[0][CAMERA_NAME]]
    current_plan_step = -1
    failure_reason = None

    try:
        for i in range(config.task_horizon):
            action_cmd = policy.get_action(observation)
            phase = policy.get_phase()
            new_step = plan_step_for_phase(phase, max(current_plan_step, 0))
            if new_step != current_plan_step:
                current_plan_step = new_step
                label = PLAN_STEP_PHASES[current_plan_step][0]
                step_name = (
                    plan[current_plan_step] if current_plan_step < len(plan) else "?"
                )
                print(
                    f"  [EXECUTE] plan step {current_plan_step + 1}/{len(PLAN_STEP_PHASES)}: "
                    f"{label} ({step_name})"
                )

            observation, reward, terminated, truncated, infos = task.step(action_cmd)
            frames.append(observation[0][CAMERA_NAME])

            if action_cmd.get("success") is False:
                failure_reason = f"policy signaled failure during phase {phase!r} (max retries exceeded)"
                break

            if task.is_done():
                info = task.get_info()[0]
                if not info["success"]:
                    failure_reason = f"task finished but was judged unsuccessful: {info}"
                print(f"  [EXECUTE] policy signaled done at step {i}.")
                break
        else:
            failure_reason = f"reached task_horizon ({config.task_horizon}) without finishing"
    except Exception as e:
        failure_reason = f"unhandled exception during execution: {e!r}"
        print(f"  [EXECUTE] {failure_reason}")

    obj_category = category_from_object_name(obj)
    recep_category = category_from_object_name(receptacle)
    suffix = "" if failure_reason is None else "_FAILED"
    output_path = f"demo_house{house_idx}_{obj_category}_to_{recep_category}{suffix}.mp4"

    print(f"\nRendered {len(frames)} frames. Writing {output_path}...")
    with imageio.get_writer(
        output_path, fps=FPS, codec="libx264", quality=7, macro_block_size=1
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    print("\n" + "=" * 60)
    print(f"Seed:        {seed}")
    print(f"House index: {house_idx}")
    print(f"Task:        pick up {obj!r}, place at {receptacle!r}")
    print("Plan:")
    for i, action_name in enumerate(plan, 1):
        print(f"  {i}. {action_name}")
    print(f"Video:       {output_path}")
    print("=" * 60)

    if failure_reason is not None:
        print(f"\nFAILED during execution: {failure_reason}")
        return 1

    print("\nSUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
