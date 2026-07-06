#!/usr/bin/env python3
"""
Fully automatic single-object pick-and-place demo (no hand-tuned coordinates).

Unlike planned_single_robot_demo.py (which uses hand-typed (x, y, theta)
waypoints and item positions), this script asks MolmoSpaces' own task/policy
framework to figure out everything geometric on its own:
  - which object to pick up (chosen by semantic type, e.g. "Cup")
  - where the robot should stand (via occupancy-map based placement,
    env.place_robot_near, collision-checked)
  - the grasp pose (via pre-baked per-asset grasp annotations + IK/collision
    filtering, get_pickup_grasps / select_grasp_pose)
  - the place pose (a nearby valid receptacle position)

This is the mechanism that (once validated here) should back each
(object, destination) step of a railroad-generated plan, instead of us
guessing coordinates by hand.

Output: auto_pick_and_place_demo.mp4
"""

import warnings
warnings.filterwarnings("ignore")

import imageio

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickAndPlaceDroidDataGenConfig,
)

OUTPUT_PATH = "auto_pick_and_place_demo.mp4"
FPS = 15  # matches policy_dt_ms ~66ms per task.step()
# exo_camera_1 is base-mounted and specifically kept pointed at the task
# object by place_robot_near's visibility check — unlike wrist_camera (which
# whips around with the arm), it gives a stable view of the whole action.
# (A second, independent mujoco.Renderer/camera collides with the GL context
# CameraManager already owns — BadAccess on GLX MakeCurrent — so we reuse
# the task's own camera observations instead of rendering separately.)
CAMERA_NAME = "exo_camera_1"


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
    config.task_sampler_config.pickup_types = None  # let the sampler use any pickup-able object
    config.task_sampler_config.enable_texture_randomization = False

    task_sampler_config = config.task_sampler_config
    task_sampler = task_sampler_config.task_sampler_class(config)
    task_sampler.reset()
    policy_config = config.policy_config

    from molmo_spaces.tasks.task_sampler_errors import RetriableError, HouseInvalidForTask
    sampling_failures = (ValueError, RetriableError, HouseInvalidForTask)

    max_attempts = 8
    frames = None
    for attempt in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt}: sampling a pick-and-place task "
              f"(object/robot pose chosen automatically)...")
        try:
            task = task_sampler.sample_task()
            pickup_name = task.config.task_config.pickup_obj_name
            place_name = task.config.task_config.place_receptacle_name
            print(f"  Sampled: pick up {pickup_name!r}, place at {place_name!r}")

            policy = policy_config.policy_factory(config, task)
            task.register_policy(policy)

            observation, _info = task.reset()
            attempt_frames = [observation[0][CAMERA_NAME]]
            for i in range(config.task_horizon):
                action_cmd = policy.get_action(observation)
                observation, reward, terminated, truncated, infos = task.step(action_cmd)
                attempt_frames.append(observation[0][CAMERA_NAME])
                if task.is_done():
                    print(f"  Policy signaled done at step {i}.")
                    break
            else:
                print(f"  Reached task_horizon ({config.task_horizon}) without "
                      f"policy signaling done.")
            frames = attempt_frames
            break
        except sampling_failures as e:
            print(f"  Attempt {attempt} failed ({e!r}); resampling a new task...")

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
