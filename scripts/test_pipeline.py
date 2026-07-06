#!/usr/bin/env python3
"""
Load a random ProcTHOR house in MolmoSpaces and print:
  - every object in the scene that can be picked up (has a free joint)
  - every named location the robot can navigate to and interact with
    (receptacle furniture/surfaces -- tables, counters, shelves, etc.)
  - the railroad initial-fluent set built from that real scene (bridge.py's
    build_initial_fluents_from_scene): robot free at "start", plus
    "at <object> <receptacle>" for every pickupable object physically
    resting on a detected receptacle

No task is planned or executed here -- this is just scene introspection.

Run: python scripts/test_pipeline.py
"""

import random
import warnings

warnings.filterwarnings("ignore")

from bridge import (
    build_initial_fluents_from_scene,
    compute_pairwise_travel_distances,
    plan_move_one_item,
)

HOUSE_INDEX_POOL = 50


def main() -> None:
    from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
        FrankaPickAndPlaceDroidDataGenConfig,
    )

    seed = random.SystemRandom().randint(0, 2**31 - 1)
    print(f"[SEED] {seed}")

    config = FrankaPickAndPlaceDroidDataGenConfig()
    config.use_passive_viewer = False
    config.profile = False
    config.use_wandb = False
    config.seed = seed
    config.task_sampler_config.pickup_types = None
    config.task_sampler_config.enable_texture_randomization = False

    task_sampler = config.task_sampler_config.task_sampler_class(config)
    task_sampler.reset()

    house_idx = random.Random(seed).randrange(HOUSE_INDEX_POOL)
    print(f"[HOUSE] index {house_idx}")

    # sample_task() is the sampler's only entry point for loading a house; it
    # also selects one pickup object and spawns one synthetic receptacle, but
    # that does not affect what we list below -- get_pickup_candidates() and
    # get_thormap() both introspect the whole loaded scene, not just the
    # sampler's own task selection.
    task = task_sampler.sample_task(house_index=house_idx)
    env = task.env
    om = env.object_managers[env.current_batch_index]

    pickupable = om.get_pickup_candidates()
    print(f"\n[PICKUPABLE OBJECTS] {len(pickupable)} object(s) with a free joint:")
    for obj in pickupable:
        print(f"  - {obj.name}")

    receptacles = om.get_receptacles()
    print(f"\n[NAMED LOCATIONS] {len(receptacles)} receptacle/surface(s) the robot can navigate to:")
    for recep in receptacles:
        print(f"  - {recep.name}")

    fluents, unplaced = build_initial_fluents_from_scene(om)
    print(f"\n[INITIAL FLUENTS] {len(fluents)} fluent(s) built from the real scene:")
    for fluent in sorted(fluents, key=str):
        print(f"  - {fluent}")
    if unplaced:
        print(
            f"\n  ({len(unplaced)} pickupable object(s) had no detected supporting "
            f"receptacle, so are not localized above -- e.g. on the floor, inside "
            f"a closed container, or below the contact-detection threshold):"
        )
        for name in unplaced:
            print(f"    - {name}")

    receptacle_names = [recep.name for recep in receptacles]
    distances = compute_pairwise_travel_distances(env, receptacle_names)
    print(
        f"\n[PAIRWISE TRAVEL DISTANCES] shortest collision-free distance (m) "
        f"between all {len(receptacle_names)} named locations:"
    )
    for (name_a, name_b), distance_m in sorted(distances.items(), key=lambda kv: -kv[1]):
        distance_str = f"{distance_m:.2f}m" if distance_m != float("inf") else "unreachable"
        print(f"  - {name_a}  <->  {name_b}: {distance_str}")

    task_obj = random.choice(pickupable).name
    task_receptacle = random.choice(receptacles).name
    print(f"\n[TASK] at {task_obj!r} {task_receptacle!r}")
    plan = plan_move_one_item(task_obj, task_receptacle, env)
    print(f"\n[PLAN] railroad MCTSPlanner produced {len(plan)} step(s):")
    for i, action_name in enumerate(plan, 1):
        print(f"  {i}. {action_name}")


if __name__ == "__main__":
    main()
