import gc
import random
import weakref

import pytest

from interruption.constants import PROCTHOR_SEED
from interruption.environments import (
    KitchenProcTHOREnvironment,
    construct_procthor_kitchen_environment,
)
from railroad.core import get_action_by_name
from railroad.environment.procthor.resources import get_procthor_10k_dir

# a fork needs a real scene, and building one without its cache would launch Unity
_CACHE = get_procthor_10k_dir() / "cache" / "deduped" / f"scene_{PROCTHOR_SEED}.pkl"
pytestmark = pytest.mark.skipif(not _CACHE.exists(), reason=f"no cached scene at {_CACHE}")


def _build() -> KitchenProcTHOREnvironment:
    env = construct_procthor_kitchen_environment(PROCTHOR_SEED, remove_duplicates=True)
    env.get_actions()  # ground once, as the datagen pipeline does before its first fork
    return env


def _snapshot(env: KitchenProcTHOREnvironment):
    return (
        set(env.fluents),
        env.time,
        {loc: set(objs) for loc, objs in env.objects_at_locations.items()},
        {loc: set(objs) for loc, objs in env.scene.object_locations.items()},
        list(env.scene.scene_graph.edges),
        {i: n["position"] for i, n in env.scene.scene_graph.nodes.items()},
        env._rng.getstate(),
    )


def test_fork_advancing_or_shuffling_leaves_the_original_untouched():
    env = _build()
    loc = next(iter(env.scene.object_locations))
    before = _snapshot(env)

    fork = env.fork()
    move = get_action_by_name(fork.get_actions(), f"move robot1 start_loc {loc}")
    fork.act(move)
    fork.update_scene_graph(move)
    fork.randomize_object_locations(random.Random(0))

    assert _snapshot(env) == before
    assert _snapshot(fork) != before


def test_fork_starts_as_an_exact_copy_with_the_same_grounded_actions():
    env = _build()
    fork = env.fork()

    assert _snapshot(fork) == _snapshot(env)
    assert sorted(a.name for a in fork.get_actions()) == sorted(a.name for a in env.get_actions())


def test_fork_shares_per_scene_data_but_copies_everything_a_step_mutates():
    env = _build()
    fork = env.fork()
    env_thor, fork_thor = env.scene._thor, fork.scene._thor

    for shared in ("cached_data", "g2p_map", "occupancy_grid", "scene"):
        assert getattr(fork_thor, shared) is getattr(env_thor, shared)
    for copied in (fork.fluents, fork.objects_at_locations, fork.scene.object_locations,
                   fork.scene.scene_graph, fork._rng):
        assert all(copied is not original for original in (
            env.fluents, env.objects_at_locations, env.scene.object_locations,
            env.scene.scene_graph, env._rng,
        ))


def test_a_fork_does_not_keep_its_ancestors_alive():
    env = _build()
    parent = env.fork()
    child = parent.fork()
    parent_ref = weakref.ref(parent)

    del parent
    gc.collect()

    assert parent_ref() is None
    assert child.get_actions()  # the child's operators must not need the freed parent
