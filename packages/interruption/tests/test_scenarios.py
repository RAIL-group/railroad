import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

import interruption.scenarios as scenarios_module
from interruption.experiments import RunResult, initialize_experiment_data
from interruption.planners import MyopicPlanner
from interruption.scenarios import ExperimentSeeds, Scenario
from railroad.core import Fluent as F, LiteralGoal, State


def _scenario(**kwargs) -> Scenario:
    settings: dict[str, Any] = dict(
        seeds=ExperimentSeeds(),
        goal=LiteralGoal(F("at spoon shelvingunit")),
        interrupting_task_dist=([LiteralGoal(F("at mug countertop"))], [1.0]),
        task_arrival_fn=lambda cost: 0.1,
    )
    return Scenario(**{**settings, **kwargs})


def test_a_scenario_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(_scenario(), "num_task_sequence", 3)


def test_initializing_a_run_maps_the_goals_without_changing_the_scenario(monkeypatch):
    scene = SimpleNamespace(
        objects={"spoon_14", "mug_3"},
        locations={"start_loc": (0, 0), "shelvingunit_2": (1, 0), "countertop_5": (2, 0)},
    )
    env = SimpleNamespace(scene=scene, get_actions=lambda: [], state=State(0, set()))
    monkeypatch.setattr(Scenario, "build_environment", lambda self: env)
    scenario = _scenario()
    before = dataclasses.replace(scenario)

    data = initialize_experiment_data(scenario, MyopicPlanner())

    assert scenario == before
    assert scenario.goal == LiteralGoal(F("at spoon shelvingunit"))
    assert data.goal == LiteralGoal(F("at-category spoon_14 shelvingunit"))
    assert data.interrupting_task_dist == (
        [LiteralGoal(F("at-category mug_3 countertop"))], [1.0]
    )
    assert data.failure_cost == scenario.failure_cost


@pytest.fixture
def procthor_loaders(monkeypatch):
    """Stubs the scene data Scenario.procthor loads; records the remap calls."""
    tasks = [LiteralGoal(F(f"at obj{i} loc")) for i in range(4)]
    remaps = []

    def use_remapped_scenes(remap_hash, seed):
        remaps.append((remap_hash, seed))
        return f"remapped/{remap_hash}"

    monkeypatch.setattr(scenarios_module, "use_remapped_scenes", use_remapped_scenes)
    monkeypatch.setattr(scenarios_module, "get_expected_time_next_arrival", lambda f, key: 42.0)
    monkeypatch.setattr(
        scenarios_module, "get_scene_task_distribution",
        lambda seed, remap_dir: (list(tasks), [0.25] * 4),
    )
    monkeypatch.setattr(scenarios_module, "extract_relevant_objects", lambda goals: ["kept"])
    return SimpleNamespace(tasks=tasks, remaps=remaps)


def test_procthor_scenario_on_a_remapped_scene(procthor_loaders):
    scenario = Scenario.procthor(
        7, arrival_exceedance=0.5, experiment_seed=11, task_sample_seed=3,
        remap_hash="abc", object_placement_seed=99, filter_objects=True,
    )

    assert procthor_loaders.remaps == [("abc", 7)]
    assert scenario.seeds == ExperimentSeeds(
        procthor_seed=7, experiment_seed=11, task_sample_seed=3,
        # a remapped scene keeps the object placement it was generated with
        object_placement_seed=None,
    )
    # the shuffled order, its last task first
    tasks, probs = scenario.interrupting_task_dist
    assert sorted(map(str, tasks)) == sorted(map(str, procthor_loaders.tasks))
    assert scenario.goal == tasks[-1]
    assert scenario.time_between_arrivals == 42.0
    assert scenario.relevant_objects == ["kept"]
    assert scenario.remap_hash == "abc"


def test_procthor_scenario_without_shuffling_starts_with_the_example_goal(procthor_loaders):
    scenario = Scenario.procthor(
        7, arrival_exceedance=0.5, experiment_seed=11, remap_hash=None,
        object_placement_seed=99, filter_objects=False,
    )

    assert procthor_loaders.remaps == []
    assert scenario.seeds.object_placement_seed == 99
    assert scenario.goal == scenarios_module.get_example_procthor_goal()
    assert scenario.interrupting_task_dist == (procthor_loaders.tasks, [0.25] * 4)
    assert scenario.relevant_objects is None


def test_the_same_scenario_builds_the_same_task_distribution_each_time(procthor_loaders):
    first = Scenario.procthor(7, arrival_exceedance=0.5, experiment_seed=11, task_sample_seed=3)
    second = Scenario.procthor(7, arrival_exceedance=0.5, experiment_seed=11, task_sample_seed=3)

    assert first.goal == second.goal
    assert first.interrupting_task_dist == second.interrupting_task_dist


def test_run_result_metrics_keep_the_benchmark_keys():
    result = RunResult(
        success=True, cost=12.5, actions=["move a b", "Failed Task"], task_sequence=[],
        search_retries=2, object_filtered_searches=0, wall_time=1.0,
    )

    assert result.had_failed_search
    assert result.metrics() == {
        "success": True, "wall_time": 1.0, "plan_cost": 12.5, "actions_count": 2,
        "required_search_retry": True, "search_retries": 2,
        "required_object_filtering": False, "object_filtered_searches": 0,
        "had_failed_search": True, "actions": ["move a b", "Failed Task"],
        "log_html": "", "log_plot": b'',
    }


def test_a_scenario_builds_its_environment_with_its_operators(monkeypatch):
    built = {}
    monkeypatch.setattr(
        scenarios_module, "construct_procthor_kitchen_environment",
        lambda seed, **kwargs: built.update(kwargs, seed=seed) or "env",
    )

    def operators(env):
        return []

    scenario = _scenario(operators=operators, relevant_objects=["spoon_14"])

    assert scenario.build_environment() == "env"
    assert built == {
        "seed": scenario.seeds.procthor_seed,
        "object_seed": scenario.seeds.object_placement_seed,
        "relevant_objects": ["spoon_14"],
        "remove_duplicate_objects": True,
        "operators": operators,
    }
