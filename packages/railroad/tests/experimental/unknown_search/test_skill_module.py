"""Tests for navigation move skills and skill-module compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from railroad._bindings import Fluent as F
from railroad.core import Effect, Operator
from railroad.environment.skill import InterruptibleNavigationMoveSkill, NavigationMoveSkill
from railroad.environment.symbolic import LocationRegistry

if TYPE_CHECKING:
    from railroad.environment import Environment


def _move_action():
    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at", "?robot", "?from"), F("free", "?robot")],
        effects=[
            Effect(time=0.0, resulting_fluents={~F("free", "?robot")}),
            Effect(
                time=10.0,
                resulting_fluents={
                    ~F("at", "?robot", "?from"),
                    F("at", "?robot", "?to"),
                    F("free", "?robot"),
                },
            ),
        ],
    )
    actions = move_op.instantiate({"robot": ["r1"], "location": ["start", "goal"]})
    return next(a for a in actions if a.name == "move r1 start goal")


class _FakeNavigationEnv:
    def __init__(self, *, destination_known: bool) -> None:
        self.fluents = {F("at", "r1", "start"), F("free", "r1")}
        self.objects_by_type = {"location": {"start"}}
        registry_locations: dict[str, np.ndarray] = {
            "start": np.array([0.0, 0.0], dtype=float),
        }
        if destination_known:
            self.objects_by_type["location"].add("goal")
            registry_locations["goal"] = np.array([0.0, 2.0], dtype=float)
        self.location_registry = LocationRegistry(registry_locations)
        self.robot_pose_updates: dict[str, object] = {}

    def compute_move_path(
        self,
        loc_from: str,
        loc_to: str,
        robot: str | None = None,
    ) -> np.ndarray:
        del loc_from, loc_to, robot
        return np.array([[0, 0, 0], [0, 1, 2]], dtype=int)

    def set_robot_pose(self, robot: str, pose: object) -> None:
        self.robot_pose_updates[robot] = pose

    def apply_effect(self, effect):
        for fluent in effect.resulting_fluents:
            if fluent.negated:
                self.fluents.discard(~fluent)
            else:
                self.fluents.add(fluent)
        return []


def test_navigation_skill_interrupt_ignored_when_destination_not_stale():
    env = _FakeNavigationEnv(destination_known=True)
    typed_env = cast("Environment", env)
    skill = NavigationMoveSkill(action=_move_action(), start_time=0.0, env=env)

    skill.advance(0.0, typed_env)
    skill.advance(5.0, typed_env)
    skill.interrupt(typed_env)

    assert not skill.is_done
    assert F("free", "r1") not in env.fluents
    assert F("at", "r1", "r1_loc") not in env.fluents


def test_navigation_skill_interrupt_rewrites_when_destination_stale():
    env = _FakeNavigationEnv(destination_known=False)
    typed_env = cast("Environment", env)
    skill = NavigationMoveSkill(action=_move_action(), start_time=0.0, env=env)

    skill.advance(0.0, typed_env)
    skill.advance(5.0, typed_env)
    skill.interrupt(typed_env)

    assert skill.is_done
    assert F("free", "r1") in env.fluents
    assert F("at", "r1", "r1_loc") in env.fluents
    assert F("at", "r1", "goal") not in env.fluents


def test_interruptible_navigation_skill_interrupts_even_when_destination_not_stale():
    env = _FakeNavigationEnv(destination_known=True)
    typed_env = cast("Environment", env)
    skill = InterruptibleNavigationMoveSkill(action=_move_action(), start_time=0.0, env=env)

    skill.advance(0.0, typed_env)
    skill.advance(5.0, typed_env)
    skill.interrupt(typed_env)

    assert skill.is_done
    assert F("at", "r1", "r1_loc") in env.fluents


def test_navigation_skill_constructor_has_runtime_guard_for_missing_compute_move_path():
    class _MissingPathEnv:
        pass

    with pytest.raises(TypeError, match="requires env.compute_move_path"):
        NavigationMoveSkill(action=_move_action(), start_time=0.0, env=_MissingPathEnv())  # type: ignore[arg-type]


def test_navigation_skill_import_paths_remain_backward_compatible():
    from railroad.environment.skill.navigation import NavigationMoveSkill as FromSkillModule

    assert FromSkillModule is NavigationMoveSkill
