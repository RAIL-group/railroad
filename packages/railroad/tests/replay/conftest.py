"""Shared fixtures for offline-replay tests.

The ASCII-grid parser keeps map-based tests readable and hand-verifiable:
maps are written inline, markers (start / goal / frontier points) are
recovered by character.
"""

from __future__ import annotations

import dataclasses
import textwrap
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest

from railroad.experimental.unknown_search import NavigationConfig
from railroad.navigation.constants import COLLISION_VAL, FREE_VAL, UNOBSERVED_VAL
from railroad.replay.types import RolloutLog

_TERRAIN = {"#": COLLISION_VAL, ".": FREE_VAL, "?": UNOBSERVED_VAL, " ": FREE_VAL}

# A deployment-style NavigationConfig recorded into synthetic test logs. Replay
# has no default config — it always reads sensing params from the log — so test
# logs (which stand in for a deployment) must carry one, just as a real
# recording does.
REPLAY_TEST_CONFIG = dataclasses.asdict(
    NavigationConfig(
        sensor_range=60.0,
        max_move_action_time=10_000.0,
        interrupt_min_new_cells=30000,
        interrupt_min_dt=30000.0,
    )
)

Markers = Dict[str, List[Tuple[int, int]]]


def parse_ascii_grid(text: str) -> Tuple[np.ndarray, Markers]:
    """Parse an ASCII map into ``(grid, markers)``.

    Terrain: ``#`` collision, ``.`` free, ``?`` unobserved, `` `` free. Any
    other character marks a cell that is otherwise ``FREE`` and is returned
    in ``markers[char]`` as a list of ``(row, col)`` tuples (in row-major
    reading order). The text is dedented and short lines padded with free
    cells, so maps can be written indented inside a test.
    """
    lines = textwrap.dedent(text).strip("\n").splitlines()
    height = len(lines)
    width = max(len(line) for line in lines)
    grid = np.full((height, width), FREE_VAL, dtype=float)
    markers: Markers = {}
    for row, line in enumerate(lines):
        for col in range(width):
            char = line[col] if col < len(line) else "."
            if char in _TERRAIN:
                grid[row, col] = _TERRAIN[char]
            else:
                grid[row, col] = FREE_VAL
                markers.setdefault(char, []).append((row, col))
    return grid, markers


@pytest.fixture
def parse_grid() -> Callable[[str], Tuple[np.ndarray, Markers]]:
    return parse_ascii_grid


def build_log_from_ascii(text: str, *, robot: str = "robot1") -> RolloutLog:
    """Build a navigation :class:`RolloutLog` from an ASCII map.

    ``S`` marks the (single robot) start, ``G`` the goal; the recorded grid
    is the parsed terrain (``?`` regions become frontiers when adjacent to
    free space).
    """
    grid, markers = parse_ascii_grid(text)
    start = markers["S"][0]
    goal = markers["G"][0]
    return RolloutLog(
        recorded_grid=grid,
        goal_cell=goal,
        robot_starts={robot: (float(start[0]), float(start[1]), 0.0)},
        config=dict(REPLAY_TEST_CONFIG),
    )


def explore_first_select(env, actions, goal) -> str:
    """Deterministic test policy: explore every reachable frontier, then goal.

    Forces ``lsp-explore`` commitments (so the intercept and bound machinery
    are exercised) regardless of whether the goal is already visible.
    """
    applicable = [a for a in actions if env.state.satisfies_precondition(a)]
    explored = {
        f.args[0]
        for f in env.state.fluents
        if f.name == "explored" and not f.negated and f.args
    }

    def first(predicate) -> str | None:
        chosen = sorted(
            (a for a in applicable if predicate(a)), key=lambda a: a.name
        )
        return chosen[0].name if chosen else None

    def parts(action) -> list:
        return action.name.split()

    return (
        first(lambda a: parts(a)[0] == "lsp-explore")
        or first(
            lambda a: parts(a)[0] == "move"
            and parts(a)[-1] in env.frontiers
            and parts(a)[-1] not in explored
        )
        or first(lambda a: parts(a)[0] == "move" and parts(a)[-1] == "goal")
        or first(lambda a: parts(a)[0] in ("move", "move-to-goal"))
        or "NONE"
    )


@pytest.fixture
def make_log() -> Callable[[str], RolloutLog]:
    return build_log_from_ascii


@pytest.fixture
def explore_first() -> Callable:
    return explore_first_select
