"""Known-map object-search replay (design doc §7 / §7.1).

The complement of :mod:`~railroad.replay.search_replay_env`: there the map is
*unknown* (frontier exploration + confinement sensing); here the map is **fully
known** (e.g. a ProcTHOR floorplan) and only the objects' *presence in
containers* is unrecorded.

Because the map is known, **travel is exact** — there is no unseen-as-free
optimism in the travel term. Because the deployment revealed the truth (it found
the target), every alternative policy's cost is computable **exactly**: it
searches containers in its own order until it hits the true one, and every other
container's emptiness is known too. So the cost reported here is the
alternative's *exact* counterfactual cost, not a lower bound (§7.1).

Consequently the two LSP-style bounds do **not** apply: the optimistic vs.
simply-connected gap exists only to bracket *unobserved-space* uncertainty (a
possible shortcut-to-goal through unseen cells), and here there is none. Both
``Bounds`` slots therefore collapse onto the single exact cost.

The key simplification: :class:`~railroad.environment.symbolic.SymbolicEnvironment`
already resolves a ``search`` deterministically from ``_objects_at_locations``.
So replay needs no bespoke intercept — it simply restricts that map to the
**recorded** contents, and the existing resolution becomes exact replay from the
recording. A container the deployment never inspected has empty recorded contents
→ ``found`` resolves false there (correct: the target was found elsewhere).
"""

from __future__ import annotations

from typing import Any, Callable, Collection, Dict, List, Mapping, Set, Tuple

import numpy as np

from railroad import operators as _operators
from railroad._bindings import Fluent, Goal, GroundedEffect, State
from railroad.core import Operator
from railroad.environment.symbolic import LocationRegistry, SymbolicEnvironment
from railroad.navigation import OccupancyGridPathingMixin

from .cost import Commit
from .types import ReplayResult, RolloutLog, SubgoalRecord

ProbFn = Callable[[str, str, str], float]
START_NAME = "start_loc"
DEFAULT_SEARCH_TIME = 10.0
DEFAULT_SPEED = 1.0


class KnownMapSearchReplayEnvironment(OccupancyGridPathingMixin, SymbolicEnvironment):
    """Known-map object-search replay: ``move`` + ``search`` over a known grid.

    Search outcomes resolve from *recorded_object_locations* (the deployment's
    revealed contents) via the inherited deterministic resolution.
    ``container_find_prob`` is the swappable candidate policy; it only drives MCTS
    belief, never the outcome.

    The map is known so travel is exact, but object *presence* is only known where
    the deployment **searched**. We do not assume an object lives in a single
    container, so a revealed-but-unsearched container's emptiness cannot be
    inferred from the object being found elsewhere: searching it forces not-found
    and logs an optimistic commit (``optimistic_to_goal = 0`` — the object could be
    right here; there is no unobserved space, hence no frontier term). Thus the
    cost is a **commit-based lower bound** (``optimistic_lb`` vs. makespan), not an
    exact value (design §7.1, updated).
    """

    def __init__(
        self,
        *,
        known_grid: np.ndarray,
        recorded_object_locations: Dict[str, Set[str]],
        container_find_prob: ProbFn,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        location_registry: LocationRegistry,
        searched_sites: Collection[str] = (),
        search_time: float = DEFAULT_SEARCH_TIME,
        speed_cells_per_sec: float = DEFAULT_SPEED,
    ) -> None:
        self._known_grid = np.asarray(known_grid, dtype=float).copy()
        self._speed = float(speed_cells_per_sec)
        self._search_time = float(search_time)
        self._search_log: List[Tuple[str, float, bool]] = []
        # Containers the deployment searched (outcome known → replay exact).
        self._searched_sites = set(searched_sites)
        # One commit per not-found search at an UNsearched container.
        self._replay_commits: List[Commit] = []
        # Bootstrap pathing state (used by estimate_move_time captured below)
        # before super().__init__.
        self._location_registry = location_registry

        operators_list = self._build_operators(container_find_prob, search_time)
        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=operators_list,
            true_object_locations=recorded_object_locations,
            location_registry=location_registry,
        )

    def _build_operators(self, prob_fn: ProbFn, search_time: float) -> List[Operator]:
        return [
            _operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0),
            _operators.construct_move_operator_blocking(self.estimate_move_time),
            _operators.construct_search_operator(prob_fn, search_time),
        ]

    # -- pathing hooks ------------------------------------------------
    @property
    def occupancy_grid(self) -> np.ndarray:
        return self._known_grid

    @property
    def _pathing_unknown_as_obstacle(self) -> bool:
        return False  # the map is fully known free/occupied space

    @property
    def _pathing_speed_cells_per_sec(self) -> float:
        return self._speed

    @property
    def search_log(self) -> List[Tuple[str, float, bool]]:
        """(location, sim_time, found) per executed search."""
        return self._search_log

    @property
    def replay_commits(self) -> List[Commit]:
        """One commit per not-found search at an unsearched container."""
        return self._replay_commits

    # -- record search provenance + optimistic commits -------------------
    def resolve_probabilistic_effect(
        self, effect: GroundedEffect, current_fluents: Set[Fluent]
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        effects, fluents = super().resolve_probabilistic_effect(effect, current_fluents)
        if effect.is_probabilistic:
            loc = _search_location(effect)
            if loc is not None:
                accrued = float(self._time)
                found = any(
                    f.name == "found" and not f.negated
                    for eff in effects
                    for f in eff.resulting_fluents
                )
                self._search_log.append((loc, accrued, found))
                # Commit only at a container the deployment did NOT search: its
                # contents are unknown, so replay forces not-found and this is an
                # unverified subgoal (object could be here → optimistic_to_goal=0).
                # A searched container's outcome is known → replay exact, no commit.
                if not found and loc not in self._searched_sites:
                    self._replay_commits.append(
                        Commit(
                            cost_accrued=accrued,
                            optimistic_to_goal=0.0,
                            robot=_robot_of(effects),
                            frontier_signature=loc,
                        )
                    )
        return effects, fluents

    # -- arena handle for run_known_map_search_replay -----------------
    _source_log: "RolloutLog | None" = None
    _target_object: str = ""

    @classmethod
    def from_log(
        cls,
        log: RolloutLog,
        *,
        target_object: str,
        container_find_prob: ProbFn | None = None,
    ) -> "KnownMapSearchReplayEnvironment":
        """Build a known-map search replay arena from a recorded *log*.

        Searchable locations (coords) and recorded contents come from the log's
        subgoals; a neutral prior is used unless *container_find_prob* is given
        (the candidate policy is normally supplied to
        :func:`run_known_map_search_replay`).
        """
        if container_find_prob is None:
            container_find_prob = lambda r, loc, o: 0.5  # noqa: E731
        env = build_known_map_search_replay_env(
            log, container_find_prob=container_find_prob
        )
        env._source_log = log
        env._target_object = target_object
        return env


def _search_location(effect: GroundedEffect) -> str | None:
    """The location of a ``search`` prob-effect (from its success branch)."""
    for _, branch in effect.prob_effects:
        for eff in branch:
            for f in eff.resulting_fluents:
                if f.name == "at" and not f.negated and len(f.args) >= 2:
                    return f.args[1]
    return None


def _robot_of(effects: List[GroundedEffect]) -> str:
    """The robot named by a ``free ?robot`` effect among *effects* (provenance)."""
    for eff in effects:
        for f in eff.resulting_fluents:
            if f.name == "free" and not f.negated and f.args:
                return f.args[0]
    return ""


# ----------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------


def _pose_tuple(pose: Any) -> Tuple[float, float, float]:
    if isinstance(pose, (tuple, list)):
        return (float(pose[0]), float(pose[1]), float(pose[2]) if len(pose) > 2 else 0.0)
    return (float(pose.x), float(pose.y), float(getattr(pose, "yaw", 0.0)))


def build_known_map_search_log(
    env,
    *,
    robot_starts: Mapping[str, Any],
    env_name: str = "",
    seed: int | None = None,
    target_object: str = "",
) -> RolloutLog:
    """Snapshot a known-map search deployment into a :class:`RolloutLog`.

    Records the **known grid**, every searchable location's coordinates, and —
    for the locations the deployment actually inspected — their true contents
    (``contents`` empty for uninspected locations: their emptiness for the found
    target is implied by the deployment having revealed the truth elsewhere).
    """
    grid = np.asarray(env.occupancy_grid, dtype=float).copy()
    registry = env.location_registry
    contents_map = getattr(env, "_objects_at_locations", {})
    searched = {
        f.args[0]
        for f in env.state.fluents
        if f.name == "searched" and not f.negated and f.args
    }
    searchable = sorted(env.objects_by_type.get("location", set()) - {START_NAME})

    subgoals: List[SubgoalRecord] = []
    for loc in searchable:
        coords = registry.get(loc) if registry is not None else None
        row, col = (int(coords[0]), int(coords[1])) if coords is not None else (0, 0)
        was_searched = loc in searched
        contents = (
            tuple(sorted(contents_map.get(loc, set()))) if was_searched else ()
        )
        subgoals.append(
            SubgoalRecord(
                signature=loc,
                centroid=(row, col),
                cells=np.array([[row], [col]], dtype=int),
                contents=contents,
                searched=was_searched,
            )
        )

    starts = {r: _pose_tuple(p) for r, p in robot_starts.items()}
    start_xy = next(iter(starts.values()))[:2]
    speed = float(getattr(env, "_pathing_speed_cells_per_sec", DEFAULT_SPEED))
    config = {"speed_cells_per_sec": speed, "search_time": DEFAULT_SEARCH_TIME}

    return RolloutLog(
        recorded_grid=grid,
        goal_cell=(int(start_xy[0]), int(start_xy[1])),
        robot_starts=starts,
        problem_class="known-map-search",
        env_name=env_name,
        seed=seed,
        target_object=target_object,
        subgoals=subgoals,
        actual_total_cost=float(getattr(getattr(env, "state", None), "time", 0.0)),
        config=config,
    )


# ----------------------------------------------------------------------
# Builder + driver
# ----------------------------------------------------------------------


def build_known_map_search_replay_env(
    log: RolloutLog,
    *,
    container_find_prob: ProbFn,
    target_object: str | None = None,
) -> KnownMapSearchReplayEnvironment:
    """Construct a :class:`KnownMapSearchReplayEnvironment` from a recorded *log*."""
    robots = log.robots
    start = log.robot_starts[robots[0]]
    coords: Dict[str, np.ndarray] = {
        START_NAME: np.array(start[:2], dtype=float)
    }
    recorded: Dict[str, Set[str]] = {}
    searched_sites: Set[str] = set()
    for s in log.subgoals:
        coords[s.signature] = np.array(s.centroid, dtype=float)
        recorded[s.signature] = set(s.contents)
        if s.searched:
            searched_sites.add(s.signature)

    targets = {target_object} if target_object else set()
    fluents: Set[Fluent] = {Fluent(f"revealed {START_NAME}")}
    poses: Dict[str, object] = {}
    for robot in robots:
        fluents |= {Fluent(f"at {robot} {START_NAME}"), Fluent(f"free {robot}")}
        from railroad.environment.types import Pose

        poses[robot] = Pose(*log.robot_starts[robot])

    env = KnownMapSearchReplayEnvironment(
        known_grid=log.recorded_grid,
        recorded_object_locations=recorded,
        container_find_prob=container_find_prob,
        state=State(0.0, fluents, []),
        objects_by_type={
            "robot": set(robots),
            "location": {START_NAME} | set(coords) - {START_NAME} | set(recorded),
            "object": targets,
        },
        location_registry=LocationRegistry(coords),
        searched_sites=searched_sites,
        search_time=float(log.config.get("search_time", DEFAULT_SEARCH_TIME)),
        speed_cells_per_sec=float(log.config.get("speed_cells_per_sec", DEFAULT_SPEED)),
    )
    return env


ActionSelector = Callable[[Any, list, "Goal | Fluent"], str]


def run_known_map_search_replay(
    arena: "KnownMapSearchReplayEnvironment | RolloutLog",
    *,
    container_find_prob: ProbFn,
    target_object: str | None = None,
    select_action: ActionSelector | None = None,
    max_planning_iterations: int = 60,
    mcts_iterations: int = 4000,
    mcts_c: float = 300.0,
    mcts_max_depth: int = 20,
    mcts_heuristic_multiplier: float = 2.0,
) -> ReplayResult:
    """Replay one candidate policy over a known-map search recording.

    A thin wrapper over the unified :func:`~railroad.replay.domains.replay`
    (known-map search domain). The map is known and the truth was revealed, so
    the realized cost is the candidate's **exact** counterfactual makespan —
    there is no optimism gap to bound (no unobserved space, hence no
    shortcut-to-goal: the only unknown, which container holds the target, was
    revealed by the deployment). The two LSP-style bounds therefore collapse:
    ``optimistic_lb == simply_connected_lb == total_cost``, the exact cost in
    deployment units (seconds), comparable to ``log.actual_total_cost``. See
    ``replay_design.md`` §7.1.
    """
    from .domains import MctsParams, replay
    from .policy import CandidatePolicy

    if isinstance(arena, KnownMapSearchReplayEnvironment):
        log = arena._source_log
        target = target_object or arena._target_object
    else:
        log = arena
        target = target_object
    assert isinstance(log, RolloutLog)
    assert target, "target_object is required (pass it or use from_log)"

    return replay(
        log,
        CandidatePolicy(container_find_prob=container_find_prob),
        target_object=target,
        select_action=select_action,
        max_planning_iterations=max_planning_iterations,
        mcts=MctsParams(
            mcts_iterations, mcts_c, mcts_max_depth, mcts_heuristic_multiplier
        ),
    )
