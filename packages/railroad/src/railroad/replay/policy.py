"""Policies: the belief models a study chooses between, and how to build them.

A *policy* is not a state->action mapping. It is the **belief model that
parameterizes the planner's actions** — "how likely is this frontier to lead to
the goal?", "how likely is the object to be in this container?". MCTS consumes
those numbers and does the choosing.

Beliefs never decide outcomes. What actually happens comes from the environment:
ground truth when deploying, the recording when replaying. Belief only steers
*which subgoal is tried next*. That is why an **oracle** may consult ground truth
even in replay — it is a black box to the bound, while the replayed cost
accounting still reads only what the deployment recorded.

**A policy is an estimator, and which estimator depends on the problem.** There
is no universal policy object, because the three problem classes genuinely
consume different things:

===========================  =========================================
problem                      what a policy is
===========================  =========================================
point-goal navigation        a ``FrontierStatisticsEstimator``
unknown-map object search    an ``ObjectFindEstimator`` (both accessors)
known-map object search      an ``ObjectFindEstimator`` (containers only)
===========================  =========================================

Each environment takes the estimator its own problem consumes, so a mismatch
fails rather than degrading quietly to a neutral prior — and a navigation study
never constructs search machinery, nor a known-map study a frontier concept like
``exploration_cost``.

This module supplies the pieces that are genuinely shared across studies: the
object-find estimators, a stand-in for a trained network, and per-family builders
that read ground truth off a scene. *Which* policies a study compares, under what
names and tuning, is an experiment choice — see ``build_policies`` in each
``scripts/replay/*.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from railroad.experimental.unknown_search.statistics import ObjectFindEstimator
from railroad.lsp.frontier_statistics import (
    LearnedFrontierStatistics,
    OracleFrontierStatistics,
)
from railroad.lsp.oracle import compute_oracle_frontier_labels
from railroad.lsp.types import FrontierObservation, FrontierStatistics
from railroad.navigation.constants import UNOBSERVED_VAL

# Reserved location names that are never searchable containers.
_NON_CONTAINER = ("start_loc", "goal_loc")


# ----------------------------------------------------------------------
# Object-find estimators
# ----------------------------------------------------------------------


class OracleObjectFind(ObjectFindEstimator):
    """Perfect object-search knowledge: ground truth for both subgoal kinds.

    - **containers**: decisive 1/0 — the object is exactly where it truly is.
    - **frontiers**: does a still-hidden target container lie behind this one?
      The navigation oracle asks whether a *point goal* is reachable through a
      frontier; here the same machinery runs once per target container (mask
      every other frontier, ask whether a path through this one alone reaches
      it), and a frontier takes the best result over the hidden targets.

    Only containers whose cell is **still unobserved** count toward the frontier
    belief. Once a container has been seen there is nothing to explore *toward* —
    the robot can navigate straight to it — and including it would make every
    frontier look feasible via a path back through observed space.

    ``refresh`` is called by the environment on every frontier change (both the
    live search environment and the replay arena), so there is no attach step.
    """

    def __init__(
        self,
        true_grid: np.ndarray,
        true_object_locations: Mapping[str, set],
        target_container_cells: Mapping[str, Tuple[int, int]],
    ) -> None:
        self._true_grid = np.asarray(true_grid, dtype=float)
        self._truth = dict(true_object_locations)
        self._target_cells = dict(target_container_cells)
        self._frontier_probabilities: Dict[str, float] = {}

    def probability(self, robot: str, subgoal: str, obj: str) -> float:
        # Only reached for subgoal kinds without a dedicated accessor.
        del robot, subgoal, obj
        return 0.0

    def container_probability(self, robot: str, container: str, obj: str) -> float:
        del robot
        return 1.0 if obj in self._truth.get(container, ()) else 0.0

    def frontier_probability(self, robot: str, frontier: str, obj: str) -> float:
        del robot, obj  # the oracle scores a frontier, not an (object, frontier) pair
        return self._frontier_probabilities.get(frontier, 0.0)

    def refresh(self, environment: Any) -> None:
        self._frontier_probabilities = {}
        frontiers = getattr(environment, "frontiers", {}) or {}
        if not frontiers:
            return
        observed_grid = np.asarray(environment.observed_grid, dtype=float)
        hidden = self._hidden_target_cells(observed_grid)
        if not hidden:
            # Every target container is visible: exploring reveals nothing needed.
            self._frontier_probabilities = {fid: 0.0 for fid in frontiers}
            return
        best: Dict[str, float] = {}
        for cell in hidden:
            labels = compute_oracle_frontier_labels(
                self._true_grid, observed_grid, frontiers, cell
            )
            for fid, label in labels.items():
                probability = float(label.prob_feasible)
                if probability > best.get(fid, 0.0):
                    best[fid] = probability
        self._frontier_probabilities = best

    def _hidden_target_cells(self, observed_grid: np.ndarray) -> list:
        rows, cols = observed_grid.shape
        return [
            (row, col)
            for row, col in self._target_cells.values()
            if 0 <= row < rows
            and 0 <= col < cols
            and float(observed_grid[row, col]) == UNOBSERVED_VAL
        ]


# ----------------------------------------------------------------------
# A stand-in for a trained network
# ----------------------------------------------------------------------


@dataclass
class ConstantFrontierStatisticsModel:
    """A ``FrontierStatisticsModel`` that returns the same statistics for every
    observation.

    Same protocol as the trained ``LSPFrontierNet`` wrapper, so it is
    interchangeable with ``load_frontier_statistics_model(...)``. It still
    *receives* the real observations — best-vantage selection, panorama serving,
    egocentric geometry all run — it just ignores them. That is the point: it
    exercises the entire served-vantage pipeline while faking only the numbers,
    so a trained network drops in at the same call site with no other change.
    """

    prob_feasible: float = 0.5
    delta_success_cost: float = 0.0
    exploration_cost: float = 10.0

    def __call__(
        self, observations: Sequence[FrontierObservation]
    ) -> List[FrontierStatistics]:
        statistics = FrontierStatistics(
            prob_feasible=self.prob_feasible,
            delta_success_cost=self.delta_success_cost,
            exploration_cost=self.exploration_cost,
        )
        return [statistics for _ in observations]


# ----------------------------------------------------------------------
# Reading ground truth off a scene
# ----------------------------------------------------------------------


def scene_goal_cell(scene: Any) -> Optional[Tuple[int, int]]:
    """The point-goal cell, if this scene has one (navigation scenes do)."""
    coord = dict(getattr(scene, "locations", {}) or {}).get("goal_loc")
    return None if coord is None else (int(coord[0]), int(coord[1]))


def target_container_cells(
    scene: Any, target_objects: Sequence[str] = ()
) -> Dict[str, Tuple[int, int]]:
    """Cells of the containers that truly hold a target object.

    Empty *target_objects* means "any object".
    """
    truth = dict(getattr(scene, "object_locations", {}) or {})
    targets = set(target_objects)
    return {
        name: (int(coord[0]), int(coord[1]))
        for name, coord in (dict(getattr(scene, "locations", {}) or {})).items()
        if name not in _NON_CONTAINER
        and (contents := set(truth.get(name, ())))
        and (not targets or contents & targets)
    }


# ----------------------------------------------------------------------
# Per-family builders
# ----------------------------------------------------------------------
#
# Two families, so two of each: one producing a FrontierStatisticsEstimator for
# point-goal navigation, one producing an ObjectFindEstimator for object search.
# Constant-belief policies need no builder at all — FixedPriorFrontierStatistics
# and FixedObjectFind are already exactly that.


def oracle_frontier_statistics(
    scene: Any, *, goal_cell: Optional[Tuple[int, int]] = None
) -> OracleFrontierStatistics:
    """Navigation oracle: frontier labels against *scene*'s true map.

    Takes the scene rather than a seed so no second world (and, for railsim, no
    second GL context) is created: the caller already holds the scene it deployed
    in, and the oracle must see exactly that world. *goal_cell* overrides the
    scene's ``goal_loc``.
    """
    return OracleFrontierStatistics(
        np.asarray(scene.grid, dtype=float),
        goal_cell=goal_cell or scene_goal_cell(scene),
    )


def oracle_object_find(
    scene: Any, *, target_objects: Sequence[str] = ()
) -> OracleObjectFind:
    """Search oracle: decisive container truth, plus frontier truth if any.

    *target_objects* narrows it to the objects this run is looking for; empty
    means any object. On a known map the frontier half is simply never consulted.
    """
    return OracleObjectFind(
        np.asarray(scene.grid, dtype=float),
        dict(getattr(scene, "object_locations", {}) or {}),
        target_container_cells(scene, target_objects),
    )


def constant_frontier_statistics(
    prob_feasible: float,
    *,
    exploration_cost: float = 10.0,
    delta_success_cost: float = 0.0,
) -> LearnedFrontierStatistics:
    """A constant navigation belief that still runs the **real** perception path.

    A :class:`ConstantFrontierStatisticsModel` behind ``LearnedFrontierStatistics``:
    best-vantage selection and panorama serving all happen, only the numbers are
    faked, so :func:`learned_frontier_statistics` is a drop-in at the same site.

    Distinct from ``FixedPriorFrontierStatistics(prob_feasible)``, which bypasses
    perception entirely — that one is the control for "is perception doing
    anything?".
    """
    return LearnedFrontierStatistics(
        ConstantFrontierStatisticsModel(
            prob_feasible=prob_feasible,
            delta_success_cost=delta_success_cost,
            exploration_cost=exploration_cost,
        )
    )


def learned_frontier_statistics(network_file: Any) -> LearnedFrontierStatistics:
    """Navigation belief from a trained ``LSPFrontierNet`` at *network_file*.

    Predicts frontier statistics from the best-vantage panorama, so it needs an
    environment that records panoramas (railsim supplies them live; replay serves
    the recorded ones). For a learned *container* belief — a different model
    answering a different question — see :func:`learned_container_find`.
    """
    if network_file is None:
        raise ValueError(
            "a learned policy needs trained weights; pass network_file "
            "(e.g. the LSPFrontierNet.pt that 'railroad lsp train-network' saves)"
        )
    # torch lives behind this import; keep the module torch-free until asked.
    from railroad.lsp.model import load_frontier_statistics_model

    return LearnedFrontierStatistics(load_frontier_statistics_model(network_file))


def learned_container_find(scene: Any, network_file: Any = None) -> ObjectFindEstimator:
    """Learned container belief: "does THIS container hold the object?".

    A different model from the panorama frontier net: ProcTHOR's
    ``FCNNforObjectSearch`` scores each (room, container, object) triple from
    their sentence embeddings, so it needs the scene's *graph* rather than any
    observation the robot made. `ProcTHORScene.get_object_find_prob_fn` owns the
    plumbing (graph -> features -> net, cached per object) and hands back exactly
    the ``(robot, location, object) -> float`` shape an estimator wraps.

    *network_file* defaults to the trained checkpoint packaged with ProcTHOR, so
    this works out of the box — no weights to supply.

    Note this is belief only: it never sees where objects actually are, so it is
    honest in replay for the same reason every other policy is.
    """
    from railroad.experimental.unknown_search import CallableObjectFind

    if network_file is None:
        from railroad.environment.procthor.learning.utils import (
            get_default_fcnn_model_path,
        )

        network_file = get_default_fcnn_model_path()
    return CallableObjectFind(scene.get_object_find_prob_fn(str(network_file)))
