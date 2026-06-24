"""LSP point-goal operators.

The goal is treated like an object whose spatial location is known in
advance: it lives in ``objects_by_type["goal"]`` (not ``"location"``)
and its coordinates are registered in the location registry from the
start. Two fluents track its status, deliberately distinct:

* ``reachable goal`` — exploring a frontier succeeded, so a route to the
  goal is known to exist beyond it. The robot has *not* yet sensed the
  goal cell. The gated ``move-to-goal`` action drives there along the
  optimistic known-location path. Set by ``lsp-explore`` (success).
* ``revealed goal`` — the goal cell has been *directly observed* and the
  goal is now a real ``location``; the ordinary navigable ``move`` can
  target it and ``move-to-goal`` steps aside. Set by the environment when
  the cell enters the observed map (see :mod:`railroad.lsp.env_mixin`).

Exploring a frontier never relocates the robot.
"""

from __future__ import annotations

from typing import Callable

from railroad.core import Effect, Fluent, Operator
from railroad.operators._utils import Numeric, OptNumeric, _to_numeric

from .frontier_statistics import FrontierStatisticsEstimator

F = Fluent


def construct_lsp_explore_operator(
    statistics: FrontierStatisticsEstimator,
    optimistic_goal_cost: Callable[[str, str], float],
    *,
    speed_cells_per_sec: float = 2.0,
    goal_name: str = "goal",
    min_time: float = 0.1,
    prob_clamp: tuple[float, float] = (0.0, 1.0),
) -> Operator:
    """Construct ``(lsp-explore ?robot ?frontier)`` for point-goal navigation.

    Exploring a frontier succeeds with the estimated ``prob_feasible``.
    The two outcomes complete at different times:

    * **success** at ``success_time = (optimistic_goal_cost + delta_success_cost)
      / speed``. ``optimistic_goal_cost(robot, frontier)`` is the lower-bound
      travel cost to the goal through the frontier — its length on the map with
      all *unseen* space treated as free — and ``delta_success_cost`` is the
      learned/oracle *extra* cost beyond that optimistic bound. The statistics
      only carry the delta, so the bound must be added back here; omitting it
      badly under-costs success. On success the goal is marked ``reachable``.
    * **failure** at ``failure_time = exploration_cost / speed``: the robot
      returns empty-handed.

    Both times are floored at *min_time*. Either way the frontier is marked
    explored and the robot stays put — reaching the goal always happens through
    a real move action.

    The probabilistic effect (where the outcome becomes known) is scheduled at
    ``min(success_time, failure_time)``, the earliest instant the two outcomes
    could diverge. Charging it any earlier would hand the robot privileged
    knowledge of the result before it could possibly have it. The branch
    effects themselves fire at ``success_time - branch_point`` /
    ``failure_time - branch_point`` *after* that point, so each outcome still
    completes at its absolute ``success_time`` / ``failure_time`` (one offset is
    always zero, the other the gap between them).

    Success sets ``reachable goal``, *not* ``revealed goal`` — the goal cell has
    not been sensed yet, so the goal is not yet a real ``location``.
    ``revealed`` is reserved for direct observation.

    Probabilities are clamped to *prob_clamp* (``[0, 1]`` by default).
    """
    speed = max(speed_cells_per_sec, 1e-6)
    lo, hi = prob_clamp

    def _success_time(r: str, f: str) -> float:
        cost = optimistic_goal_cost(r, f) + statistics.get(r, f).delta_success_cost
        return max(min_time, cost / speed)

    def _failure_time(r: str, f: str) -> float:
        return max(min_time, statistics.get(r, f).exploration_cost / speed)

    prob_fn = Numeric(
        lambda r, f: min(hi, max(lo, statistics.get(r, f).prob_feasible))
    )
    branch_point_fn = Numeric(
        lambda r, f: min(_success_time(r, f), _failure_time(r, f))
    )
    success_offset_fn = Numeric(
        lambda r, f: _success_time(r, f)
        - min(_success_time(r, f), _failure_time(r, f))
    )
    failure_offset_fn = Numeric(
        lambda r, f: _failure_time(r, f)
        - min(_success_time(r, f), _failure_time(r, f))
    )

    return Operator(
        name="lsp-explore",
        parameters=[("?r", "robot"), ("?f", "frontier")],
        preconditions=[
            F("at ?r ?f"),
            F("free ?r"),
            ~F("explored ?f"),
            ~F("lock-explore ?f"),
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={F("not free ?r"), F("lock-explore ?f")},
            ),
            Effect(
                time=(branch_point_fn, ["?r", "?f"]),
                resulting_fluents=set(),
                prob_effects=[
                    (
                        (prob_fn, ["?r", "?f"]),
                        [Effect(
                            time=(success_offset_fn, ["?r", "?f"]),
                            resulting_fluents={
                                F("free ?r"),
                                F(f"reachable {goal_name}"),
                                F("explored ?f"),
                                F("not lock-explore ?f"),
                            },
                        )],
                    ),
                    (
                        (1 - prob_fn, ["?r", "?f"]),
                        [Effect(
                            time=(failure_offset_fn, ["?r", "?f"]),
                            resulting_fluents={
                                F("free ?r"),
                                F("explored ?f"),
                                F("not lock-explore ?f"),
                            },
                        )],
                    ),
                ],
            ),
        ],
    )


def construct_move_to_goal_operator(
    move_time: OptNumeric,
    *,
    goal_type: str = "goal",
) -> Operator:
    """Construct a move toward the ``reachable``-but-not-yet-``revealed`` goal.

    This is the planning *landmark* that lets a robot head for the goal
    once exploration has shown it is reachable, before its cell has been
    directly observed. It is gated on ``reachable ?to`` (set by an
    lsp-explore success) and on the goal *not* yet being ``revealed``: the
    moment the goal cell is observed it becomes a real ``location`` and the
    ordinary navigable ``move`` takes over, so this operator deactivates to
    avoid two ways of reaching the same destination.

    The goal's coordinates are known in advance, so the action grounds and
    time-estimates (optimistically) before discovery; reachability of the
    dispatched move is still enforced against the observed map by the
    environment.

    Args:
        move_time: Time or function to compute movement duration.
            Function signature: (robot, from_location, to_location) -> float
        goal_type: Object type holding the goal name.
    """
    move_time_fn = _to_numeric(move_time)
    return Operator(
        # Distinct from the navigable "move" operator: once the goal is
        # observed it also joins the location set, so both operators would
        # otherwise ground the same "move <r> <from> goal" name. A unique,
        # descriptive name keeps get_action_by_name / the planner
        # unambiguous; the ~revealed gate ensures only one is ever active.
        name="move-to-goal",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", goal_type)],
        preconditions=[
            F("at ?r ?from"),
            F("free ?r"),
            F("reachable ?to"),
            ~F("revealed ?to"),
            ~F("just-moved ?r"),
        ],
        effects=[
            Effect(
                time=0,
                resulting_fluents={F("not free ?r"), F("not at ?r ?from")},
            ),
            Effect(
                time=(move_time_fn, ["?r", "?from", "?to"]),
                resulting_fluents={
                    F("free ?r"),
                    F("at ?r ?to"),
                    F("just-moved ?r"),
                },
            ),
            Effect(
                time=(move_time_fn + 0.1, ["?r", "?from", "?to"]),
                resulting_fluents={~F("just-moved ?r")},
            ),
        ],
    )
