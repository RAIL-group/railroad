"""Symbolic planning `Domain` for the MolmoSpaces box-station scene.

Builds the same box-station gift-wrap domain as
pick_and_place_astar_boxstation.py (workstation1/workstation2 own by each
robot, tool_space/box_station shared), but the `move` operator's duration
comes from the *real* Euclidean distances between the scene's tables
(molmospace_scene.WORKSTATION_LOCATIONS) instead of a hardcoded layout.

Only the domain-specific operators live here (move + the 4 work steps);
pick/place/wait/no_op are deliberately left out — see common.Domain's
docstring — each planning method in decentralized.py/centralized.py builds
whichever variant of those it needs.

`build_domain()` only needs the table *coordinates*, not a compiled
MolmoSpaceScene -- so planner statistics (comparing plan_reactive /
plan_reservation / plan_no_op_blind / plan_joint_astar / plan_joint_mcts,
sweeping goals, timing runs, ...) can import this module and call
build_domain() directly without ever installing ProcTHOR assets or
compiling a MuJoCo model. Only molmospace_executor.PlanExecutor (replaying
a plan as an actual simulation) needs a real built scene.
"""

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from railroad.core import Operator, Effect, Fluent as F
from railroad.operators._utils import Numeric

from common import Domain  # pyrefly: ignore [missing-import]
from molmospace_scene import ROBOT_HOME, WORKSTATION_LOCATIONS  # pyrefly: ignore [missing-import]

# Not specified by the original box-station domain description — tune here.
WRAP_GIFT_TIME = 10.0
CUT_TIME = 20.0
COMPLETE_JOB_TIME = 10.0

# extra_cost_fn (see expected_cost() below) weight -- how heavily the search
# is biased away from successor states where the scissors aren't at
# tool_space. 0 disables the penalty entirely. Only affects the 4
# decentralized planners (they route through CoordinationAStarPlanner);
# plan_joint_astar/plan_joint_mcts never read Domain.extra_cost_fn at all.
SCISSORS_AWAY_PENALTY = 500.0

# Robots travel at this speed (m/s) between tables; move duration = real
# Euclidean distance between two tables' positions divided by this.
ROBOT_SPEED = 0.5


def _make_move_time(location_positions: Dict[str, tuple]) -> Numeric:
    coords = {loc: np.array(pos) for loc, pos in location_positions.items()}

    def _move_time(_robot: str, loc_from: str, loc_to: str) -> float:
        return float(np.linalg.norm(coords[loc_from] - coords[loc_to])) / ROBOT_SPEED

    return Numeric(_move_time)


def construct_accessible_move_operator(location_positions: Dict[str, tuple]) -> Operator:
    """Move with workstation exclusivity enforced by the 'accessible' precondition."""
    move_time = _make_move_time(location_positions)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[
            F("at ?r ?from"), F("free ?r"), ~F("just-moved ?r"),
            F("accessible ?r ?to"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={F("free ?r"), F("at ?r ?to"), F("just-moved ?r")},
            ),
            Effect(
                time=(move_time + 0.1, ["?r", "?from", "?to"]),
                resulting_fluents={~F("just-moved ?r")},
            ),
        ],
    )


def construct_cut_paper_operator() -> Operator:
    return Operator(
        name="cut_paper",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), F("holding ?r scissors"),
            F("at ?gift ?ws"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
            ~F("paper_cut ?r ?gift"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=CUT_TIME, resulting_fluents={F("free ?r"), F("paper_cut ?r ?gift")}),
        ],
    )


def construct_wrap_gift_operator() -> Operator:
    return Operator(
        name="wrap_gift",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), ~F("hand-full ?r"),
            F("paper_cut ?r ?gift"), F("at ?gift ?ws"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
            ~F("paper_wrapped ?r ?gift"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=WRAP_GIFT_TIME, resulting_fluents={F("free ?r"), F("paper_wrapped ?r ?gift")}),
        ],
    )


def construct_cut_ribbon_operator() -> Operator:
    return Operator(
        name="cut_ribbon",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), F("holding ?r scissors"),
            F("paper_wrapped ?r ?gift"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
            ~F("ribbon_cut ?r ?gift"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=CUT_TIME, resulting_fluents={F("free ?r"), F("ribbon_cut ?r ?gift")}),
        ],
    )


def construct_complete_job_operator() -> Operator:
    return Operator(
        name="complete_job",
        parameters=[("?r", "robot"), ("?ws", "location"), ("?gift", "gift")],
        preconditions=[
            F("at ?r ?ws"), F("free ?r"), ~F("hand-full ?r"),
            F("ribbon_cut ?r ?gift"),
            F("is_workstation ?ws"), F("accessible ?r ?ws"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=COMPLETE_JOB_TIME, resulting_fluents={F("free ?r"), F("wrapped_gift ?r ?gift")}),
        ],
    )


def default_robot_goals() -> Dict[str, List[str]]:
    """One gift per robot — robot1 wraps cube_1, robot2 wraps cube_2, both
    returning the scissors to tool_space so the other robot can acquire it.
    """
    return {
        "robot1": ["wrapped_gift robot1 cube_1 & at scissors tool_space"],
        "robot2": ["wrapped_gift robot2 cube_2 & at scissors tool_space"],
    }


def expected_cost(planner, action, successor) -> float:
    """extra_cost_fn for Domain.extra_cost_fn -- called by
    CoordinationAStarPlanner._search (planner_interface.py) on every
    successor state expanded during search, added as the `w` term to
    f = g + h + w. Doesn't change a plan's real reported cost/duration
    (that's still just g); it only biases *which* plan the search prefers.

    Penalizes the specific action of setting the scissors down anywhere
    other than tool_space -- in this domain that means "place robot1
    workstation1 scissors" / "place robot2 workstation2 scissors", the
    intermediate parking move between cut_paper/cut_ribbon and
    wrap_gift/complete_job (those two need empty hands). Checks the
    grounded action name directly (e.g. "place robot1 workstation1
    scissors" -> parts ["place", "robot1", "workstation1", "scissors"])
    rather than the successor state, since that's a much more targeted
    condition than "scissors isn't at tool_space right now" -- the earlier
    version of this function penalized *every* state where the scissors
    were merely being carried or actively in use, not just this one
    parking action. Tune via SCISSORS_AWAY_PENALTY above.
    """
    parts = action.name.split()
    if parts[0] == "place" and parts[3] == "scissors" and parts[2] != "tool_space":
        return SCISSORS_AWAY_PENALTY
    return 0.0


def build_domain(
    location_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    robot_goals: Optional[Dict[str, List[str]]] = None,
    pick_time: float = 1.0,
    place_time: float = 1.0,
) -> Domain:
    """Build the box-station `Domain`, using real table positions for move
    durations.

    `location_positions` defaults to WORKSTATION_LOCATIONS (the scene's own
    layout) -- pass `scene.location_positions` if you have a built
    MolmoSpaceScene and want to guarantee an exact match, but nothing here
    actually requires one; this only needs the table coordinates.
    """
    location_positions = location_positions if location_positions is not None else WORKSTATION_LOCATIONS
    robots = set(ROBOT_HOME.keys())

    objects_by_type: Dict[str, Set[str]] = {
        "robot": robots,
        "location": set(location_positions.keys()),
        "object": {"scissors", "cube_1", "cube_2"},
        "gift": {"cube_1", "cube_2"},
    }

    initial_state: Set[str] = {
        "at scissors tool_space",
        "at cube_1 box_station",
        "at cube_2 box_station",
        "is_workstation workstation1",
        "is_workstation workstation2",
    }
    for robot, home in ROBOT_HOME.items():
        initial_state.add(f"free {robot}")
        initial_state.add(f"at {robot} {home}")
        # Each robot may only enter its own workstation; tool_space/box_station
        # are shared/common to both robots.
        initial_state.add(f"accessible {robot} {home}")
        initial_state.add(f"accessible {robot} tool_space")
        initial_state.add(f"accessible {robot} box_station")

    base_operators = [
        construct_accessible_move_operator(location_positions),
        construct_cut_paper_operator(),
        construct_wrap_gift_operator(),
        construct_cut_ribbon_operator(),
        construct_complete_job_operator(),
    ]

    return Domain(
        objects_by_type=objects_by_type,
        initial_state=initial_state,
        robots=sorted(robots),
        base_operators=base_operators,
        robot_goals=robot_goals if robot_goals is not None else default_robot_goals(),
        contested_resources={"scissors": "tool_space"},
        pick_time=pick_time,
        place_time=place_time,
        extra_cost_fn=expected_cost,
    )
