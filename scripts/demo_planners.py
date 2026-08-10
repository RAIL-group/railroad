"""Example: building a Domain externally and calling each planning method.

Reuses the box-station gift-wrap world (locations, gift boxes, the
cut_paper work operator) from pick_and_place_astar_boxstation.py as the
"domain built outside" — this file only adapts it into the generic Domain
contract from common.py and calls each of the 5 planning methods against it.

robot_goals is a *queue* per robot here (2 tasks each), matching the
ROBOT_TASKS shape from pick_and_place_astar_boxstation.py — the
decentralized methods (plan_reactive/plan_reservation/plan_no_op_blind)
work through each robot's whole queue in one call. Uses the reduced
paper_cut-only goal per task (not the full wrapped_gift workflow) so this
stays fast — some of these methods take 100+ seconds *per task* on the full
goal, and plan_joint_astar doesn't finish even on a single reduced task (see
below). Centralized methods (plan_joint_mcts) only ever plan for each
robot's *first* queued task — see centralized.py's _combined_goal.

Run directly: `uv run scripts/demo_planners.py`
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pick_and_place_astar_boxstation import (  # pyrefly: ignore [missing-import]
    objects_by_type,
    initial_world_state,
    move_op,
    cut_paper_op,
    wrap_gift_op,
    cut_ribbon_op,
    complete_job_op,
)

from common import Domain  # pyrefly: ignore [missing-import]
from decentralized import (  # pyrefly: ignore [missing-import]
    plan_reactive,
    plan_reservation,
    plan_no_op_blind,
)
from centralized import plan_joint_mcts  # pyrefly: ignore [missing-import]
# plan_joint_astar (provably optimal, exhaustive) is intentionally not run
# here — it's empirically intractable even at this reduced scope (measured:
# memory grew past 6GB in under 30s with no sign of converging). Its call
# signature is identical to plan_joint_mcts's; see centralized.py.


def build_domain() -> Domain:
    return Domain(
        objects_by_type={
            **objects_by_type,
            "gift": {"gift1", "gift2", "gift4", "gift5"},
            "object": {"scissors", "gift1", "gift2", "gift4", "gift5"},
        },
        initial_state=initial_world_state,
        robots=["robot1", "robot2"],
        base_operators=[move_op, cut_paper_op, wrap_gift_op, cut_ribbon_op, complete_job_op],
        robot_goals={
            "robot1": [
                "wrapped_gift robot1 gift1 & at scissors tool_space",
                "wrapped_gift robot1 gift2 & at scissors tool_space",
            ],
            "robot2": [
                "wrapped_gift robot2 gift4 & at scissors tool_space",
                "wrapped_gift robot2 gift5 & at scissors tool_space",
            ],
        },
        contested_resources={"scissors": "tool_space"},
    )


def main() -> None:
    domain = build_domain()

    methods = [
        ("reactive", plan_reactive),
        ("reservation", plan_reservation),
        ("no_op_blind", plan_no_op_blind),
        # ("joint_mcts", plan_joint_mcts),
    ]

    for name, fn in methods:
        result = fn(domain, verbose=True)
        status = "OK" if result.success else "FAILED"
        cost = f"{result.cost:.1f}" if result.cost is not None else "-"
        print(f"{name:24s} {status:8s} cost={cost:>8s}  steps={len(result.steps):3d}  {result.message}")


if __name__ == "__main__":
    main()
