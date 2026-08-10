#!/usr/bin/env python3
"""
End-to-end demo: plan the box-station gift-wrap task with `railroad`,
then replay the resulting plan as a MobileFranka simulation in MolmoSpaces.

Three independently reusable pieces, wired together here:
  molmospace_scene.py    -- builds the MuJoCo scene (workstation1,
                             workstation2, tool_space, box_station tables;
                             2 gift cubes; scissors; 2 MobileFranka robots).
  molmospace_domain.py    -- builds the symbolic `Domain` (see common.py)
                             from that scene's real table positions.
  molmospace_executor.py  -- replays any `PlanResult.steps` (see common.py)
                             against a built scene and renders a video.

Swap `PLANNER` below for any of decentralized.py's / centralized.py's other
4 planning methods to compare against the same scene, or pass different
`robot_goals` to molmospace_domain.build_domain -- the scene only needs to
be built once.

Run: `uv run python scripts/molmospace_demo.py`
Output: molmospace_demo.mp4
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decentralized import plan_reservation, plan_reactive, plan_myopic_reactive  # pyrefly: ignore [missing-import]
from molmospace_domain import build_domain, default_robot_goals  # pyrefly: ignore [missing-import]
from molmospace_executor import PlanExecutor  # pyrefly: ignore [missing-import]
from molmospace_scene import build_scene  # pyrefly: ignore [missing-import]

PLANNER = plan_reactive
OUTPUT_PATH = "molmospace_demo_reactive.mp4"


def main() -> None:
    print("Building MolmoSpaces scene...")
    scene = build_scene()

    print("\nBuilding symbolic domain from the scene's real table positions...")
    domain = build_domain(scene.location_positions, robot_goals=default_robot_goals())

    print(f"\nPlanning with {PLANNER.__name__}...")
    result = PLANNER(domain, verbose=True)
    print(f"\nsuccess={result.success}  cost={result.cost}  message={result.message!r}")
    if not result.success:
        print("No plan to replay -- exiting.")
        return

    print(f"\nReplaying {len(result.steps)} plan steps in the MolmoSpaces scene...")
    executor = PlanExecutor(scene)
    executor.render_video(result.steps, OUTPUT_PATH)


if __name__ == "__main__":
    main()
