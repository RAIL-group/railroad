#!/usr/bin/env python3
"""Planner statistics for the MolmoSpaces box-station domain -- no MuJoCo,
no ProcTHOR scene, no video.

Only molmospace_domain.build_domain() is used (the same real table
distances the video demo uses), and it only needs WORKSTATION_LOCATIONS --
molmospace_scene.build_scene() / molmospace_executor.PlanExecutor are never
touched, so this runs in seconds instead of minutes.

Run: `uv run python scripts/molmospace_stats.py`
"""

import os
import sys
import time
from typing import Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import Domain, PlanResult  # pyrefly: ignore [missing-import]
from decentralized import (  # pyrefly: ignore [missing-import]
    plan_reactive,
    plan_reservation,
    plan_no_op_blind,
    plan_myopic_reactive,
)
from centralized import plan_joint_mcts  # pyrefly: ignore [missing-import]
# plan_joint_astar (centralized.py) is intentionally excluded by default --
# provably optimal but empirically intractable even on this domain (see
# scripts/README.md: memory grew past 6GB in under 30s with no sign of
# converging on a single reduced task). Import and add it below yourself
# if you want to confirm that firsthand.
from molmospace_domain import build_domain, default_robot_goals  # pyrefly: ignore [missing-import]

PLANNERS: Dict[str, Callable[..., PlanResult]] = {
    "reactive": plan_reactive,
    "myopic_reactive": plan_myopic_reactive,
    "reservation": plan_reservation,
    "no_op_blind": plan_no_op_blind,
    "joint_mcts": plan_joint_mcts,
}


def run_once(name: str, planner: Callable[..., PlanResult], domain: Domain) -> dict:
    start = time.perf_counter()
    result = planner(domain, verbose=False)
    elapsed = time.perf_counter() - start
    return {
        "planner": name,
        "success": result.success,
        "cost": result.cost,
        "n_steps": len(result.steps),
        "wall_time_s": elapsed,
        "message": result.message,
        "robot_results": result.robot_results,
        "search_stats": result.search_stats,
    }


def _fmt_cost(cost) -> str:
    return f"{cost:.2f}" if cost is not None else "-"


def print_summary_table(rows: List[dict]) -> None:
    header = f"{'planner':16s} {'success':8s} {'cost':>10s} {'steps':>7s} {'wall_time_s':>12s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['planner']:16s} {str(r['success']):8s} {_fmt_cost(r['cost']):>10s} "
            f"{r['n_steps']:>7d} {r['wall_time_s']:>12.3f}"
            + (f"   {r['message']}" if r["message"] else "")
        )


def print_per_robot_table(rows: List[dict]) -> None:
    header = f"{'planner':16s} {'robot':10s} {'success':8s} {'cost':>10s}  message"
    print(header)
    print("-" * len(header))
    for r in rows:
        for robot, outcome in r["robot_results"].items():
            print(
                f"{r['planner']:16s} {robot:10s} {str(outcome['success']):8s} "
                f"{_fmt_cost(outcome['cost']):>10s}  {outcome['message']}"
            )


def print_search_stats_table(rows: List[dict]) -> None:
    header = f"{'planner':16s} {'robot':10s} {'expanded':>9s} {'generated':>10s}  goal"
    print(header)
    print("-" * len(header))
    for r in rows:
        if not r["search_stats"]:
            print(f"{r['planner']:16s} (no per-task search stats for this planner)")
            continue
        for robot, attempts in r["search_stats"].items():
            for a in attempts:
                expanded = a["states_expanded"] if a["states_expanded"] is not None else "-"
                generated = a["states_generated"] if a["states_generated"] is not None else "-"
                print(f"{r['planner']:16s} {robot:10s} {expanded!s:>9s} {generated!s:>10s}  {a['goal']}")


def main() -> None:
    domain = build_domain(robot_goals=default_robot_goals())

    rows = []
    for name, planner in PLANNERS.items():
        print(f"Running {name}...")
        rows.append(run_once(name, planner, domain))

    print("\n=== Overall (aggregate: did every robot finish) ===")
    print_summary_table(rows)

    print("\n=== Per-robot (each robot's own outcome) ===")
    print_per_robot_table(rows)

    print("\n=== Per-task search effort (one row per planning attempt) ===")
    print_search_stats_table(rows)


if __name__ == "__main__":
    main()
