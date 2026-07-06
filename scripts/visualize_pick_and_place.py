"""
visualize_pick_and_place.py — Simulate and render pre-computed robot plans as a video.

Public API
----------
    simulate_video(robot_plans, initial_state, ...)

    Given per-robot SimpleAction lists and an initial world state, this:
      1. Runs a pure execution simulation (no replanning) with blocking/retry.
      2. Renders every frame with matplotlib and writes an MP4 via cv2.

Usage
-----
    uv run python scripts/visualize_pick_and_place.py            # → pick_and_place.mp4
    uv run python scripts/visualize_pick_and_place.py --speed 8  # 8× speedup

Example from another script
---------------------------
    from visualize_pick_and_place import simulate_video
    from planner_interface import call_planner

    plan_r1 = call_planner(world_state, goal1, objects, ops, "robot1")
    plan_r2 = call_planner(world_state, goal2, objects, ops, "robot2")

    simulate_video(
        robot_plans   = {"robot1": plan_r1, "robot2": plan_r2},
        initial_state = world_state,
        task_starts   = {"robot1": 0.0, "robot2": 20.0},
        out_path      = "my_sim.mp4",
    )
"""

from __future__ import annotations

import argparse
import heapq
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner_interface import SimpleAction  # pyrefly: ignore [missing-import]

# ── Layout constants (specific to the 3-location gift-wrap scene) ────────────

_LOC_X: Dict[str, float] = {}         # populated in main() or by caller
_LOC_LABEL: Dict[str, str] = {}
_LOC_BOX_CFG: Dict[str, dict] = {}

_DEFAULT_LOC_X = {
    "workstation1": 0.0,
    "tool_space":   5.0,
    "workstation2": 10.0,
}
_DEFAULT_LOC_LABEL = {
    "workstation1": "Workstation 1",
    "tool_space":   "Tool Space",
    "workstation2": "Workstation 2",
}
_DEFAULT_LOC_BOX_CFG = {
    "workstation1": {"face": "#E8F5E9", "edge": "#388E3C"},
    "tool_space":   {"face": "#FFF8E1", "edge": "#E65100"},
    "workstation2": {"face": "#E3F2FD", "edge": "#1565C0"},
}

BOX_W, BOX_H = 1.7, 2.4

ROBOT_COLORS = ["#1565C0", "#B71C1C", "#558B2F", "#6A1B9A"]
ROBOT_Y_POSITIONS = [0.45, -0.45, 0.45, -0.45]    # y for up to 4 robots
LABEL_Y_OFFSETS   = [+1.55, -1.55, +1.55, -1.55]
NAME_Y_OFFSETS    = [+1.85, -1.85, +1.85, -1.85]

STATUS_CLR = {
    "moving":  "#42A5F5",
    "working": "#66BB6A",
    "waiting": "#FF7043",
    "idle":    "#BDBDBD",
}

FIG_W, FIG_H, DPI = 14, 7, 100    # → 1400 × 700 px

# ── Internal event type ───────────────────────────────────────────────────────

@dataclass(order=True)
class _Ev:
    time:       float
    robot_id:   str
    event_type: str = field(compare=False)
    data:       Any = field(compare=False)


def _check_preconditions(action: SimpleAction, state: Set[str]) -> Optional[str]:
    for pre in action.preconditions:
        if pre.startswith("not "):
            if pre[4:] in state:
                return pre
        else:
            if pre not in state:
                return pre
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Execution
# ═══════════════════════════════════════════════════════════════════════════════

def execute_plans(
    robot_plans: Dict[str, List[SimpleAction]],
    initial_state: Set[str],
    task_starts: Optional[Dict[str, float]] = None,
) -> Tuple[List[dict], float]:
    """Execute pre-computed plans and return (event_log, sim_end_time).

    No planning is done here — actions are taken exactly as given.
    If a precondition fails (e.g. scissors not yet available) the robot
    blocks and retries each time another robot completes an action.

    Args:
        robot_plans:  per-robot ordered list of SimpleActions to execute.
        initial_state: set of fluent strings describing the starting world.
        task_starts:  sim-time at which each robot begins its plan
                      (default 0.0 for every robot).

    Returns:
        (log, sim_end) where log is a flat list of event dicts and
        sim_end is the final simulation time + a 3-second tail.
    """
    log: List[dict]   = []
    world_state        = set(initial_state)
    robots             = list(robot_plans.keys())
    plans              = {r: list(acts) for r, acts in robot_plans.items()}
    blocked            = {r: False for r in robots}
    queue: List[_Ev]  = []
    starts             = task_starts or {r: 0.0 for r in robots}

    for robot, t0 in starts.items():
        heapq.heappush(queue, _Ev(t0, robot, "TASK_START", None))

    def try_start(t: float, robot: str) -> None:
        if not plans[robot]:
            blocked[robot] = False
            log.append({"t": t, "ev": "idle", "robot": robot})
            return
        action = plans[robot][0]
        fail   = _check_preconditions(action, world_state)
        if fail is None:
            blocked[robot] = False
            for d in action.del_effects:
                world_state.discard(d)
            log.append({
                "t": t, "ev": "start", "robot": robot,
                "action": action.name, "dur": action.duration,
            })
            heapq.heappush(queue, _Ev(t + action.duration, robot, "ACTION_END", action))
        else:
            blocked[robot] = True
            log.append({"t": t, "ev": "waiting", "robot": robot})

    while queue:
        ev            = heapq.heappop(queue)
        t, robot      = ev.time, ev.robot_id

        if ev.event_type == "TASK_START":
            log.append({"t": t, "ev": "arrival", "robot": robot})
            try_start(t, robot)

        elif ev.event_type == "ACTION_END":
            action: SimpleAction = ev.data
            for f in action.add_effects:
                world_state.add(f)
            log.append({"t": t, "ev": "end", "robot": robot, "action": action.name})
            plans[robot].pop(0)
            try_start(t, robot)
            for other in robots:
                if other != robot and blocked[other]:
                    try_start(t, other)

    end_t = (max(e["t"] for e in log) + 3.0) if log else 3.0
    return log, end_t


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — State interpolation (log → visual state at time t)
# ═══════════════════════════════════════════════════════════════════════════════

def _ease(p: float) -> float:
    p = max(0.0, min(1.0, p))
    return 3 * p * p - 2 * p * p * p


def get_state_at(
    t: float,
    log: List[dict],
    robot_init_locs: Dict[str, str],
    robot_y: Dict[str, float],
    loc_x: Dict[str, float],
) -> dict:
    """Compute positions and labels for every entity at simulation time t."""
    cur_loc  = dict(robot_init_locs)
    active   = {r: None for r in robot_init_locs}
    mv_from  = {}
    mv_to    = {}
    status   = {r: "idle" for r in robot_init_locs}
    sc_holder: Optional[str] = None
    sc_loc: str = "tool_space"

    for e in log:
        if e["t"] > t:
            break
        robot = e.get("robot", "")
        ev    = e["ev"]

        if ev == "start":
            active[robot] = e
            parts = e["action"].split()
            verb  = parts[0]
            status[robot] = "moving" if verb == "move" else "working"
            if verb == "move":
                mv_from[robot], mv_to[robot] = parts[2], parts[3]
            if verb == "pick" and "scissors" in parts:
                sc_holder, sc_loc = robot, ""

        elif ev == "end":
            parts = e["action"].split()
            verb  = parts[0]
            if verb == "move":
                cur_loc[robot] = mv_to.get(robot, cur_loc[robot])
            if verb == "place" and "scissors" in parts:
                sc_holder, sc_loc = None, parts[2]
            active[robot] = None
            status[robot] = "idle"

        elif ev in ("waiting", "idle"):
            active[robot] = None
            status[robot] = ev

    out: dict = {}
    for robot in robot_init_locs:
        act = active[robot]
        if act and act["action"].startswith("move"):
            p   = _ease((t - act["t"]) / max(act["dur"], 1e-6))
            x   = loc_x[mv_from[robot]] + p * (loc_x[mv_to[robot]] - loc_x[mv_from[robot]])
            lbl = f"→ {mv_to[robot].replace('_', ' ')}"
        else:
            x   = loc_x[cur_loc[robot]]
            if act:
                verb = act["action"].split()[0]
                pct  = int(_ease((t - act["t"]) / max(act["dur"], 1e-6)) * 100)
                lbl  = f"{verb.replace('_', ' ')} {pct}%"
            elif status[robot] == "waiting":
                lbl = "waiting…"
            else:
                lbl = "idle"
        out[robot] = {"x": x, "y": robot_y[robot], "label": lbl, "status": status[robot]}

    # Scissors: snap to robot (slightly offset) when held, otherwise at location
    if sc_holder and sc_holder in out:
        ry = out[sc_holder]["y"]
        out["scissors"] = {"x": out[sc_holder]["x"], "y": ry + (-0.28 if ry > 0 else 0.28)}
    else:
        out["scissors"] = {"x": loc_x.get(sc_loc, 5.0), "y": 0.0}

    # Task-arrival banner (visible for 4 sim-seconds after event)
    banner = ""
    for e in reversed(log):
        if e["ev"] == "arrival" and e["t"] <= t:
            if t - e["t"] < 4.0:
                banner = f"Task started for {e['robot']}  (t = {e['t']:.0f} s)"
            break
    out["banner"] = banner
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — Rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_figure(robot_colors: Dict[str, str], loc_x: Dict[str, float]) -> tuple:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("#F0F4F8")
    ax.set_facecolor("#E8EDF2")
    ax.set_xlim(-1.8, max(loc_x.values()) + 1.8)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")

    loc_cfg = _LOC_BOX_CFG or _DEFAULT_LOC_BOX_CFG
    loc_lbl = _LOC_LABEL    or _DEFAULT_LOC_LABEL
    for loc, cfg in loc_cfg.items():
        cx = loc_x[loc]
        ax.add_patch(FancyBboxPatch(
            (cx - BOX_W / 2, -BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.08", lw=2,
            edgecolor=cfg["edge"], facecolor=cfg["face"], zorder=2,
        ))
        ax.text(cx, BOX_H / 2 + 0.12, loc_lbl.get(loc, loc),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=cfg["edge"], zorder=3)

    locs = sorted(loc_x, key=loc_x.get)
    for a, b in zip(locs, locs[1:]):
        ax.plot([loc_x[a] + BOX_W / 2, loc_x[b] - BOX_W / 2], [0, 0],
                color="#90A4AE", lw=2, ls="--", zorder=1)

    legend = [mpatches.Patch(facecolor=c, label=r.replace("robot", "Robot "))
              for r, c in robot_colors.items()]
    legend.append(mpatches.Patch(facecolor="#FFA000", label="scissors"))
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Gift Wrap — Two-Robot Coordination with A* + Expected Cost",
                 fontsize=11, fontweight="bold", color="#263238", pad=6)
    return fig, ax


def _create_artists(
    ax,
    robot_colors: Dict[str, str],
    robot_y: Dict[str, float],
    label_y: Dict[str, float],
    name_y: Dict[str, float],
    robot_init_locs: Dict[str, str],
    loc_x: Dict[str, float],
) -> dict:
    artists: dict = {}
    for robot, clr in robot_colors.items():
        x0 = loc_x[robot_init_locs[robot]]
        ry = robot_y[robot]
        ly = label_y[robot]
        ny = name_y[robot]

        c = plt.Circle((x0, ry), 0.30, color=clr, ec="white", lw=2, zorder=5)
        ax.add_patch(c)
        artists[f"{robot}_circle"] = c
        artists[f"{robot}_inner"] = ax.text(
            x0, ry, robot[-1], ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=6)
        artists[f"{robot}_name"] = ax.text(
            x0, ny, robot.replace("robot", "Robot "),
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=clr, zorder=6)
        artists[f"{robot}_label"] = ax.text(
            x0, ly, "idle", ha="center", va="center",
            fontsize=7.5, color="#424242", zorder=6)
        bar_y = BOX_H / 2 + 0.44 if ry > 0 else -(BOX_H / 2 + 0.44)
        bar = plt.Rectangle((x0 - 0.55, bar_y - 0.055), 1.1, 0.11,
                             color=STATUS_CLR["idle"], zorder=4)
        ax.add_patch(bar)
        artists[f"{robot}_bar"] = bar

    sc = ax.scatter([loc_x.get("tool_space", 5.0)], [0.0],
                    s=400, marker="*", color="#FFA000",
                    edgecolors="#E65100", linewidths=1.5, zorder=7)
    artists["scissors"] = sc
    artists["clock"] = ax.text(
        0.01, 0.97, "t = 0.0 s", transform=ax.transAxes,
        ha="left", va="top", fontsize=12, fontweight="bold", color="#263238",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B0BEC5", alpha=0.9))
    artists["banner"] = ax.text(
        0.5, 0.03, "", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=9, color="#E65100",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E1", ec="#E65100", alpha=0.0))
    return artists


def _update_artists(
    state: dict, t: float, artists: dict,
    robot_y: Dict[str, float], label_y: Dict[str, float], name_y: Dict[str, float],
) -> None:
    for robot, rs in state.items():
        if robot in ("scissors", "banner"):
            continue
        x = rs["x"]
        ry, ly, ny = robot_y[robot], label_y[robot], name_y[robot]
        artists[f"{robot}_circle"].center = (x, ry)
        artists[f"{robot}_inner"].set_position((x, ry))
        artists[f"{robot}_name"].set_position((x, ny))
        artists[f"{robot}_label"].set_position((x, ly))
        artists[f"{robot}_label"].set_text(rs["label"])
        bar_y = BOX_H / 2 + 0.44 if ry > 0 else -(BOX_H / 2 + 0.44)
        artists[f"{robot}_bar"].set_xy((x - 0.55, bar_y - 0.055))
        artists[f"{robot}_bar"].set_color(STATUS_CLR[rs["status"]])

    sc = state["scissors"]
    artists["scissors"].set_offsets([[sc["x"], sc["y"]]])
    artists["clock"].set_text(f"t = {t:.1f} s")
    banner = state["banner"]
    artists["banner"].set_text(banner)
    artists["banner"].get_bbox_patch().set_alpha(0.85 if banner else 0.0)


def render_video(
    log: List[dict],
    sim_end: float,
    robot_init_locs: Dict[str, str],
    out_path: str = "simulation.mp4",
    fps: int = 30,
    speedup: float = 5.0,
    loc_x: Optional[Dict[str, float]] = None,
    robot_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Render an event log produced by execute_plans() to an MP4 file.

    Args:
        log:            event log from execute_plans()
        sim_end:        simulation end time (also from execute_plans())
        robot_init_locs: starting location for each robot
        out_path:       output file path
        fps:            video frame rate
        speedup:        simulation time per video second
        loc_x:          x-coordinate of each location (default: gift-wrap layout)
        robot_colors:   hex colour per robot (auto-assigned by default)
    """
    loc_x = loc_x or _DEFAULT_LOC_X
    robots = list(robot_init_locs.keys())
    robot_colors = robot_colors or {r: ROBOT_COLORS[i % len(ROBOT_COLORS)]
                                    for i, r in enumerate(robots)}
    robot_y  = {r: ROBOT_Y_POSITIONS[i % len(ROBOT_Y_POSITIONS)] for i, r in enumerate(robots)}
    label_y  = {r: LABEL_Y_OFFSETS[i % len(LABEL_Y_OFFSETS)]     for i, r in enumerate(robots)}
    name_y   = {r: NAME_Y_OFFSETS[i % len(NAME_Y_OFFSETS)]       for i, r in enumerate(robots)}

    n_frames = max(1, int(sim_end / speedup * fps))
    dt       = sim_end / n_frames

    fig, ax = _setup_figure(robot_colors, loc_x)
    artists = _create_artists(ax, robot_colors, robot_y, label_y, name_y,
                               robot_init_locs, loc_x)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()

    w_px = int(fig.get_figwidth()  * DPI)
    h_px = int(fig.get_figheight() * DPI)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (w_px, h_px))

    print(f"Rendering {n_frames} frames ({speedup}×) → {out_path}")
    for i in range(n_frames):
        t     = i * dt
        state = get_state_at(t, log, robot_init_locs, robot_y, loc_x)
        _update_artists(state, t, artists, robot_y, label_y, name_y)
        fig.canvas.draw()
        buf   = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(h_px, w_px, 4)[:, :, :3]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i % (fps * 5) == 0:
            print(f"  frame {i}/{n_frames}  (t={t:.1f}s)")

    writer.release()
    plt.close(fig)
    print(f"Saved → {out_path}  ({os.path.getsize(out_path) // 1024} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_video(
    robot_plans: Dict[str, List[SimpleAction]],
    initial_state: Set[str],
    robot_init_locs: Optional[Dict[str, str]] = None,
    task_starts: Optional[Dict[str, float]] = None,
    out_path: str = "simulation.mp4",
    fps: int = 30,
    speedup: float = 5.0,
    loc_x: Optional[Dict[str, float]] = None,
    robot_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Execute pre-computed plans and render a video.

    Args:
        robot_plans:     per-robot lists of SimpleActions to execute.
        initial_state:   set of fluent strings describing the starting world.
        robot_init_locs: starting location for each robot. Defaults to the
                         first 'at <robot> <loc>' fluent found in initial_state.
        task_starts:     sim-time at which each robot begins its plan
                         (default 0.0 for every robot).
        out_path:        output video file path (.mp4).
        fps:             video frame rate (default 30).
        speedup:         how many simulation seconds equal one video second
                         (default 5.0 → 5× faster than wall-clock).
        loc_x:           x-coordinate of each location name. Defaults to the
                         3-location gift-wrap layout (workstation1=0, etc.)
        robot_colors:    hex colour per robot. Auto-assigned if not provided.
    """
    # Infer initial locations from world state if not given explicitly
    if robot_init_locs is None:
        robot_init_locs = {}
        for robot in robot_plans:
            for fluent in initial_state:
                parts = fluent.split()
                if parts[0] == "at" and parts[1] == robot:
                    robot_init_locs[robot] = parts[2]
                    break

    print("Executing plans…")
    log, sim_end = execute_plans(robot_plans, initial_state, task_starts)
    print(f"  {len(log)} events, sim ends at t={sim_end - 3:.1f}s")

    render_video(log, sim_end, robot_init_locs, out_path, fps, speedup,
                 loc_x=loc_x, robot_colors=robot_colors)


# ═══════════════════════════════════════════════════════════════════════════════
# Demo — called when run directly
# ═══════════════════════════════════════════════════════════════════════════════

def main(fps: int = 30, speedup: float = 5.0, out_path: str = "pick_and_place.mp4") -> None:
    """Compute plans for the gift-wrap scenario and render a video."""
    from pick_and_place_astar import (  # pyrefly: ignore [missing-import]
        LOCATION_COORDS, PLANNER, TASKS,
        expected_cost, initial_world_state, my_operators, objects_by_type,
    )
    from planner_interface import call_planner  # pyrefly: ignore [missing-import]

    loc_x = {k: float(v[0]) for k, v in LOCATION_COORDS.items()}

    # ── Plan for each robot ──────────────────────────────────────────────────
    # Apply each robot's plan effects to the state before planning the next one
    # so later robots can reason about earlier robots' committed futures.
    broadcast = set(initial_world_state)
    robot_plans: Dict[str, List[SimpleAction]] = {}
    task_starts: Dict[str, float] = {}

    for arrival_t, robot, goal in TASKS:
        gift_m = re.search(r"gift\d+", goal)
        objs   = {**objects_by_type, "gift": {gift_m.group()}} if gift_m else objects_by_type
        plan   = call_planner(broadcast, goal, objs, my_operators, robot,
                              planner_cls=PLANNER, extra_cost_fn=expected_cost)
        if not plan:
            print(f"WARNING: could not plan for {robot}")
            continue
        robot_plans[robot] = plan
        task_starts[robot] = arrival_t
        # Fold this robot's future effects into the broadcast state
        for action in plan:
            for d in action.del_effects: broadcast.discard(d)
            for f in action.add_effects:  broadcast.add(f)

    simulate_video(
        robot_plans    = robot_plans,
        initial_state  = initial_world_state,
        task_starts    = task_starts,
        out_path       = out_path,
        fps            = fps,
        speedup        = speedup,
        loc_x          = loc_x,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fps",   type=int,   default=30)
    p.add_argument("--speed", type=float, default=5.0)
    p.add_argument("--out",   type=str,   default="pick_and_place.mp4")
    a = p.parse_args()
    main(fps=a.fps, speedup=a.speed, out_path=a.out)
