# Runs the experiments: demo, then benchmark.
# The command for each is written above its section.

import time
from dataclasses import replace
from itertools import product
from pathlib import Path

from railroad.bench import benchmark, BenchmarkCase
from railroad.dashboard import PlannerDashboard
from rich.console import Console

from resilient_mrp.experiments import (
    ALL_PLANNERS, Spec, build_instance, run_trial, start_trial,
)
from resilient_mrp.experiments.mission import print_mission_summary
from resilient_mrp.analysis.graph_viz import GraphVisualizer

RISK_MULTIPLIER: list[float] = [1.0, 2.0, 3.0]

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_DEMO_VIDEO_DIR = _RESULTS_DIR / "demo_videos"
_LABEL_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# Writes the planner's name into the top-left of a saved video, using ffmpeg.
# Leaves the video as it was if ffmpeg or the font is missing.
def _overlay_label(path: Path, label: str) -> None:
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not Path(_LABEL_FONT).exists():
        return
    text = label.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    drawtext = (f"drawtext=fontfile={_LABEL_FONT}:text='{text}':x=16:y=14:fontsize=30:"
                f"fontcolor=black:box=1:boxcolor=white@0.7:boxborderw=10")
    tmp = path.with_suffix(".labeled.mp4")
    try:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
                        "-vf", drawtext, "-codec:a", "copy", str(tmp)],
                       check=True, capture_output=True)
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"label overlay skipped ({type(e).__name__})")


def _fluent_filter(f):
    return any(kw in f.name for kw in ["at", "safely_visited", "operational"])


def _render_graph(inst, show: bool) -> None:
    spec = inst.spec
    title = (
        f"{spec.graph_type} graph (n={spec.graph_size}, risk_scale={spec.risk_scale})\n"
        f"{len(inst.graph.nodes)} nodes · {len(inst.graph.edges) // 2} edges · "
        f"{len(inst.goal_sites)} goals · {len(inst.robots)} robots"
    )
    save_path = _RESULTS_DIR / "diagrams" / f"{spec.graph_type}_n{spec.graph_size}_scale{spec.risk_scale}.png"
    GraphVisualizer(inst.graph).render(title, save_path, show=show)


# Demos: watch a mission, or record videos. One graph, one or more planners.
# Runs whatever is uncommented under __main__ at the bottom.
# uv run python packages/resilient_mrp/scripts/playground.py

# With more than one planner they all hit the same failures, so the videos can be compared.
# video: a filename for one planner, or True to name a file per planner.
def run_demo(spec: Spec,
             planners: tuple[str, ...] = ("failure_aware_split",),
             video: str | bool | None = None,
             show_dashboard: bool = False,
             show_graph: bool = False,
             render_graph: bool = True) -> None:

    inst = build_instance(spec)
    initial_paths = inst.graph.get_available_path_fluents()
    if render_graph:
        _render_graph(inst, show=show_graph)

    # planners only compare if they hit the same failures
    exec_seed = spec.seed if len(planners) > 1 else None

    for planner in planners:
        if video is True:
            out = _DEMO_VIDEO_DIR / f"{spec.graph_type}_n{spec.graph_size}_seed{spec.seed}_{planner}.mp4"
        elif isinstance(video, str):
            out = _DEMO_VIDEO_DIR / video
        else:
            out = None

        env = start_trial(inst, exec_seed)
        if show_dashboard or out is not None:
            # Live dashboard when watching; either way the run is recorded so a video can be saved.
            with PlannerDashboard(inst.goal_fluent, env, fluent_filter=_fluent_filter,
                                  force_interactive=(show_dashboard and out is None)) as dashboard:
                outcome = run_trial(inst, planner, env, dashboard=dashboard)
                if len(planners) > 1:
                    print(f"\n=== {planner}  |  travel {outcome.travel:.1f} ===")
                if out is not None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    dashboard.save_video(str(out), location_coords=inst.graph.node_coords)
                    _overlay_label(out, planner)
                    print(f"Video saved: {out}")
        else:
            outcome = run_trial(inst, planner, env)

        print_mission_summary(outcome.env, inst.goal_sites, initial_paths, outcome.travel)


# Benchmark: statistical performance across a risk-scale sweep.
# uv run railroad benchmarks run   --tags resilient_mrp   --include packages/resilient_mrp/scripts/playground.py  --parallel 2 
# uv run railroad benchmarks dashboard

# All planners see the same graph (size, scale, trial) for a fair comparison.
BENCH_SPEC = Spec(
    graph_type="sctp_island",   # "sctp_random" or "sctp_island" or "small_scale"
    graph_size=10,
    num_robots=2,
    num_goals=2,
    mcts_iterations=10000,
    max_depth=40,
    max_steps=50,
)
_BENCH_GRAPH_SIZES = [10]
_BENCH_NUM_TRIALS = 100
_BENCH_BASE_SEED = 43
_BENCH_RUNS_PER_TOPOLOGY = 10   # runs per graph; number of graphs = trials / this, so 100/10 = 10
_BENCH_PLANNER_KEYS = ALL_PLANNERS
# Both failure models, because which baseline wins depends on it rather than on the planners.
_BENCH_BLOCKING = [True, False]


@benchmark(
    name="risk_scale_sweep",
    description="Baselines vs failure_aware planners across a risk-scale sweep.",
    tags=["resilient_mrp", "risk_sweep"],
    repeat=_BENCH_NUM_TRIALS,
    timeout=600.0,
)
def bench_risk_scale_sweep(case: BenchmarkCase) -> dict:
    planner = case.params["planner"]
    trial = case.repeat_idx

    # One seed gives one graph with one set of goal sites, so every planner on trial t solves the
    # same problem. Each block of RUNS_PER_TOPOLOGY trials in a row shares a graph.
    topo_seed = _BENCH_BASE_SEED + trial // _BENCH_RUNS_PER_TOPOLOGY
    inst = build_instance(replace(
        BENCH_SPEC,
        graph_type=case.params["graph_type"],
        graph_size=case.params["graph_size"],
        risk_scale=case.params["risk_scale"],
        blocks_on_failure=case.params["blocks_on_failure"],
        seed=topo_seed,
    ))

    # record the run so each one gets a clickable log.html in the dashboard
    console = Console(record=True)
    t0 = time.perf_counter()
    # every planner on this trial hits the same failures, and each trial differs, so the two
    # baselines are not just repeating one run over and over
    env = start_trial(inst, _BENCH_BASE_SEED + trial)
    with PlannerDashboard(inst.goal_fluent, env, fluent_filter=_fluent_filter, print_on_exit=False,
                          force_interactive=False, console=console) as dashboard:
        outcome = run_trial(inst, planner, env, dashboard=dashboard)
        dashboard.print_history()

    wall_time = time.perf_counter() - t0

    # makespan:   the clock at the end when the last robot stops
    # travel:     every edge any robot travels, added up. it is larger than makespan when they overlap.
    # trial_cost: this run's score. Averaging it over a case's runs gives the expected cost.
    plan_cost = outcome.makespan  # primary cost is makespan
    return {
        "success": outcome.success,
        "failed": not outcome.success,
        "goals_visited": outcome.visited,
        "num_goals": outcome.num_goals,
        "num_robots": len(inst.robots),   # logged, not assumed, so a team sweep groups correctly
        "exec_seed": _BENCH_BASE_SEED + trial,
        "plan_cost": plan_cost,
        "travel": outcome.travel,  # secondary: total edge cost walked
        "cost_at_failure": (outcome.travel if not outcome.success else None),
        "failure_time": (outcome.makespan if not outcome.success else None),
        "c_fail": inst.c_fail,
        "trial_cost": outcome.trial_cost,
        "topo_seed": topo_seed,   # which graph this trial ran on, so cost can be read per graph
        "blocks_on_failure": inst.spec.blocks_on_failure,  # logged: it reorders the planners
        "wall_time": wall_time,
        "log_html": console.export_html(inline_styles=True),
    }


# One case per (graph size, risk scale, failure model, planner). Generated graphs only for now.
def _bench_cases() -> list:
    # the hand-built 7-node graph is off; generated graphs only
    #small = [{"graph_type": "small_scale", "graph_size": 7, "risk_scale": s, "planner": p}
    #         for s, p in product(RISK_MULTIPLIER, _BENCH_PLANNER_KEYS)]
    sctp = [{"graph_type": BENCH_SPEC.graph_type, "graph_size": gs, "risk_scale": s,
             "blocks_on_failure": b, "planner": p}
            for gs, s, b, p in product(_BENCH_GRAPH_SIZES, RISK_MULTIPLIER,
                                       _BENCH_BLOCKING, _BENCH_PLANNER_KEYS)]
    return sctp

bench_risk_scale_sweep.add_cases(_bench_cases())


if __name__ == "__main__":

    # Single mission with the live dashboard and graph.
    run_demo(Spec(graph_type="sctp_random", graph_size=20, num_robots=2, num_goals=3,
                  risk_scale=2.0, seed=99),
             show_dashboard=True, show_graph=True, video="mission_test.mp4")

    # sctp_island: graph_size is the island count, not a vertex count
    #run_demo(Spec(graph_type="sctp_island", graph_size=3, num_robots=2, num_goals=2,
    #              risk_scale=2.0, seed=44),
    #         show_dashboard=True, show_graph=True, video="mission_test_island.mp4")

    # One video per planner on the same graph and the same failure draws.
    #for seed in (43, 7):
    #    run_demo(Spec(graph_type="sctp_random", graph_size=20, num_robots=2, num_goals=2,
    #                  risk_scale=2.0, seed=seed),
    #             planners=ALL_PLANNERS, video=True, render_graph=False)
