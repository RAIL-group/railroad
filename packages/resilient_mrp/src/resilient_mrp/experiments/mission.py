# Runs one mission from start to finish and reports how it went.

from typing import Any, Callable

from rich.console import Console
from rich.table import Table

from railroad.core import Action, Fluent as F, get_action_by_name
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner

from resilient_mrp.planning.core import ResilientGraph, parse_state


# At the end of a run, prints who survived and which goals got done.
def print_mission_summary(
    env: SymbolicEnvironment,
    goal_sites: list[str],
    initial_paths: set,
    travel: float | None = None,
) -> None:
    fluents = env.state.fluents
    robots = sorted(env.objects_by_type.get("robot", set()))
    removed_paths = initial_paths - fluents
    console = Console()

    robot_table = Table(title="Robot Outcomes")
    robot_table.add_column("Robot", style="cyan")
    robot_table.add_column("Status")
    for robot in robots:
        operational = F(f"operational {robot}") in fluents
        status = "[green]OPERATIONAL[/green]" if operational else "[red]FAILED[/red]"
        robot_table.add_row(robot, status)

    goal_table = Table(title="Goal Coverage")
    goal_table.add_column("Goal", style="cyan")
    goal_table.add_column("Covered")
    goals_done = sum(1 for g in goal_sites if F(f"safely_visited {g}") in fluents)
    for g in goal_sites:
        done = F(f"safely_visited {g}") in fluents
        goal_table.add_row(g, "[green]YES[/green]" if done else "[red]NO[/red]")

    console.print(robot_table)
    console.print(goal_table)

    mission_ok = goals_done == len(goal_sites)
    status_str = "[green]SUCCESS[/green]" if mission_ok else "[red]INCOMPLETE[/red]"

    # makespan is what the trial cost charges; travel is the secondary metric and exceeds it under
    # concurrency, since it adds up edges the robots walked at the same time
    line = f"\nMission: {status_str}  |  Goals: {goals_done}/{len(goal_sites)}"
    if travel is not None:
        line += f"  |  Travel: {travel:.1f}"
    line += f"  |  Makespan: {env.state.time:.1f}"
    console.print(line)


# stand-in so the dashboard can log a route-policy step, which has no search tree to show
class _NoTracePlanner:
    def get_trace_from_last_mcts_tree(self):
        return ""

    def heuristic(self, state, goal):
        return 0.0


# The next action a route policy takes: mark a goal the robot is standing on, else step along the
# route. Returns None when nothing applies, which ends the mission.
def _route_policy_action(env: SymbolicEnvironment, real_actions: list, goal_sites: list[str],
                         route_policy) -> Action | None:
    mark = next((a for a in real_actions if a.name.startswith("safely_visited")
                 and env.state.satisfies_precondition(a)), None)
    if mark is not None:
        return mark

    applicable = [a for a in real_actions if env.state.satisfies_precondition(a)
                  and a.name.startswith("risk_move")]

    fluents = env.state.fluents
    visited = {g for g in goal_sites if F(f"safely_visited {g}") in fluents}
    # a failure closes the edge it happened on, so re-read the map before routing anything
    route_policy.observe(env.state)
    # operational robots only, so a failed robot drops out and assign() replans with the survivors
    pos, _, _ = parse_state(env.state)
    positions = {r: u for r, u in pos.items() if F(f"operational {r}") in fluents}
    queues = route_policy.assign(positions, visited)

    targets: dict = {}
    for a in applicable:
        _, r, u, v = a.name.split()
        targets.setdefault((r, u), set()).add(v)

    for (r, u), cands in targets.items():
        q = queues.get(r) or []
        mv = route_policy.step_toward(u, r, q[0] if q else None, candidates=cands)
        match = [a for a in applicable if a.name == mv]
        if match:
            return match[0]

    # no free robot has an assigned move: if another robot is still finishing one, an idle robot
    # waits so time advances and it arrives; otherwise nothing more can happen and the mission ends
    in_transit = any(F(f"operational {r}") in fluents and F(f"free {r}") not in fluents
                     for r in env.objects_by_type.get("robot", ()))
    if in_transit:
        return next((a for a in real_actions if a.name.startswith("no_op")
                     and env.state.satisfies_precondition(a)), None)
    return None


# MCTS over the real actions, or over planning_operators when the planner searches a different
# world model than the one it acts in. Returns the planner so the dashboard can show its tree.
def _mcts_action(env: SymbolicEnvironment, real_actions: list, goal: F, planning_operators: list | None,
                 max_iterations: int, max_depth: int, c: float,
                 heuristic_fn: Callable | None, heuristic_multiplier: float,
                 unreachable_penalty: float, dead_end_penalty: float | None = None):
    # dead_end_penalty is a constructor argument, not a call argument: it is a property of how this
    # planner scores a lost branch, and it has to be the same c_fail the trial metric charges.
    if planning_operators is not None:
        planning_env = SymbolicEnvironment(
            state=env.state,
            objects_by_type=env.objects_by_type,
            operators=planning_operators,
        )
        mcts = MCTSPlanner(planning_env.get_actions(), dead_end_penalty=dead_end_penalty)
    else:
        mcts = MCTSPlanner(real_actions, dead_end_penalty=dead_end_penalty)

    action_name = mcts(env.state, goal, max_iterations=max_iterations,
                       c=c, max_depth=max_depth,
                       heuristic_multiplier=heuristic_multiplier,
                       heuristic_fn=heuristic_fn,
                       unreachable_penalty=unreachable_penalty)
    if action_name == "NONE":
        return mcts, None
    return mcts, get_action_by_name(real_actions, action_name)


# Runs one mission and returns (goals safely visited, travel cost). Each step whoever is free acts:
# a route_policy robot follows its route without searching, otherwise MCTS picks the move.
def run_episode(
    env: SymbolicEnvironment,
    goal: F,
    max_iterations: int,
    max_depth: int,
    goal_sites: list[str],
    planning_operators: list | None = None,
    c: float = 100,
    max_steps: int = 50,
    heuristic_fn: Callable | None = None,
    heuristic_multiplier: float = 5.0,
    unreachable_penalty: float = 0.0,
    dead_end_penalty: float | None = None,
    route_policy: Any = None,
    dashboard: Any = None,
    graph: ResilientGraph | None = None,
) -> tuple[int, float]:
    travel = 0.0

    # sum of edge costs actually taken, the secondary metric; makespan (env.state.time) is what the
    # trial cost is built from
    def charge(action_name: str) -> None:
        nonlocal travel
        if graph is None or not action_name.startswith("risk_move"):
            return
        _, _, u, v = action_name.split()
        travel += graph.edges.get((u, v), {}).get("cost", 0.0)

    for _ in range(max_steps):
        if goal.evaluate(env.state.fluents):
            break

        real_actions = env.get_actions()
        if not real_actions:
            break

        if route_policy is not None:
            action = _route_policy_action(env, real_actions, goal_sites, route_policy)
            planner = _NoTracePlanner()
        else:
            planner, action = _mcts_action(env, real_actions, goal, planning_operators,
                                           max_iterations, max_depth, c,
                                           heuristic_fn, heuristic_multiplier,
                                           unreachable_penalty, dead_end_penalty)
        if action is None:
            break

        charge(action.name)
        env.act(action)
        if dashboard is not None:
            dashboard.update(planner, action.name)

    visited = sum(1 for g in goal_sites if F(f"safely_visited {g}") in env.state.fluents)
    return visited, travel
