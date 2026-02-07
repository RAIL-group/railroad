"""
Benchmark: Heterogeneous Robots in an environment.
"""

import time
import itertools
import numpy as np
from railroad.core import Fluent as F, State, get_action_by_name
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard
from railroad.experimental.environment import EnvironmentInterface, SimpleEnvironment
from railroad import operators
from rich.console import Console
from railroad.bench import benchmark, BenchmarkCase

SKILLS_TIME = {
    'rover': {
        'pick': 10,
        'place': 10,
        'search': 10
    },
    'crawler': {
        'pick': 10,
        'place': 10,
        'search': 10
    },
    'drone': {
        'search': 10
    }  # drone can only search (in addition to move)
}


class DemoEnvironment(SimpleEnvironment):

    def _get_move_cost_fn(self):
        """Return a function that computes movement time between locations."""
        def get_move_time(robot, loc_from, loc_to):
            distance = np.linalg.norm(
                self.locations[loc_from] - self.locations[loc_to]
            )
            return 0.5 * distance if robot == 'drone' else distance  # drone is faster
        return get_move_time

    def get_skills_time_fn(self, skill_name: str):
        if skill_name == 'move':
            return super()._get_move_cost_fn()
        else:
            def get_skill_time(robot_name, *args, **kwargs):
                return SKILLS_TIME[robot_name].get(skill_name, float('inf'))
            return get_skill_time


@benchmark(
    name="heterogeneous_robots_search_and_deliver",
    description="Heterogeneous robots must search different locations to find supplies and bring to the base.",
    tags=["multi-agent", "heterogeneous", "search"],
    timeout=120.0,
)
def bench_heterogeneous_robots(case: BenchmarkCase):
    # Define locations with coordinates (for move cost calculation)
    locations = {
        "start": np.array([0, 0]),
        "location1": np.array([10, 9]),
        "location2": np.array([9, 0]),
        "location3": np.array([1, 2]),
        "location4": np.array([1, 10]),
    }
    objects_at_locations = {
        "start": {"object": set()},
        "location1": {"object": set()},
        "location2": {"object": {"supplies"}},
        "location3": {"object": set()},
        "location4": {"object": set()},
    }

    available_robots = ['rover', 'drone', 'crawler']

    # Initialize environment
    robot_locations = {available_robots[i]: "start" for i in range(min(case.num_robots, len(available_robots)))}
    env = DemoEnvironment(locations, objects_at_locations, robot_locations)

    initial_fluents = {
        F("revealed start"),
    }

    robot_names = []
    for robot_name in robot_locations.keys():
        # Free all robots and put in the start
        robot_names.append(robot_name)
        initial_fluents.add(F(f"free {robot_name}"))
        initial_fluents.add(F(f"at {robot_name} start"))

    initial_state = State(
        time=0,
        fluents=initial_fluents,
    )

    # Define goal: find supplies and have them at the start location
    goal = F("found supplies") & F("at supplies start")

    objects_by_type = {
        "robot": robot_names,
        "location": env.locations.keys(),
        "object": ["supplies"],
    }

    # Create operators
    move_time_fn = env.get_skills_time_fn('move')
    move_op = operators.construct_move_operator(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob=lambda r, loc, o: 1.0,
                                                    search_time=env.get_skills_time_fn('search'))
    no_op = operators.construct_no_op_operator(
        no_op_time=env.get_skills_time_fn('no_op'),
        extra_cost=100
    )
    pick_op = operators.construct_pick_operator_blocking(
        pick_time=env.get_skills_time_fn('pick')
    )

    place_op = operators.construct_place_operator_blocking(
        place_time=env.get_skills_time_fn('place')
    )

    # Create environment interface
    env_interface = EnvironmentInterface(
        initial_state,
        objects_by_type,
        [no_op, pick_op, place_op, move_op, search_op],
        env
    )

    # Planning loop
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Run planning loop
    start_time = time.perf_counter()

    # Dashboard with recording console
    recording_console = Console(record=True, force_terminal=True, width=120)
    fluent_filter = lambda f: any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
    dashboard = PlannerDashboard(goal, env_interface, fluent_filter=fluent_filter, print_on_exit=False, console=recording_console)

    for iteration in range(max_iterations):
        # Check if goal is reached
        if goal.evaluate(env_interface.state.fluents):
            break

        # Get available actions
        all_actions = env_interface.get_actions()
        # Plan next action
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(env_interface.state, goal,
                           max_iterations=case.mcts.iterations,
                           c=case.mcts.c,
                           max_depth=20)

        if action_name == 'NONE':
            dashboard.console.print("No more actions available. Goal may not be achievable.")
            break

        # Execute action
        action = get_action_by_name(all_actions, action_name)
        env_interface.advance(action, do_interrupt=case.do_interrupt)
        dashboard.update(mcts, action_name)

    # Export the recorded console output as HTML
    actions_taken = [name for name, _ in dashboard.actions_taken]
    dashboard.print_history()
    html_output = recording_console.export_html(inline_styles=True)

    return {
        "success": goal.evaluate(env_interface.state.fluents),
        "wall_time": time.perf_counter() - start_time,
        "plan_cost": float(env_interface.state.time),
        "actions_count": len(actions_taken),
        "actions": actions_taken,
        "log_html": html_output,  # Will be logged as HTML artifact
    }


# Register parameter combinations
bench_heterogeneous_robots.add_cases([
    {
        "mcts.iterations": iterations,
        "mcts.c": c,
        "mcts.h_mult": h_mult,
        "num_robots": num_robots,
        "do_interrupt": do_interrupt,
    }
    for c, num_robots, h_mult, iterations, do_interrupt in itertools.product(
        [100, 300],                 # mcts.c
        [1, 2, 3],                  # num_robots
        [1, 2, 5],                  # mcts.h_mult
        [400, 1000, 4000],          # mcts.iterations
        [True, False],              # do_interrupt
    )
])
