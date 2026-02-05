"""Integration tests for ProcTHOR visualization with multi-robot planning.

These tests verify end-to-end planning with MCTS and trajectory visualization.
They were originally in packages/environments/tests/test_visualization.py and
were migrated to use the consolidated railroad.environment.procthor module.
"""

import random
from functools import reduce
from operator import and_
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Headless backend for tests
import matplotlib.pyplot as plt
import pytest

from railroad import operators
from railroad._bindings import State
from railroad.core import Fluent as F, get_action_by_name
from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
from railroad.environment.procthor.plotting import (
    extract_robot_poses,
    plot_multi_robot_trajectories,
    plot_robot_trajectory,
    plot_grid,
)
from railroad.environment.procthor.utils import get_trajectory
from railroad.planner import MCTSPlanner


# Test configuration
SEED = 7005
SAVE_DIR = Path('./data/test_logs')


@pytest.fixture
def scene():
    """Create ProcTHOR scene for tests."""
    random.seed(SEED)
    return ProcTHORScene(seed=SEED, resolution=0.05)


@pytest.fixture
def target_objects(scene):
    """Select two objects from the scene that have known locations."""
    # Pick objects that actually exist in scene.object_locations
    objects_with_locations = []
    for loc, objs in scene.object_locations.items():
        for obj in objs:
            if obj not in objects_with_locations:
                objects_with_locations.append(obj)
            if len(objects_with_locations) >= 2:
                break
        if len(objects_with_locations) >= 2:
            break

    if len(objects_with_locations) < 2:
        pytest.skip("Not enough objects with known locations in scene")

    return objects_with_locations[:2]


@pytest.fixture
def target_locations(scene, target_objects):
    """Select target locations for placing objects (different from where they are)."""
    # Find where objects currently are
    obj_current_locs = set()
    for loc, objs in scene.object_locations.items():
        for obj in target_objects:
            if obj in objs:
                obj_current_locs.add(loc)

    # Pick target locations that are different from current locations
    all_locs = list(scene.locations.keys())
    targets = [loc for loc in all_locs if loc != 'start_loc' and loc not in obj_current_locs]

    if len(targets) < 2:
        # Fall back to any non-start locations
        targets = [loc for loc in all_locs if loc != 'start_loc']

    return targets[:2] if len(targets) >= 2 else [targets[0], targets[0]]


def compute_trajectory_cost(grid, waypoints):
    """Compute total trajectory cost through waypoints."""
    trajectory = get_trajectory(grid, waypoints)
    if len(trajectory) < 2:
        return 0.0, trajectory

    # Cost is total path length
    total_cost = 0.0
    for i in range(len(trajectory) - 1):
        dx = trajectory[i + 1][0] - trajectory[i][0]
        dy = trajectory[i + 1][1] - trajectory[i][1]
        total_cost += (dx * dx + dy * dy) ** 0.5

    return total_cost, trajectory


@pytest.mark.timeout(120)
def test_single_robot_plotting(scene, target_objects, target_locations):
    """Test visualization for single robot pick-and-place scenario.

    Creates a planning scenario where one robot must pick objects and place
    them at target locations, then verifies trajectory plotting works.
    """
    obj1, obj2 = target_objects
    loc1, loc2 = target_locations

    # Create operators
    move_cost_fn = scene.get_move_cost_fn()
    search_time_fn = lambda r, l, o: 10.0
    pick_time_fn = lambda r, l, o: 5.0
    place_time_fn = lambda r, l, o: 5.0
    object_find_prob = lambda r, l, o: 1.0  # Perfect search for faster tests

    move_op = operators.construct_move_operator_blocking(move_cost_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time_fn)
    pick_op = operators.construct_pick_operator_blocking(pick_time_fn)
    place_op = operators.construct_place_operator_blocking(place_time_fn)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initial state
    initial_state = State(0.0, {
        F("revealed start_loc"),
        F("at robot1 start_loc"),
        F("free robot1"),
    }, [])

    # Goal: place objects at target locations
    goal = F(f"at {obj1} {loc1}") & F(f"at {obj2} {loc2}")

    # Create environment
    env = ProcTHOREnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": {"robot1"},
            "location": set(scene.locations.keys()),
            "object": {obj1, obj2},
        },
        operators=[move_op, search_op, pick_op, place_op, no_op],
    )

    # Planning loop
    actions_taken = []
    all_actions = env.get_actions()
    mcts = MCTSPlanner(all_actions)

    for _ in range(50):
        all_actions = env.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(
            env.state, goal,
            max_iterations=4000,
            c=300,
            heuristic_multiplier=2
        )

        if action_name == 'NONE':
            break

        action = get_action_by_name(all_actions, action_name)
        env.act(action)
        actions_taken.append(action_name)

        if goal.evaluate(env.state.fluents):
            break

    # Extract robot trajectory
    robot_locations = {'robot1': 'start_loc'}
    robot_poses_dict = extract_robot_poses(actions_taken, robot_locations, scene.locations)
    robot_waypoints = robot_poses_dict.get('robot1', [])

    # Compute trajectory cost
    cost, trajectory = compute_trajectory_cost(scene.grid, robot_waypoints)

    # Plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    plot_grid(ax, scene.grid)
    if len(robot_waypoints) >= 2:
        plot_robot_trajectory(ax, robot_waypoints, scene.grid, scene.scene_graph, "robot1")
    plt.title(f"Single Robot Trajectory Cost: {cost:.1f}\nGoal: {goal}")

    # Save
    figpath = SAVE_DIR / f'test_visualization_single_robot_{SEED}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=300)
    plt.close()

    # Verify goal reached
    assert goal.evaluate(env.state.fluents), f"Goal not reached. Final fluents: {env.state.fluents}"


@pytest.mark.timeout(180)
def test_multi_robot_unknown_plotting(scene, target_objects, target_locations):
    """Test multi-robot planning with unknown object locations.

    Two robots search for objects (locations unknown) and move them to targets.
    Tests the full search -> find -> pick -> place workflow.
    """
    obj1, obj2 = target_objects
    loc1, loc2 = target_locations

    # Create operators
    move_cost_fn = scene.get_move_cost_fn()
    search_time_fn = lambda r, l, o: 10.0
    pick_time_fn = lambda r, l, o: 5.0
    place_time_fn = lambda r, l, o: 5.0

    # Probability based on ground truth
    def object_find_prob(robot: str, location: str, obj: str) -> float:
        for loc, objs in scene.object_locations.items():
            if obj in objs:
                return 0.9 if loc == location else 0.1
        return 0.1

    move_op = operators.construct_move_operator_blocking(move_cost_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time_fn)
    pick_op = operators.construct_pick_operator_blocking(pick_time_fn)
    place_op = operators.construct_place_operator_blocking(place_time_fn)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initial state - objects locations unknown
    initial_state = State(0.0, {
        F("revealed start_loc"),
        F("at robot1 start_loc"), F("free robot1"),
        F("at robot2 start_loc"), F("free robot2"),
    }, [])

    goal = F(f"at {obj1} {loc1}") & F(f"at {obj2} {loc2}")

    env = ProcTHOREnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": {"robot1", "robot2"},
            "location": set(scene.locations.keys()),
            "object": {obj1, obj2},
        },
        operators=[move_op, search_op, pick_op, place_op, no_op],
    )

    # Planning loop with early termination on consecutive no-ops
    actions_taken = []
    consecutive_no_op = 0
    all_actions = env.get_actions()
    mcts = MCTSPlanner(all_actions)

    for _ in range(60):
        all_actions = env.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(
            env.state, goal,
            max_iterations=4000,
            c=300,
            heuristic_multiplier=2
        )

        if action_name == 'NONE':
            break

        action = get_action_by_name(all_actions, action_name)
        env.act(action)
        actions_taken.append(action_name)

        # Track consecutive no-ops for early termination
        if action_name.split()[0] == 'no_op':
            consecutive_no_op += 1
            if consecutive_no_op > 4:
                break
        else:
            consecutive_no_op = 0

        if goal.evaluate(env.state.fluents):
            break

    # Extract trajectories for both robots
    robot_locations = {'robot1': 'start_loc', 'robot2': 'start_loc'}
    robot_poses_dict = extract_robot_poses(actions_taken, robot_locations, scene.locations)

    # Compute total cost
    total_cost = 0.0
    for robot_name, waypoints in robot_poses_dict.items():
        cost, _ = compute_trajectory_cost(scene.grid, waypoints)
        total_cost += cost

    # Plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    plot_multi_robot_trajectories(ax, scene.grid, robot_poses_dict, scene.scene_graph)
    plt.title(f"Multi Robot (Unknown) Trajectory Cost: {total_cost:.1f}\nGoal: {goal}")

    figpath = SAVE_DIR / f'test_visualization_unknown_multi_robot_{SEED}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=300)
    plt.close()

    assert goal.evaluate(env.state.fluents), f"Goal not reached. Final fluents: {env.state.fluents}"


@pytest.mark.timeout(120)
def test_multi_robot_known_plotting(scene, target_objects, target_locations):
    """Test multi-robot planning with known object locations.

    Two robots know where objects are located and only need to pick/place.
    No search operator needed.
    """
    obj1, obj2 = target_objects
    loc1, loc2 = target_locations

    # Find where objects actually are in the scene
    obj1_loc = None
    obj2_loc = None
    for location, objects in scene.object_locations.items():
        if obj1 in objects:
            obj1_loc = location
        if obj2 in objects:
            obj2_loc = location

    # Skip if we can't find objects (shouldn't happen)
    if obj1_loc is None or obj2_loc is None:
        pytest.skip("Could not find target objects in scene")

    # Create operators (no search needed)
    move_cost_fn = scene.get_move_cost_fn()
    pick_time_fn = lambda r, l, o: 5.0
    place_time_fn = lambda r, l, o: 5.0

    move_op = operators.construct_move_operator_blocking(move_cost_fn)
    pick_op = operators.construct_pick_operator_blocking(pick_time_fn)
    place_op = operators.construct_place_operator_blocking(place_time_fn)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initial state with known object locations
    initial_state = State(0.0, {
        F("revealed start_loc"),
        F("at robot1 start_loc"), F("free robot1"),
        F("at robot2 start_loc"), F("free robot2"),
        F(f"at {obj1} {obj1_loc}"),
        F(f"at {obj2} {obj2_loc}"),
    }, [])

    goal = F(f"at {obj1} {loc1}") & F(f"at {obj2} {loc2}")

    env = ProcTHOREnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": {"robot1", "robot2"},
            "location": set(scene.locations.keys()),
            "object": {obj1, obj2},
        },
        operators=[move_op, pick_op, place_op, no_op],
    )

    # Planning loop
    actions_taken = []
    all_actions = env.get_actions()
    mcts = MCTSPlanner(all_actions)

    for _ in range(20):
        all_actions = env.get_actions()
        mcts = MCTSPlanner(all_actions)
        action_name = mcts(
            env.state, goal,
            max_iterations=4000,
            c=300,
            heuristic_multiplier=2
        )

        if action_name == 'NONE':
            break

        action = get_action_by_name(all_actions, action_name)
        env.act(action)
        actions_taken.append(action_name)

        if goal.evaluate(env.state.fluents):
            break

    # Extract trajectories
    robot_locations = {'robot1': 'start_loc', 'robot2': 'start_loc'}
    robot_poses_dict = extract_robot_poses(actions_taken, robot_locations, scene.locations)

    # Compute total cost
    total_cost = 0.0
    for robot_name, waypoints in robot_poses_dict.items():
        cost, _ = compute_trajectory_cost(scene.grid, waypoints)
        total_cost += cost

    # Plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    plot_multi_robot_trajectories(ax, scene.grid, robot_poses_dict, scene.scene_graph)
    plt.title(f"Multi Robot (Known) Trajectory Cost: {total_cost:.1f}\nGoal: {goal}")

    figpath = SAVE_DIR / f'test_visualization_known_multi_robot_{SEED}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=300)
    plt.close()

    assert goal.evaluate(env.state.fluents), f"Goal not reached. Final fluents: {env.state.fluents}"
