"""Integration tests for ProcTHOR visualization with multi-robot planning.

These tests verify end-to-end planning with MCTS and trajectory visualization
using PlannerDashboard.plot_trajectories().
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Headless backend for tests
import matplotlib.pyplot as plt
import pytest

from railroad import operators
from railroad._bindings import State
from railroad.core import Fluent as F, get_action_by_name
from railroad.dashboard import PlannerDashboard
from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
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
    def search_time_fn(r, loc, o):
        return 10.0

    def pick_time_fn(r, loc, o):
        return 5.0

    def place_time_fn(r, loc, o):
        return 5.0

    def object_find_prob(r, loc, o):
        return 1.0  # Perfect search for faster tests

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

    # Planning loop with dashboard
    with PlannerDashboard(goal, env, force_interactive=False, print_on_exit=False) as dashboard:
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
            dashboard.update(mcts, action_name)

            if goal.evaluate(env.state.fluents):
                break

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    dashboard.plot_trajectories(ax=ax)
    plt.title(f"Single Robot Trajectory\nGoal: {goal}")

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
    def search_time_fn(r, loc, o):
        return 10.0

    def pick_time_fn(r, loc, o):
        return 5.0

    def place_time_fn(r, loc, o):
        return 5.0

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

    # Planning loop with dashboard
    consecutive_no_op = 0
    with PlannerDashboard(goal, env, force_interactive=False, print_on_exit=False) as dashboard:
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
            dashboard.update(mcts, action_name)

            # Track consecutive no-ops for early termination
            if action_name.split()[0] == 'no_op':
                consecutive_no_op += 1
                if consecutive_no_op > 4:
                    break
            else:
                consecutive_no_op = 0

            if goal.evaluate(env.state.fluents):
                break

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    dashboard.plot_trajectories(ax=ax)
    plt.title(f"Multi Robot (Unknown) Trajectory\nGoal: {goal}")

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
    def pick_time_fn(r, loc, o):
        return 5.0

    def place_time_fn(r, loc, o):
        return 5.0

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

    # Planning loop with dashboard
    with PlannerDashboard(goal, env, force_interactive=False, print_on_exit=False) as dashboard:
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
            dashboard.update(mcts, action_name)

            if goal.evaluate(env.state.fluents):
                break

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    dashboard.plot_trajectories(ax=ax)
    plt.title(f"Multi Robot (Known) Trajectory\nGoal: {goal}")

    figpath = SAVE_DIR / f'test_visualization_known_multi_robot_{SEED}.png'
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=300)
    plt.close()

    assert goal.evaluate(env.state.fluents), f"Goal not reached. Final fluents: {env.state.fluents}"
