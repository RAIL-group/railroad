import numpy as np
import random

import common
import gridmap


def _make_sample_two_frontier_grid(add_blocking_wall=False,
                                   add_nonblocking_wall=False):
    """Generates a simple grid with two frontiers. Known space is on the left side
of the map and both frontiers lead to the same open portion of the map. The
returned dictionary object contains a goal pose that both frontiers can reach.

Two walls can be added to the map. The first, the 'blocking wall', separates the
unknown space so that only one of the two frontiers can be reached. The second,
the 'nonblocking wall' does not separate the unknown space but introduces an
obstacle so that the delta_success_cost should be non-zero for both frontiers.

Finally, one of the frontiers is considerably smaller than the other, so
intermediate levels of inflating the grid should block off one before the other.
    """

    # Generate a simple grid
    boundary_ind = 75
    grid = gridmap.constants.UNOBSERVED_VAL * np.ones([150, 150])
    grid[1:-2, 1:boundary_ind] = gridmap.constants.FREE_VAL
    grid[1, :] = gridmap.constants.COLLISION_VAL
    grid[-2, :] = gridmap.constants.COLLISION_VAL
    grid[:, 1] = gridmap.constants.COLLISION_VAL
    grid[:, -2] = gridmap.constants.COLLISION_VAL
    grid[:, boundary_ind] = gridmap.constants.COLLISION_VAL
    grid[130:140, boundary_ind] = gridmap.constants.FREE_VAL
    grid[20:21, boundary_ind] = gridmap.constants.FREE_VAL

    known_grid = grid.copy()
    known_grid[known_grid ==
               gridmap.constants.UNOBSERVED_VAL] = gridmap.constants.FREE_VAL

    if add_blocking_wall:
        # The known_grid should now have a "wall" that prevents us from reaching the
        # goal from one of the frontiers.
        known_grid[boundary_ind,
                   boundary_ind:] = gridmap.constants.COLLISION_VAL

    if add_nonblocking_wall:
        # Add a wall so that the delta_success_cost will be nonzero
        known_grid[10:-10, boundary_ind + 25] = gridmap.constants.COLLISION_VAL

    # Define the robot pose and the goal
    pose = common.Pose(x=20.0, y=30.0, yaw=np.pi / 3)
    goal = common.Pose(x=20.0, y=120.0, yaw=0)

    return {'grid': grid, 'known_grid': known_grid, 'pose': pose, 'goal': goal}


# Testing the sensor functionality
def test_basic_sensor_insertion():
    random.seed(8616)
    np.random.seed(8616)

    # Generate a simple grid
    boundary_ind = 75
    grid = np.zeros([150, 150])
    grid[1, :] = gridmap.constants.COLLISION_VAL
    grid[-2, :] = gridmap.constants.COLLISION_VAL
    grid[:, 1] = gridmap.constants.COLLISION_VAL
    grid[:, -2] = gridmap.constants.COLLISION_VAL
    grid[:, boundary_ind] = gridmap.constants.COLLISION_VAL
    grid[40:50, 40:50] = gridmap.constants.COLLISION_VAL

    directions = gridmap.laser.get_laser_scanner_directions(
        num_points=512, field_of_view_rad=2 * np.pi / 3)

    for _ in range(10):
        pose = common.Pose(x=random.uniform(3.0, 60.0),
                           y=random.uniform(3.0, 60.0),
                           yaw=random.uniform(0, 2 * np.pi))
        ranges = gridmap.laser.simulate_sensor_measurement(grid,
                                                           directions,
                                                           max_range=102,
                                                           sensor_pose=pose)

        empty_grid = gridmap.constants.UNOBSERVED_VAL * np.ones(grid.shape)
        scan_inserted_grid = gridmap.mapping.insert_scan(
            empty_grid,
            directions,
            ranges,
            max_range=100,
            sensor_pose=pose,
            connect_neighbor_distance=3)

        # Assert that few "occupied" points are out of place
        assert np.sum(
            np.logical_and(
                scan_inserted_grid == gridmap.constants.COLLISION_VAL, grid
                == gridmap.constants.FREE_VAL)) < 20
        # Assert that few "unoccupied" points exist where they should not
        assert np.sum(
            np.logical_and(scan_inserted_grid == gridmap.constants.FREE_VAL,
                           grid == gridmap.constants.COLLISION_VAL)) < 5
        assert np.sum(scan_inserted_grid[:, boundary_ind:] ==
                      gridmap.constants.FREE_VAL) == 0


def test_grid_inflation():
    """Confirm that inflating the grid generates plans incapable of going through small gaps"""

    # Get the basic grid
    data = _make_sample_two_frontier_grid(add_blocking_wall=False,
                                          add_nonblocking_wall=False)
    grid = data['grid']
    pose = data['pose']
    goal = data['goal']

    # Initialize the grids
    uninflated_grid = grid.copy()
    inflated_grid = gridmap.utils.inflate_grid(grid, inflation_radius=2.5)
    very_inflated_grid = gridmap.utils.inflate_grid(grid, inflation_radius=8.0)

    # The plan should succeed and find the path through the small gap
    _, uninflated_get_costs = gridmap.planning.compute_cost_grid_from_position(
        uninflated_grid, [pose.x, pose.y])
    did_plan, path_sparse = uninflated_get_costs([goal.x, goal.y],
                                                 do_sparsify=True)
    assert did_plan
    assert common.compute_path_length(path_sparse) < 100.0

    # The plan should succeed but have to go through the larger far gap
    _, inflated_get_costs = gridmap.planning.compute_cost_grid_from_position(
        inflated_grid, [pose.x, pose.y])
    did_plan, path_sparse = inflated_get_costs([goal.x, goal.y],
                                               do_sparsify=True)
    assert did_plan
    assert common.compute_path_length(path_sparse) > 100.0

    # The plan should not succeed
    _, very_inflated_get_costs = gridmap.planning.compute_cost_grid_from_position(
        very_inflated_grid, [pose.x, pose.y])
    did_plan, path_sparse = very_inflated_get_costs([goal.x, goal.y],
                                                    do_sparsify=True)
    assert not did_plan
