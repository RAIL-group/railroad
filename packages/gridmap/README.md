# `gridmap`: Mapping and Planning with Occupancy Grids

`gridmap` is a Python module designed to provide utilities for robot mapping and planning in occupancy grids. Functionality is split across four files:

- `laser.py` For representing laser scanns and ray casting into an occupancy grid. It allows users to get directions from a laser scanner, simulate sensor measurements, generate a line between two points using Bresenham's line algorithm, and perform ray casting in the occupancy grid.
- `mapping.py` For constructing an occupancy grid from planer laser scans. It offers two main functionalities: insertion of a scan into the occupancy grid, and getting a fully-connected observed grid from a given pose. The latter function returns a grid where any components not connected to the robot's region are marked as 'unobserved', which can be crucial for avoiding invalid planning to unreachable frontiers.
- `planning.py` For planning in the occupancy grid. It provides functionality to use Dijkstra's algorithm to generate paths in the grid and optionally *sparsify* them: shortening them while still respecting occupancy.
- `utils.py` Provides a utility function for inflating obstacles in the occupancy grid, useful to create a safe distance between robot and obstacles during planning.

Our [Jupyter Notebook onboarding tutorials](../../resources/notebooks/) make extensive use of the `gridmap` package for simulated robot navigation through previously-unmapped simulated environments. Look there for worked examples of the use of `gridmap`.
