import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection

from railroad.environment.procthor.plotting import make_plotting_grid


def plot_grid(ax, grid):
    plotting_grid = make_plotting_grid(grid.T)
    ax.imshow(plotting_grid, cmap='gray', origin='upper')


def plot_single_robot_trajectory(ax, robot_all_poses, trajectory, graph, robot_name, color_map_name='viridis', robot_id=0):
    # trajectory is likely (2, N)
    x = trajectory[0]
    y = trajectory[1]

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, len(x))
    lc = LineCollection(segments.tolist(), cmap=color_map_name, norm=norm)

    # Set the values used for colormapping
    lc.set_array(np.arange(len(x)))
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # Use a distinct color for text that stands out (simpler than matching gradient)
    text_color = 'brown'

    # Add a marker for the start
    if len(robot_all_poses) > 0:
        ax.text(robot_all_poses[0].x, robot_all_poses[0].y, f'{robot_id} - {robot_name}', color=text_color, size=4, weight='bold')

    for i, pose in enumerate(robot_all_poses[1:]):
        idx = graph.get_node_idx_by_position([pose.x, pose.y])
        if idx is not None:
            name = graph.get_node_name_by_idx(idx)
            ax.text(pose.x, pose.y, f'{i + 1} - {name}', color=text_color, size=4, weight='bold')
        else:
            print(f'Plotting warning: No node found in graph for pose [{pose.x:.2f}, {pose.y:.2f}]')


def plot_grid_with_robot_trajectory(ax, grid, robot_all_poses, trajectory, graph, cmap_name='viridis'):
    """Backward compatibility wrapper for single robot."""
    plot_grid(ax, grid)
    plot_single_robot_trajectory(ax, robot_all_poses, trajectory, graph, "ROBOT", cmap_name)


def plot_multi_robot_trajectories(ax, grid, robots_data, graph):
    """
    robots_data: dict mapping robot_name -> (poses, trajectory)
    """
    plot_grid(ax, grid)

    # Predefined colormaps for different robots to be distinct
    # Viridis-like (perceptually uniform sequential) colormaps
    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'spring', 'summer', 'autumn', 'winter', 'cool']

    for i, (robot_name, (poses, trajectory)) in enumerate(robots_data.items()):
        cmap = colormaps[i % len(colormaps)]
        plot_single_robot_trajectory(ax, poses, trajectory, graph, robot_name, cmap, robot_id=i)


def save_navigation_video(trajectory, thor_interface, video_file_path, fig_title):
    video_file_path = Path(video_file_path)
    video_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    writer = animation.FFMpegWriter(12)
    writer.setup(fig, video_file_path, 500)
    for step, grid_coord in enumerate(list(zip(trajectory[0], trajectory[1]))[::5]):
        position = thor_interface.g2p_map[grid_coord]
        thor_interface.controller.step(action="Teleport", position=position, horizon=30)
        plt.clf()
        top_down_image = thor_interface.get_top_down_image(orthographic=False)
        plt.imshow(top_down_image)
        plt.axis('off')
        plt.title(f'{fig_title} [Step: {step}]', fontsize='10')
        writer.grab_frame()
    writer.finish()


def plot_plan_progression(ax, plan):
    textstr = ''
    for p in plan:
        textstr += str(p) + '\n'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place a text box in upper left in axes coords
    ax.text(0, 1, textstr, transform=ax.transAxes, fontsize=5,
            verticalalignment='top', bbox=props)
    ax.box(False)
    # Hide x and y ticks
    ax.xticks([])
    ax.yticks([])

    # Add labels and title
    ax.title.set_text('Plan progression')
    ax.title.set_fontsize(6)
