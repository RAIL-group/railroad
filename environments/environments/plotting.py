import matplotlib.pyplot as plt
import matplotlib.animation as animation
import procthor
from pathlib import Path


def plot_grid_with_robot_trajectory(ax, grid, robot_all_poses, trajectory, graph):
    plotting_grid = procthor.plotting.make_plotting_grid(grid.T)
    ax.imshow(plotting_grid)
    ax.plot(trajectory[0], trajectory[1])
    ax.text(robot_all_poses[0].x, robot_all_poses[0].y, '0 - ROBOT', color='brown', size=4)
    for i, pose in enumerate(robot_all_poses[1:]):
        idx = graph.get_node_idx_by_position([pose.x, pose.y])
        if idx is not None:
            name = graph.get_node_name_by_idx(idx)
            ax.text(pose.x, pose.y, f'{i+1} - {name}', color='brown', size=4)
        else:
            print(f'Plotting warning: No node found in graph for pose [{pose.x:.2f}, {pose.y:.2f}]')


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
