"""Matplotlib canvas for drawing and recording PyRoboSim worlds."""

from __future__ import annotations


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyrobosim.gui.options import WorldCanvasOptions
from pyrobosim.gui.world_canvas import WorldCanvas
from pyrobosim.navigation.visualization import plot_path_planner



class MatplotlibWorldCanvas(WorldCanvas):
    """
    Matplotlib-based visualization canvas for PyRoboSim worlds.
    Replaces Qt-based WorldCanvas to support non-blocking plotting.
    """

    class MockSignal:
        def __init__(self, callback):
            self.callback = callback

        def emit(self, *args, **kwargs):
            self.callback()

    class MockMainWindow:
        def get_current_robot(self):
            return None

    def __init__(self, world, show_plot: bool = True, record_video: bool = False):
        self.world = world
        self.show_plot = show_plot
        if not self.show_plot:
            plt.switch_backend("Agg")
        self.record_video = record_video
        self.options = WorldCanvasOptions()
        self.main_window = self.MockMainWindow()
        self.fig, self.axes = plt.subplots(
            dpi=self.options.dpi,
            tight_layout=True
        )
        plt.ion()

        # Hijack the signals BEFORE calling any methods like show()
        # This prevents the "Signal source has been deleted" error
        self.draw_signal = self.MockSignal(self.draw_signal_callback)  # ty: ignore[invalid-assignment]
        # Add other signals as needed to prevent attribute errors
        self.show_robots_signal = self.MockSignal(self.show_robots)  # ty: ignore[invalid-assignment]
        self.show_planner_and_path_signal = self.MockSignal(self._show_all_paths)  # ty: ignore[invalid-assignment]

        # Manually initialize the artist lists from WorldCanvas since we do not call its __init__
        self.path_artists_storage = {}
        self.robot_bodies = []
        self.robot_dirs = []
        self.robot_lengths = []
        self.robot_texts = []
        self.robot_sensor_artists = []
        self.obj_patches = []
        self.obj_texts = []
        self.hallway_patches = []
        self.room_patches = []
        self.room_texts = []
        self.location_patches = []
        self.location_texts = []
        self.path_planner_artists = {"graph": [], "path": []}

        self.show()
        self.axes.autoscale()
        self.axes.axis("equal")
        self._plot_frames = []

    def _show_all_paths(self):
        """Custom method to draw paths for EVERY robot in the world."""
        for robot in self.world.robots:
            self._draw_single_robot_path(robot)

    def _draw_single_robot_path(self, robot):
        """Helper to draw a specific robot's path without clearing others."""
        if not robot.path_planner:
            return

        # Get the path from the robot
        path = robot.path_planner.get_latest_path()
        if not path:
            return

        # Clear OLD artists for THIS specific robot only
        if robot.name in self.path_artists_storage:
            for artist in self.path_artists_storage[robot.name]:
                try:
                    artist.remove()
                except Exception:
                    pass

        # Note: plot_path_planner returns a dict of lists of artists
        new_artists_dict = plot_path_planner(
            self.axes,
            graphs=[],  # Set to graphs=robot.path_planner.get_graphs() if you want the RRT/PRM trees
            path=path,
            path_color=robot.color
        )

        # Store these artists so we can remove them in the next frame
        flat_artists = new_artists_dict.get("path", []) + new_artists_dict.get("graph", [])
        self.path_artists_storage[robot.name] = flat_artists

    def show(self) -> None:
        """Overriding show to ensure all robots are processed."""
        self.show_rooms()
        self.show_hallways()
        self.show_locations()
        self.show_objects()
        self.show_robots()
        self.update_robots_plot()

        self._show_all_paths()  # Show paths for all robots

        self.axes.autoscale()
        self.axes.axis("equal")

    def draw_signal_callback(self):
        """Replacement for the Qt signal execution."""
        if hasattr(self, "fig"):
            self.fig.canvas.draw_idle()

    def update(self):
        """Updates the world visualization in a loop."""
        self.show()
        self.fig.canvas.draw_idle()
        if self.record_video:
            self._plot_frames.append(self._get_frame())
        if self.show_plot:
            plt.pause(self.options.animation_dt)

    def wait_for_close(self):
        """Blocks until the plot window is closed."""
        if not self.show_plot:
            return
        plt.ioff()
        plt.show()

    def _get_frame(self):
        """Captures the current frame as an image array."""
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        width = int(renderer.width)
        height = int(renderer.height)
        image = np.frombuffer(self.fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape((height, width, 4))
        return image[..., 1:4]  # Convert ARGB to RGB

    def save_video(self, path):
        """Saves the recorded frames as a video file.

        Follows the dashboard's videos: matplotlib's FFMpegWriter (the
        container follows the extension, e.g. .mp4) and the same console
        messages.
        """
        if not self._plot_frames:
            import warnings
            warnings.warn("No frames recorded to save a video. Use 'record_video=True' to record frames.")
            return

        from matplotlib.animation import FFMpegWriter
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from rich.console import Console

        from railroad.dashboard._tui import _is_ci_environment

        console = Console()
        fps = int(round(1 / self.options.animation_dt))
        n_frames = len(self._plot_frames)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Replay the recorded frames on an off-screen figure the size of the
        # first frame; a frame of another size is scaled to fit it.
        height, width = self._plot_frames[0].shape[:2]
        dpi = 100
        fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        FigureCanvasAgg(fig)
        axes = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_axis_off()
        image = axes.imshow(self._plot_frames[0], aspect="auto")

        def _frames():
            """Yield the frames, with a progress bar when interactive."""
            if _is_ci_environment():
                yield from self._plot_frames
                return
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
            with Progress(
                TextColumn("[bold blue]Saving video"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("frames", total=n_frames)
                for index, frame in enumerate(self._plot_frames):
                    yield frame
                    progress.update(task, completed=index + 1)

        console.print(
            f"Writing [bold]{n_frames / fps:.1f}s[/bold] of simulator video at {fps} fps — {n_frames} frames."
        )
        writer = FFMpegWriter(fps=fps)
        with writer.saving(fig, str(path), dpi):
            for frame in _frames():
                image.set_data(frame)
                writer.grab_frame()
        console.print(f"Saved video to [yellow]{path}[/yellow]")
