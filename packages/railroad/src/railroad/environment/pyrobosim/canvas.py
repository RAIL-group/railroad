"""Matplotlib-based visualization canvas for PyRoboSim worlds."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pyrobosim.gui.world_canvas import WorldCanvas
from pyrobosim.gui.options import WorldCanvasOptions
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

    def __init__(self, world, show_plot: bool = True, record_plots: bool = False):
        self.world = world
        self.show_plot = show_plot
        if not self.show_plot:
            plt.switch_backend("Agg")
        self.record_plots = record_plots
        self.options = WorldCanvasOptions()
        self.main_window = self.MockMainWindow()
        self.fig, self.axes = plt.subplots(
            dpi=self.options.dpi,
            tight_layout=True
        )
        plt.ion()

        # Hijack the signals BEFORE calling any methods like show()
        self.draw_signal = self.MockSignal(self.draw_signal_callback)
        self.show_robots_signal = self.MockSignal(self.show_robots)
        self.show_planner_and_path_signal = self.MockSignal(self._show_all_paths)

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
        self._plot_frames = [] if record_plots else None

    def _show_all_paths(self):
        for robot in self.world.robots:
            self._draw_single_robot_path(robot)

    def _draw_single_robot_path(self, robot):
        if not robot.path_planner:
            return
        path = robot.path_planner.get_latest_path()
        if not path:
            return
        if robot.name in self.path_artists_storage:
            for artist in self.path_artists_storage[robot.name]:
                try:
                    artist.remove()
                except Exception:
                    pass
        new_artists_dict = plot_path_planner(
            self.axes,
            graphs=[],
            path=path,
            path_color=robot.color
        )
        flat_artists = new_artists_dict.get("path", []) + new_artists_dict.get("graph", [])
        self.path_artists_storage[robot.name] = flat_artists

    def show(self) -> None:
        self.show_rooms()
        self.show_hallways()
        self.show_locations()
        self.show_objects()
        self.show_robots()
        self.update_robots_plot()
        self._show_all_paths()
        self.axes.autoscale()
        self.axes.axis("equal")

    def draw_signal_callback(self):
        if hasattr(self, "fig"):
            self.fig.canvas.draw_idle()

    def update(self):
        self.show()
        self.fig.canvas.draw_idle()
        if self._plot_frames is not None:
            self._plot_frames.append(self._get_frame())
        if self.show_plot:
            plt.pause(self.options.animation_dt)

    def wait_for_close(self):
        if not self.show_plot:
            return
        plt.ioff()
        plt.show()

    def _get_frame(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        width = int(renderer.width)
        height = int(renderer.height)
        image = np.frombuffer(self.fig.canvas.tostring_argb(), dtype='uint8')
        image = image.reshape((height, width, 4))
        return image[..., 1:4]

    def save_animation(self, filepath):
        if not self._plot_frames:
            import warnings
            warnings.warn("No frames recorded to save animation.")
            return
        import imageio
        from PIL import Image
        fps = int(round(1 / self.options.animation_dt))
        target_size = (self._plot_frames[0].shape[1], self._plot_frames[0].shape[0])
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath_str = filepath if filepath.as_posix().startswith(("/", "./", "../")) else f"./{filepath}"
        writer = imageio.get_writer(filepath, format="ffmpeg", mode="I", fps=fps, codec="libx264", macro_block_size=None)  # type: ignore[arg-type]  # imageio stubs incorrectly type `format`
        for frame in self._plot_frames:
            frame_uint8 = frame.astype("uint8")
            if (frame.shape[1], frame.shape[0]) != target_size:
                img = Image.fromarray(frame_uint8)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                frame_uint8 = np.array(img)
            writer.append_data(frame_uint8)
        writer.close()
        print(f"Animation saved to {filepath_str}")
