from __future__ import annotations

import subprocess
from typing import Any, TYPE_CHECKING

from ._goals import format_goal
from ._tui import _generate_coordinates, _is_headless_environment

if TYPE_CHECKING:
    from .dashboard import PlannerDashboard


class _PlottingMixin:
    """Mixin containing all matplotlib plotting methods for PlannerDashboard."""

    _COLORMAPS = [
        "Reds", "Blues", "Greens", "Oranges", "Purples",
        "YlOrBr", "BuGn", "RdPu", "GnBu", "OrRd",
    ]
    _TRAIL_SIZE_START = 25.0
    _TRAIL_SIZE_END = 2.0

    @staticmethod
    def _get_cmap(idx: int):
        """Return a truncated colormap starting at 0.25 for the idx-th entry."""
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        cmap_name = _PlottingMixin._COLORMAPS[idx % len(_PlottingMixin._COLORMAPS)]
        base = plt.get_cmap(cmap_name)
        colors = base(np.linspace(0.25, 1.0, 256))
        return LinearSegmentedColormap.from_list(f"{cmap_name}_t", colors)

    @staticmethod
    def _marker_color(idx: int) -> tuple[float, float, float, float]:
        """Return a marker color by sampling the end of the idx-th colormap."""
        return _PlottingMixin._get_cmap(idx)(0.85)

    def _build_entity_trajectories(
        self: PlannerDashboard,
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
        include_objects: bool = False,
    ) -> tuple[
        dict[str, tuple[list[tuple[float, float]], list[float]]],  # entity -> (waypoints, times)
        dict[str, tuple[float, float]],  # resolved env_coords
        Any,  # grid (for plot_grid background)
    ]:
        """Build resolved and expanded entity trajectories.

        Handles: environment coordinate lookup, ``location_coords`` conflict
        check, missing-coordinate generation, robot/object separation, and
        scene ``get_trajectory`` expansion.

        Args:
            location_coords: Optional explicit location->(x,y) mapping.
                Raises ValueError if any stored positions already have
                coordinates from the environment.
            include_objects: If True, also include non-robot entities.

        Returns:
            (trajectories, env_coords, grid) where *trajectories* maps entity
            names to (waypoints, times) lists with scene-expanded paths.
        """
        # Get environment coordinates + grid
        env_coords, grid, _graph = self._get_location_coords()

        # Handle explicit location_coords
        if location_coords is not None:
            for entity, positions in self._entity_positions.items():
                for _, loc_name, stored_coords in positions:
                    if stored_coords is not None:
                        raise ValueError(
                            f"Cannot pass location_coords when positions already "
                            f"have coordinates from the environment "
                            f"(entity={entity!r}, location={loc_name!r}). "
                            f"Use location_coords only when the environment does "
                            f"not provide coordinates."
                        )
            env_coords.update(location_coords)

        # Generate coordinates for locations that still lack them
        all_location_names: set[str] = set()
        for positions in self._entity_positions.values():
            for _, loc_name, _ in positions:
                all_location_names.add(loc_name)
        missing = all_location_names - set(env_coords.keys())
        if missing:
            env_coords.update(_generate_coordinates(missing))

        # Separate entities into robots and objects
        robot_entities: dict[str, list[tuple[float, str, tuple[float, float] | None]]] = {}
        object_entities: dict[str, list[tuple[float, str, tuple[float, float] | None]]] = {}
        for entity, positions in self._entity_positions.items():
            if entity in self.known_robots:
                robot_entities[entity] = positions
            else:
                object_entities[entity] = positions

        # Scene-aware trajectory expansion
        scene = getattr(self._env, "scene", None)
        get_trajectory_fn = getattr(scene, "get_trajectory", None)

        trajectories: dict[str, tuple[list[tuple[float, float]], list[float]]] = {}

        # Build robot trajectories (with scene expansion)
        for entity, positions in sorted(robot_entities.items()):
            waypoints: list[tuple[float, float]] = []
            times: list[float] = []
            for t, loc_name, stored_coords in positions:
                if stored_coords is not None:
                    waypoints.append(stored_coords)
                    times.append(t)
                elif loc_name in env_coords:
                    waypoints.append(env_coords[loc_name])
                    times.append(t)

            if len(waypoints) < 2:
                trajectories[entity] = (waypoints, times)
                continue

            # Expand through scene trajectory if available
            if get_trajectory_fn is not None:
                import numpy as np
                expanded_wps: list[tuple[float, float]] = []
                expanded_ts: list[float] = []
                for seg_i in range(len(waypoints) - 1):
                    path = get_trajectory_fn([waypoints[seg_i], waypoints[seg_i + 1]])
                    if not path or len(path) < 2:
                        path = [waypoints[seg_i], waypoints[seg_i + 1]]
                    pts = np.array(path)
                    cum_dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
                    t0, t1 = times[seg_i], times[seg_i + 1]
                    seg_times = (t0 + (t1 - t0) * cum_dist / max(cum_dist[-1], 1e-9)).tolist()
                    start = 1 if expanded_wps else 0
                    expanded_wps.extend(path[start:])
                    expanded_ts.extend(seg_times[start:])
                waypoints = expanded_wps
                times = expanded_ts

            trajectories[entity] = (waypoints, times)

        # Build object trajectories (no scene expansion)
        if include_objects:
            for entity, positions in sorted(object_entities.items()):
                waypoints = []
                times = []
                for t, loc_name, stored_coords in positions:
                    if stored_coords is not None:
                        waypoints.append(stored_coords)
                        times.append(t)
                    elif loc_name in env_coords:
                        waypoints.append(env_coords[loc_name])
                        times.append(t)
                trajectories[entity] = (waypoints, times)

        return trajectories, env_coords, grid

    def plot_trajectories(
        self: PlannerDashboard,
        ax: Any = None,
        *,
        show_objects: bool = False,
        location_coords: dict[str, tuple[float, float]] | None = None,
    ) -> Any:
        """Plot entity trajectories collected during planning.

        Works across all environment types:
        - ProcTHOR: uses occupancy grid background and obstacle-respecting paths
        - LocationRegistry: uses Euclidean coordinates with gradient-colored lines
        - Pure symbolic: auto-generates circular layout coordinates

        Args:
            ax: Matplotlib axes. If None, a new figure/axes is created.
            show_objects: If True, also plot object trajectories as dashed lines.
            location_coords: Optional explicit location->(x,y) mapping used to
                resolve positions that lack coordinates (stored as None). If
                provided, any positions that already have non-None coordinates
                from the environment will raise a ValueError to prevent
                conflicting coordinate sources.

        Returns:
            The matplotlib axes with the plotted trajectories.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Trajectory plotting requires matplotlib: pip install matplotlib"
            )

        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Build trajectories using shared helper
        trajectories, env_coords, grid = self._build_entity_trajectories(
            location_coords=location_coords,
            include_objects=show_objects,
        )

        # Plot grid background if available (ProcTHOR tier)
        if grid is not None:
            try:
                from railroad.environment.procthor.plotting import plot_grid
                plot_grid(ax, grid)
            except ImportError:
                pass

        # Dense interpolation for smooth scatter trails
        import numpy as np
        if self._goal_time is not None:
            t_end = self._goal_time
        else:
            t_end = 0.0
            for _entity, (_wps, traj_times) in trajectories.items():
                if traj_times:
                    t_end = max(t_end, max(traj_times))
        dense_times = np.linspace(0.0, t_end, 2000) if t_end > 0 else np.array([0.0])
        dense_positions = self.get_entity_positions_at_times(
            dense_times, location_coords=location_coords,
        )

        # Plot robot trajectories as a single combined scatter sorted by time.
        # Earlier points are drawn first (lowest array index) and are larger,
        # so later (smaller) points overlay them — making double-back paths visible.
        all_xy: list[Any] = []
        all_sizes: list[Any] = []
        all_colors: list[Any] = []
        all_times: list[Any] = []

        robot_id = 0
        for entity in sorted(self.known_robots & set(trajectories.keys())):
            waypoints, times = trajectories[entity]
            if len(waypoints) < 2:
                continue

            cmap = self._get_cmap(robot_id)
            robot_id += 1

            # Get original positions for location markers/labels
            positions = self._entity_positions[entity]
            location_waypoints: list[tuple[float, float]] = []
            for _, loc_name, stored_coords in positions:
                if stored_coords is not None:
                    location_waypoints.append(stored_coords)
                elif loc_name in env_coords:
                    location_waypoints.append(env_coords[loc_name])

            if entity in dense_positions:
                pts = dense_positions[entity]
                n_pts = len(dense_times)
                # Per-point sizes: linearly decay from large (early) to small (late)
                sizes = np.linspace(self._TRAIL_SIZE_START, self._TRAIL_SIZE_END, n_pts)
                # Per-point RGBA colors from this robot's colormap
                norm_t = dense_times / t_end if t_end > 0 else np.zeros_like(dense_times)
                rgba = cmap(norm_t)
                all_xy.append(pts)
                all_sizes.append(sizes)
                all_colors.append(rgba)
                all_times.append(dense_times)

            # Plot location markers and labels
            wx = [p[0] for p in location_waypoints]
            wy = [p[1] for p in location_waypoints]
            ax.scatter(wx, wy, s=20, zorder=6, color="black")

            ax.annotate(
                entity, (wx[0], wy[0]),
                fontsize=7, fontweight="bold", color="brown",
            )

            for i, (_, loc_name, _) in enumerate(positions):
                if i < len(location_waypoints):
                    ax.annotate(
                        loc_name, location_waypoints[i],
                        fontsize=5, color="brown",
                        xytext=(3, 3), textcoords="offset points",
                    )

        # Draw one combined scatter for all robot trails, sorted by time
        if all_xy:
            combined_xy = np.concatenate(all_xy)
            combined_sizes = np.concatenate(all_sizes)
            combined_colors = np.concatenate(all_colors)
            combined_times = np.concatenate(all_times)
            order = np.argsort(combined_times)
            ax.scatter(
                combined_xy[order, 0], combined_xy[order, 1],
                s=combined_sizes[order], c=combined_colors[order],
                zorder=5, alpha=0.7,
            )

        # Optionally plot object trajectories
        if show_objects:
            object_entities = {e: trajectories[e] for e in trajectories if e not in self.known_robots}
            for entity in sorted(object_entities):
                waypoints, _times = object_entities[entity]
                if len(waypoints) < 2:
                    continue

                x = [p[0] for p in waypoints]
                y = [p[1] for p in waypoints]
                ax.plot(x, y, linestyle="--", linewidth=1, alpha=0.6, label=entity)
                ax.scatter(x, y, s=10, zorder=5, alpha=0.6)

            if object_entities:
                ax.legend(fontsize=6)

        # Auto-scale for non-grid plots
        if grid is None:
            ax.autoscale()
            ax.set_aspect("equal", adjustable="datalim")

        ax.set_title(f"Entity Trajectories  (cost = {t_end:.1f})",
                     fontfamily="monospace", fontsize=10)
        return ax

    def get_entity_positions_at_times(
        self: PlannerDashboard,
        times: Any,
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
        include_objects: bool = False,
    ) -> dict[str, Any]:
        """Interpolate entity positions at arbitrary query times.

        Uses numpy.interp on x and y independently. Clamps to first position
        before trajectory start and last position after trajectory end.

        Args:
            times: Query times as a numpy array or list of floats.
            location_coords: Optional explicit location->(x,y) mapping.
            include_objects: If True, include non-robot entities.

        Returns:
            ``{entity_name: (N, 2) ndarray}`` of interpolated positions.
        """
        import numpy as np

        query_times = np.asarray(times, dtype=float)
        trajectories, _env_coords, _grid = self._build_entity_trajectories(
            location_coords=location_coords,
            include_objects=include_objects,
        )

        result: dict[str, Any] = {}
        for entity, (waypoints, traj_times) in trajectories.items():
            if len(waypoints) < 2:
                continue
            wp_arr = np.array(waypoints)
            t_arr = np.array(traj_times)
            x_interp = np.interp(query_times, t_arr, wp_arr[:, 0])
            y_interp = np.interp(query_times, t_arr, wp_arr[:, 1])
            result[entity] = np.column_stack([x_interp, y_interp])

        return result

    def _render_overhead(self: PlannerDashboard, ax: Any) -> bool:
        """Render ProcTHOR top-down image on the given axes if available.

        Args:
            ax: Matplotlib axes to draw on.

        Returns:
            True if an overhead image was rendered, False otherwise.
        """
        scene = getattr(self._env, "scene", None)
        get_top_down = getattr(scene, "get_top_down_image", None)
        if get_top_down is None:
            return False
        top_down_image = get_top_down(orthographic=True)
        ax.imshow(top_down_image)
        ax.axis("off")
        ax.set_title("Top-down View", fontsize=8)
        return True

    def _render_sidebar(
        self: PlannerDashboard,
        sidebar_ax: Any,
        t_end: float,
        goal_snapshots_at_end: dict[str, bool],
        *,
        show_actions: bool = True,
    ) -> tuple[list[tuple[Any, str | None]], list[tuple[Any, float]]]:
        """Render the sidebar content: colorbars, goal status, and action list.

        Shared by show_plots() (static) and save_trajectory_video() (animated).

        Returns:
            (goal_text_artists, action_texts) where:
            - goal_text_artists: list of (text_artist, literal_str_or_None)
            - action_texts: list of (text_artist, start_time)
        """
        import numpy as np

        # Derive entity names from tracked robots that have positions
        entity_names = sorted(self.known_robots & set(self._entity_positions.keys()))
        n_entities = len(entity_names)
        cbar_width = 0.06
        cbar_gap = 0.03
        cbar_left = 0.05

        # Goal section: measure how tall it needs to be
        goal_str = format_goal(self.goal, compact=False)
        goal_lines = goal_str.split("\n")
        goal_line_spacing = 0.035
        goal_section_height = (len(goal_lines) + 1) * goal_line_spacing + 0.04
        goal_section_bottom = 0.03
        goal_section_top = goal_section_bottom + goal_section_height

        # Colorbars fill the space above the goal section
        cbar_bottom = goal_section_top + 0.04
        cbar_top = 0.88
        cbar_height = cbar_top - cbar_bottom

        # Total width used by colorbar strips
        cbar_total_width = n_entities * cbar_width + max(0, n_entities - 1) * cbar_gap
        actions_left = cbar_left + cbar_total_width + 0.08

        for idx, entity in enumerate(entity_names):
            cmap = self._get_cmap(idx)
            mcolor = self._marker_color(idx)
            x0 = cbar_left + idx * (cbar_width + cbar_gap)

            # Create an inset axes for each colorbar strip
            cbar_ax = sidebar_ax.inset_axes((x0, cbar_bottom, cbar_width, cbar_height))
            gradient = np.linspace(0, 1, 256).reshape(-1, 1)
            cbar_ax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower",
                           extent=(0, 1, 0, t_end))
            cbar_ax.set_xlim(0, 1)
            cbar_ax.set_ylim(t_end, 0)
            cbar_ax.set_xticks([])
            if idx == 0:
                cbar_ax.set_ylabel("time", fontsize=6)
                cbar_ax.tick_params(axis="y", labelsize=5)
            else:
                cbar_ax.set_yticks([])

            # Robot marker at top of the strip
            sidebar_ax.plot(
                x0 + cbar_width / 2, cbar_top + 0.04, "o",
                color=mcolor, markeredgecolor="black", markeredgewidth=0.8,
                markersize=8, transform=sidebar_ax.transAxes, clip_on=False,
            )
            sidebar_ax.text(
                x0 + cbar_width / 2, cbar_top + 0.08, entity,
                fontsize=5, fontfamily="monospace", fontweight="bold",
                ha="center", va="bottom",
                transform=sidebar_ax.transAxes,
            )

        # --- Goal progress section ---
        sidebar_ax.text(
            cbar_left, goal_section_top - 0.01, "Goal:",
            fontsize=7, fontfamily="monospace", fontweight="bold",
            ha="left", va="top", transform=sidebar_ax.transAxes,
        )
        goal_text_artists: list[tuple[Any, str | None]] = []
        for i, line in enumerate(goal_lines):
            y_pos = goal_section_top - 0.01 - (i + 1) * goal_line_spacing
            stripped = line.strip()
            literal_str: str | None = None
            if stripped.startswith("(") and stripped.endswith(")") and "(" not in stripped[1:]:
                literal_str = stripped
            # For static plots, show the final state colors directly
            if show_actions and literal_str is not None:
                color = "green" if goal_snapshots_at_end.get(literal_str, False) else "red"
            else:
                color = "red" if literal_str else "gray"
            txt = sidebar_ax.text(
                cbar_left, y_pos, line,
                fontsize=6, fontfamily="monospace",
                color=color,
                ha="left", va="top", transform=sidebar_ax.transAxes,
            )
            goal_text_artists.append((txt, literal_str))

        # --- Action list ---
        actions = self.actions_taken
        n_actions = len(actions)
        action_y_top = cbar_top
        action_y_bottom = cbar_bottom
        min_gap = 0.025

        action_y_positions: list[float] = []
        if n_actions > 0:
            y_range_avail = action_y_top - action_y_bottom
            for _act_name, act_time in actions:
                frac = act_time / t_end if t_end > 0 else 0.0
                action_y_positions.append(action_y_top - frac * y_range_avail)
            for i in range(1, n_actions):
                if action_y_positions[i] > action_y_positions[i - 1] - min_gap:
                    action_y_positions[i] = action_y_positions[i - 1] - min_gap

        action_texts: list[tuple[Any, float]] = []
        for i, (act_name, act_time) in enumerate(actions):
            y_pos = action_y_positions[i]
            txt = sidebar_ax.text(
                actions_left, y_pos, f"{i+1}. {act_name}",
                fontsize=5, fontfamily="monospace",
                ha="left", va="top",
                transform=sidebar_ax.transAxes,
                alpha=1.0 if show_actions else 0.0,
            )
            action_texts.append((txt, act_time))

        return goal_text_artists, action_texts

    def _create_trajectory_figure(
        self: PlannerDashboard,
        figsize: tuple[float, float],
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any, float] | None:
        """Create a GridSpec figure with main + sidebar + optional overhead axes.

        Returns (fig, main_ax, sidebar_ax, trajectories, env_coords, grid, t_end),
        or None if there are no trajectories to display.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        trajectories, env_coords, grid = self._build_entity_trajectories(
            location_coords=location_coords,
            include_objects=False,
        )

        if self._goal_time is not None:
            t_end = self._goal_time
        else:
            t_end = 0.0
            for _entity, (_wps, traj_times) in trajectories.items():
                if traj_times:
                    t_end = max(t_end, max(traj_times))
        if t_end <= 0.0:
            return None

        has_overhead = getattr(getattr(self._env, "scene", None), "get_top_down_image", None) is not None

        fig = plt.figure(figsize=figsize)
        if has_overhead:
            gs = GridSpec(1, 3, width_ratios=[1, 2, 1], figure=fig,
                         wspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
            overhead_ax = fig.add_subplot(gs[0, 0])
            main_ax = fig.add_subplot(gs[0, 1])
            sidebar_ax = fig.add_subplot(gs[0, 2])
            self._render_overhead(overhead_ax)
        else:
            gs = GridSpec(1, 2, width_ratios=[3, 1], figure=fig,
                         wspace=0.1, left=0.05, right=0.97, top=0.95, bottom=0.05)
            main_ax = fig.add_subplot(gs[0, 0])
            sidebar_ax = fig.add_subplot(gs[0, 1])
        sidebar_ax.set_axis_off()

        # Plot grid background if available
        if grid is not None:
            try:
                from railroad.environment.procthor.plotting import plot_grid
                plot_grid(main_ax, grid)
            except ImportError:
                pass

        return fig, main_ax, sidebar_ax, trajectories, env_coords, grid, t_end

    def save_trajectory_video(
        self: PlannerDashboard,
        path: str,
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
        fps: int = 60,
        duration: float = 10.0,
        figsize: tuple[float, float] = (12.8, 7.2),
        dpi: int = 150,
    ) -> None:
        """Save an animated trajectory video/GIF.

        Args:
            path: Output file path. Extension determines writer:
                ``.mp4``/``.avi`` uses FFMpegWriter.
            location_coords: Optional explicit location->(x,y) mapping.
            fps: Frames per second.
            duration: Total animation duration in seconds.
            figsize: Figure size in inches.
            dpi: Resolution in dots per inch.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        result = self._create_trajectory_figure(figsize, location_coords=location_coords)
        if result is None:
            return
        fig, ax, sidebar_ax, trajectories, env_coords, grid, t_end = result

        n_frames = int(fps * duration)
        frame_times = np.linspace(0.0, t_end, n_frames)

        # Pre-compute dense trail positions (same resolution as static plot)
        dense_trail_times = np.linspace(0.0, t_end, 2000)
        dense_positions = self.get_entity_positions_at_times(
            dense_trail_times, location_coords=location_coords,
        )
        if not dense_positions:
            plt.close(fig)
            return

        # Pre-compute low-res positions for the moving marker
        marker_positions = self.get_entity_positions_at_times(
            frame_times, location_coords=location_coords,
        )
        entity_names = sorted(dense_positions.keys())

        # Static background: location markers + labels
        plotted_locs: set[str] = set()
        for positions in self._entity_positions.values():
            for _, loc_name, stored_coords in positions:
                if loc_name in plotted_locs:
                    continue
                plotted_locs.add(loc_name)
                coord = stored_coords if stored_coords is not None else env_coords.get(loc_name)
                if coord is not None:
                    ax.plot(coord[0], coord[1], "ks", markersize=4, zorder=4)
                    ax.annotate(
                        loc_name, coord,
                        fontsize=5, color="brown",
                        xytext=(3, 3), textcoords="offset points",
                    )

        # Render sidebar (colorbars, goals, actions hidden initially)
        goal_text_artists, action_texts = self._render_sidebar(
            sidebar_ax, t_end,
            goal_snapshots_at_end={},  # video animates from empty
            show_actions=False,
        )

        # Pre-compute goal snapshots for animation
        goal_snapshots: list[tuple[float, dict[str, bool]]] = []
        for entry in self.history:
            goal_snapshots.append((entry["time"], entry["goals"]))

        # --- Pre-compute per-point sizes and RGBA colors for all robots ---
        n_dense = len(dense_trail_times)
        per_point_sizes = np.linspace(self._TRAIL_SIZE_START, self._TRAIL_SIZE_END, n_dense)
        norm_t = dense_trail_times / t_end if t_end > 0 else np.zeros(n_dense)

        # Build combined arrays for all robots
        all_dense_xy: list[Any] = []
        all_dense_sizes: list[Any] = []
        all_dense_colors: list[Any] = []
        all_dense_times: list[Any] = []

        # --- Main plot: entity artists (markers + labels only) ---
        markers = []
        labels = []
        for idx, entity in enumerate(entity_names):
            pos0 = marker_positions[entity][0]
            (marker,) = ax.plot(
                [pos0[0]], [pos0[1]], "o",
                color=self._marker_color(idx),
                markeredgecolor="black", markeredgewidth=1.0,
                markersize=11, zorder=10, label=entity,
            )
            label = ax.text(
                pos0[0], pos0[1], entity,
                fontsize=6, fontfamily="monospace", fontweight="bold",
                ha="center", va="bottom",
                zorder=11,
            )
            markers.append(marker)
            labels.append(label)

            if entity in dense_positions:
                cmap = self._get_cmap(idx)
                rgba = cmap(norm_t)
                all_dense_xy.append(dense_positions[entity])
                all_dense_sizes.append(per_point_sizes)
                all_dense_colors.append(rgba)
                all_dense_times.append(dense_trail_times)

        # Concatenate and sort by time for correct draw order
        if all_dense_xy:
            combined_xy = np.concatenate(all_dense_xy)
            combined_sizes = np.concatenate(all_dense_sizes)
            combined_colors = np.concatenate(all_dense_colors)
            combined_times = np.concatenate(all_dense_times)
            sort_order = np.argsort(combined_times)
            combined_xy = combined_xy[sort_order]
            combined_sizes = combined_sizes[sort_order]
            combined_colors = combined_colors[sort_order]
            combined_times = combined_times[sort_order]
        else:
            combined_xy = np.empty((0, 2))
            combined_sizes = np.empty(0)
            combined_colors = np.empty((0, 4))
            combined_times = np.empty(0)

        # Single shared trail scatter artist
        trail_scatter = ax.scatter([], [], s=[], zorder=5, alpha=1.0)

        ax.legend(fontsize=7, loc="upper right")
        # Fixed-width title so it doesn't jump during animation
        t_width = len(f"{t_end:.1f}")
        title_artist = ax.set_title(
            f"Entity Trajectories  (cost = {'0.0':>{t_width}} / {t_end:.1f})",
            fontfamily="monospace", fontsize=10,
        )

        if grid is None:
            ax.autoscale()
            ax.set_aspect("equal", adjustable="datalim")

        # Compute label offset in data coords (fraction of y-axis range)
        y_range = ax.get_ylim()
        label_offset = (y_range[1] - y_range[0]) * 0.02

        def _update(frame: int):
            current_time = frame_times[frame]
            title_artist.set_text(
                f"Entity Trajectories  (cost = {current_time:>{t_width}.1f} / {t_end:.1f})"
            )
            for idx, entity in enumerate(entity_names):
                # Update current position marker
                pos = marker_positions[entity]
                markers[idx].set_data([pos[frame, 0]], [pos[frame, 1]])
                # Update label position (just above the marker)
                labels[idx].set_position((pos[frame, 0], pos[frame, 1] + label_offset))
            # Update combined trail: mask points up to current time
            mask = combined_times <= current_time
            trail_scatter.set_offsets(combined_xy[mask])
            trail_scatter.set_sizes(combined_sizes[mask])
            trail_scatter.set_facecolors(combined_colors[mask])
            # Reveal actions whose start time has passed
            for txt, act_time in action_texts:
                if current_time >= act_time:
                    txt.set_alpha(1.0)
            # Update goal literal colors based on latest snapshot <= current_time
            current_goals: dict[str, bool] = {}
            for snap_time, snap_goals in goal_snapshots:
                if snap_time <= current_time:
                    current_goals = snap_goals
                else:
                    break
            for txt, literal_str in goal_text_artists:
                if literal_str is not None:
                    satisfied = current_goals.get(literal_str, False)
                    txt.set_color("green" if satisfied else "red")
            return (markers + labels + [trail_scatter, title_artist]
                    + [txt for txt, _ in action_texts]
                    + [txt for txt, _ in goal_text_artists])

        anim = FuncAnimation(fig, _update, frames=n_frames, blit=False, interval=1000 / fps)

        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps)

        interrupted = False
        try:
            if not _is_headless_environment():
                from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
                with Progress(
                    TextColumn("[bold blue]Saving video"),
                    BarColumn(),
                    TextColumn("{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                ) as progress:
                    task = progress.add_task("frames", total=n_frames)
                    def _progress_cb(current_frame: int, total_frames: int) -> None:
                        progress.update(task, completed=current_frame)
                    anim.save(path, writer=writer, dpi=dpi, progress_callback=_progress_cb)
            else:
                anim.save(path, writer=writer, dpi=dpi)
        except KeyboardInterrupt:
            interrupted = True
        except subprocess.CalledProcessError as exc:
            if isinstance(exc.__context__, KeyboardInterrupt):
                interrupted = True
            else:
                raise
        plt.close(fig)
        if interrupted:
            self.console.print("[yellow]Video generation interrupted — saved partial video.[/yellow]")

    def get_plot_image(
        self: PlannerDashboard,
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
        figsize: tuple[float, float] = (12.8, 7.2),
        dpi: int = 150,
        quality: int = 85,
    ) -> bytes | None:
        """Render the trajectory plot to JPEG bytes in memory.

        Uses the Agg backend for headless rendering (safe in subprocess workers).

        Args:
            location_coords: Optional explicit location->(x,y) mapping.
            figsize: Figure size in inches.
            dpi: Resolution in dots per inch.
            quality: JPEG quality (1-95).

        Returns:
            JPEG image bytes, or None if there are no trajectories to display.
        """
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = self._create_trajectory_figure(figsize, location_coords=location_coords)
        if result is None:
            return None
        fig, main_ax, sidebar_ax, _trajs, _env_coords, _grid, t_end = result

        self.plot_trajectories(ax=main_ax, location_coords=location_coords)

        goal_snapshots_at_end: dict[str, bool] = {}
        if self.history:
            goal_snapshots_at_end = self.history[-1].get("goals", {})

        self._render_sidebar(
            sidebar_ax, t_end,
            goal_snapshots_at_end=goal_snapshots_at_end,
            show_actions=True,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=dpi,
                    bbox_inches="tight",
                    pil_kwargs={"quality": quality})
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def show_plots(
        self: PlannerDashboard,
        *,
        save_plot: str | None = None,
        show_plot: bool = False,
        save_video: str | None = None,
        location_coords: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Convenience method that handles plot/video output based on CLI flags.

        Args:
            save_plot: If set, save the trajectory plot to this file path.
            show_plot: If True, display the trajectory plot interactively.
            save_video: If set, save a trajectory animation to this file path.
            location_coords: Optional explicit location->(x,y) mapping.
        """
        if not save_plot and not show_plot and not save_video:
            return

        if save_plot or show_plot:
            import matplotlib.pyplot as plt

            result = self._create_trajectory_figure(
                (12.8, 7.2), location_coords=location_coords,
            )
            if result is not None:
                fig, main_ax, sidebar_ax, _trajs, _env_coords, _grid, t_end = result

                # Draw trajectories on the main axes
                self.plot_trajectories(ax=main_ax, location_coords=location_coords)

                # Get final goal snapshot from history
                goal_snapshots_at_end: dict[str, bool] = {}
                if self.history:
                    goal_snapshots_at_end = self.history[-1].get("goals", {})

                self._render_sidebar(
                    sidebar_ax, t_end,
                    goal_snapshots_at_end=goal_snapshots_at_end,
                    show_actions=True,
                )

                if save_plot:
                    fig.savefig(save_plot, dpi=300, bbox_inches='tight')
                    self.console.print(f"Saved plot to [yellow]{save_plot}[/yellow]")
                if show_plot:
                    plt.show()
                else:
                    plt.close(fig)

        if save_video:
            self.save_trajectory_video(save_video, location_coords=location_coords)
            self.console.print(f"Saved video to [yellow]{save_video}[/yellow]")
