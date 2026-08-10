from __future__ import annotations

import subprocess
from typing import Any, TYPE_CHECKING

from ._goals import format_goal
from ._tui import _generate_coordinates, _is_ci_environment

if TYPE_CHECKING:
    from .dashboard import PlannerDashboard


def _canvas_buffer_writer_cls() -> Any:
    """Return an ``FFMpegWriter`` that pipes the canvas buffer as-is.

    ``MovieWriter.grab_frame`` routes through ``fig.savefig``, which
    re-rasterises the whole figure. ``save_video`` has already composited the
    frame into the canvas buffer, so those bytes go straight to ffmpeg.
    """
    from matplotlib.animation import FFMpegWriter

    class _CanvasBufferWriter(FFMpegWriter):
        def grab_frame(self, **savefig_kwargs: Any) -> None:
            if savefig_kwargs:
                # The base class forwards these to savefig; this one never
                # calls savefig, so honouring them silently is not an option.
                raise TypeError(
                    "_CanvasBufferWriter.grab_frame takes no savefig kwargs, "
                    f"got {sorted(savefig_kwargs)}"
                )
            proc: Any = self._proc  # ty: ignore[unresolved-attribute]
            canvas: Any = self.fig.canvas
            buf = canvas.buffer_rgba()
            # ffmpeg reads fixed-size frames off a raw stream, so a buffer of
            # the wrong size would shear the whole video rather than fail.
            width, height = self.frame_size
            expected = width * height * 4
            if buf.nbytes != expected:
                raise RuntimeError(
                    f"canvas buffer is {buf.nbytes} bytes but ffmpeg expects "
                    f"{expected} ({width}x{height}); figure dpi drifted from "
                    f"the writer's {self.dpi}"
                )
            proc.stdin.write(buf)

    return _CanvasBufferWriter


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

    @staticmethod
    def _try_embed_video_in_notebook(path: str) -> None:
        """Best-effort inline video display for notebook environments."""
        try:
            import importlib
            ipython_mod = importlib.import_module("IPython")
            display_mod = importlib.import_module("IPython.display")
        except Exception:
            return

        get_ipython = getattr(ipython_mod, "get_ipython", None)
        if get_ipython is None:
            return

        shell = get_ipython()
        if shell is None or getattr(shell, "kernel", None) is None:
            return

        try:
            video_cls = getattr(display_mod, "Video", None)
            display_fn = getattr(display_mod, "display", None)
            if video_cls is None or display_fn is None:
                return
            display_fn(video_cls(path, embed=True))
        except Exception:
            # Never fail video save if inline rendering is unavailable.
            return

    def _build_entity_trajectories(
        self: PlannerDashboard,
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
        include_objects: bool = False,
    ) -> tuple[
        dict[str, tuple[list[tuple[float, float]], list[float]]],  # entity -> (waypoints, times)
        dict[str, tuple[float, float]],  # resolved env_coords
    ]:
        """Build resolved and expanded entity trajectories.

        Handles: environment coordinate lookup, ``location_coords`` conflict
        check, missing-coordinate generation, robot/object separation, and
        environment ``compute_move_path`` expansion when available.

        Args:
            location_coords: Optional explicit location->(x,y) mapping.
                Overrides environment-provided coords for matching keys;
                per-position stored coordinates still take priority.
            include_objects: If True, also include non-robot entities.

        Returns:
            (trajectories, env_coords) where *trajectories* maps entity
            names to (waypoints, times) lists with path-expanded routes.
        """
        # Get environment coordinates
        env_coords = self._get_location_coords()

        # Handle explicit location_coords — overrides environment coords.
        # Per-position stored_coords still take priority at usage time.
        if location_coords is not None:
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

        # Environment-aware trajectory expansion
        compute_path_between_points_fn = getattr(self._env, "compute_path_between_points", None)
        compute_move_path_fn = getattr(self._env, "compute_move_path", None)

        trajectories: dict[str, tuple[list[tuple[float, float]], list[float]]] = {}

        # Build robot trajectories (with scene expansion)
        for entity, positions in sorted(robot_entities.items()):
            # Prefer continuous navigation positions (already obstacle-respecting)
            nav_pos = self._nav_continuous_positions.get(entity)
            if nav_pos:
                waypoints = [(x, y) for _, x, y in nav_pos]
                times = [t for t, _, _ in nav_pos]
                # Append any finalized positions after the last continuous sample
                # (handles upcoming-effect extrapolation at goal time)
                last_t = times[-1] if times else -1.0
                for t, loc_name, stored_coords in positions:
                    if t > last_t:
                        coord = stored_coords if stored_coords is not None else env_coords.get(loc_name)
                        if coord is not None:
                            waypoints.append(coord)
                            times.append(t)
                trajectories[entity] = (waypoints, times)
                continue

            waypoints = []
            times = []
            waypoint_locs: list[str] = []
            for t, loc_name, stored_coords in positions:
                if stored_coords is not None:
                    waypoints.append(stored_coords)
                    times.append(t)
                    waypoint_locs.append(loc_name)
                elif loc_name in env_coords:
                    waypoints.append(env_coords[loc_name])
                    times.append(t)
                    waypoint_locs.append(loc_name)

            if len(waypoints) < 2:
                trajectories[entity] = (waypoints, times)
                continue

            # Expand through environment pathing if available.
            if callable(compute_path_between_points_fn) or callable(compute_move_path_fn):
                import numpy as np
                expanded_wps: list[tuple[float, float]] = []
                expanded_ts: list[float] = []
                for seg_i in range(len(waypoints) - 1):
                    seg_start = waypoints[seg_i]
                    seg_end = waypoints[seg_i + 1]
                    seg_loc_from = waypoint_locs[seg_i]
                    seg_loc_to = waypoint_locs[seg_i + 1]

                    path: list[tuple[float, float]] | None = None

                    # Prefer coordinate-based pathing (uses dashboard's stored coords)
                    if callable(compute_path_between_points_fn):
                        try:
                            path_arr = compute_path_between_points_fn(seg_start, seg_end)
                            if getattr(path_arr, "size", 0) != 0 and path_arr.shape[0] == 2:
                                path = [
                                    (float(path_arr[0, j]), float(path_arr[1, j]))
                                    for j in range(path_arr.shape[1])
                                ]
                        except Exception:
                            path = None

                    # Fall back to name-based pathing
                    if (not path or len(path) < 2) and callable(compute_move_path_fn):
                        try:
                            path_arr = compute_move_path_fn(seg_loc_from, seg_loc_to)
                            if getattr(path_arr, "size", 0) != 0 and path_arr.shape[0] == 2:
                                path = [
                                    (float(path_arr[0, j]), float(path_arr[1, j]))
                                    for j in range(path_arr.shape[1])
                                ]
                        except Exception:
                            path = None

                    if not path or len(path) < 2:
                        path = [seg_start, seg_end]
                    pts = np.array(path)
                    cum_dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
                    t0, t1 = times[seg_i], times[seg_i + 1]
                    if cum_dist[-1] > 1e-9:
                        seg_times = (t0 + (t1 - t0) * cum_dist / cum_dist[-1]).tolist()
                    else:
                        # Stationary segment (e.g. robot holding position
                        # during pick/place/no_op): path collapses to a
                        # single point, so distance-based timing would drop
                        # t1 and make interpolation leak into the next move.
                        seg_times = np.linspace(t0, t1, len(path)).tolist()
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

        return trajectories, env_coords

    def _compute_trail_arrays(
        self: PlannerDashboard,
        trajectories: dict[str, tuple[list[tuple[float, float]], list[float]]],
        t_end: float,
        *,
        n_points: int = 2000,
    ) -> tuple[Any, Any, Any, Any] | None:
        """Compute dense scatter-trail arrays for all robots, sorted by time.

        Returns ``(xy, sizes, colors, times)`` or ``None`` if empty.
        """
        import numpy as np

        dense_times = np.linspace(0.0, t_end, n_points) if t_end > 0 else np.array([0.0])
        n_pts = len(dense_times)
        sizes = np.linspace(self._TRAIL_SIZE_START, self._TRAIL_SIZE_END, n_pts)
        norm_t = dense_times / t_end if t_end > 0 else np.zeros_like(dense_times)

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

            # Interpolate positions from trajectory waypoints
            wp_arr = np.array(waypoints)
            t_arr = np.array(times)
            x_interp = np.interp(dense_times, t_arr, wp_arr[:, 0])
            y_interp = np.interp(dense_times, t_arr, wp_arr[:, 1])
            pts = np.column_stack([x_interp, y_interp])

            rgba = cmap(norm_t)
            all_xy.append(pts)
            all_sizes.append(sizes)
            all_colors.append(rgba)
            all_times.append(dense_times)

        if not all_xy:
            return None

        combined_xy = np.concatenate(all_xy)
        combined_sizes = np.concatenate(all_sizes)
        combined_colors = np.concatenate(all_colors)
        combined_times = np.concatenate(all_times)
        order = np.argsort(combined_times)
        return (
            combined_xy[order],
            combined_sizes[order],
            combined_colors[order],
            combined_times[order],
        )

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
        trajectories, env_coords = self._build_entity_trajectories(
            location_coords=location_coords,
            include_objects=show_objects,
        )

        # Plot grid background if available
        occupancy_grid = getattr(self._env, 'occupancy_grid', None)
        if occupancy_grid is not None:
            from railroad.navigation.plotting import plot_grid_background
            true_grid = getattr(self._env, 'true_grid', None)
            plot_grid_background(ax, occupancy_grid, true_grid)
            # Predicted frontier probabilities (LSP environments): color
            # each frontier's cells by its prob_feasible.
            overlays = getattr(self._env, 'frontier_probability_overlays', None)
            if overlays:
                from railroad.navigation.plotting import make_frontier_overlay_rgba
                ax.imshow(
                    make_frontier_overlay_rgba(occupancy_grid.shape, overlays),
                    origin="upper", zorder=1,
                )

        if self._goal_time is not None:
            t_end = self._goal_time
        else:
            t_end = 0.0
            for _entity, (_wps, traj_times) in trajectories.items():
                if traj_times:
                    t_end = max(t_end, max(traj_times))
        trail = self._compute_trail_arrays(trajectories, t_end)

        # Plot robot location markers and labels
        for entity in sorted(self.known_robots & set(trajectories.keys())):
            waypoints, _times = trajectories[entity]
            if len(waypoints) < 2:
                continue

            # Get original positions for location markers/labels.
            # Skip frontiers: they're transient exploration targets, not
            # persistent landmarks worth marking on the trajectory.
            positions = self._entity_positions[entity]
            annotated_locs: list[tuple[str, tuple[float, float]]] = []
            for _, loc_name, stored_coords in positions:
                if loc_name.startswith("frontier_"):
                    continue
                if stored_coords is not None:
                    annotated_locs.append((loc_name, stored_coords))
                elif loc_name in env_coords:
                    annotated_locs.append((loc_name, env_coords[loc_name]))

            if not annotated_locs:
                continue

            wx = [c[0] for _, c in annotated_locs]
            wy = [c[1] for _, c in annotated_locs]
            ax.scatter(wx, wy, s=20, zorder=6, color="black")

            ax.annotate(
                entity, (wx[0], wy[0]),
                fontsize=7, fontweight="bold", color="brown",
            )

            for loc_name, coord in annotated_locs:
                ax.annotate(
                    loc_name, coord,
                    fontsize=5, color="brown",
                    xytext=(3, 3), textcoords="offset points",
                )

        # Draw one combined scatter for all robot trails, sorted by time
        if trail is not None:
            combined_xy, combined_sizes, combined_colors, _combined_times = trail
            ax.scatter(
                combined_xy[:, 0], combined_xy[:, 1],
                s=combined_sizes, c=combined_colors,
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
        if occupancy_grid is None:
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
        trajectories, _env_coords = self._build_entity_trajectories(
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

        Shared by show_plots() (static) and save_video() (animated).

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

        # Total width used by colorbar strips (plus the frontier
        # probability legend when overlays were captured)
        has_frontier_cbar = bool(self._frontier_overlay_snapshots)
        cbar_total_width = n_entities * cbar_width + max(0, n_entities - 1) * cbar_gap
        if has_frontier_cbar:
            # Extra room for the legend's right-side tick labels
            cbar_total_width += 2 * cbar_gap + cbar_width + 0.06
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

        # --- Frontier probability legend ---
        if has_frontier_cbar:
            x0 = cbar_left + n_entities * (cbar_width + cbar_gap) + 2 * cbar_gap
            cbar_ax = sidebar_ax.inset_axes((x0, cbar_bottom, cbar_width, cbar_height))
            gradient = np.linspace(0.1, 0.9, 256).reshape(-1, 1)
            cbar_ax.imshow(gradient, aspect="auto", cmap="viridis", origin="lower",
                           vmin=0.0, vmax=1.0, extent=(0, 1, 0, 1))
            cbar_ax.set_xticks([])
            cbar_ax.set_ylim(0, 1)
            cbar_ax.yaxis.tick_right()
            cbar_ax.set_yticks([0, 0.5, 1])
            cbar_ax.tick_params(axis="y", labelsize=5)
            sidebar_ax.text(
                x0 + cbar_width / 2, cbar_top + 0.04, "frontier\nP(goal)",
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
        onboard_robots: list[str] | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, float, dict[str, Any]] | None:
        """Create a GridSpec figure with main + sidebar + optional overhead axes.

        When ``onboard_robots`` is given, a bottom row is added with one axes
        per robot for onboard camera imagery.

        Returns (fig, main_ax, sidebar_ax, trajectories, env_coords, t_end,
        onboard_axes), or None if there are no trajectories to display.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        trajectories, env_coords = self._build_entity_trajectories(
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
        n_onboard = len(onboard_robots) if onboard_robots else 0

        fig = plt.figure(figsize=figsize)
        onboard_spec = None
        onboard_stacked = False
        if has_overhead:
            if n_onboard:
                # Onboard images stack in the left column, under the
                # top-down view; main plot and sidebar span both rows.
                gs = GridSpec(2, 3, width_ratios=[1, 2, 1],
                             height_ratios=[3, n_onboard], figure=fig,
                             wspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
                overhead_ax = fig.add_subplot(gs[0, 0])
                main_ax = fig.add_subplot(gs[:, 1])
                sidebar_ax = fig.add_subplot(gs[:, 2])
                onboard_spec = gs[1, 0]
                onboard_stacked = True
            else:
                gs = GridSpec(1, 3, width_ratios=[1, 2, 1], figure=fig,
                             wspace=0.1, left=0.03, right=0.97, top=0.95, bottom=0.05)
                overhead_ax = fig.add_subplot(gs[0, 0])
                main_ax = fig.add_subplot(gs[0, 1])
                sidebar_ax = fig.add_subplot(gs[0, 2])
            self._render_overhead(overhead_ax)
        else:
            # No top-down view: onboard images go in a bottom row instead.
            n_rows = 2 if n_onboard else 1
            height_ratios = [3, 1] if n_onboard else None
            gs = GridSpec(n_rows, 2, width_ratios=[3, 1], height_ratios=height_ratios,
                         figure=fig,
                         wspace=0.1, left=0.05, right=0.97, top=0.95, bottom=0.05)
            main_ax = fig.add_subplot(gs[0, 0])
            sidebar_ax = fig.add_subplot(gs[0, 1])
            if n_onboard:
                onboard_spec = gs[1, :]
        sidebar_ax.set_axis_off()

        onboard_axes: dict[str, Any] = {}
        if onboard_robots and onboard_spec is not None:
            if onboard_stacked:
                onboard_gs = GridSpecFromSubplotSpec(
                    n_onboard, 1, subplot_spec=onboard_spec, hspace=0.25,
                )
            else:
                onboard_gs = GridSpecFromSubplotSpec(
                    1, n_onboard, subplot_spec=onboard_spec, wspace=0.05,
                )
            for i, robot in enumerate(onboard_robots):
                index = (i, 0) if onboard_stacked else (0, i)
                onboard_ax = fig.add_subplot(onboard_gs[index])
                onboard_ax.set_axis_off()
                onboard_axes[robot] = onboard_ax

        return fig, main_ax, sidebar_ax, trajectories, env_coords, t_end, onboard_axes

    def save_video(
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
        import matplotlib as mpl
        import matplotlib.image as mimage
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import to_rgba

        # Onboard camera imagery (duck-typed: env exposes pano_records with
        # .robot/.time/.image, e.g. VisualUnknownSpaceEnvironment)
        onboard_by_robot: dict[str, list[Any]] = {}
        for record in getattr(self._env, "pano_records", None) or []:
            onboard_by_robot.setdefault(record.robot, []).append(record)
        onboard_robots = sorted(onboard_by_robot)

        result = self._create_trajectory_figure(
            figsize, location_coords=location_coords, onboard_robots=onboard_robots,
        )
        if result is None:
            return
        fig, ax, sidebar_ax, trajectories, env_coords, t_end, onboard_axes = result

        # Render at the output resolution from the start. The frame loop
        # composites straight out of the canvas buffer, so the figure's dpi
        # has to match the writer's rather than being applied per savefig.
        fig.set_dpi(dpi)
        # Bind an Agg canvas explicitly: the compositing below needs Agg's
        # buffer_rgba/copy_from_bbox/restore_region, and an interactive canvas
        # would re-derive dpi from its device pixel ratio and silently render
        # at a size the writer is not expecting.
        FigureCanvasAgg(fig)
        # savefig applies `savefig.facecolor` and Animation.save pre-composited
        # it onto white; rendering the canvas directly bypasses both, so do it
        # here to keep non-default figure backgrounds looking the same.
        facecolor = mpl.rcParams["savefig.facecolor"]
        if facecolor == "auto":
            facecolor = fig.get_facecolor()
        red, green, blue, alpha = to_rgba(facecolor)
        fig.set_facecolor((
            alpha * red + 1 - alpha,
            alpha * green + 1 - alpha,
            alpha * blue + 1 - alpha,
            1.0,
        ))

        # One image artist per robot; frames are selected by capture time.
        onboard_artists: dict[str, Any] = {}
        onboard_times: dict[str, Any] = {}
        onboard_frame_idx: dict[str, int] = {}
        for robot in onboard_robots:
            records = sorted(onboard_by_robot[robot], key=lambda r: r.time)
            onboard_by_robot[robot] = records
            onboard_axes[robot].set_title(
                f"{robot} onboard", fontsize=8, fontfamily="monospace",
            )
            onboard_artists[robot] = onboard_axes[robot].imshow(records[0].image)
            onboard_times[robot] = np.array([r.time for r in records])
            onboard_frame_idx[robot] = 0

        # Animated navigation grid background
        from railroad.navigation.plotting import make_plotting_grid, make_plotting_grid_rgba, _BACKGROUND_GRAY

        nav_grid_artist = None
        nav_grid_frames: list[tuple[float, Any]] = []
        occupancy_grid = getattr(self._env, 'occupancy_grid', None)
        _nav_has_true_underlay = False
        if occupancy_grid is not None:
            true_grid = getattr(self._env, "true_grid", None)
            has_unknown = bool(np.any(occupancy_grid == -1.0))
            _nav_has_true_underlay = true_grid is not None and has_unknown

            if _nav_has_true_underlay:
                assert true_grid is not None
                # Pre-compute true grid underlay blended onto gray base
                base = np.full(3, _BACKGROUND_GRAY)
                true_rgb = make_plotting_grid(true_grid.T)
                underlay = base * (1 - 0.35) + true_rgb * 0.35

                def _composite_frame(obs_grid: np.ndarray) -> np.ndarray:
                    obs_rgba = make_plotting_grid_rgba(obs_grid.T)
                    obs_alpha = obs_rgba[:, :, 3:4]
                    return underlay * (1 - obs_alpha) + obs_rgba[:, :, :3] * obs_alpha
            else:
                def _composite_frame(obs_grid: np.ndarray) -> np.ndarray:
                    return make_plotting_grid(obs_grid.T)

            if self._nav_grid_snapshots:
                nav_grid_frames = [
                    (t, _composite_frame(g))
                    for t, g in self._nav_grid_snapshots
                ]
                nav_grid_artist = ax.imshow(
                    nav_grid_frames[0][1], origin="upper", zorder=0,
                )
            else:
                nav_grid_artist = ax.imshow(
                    _composite_frame(occupancy_grid), origin="upper", zorder=0,
                )

        # Animated frontier-probability overlay (LSP environments)
        frontier_overlay_artist = None
        frontier_overlay_frames: list[tuple[float, Any]] = []
        if occupancy_grid is not None and self._frontier_overlay_snapshots:
            from railroad.navigation.plotting import make_frontier_overlay_rgba
            frontier_overlay_frames = [
                (t, make_frontier_overlay_rgba(occupancy_grid.shape, overlays))
                for t, overlays in self._frontier_overlay_snapshots
            ]
            frontier_overlay_artist = ax.imshow(
                frontier_overlay_frames[0][1], origin="upper", zorder=1,
            )

        n_frames = int(fps * duration)
        animation_times = np.linspace(0.0, t_end, n_frames)
        # Prepend the final time as frame 0 so the video thumbnail/poster
        # shows the completed plan rather than the empty initial state.
        frame_times = np.concatenate(([t_end], animation_times))
        n_frames += 1

        # Pre-compute trail arrays (dense scatter data for all robots)
        trail = self._compute_trail_arrays(trajectories, t_end)
        if trail is not None:
            combined_xy, combined_sizes, combined_colors, combined_times = trail
        else:
            combined_xy = np.empty((0, 2))
            combined_sizes = np.empty(0)
            combined_colors = np.empty((0, 4))
            combined_times = np.empty(0)

        # Derive entity names from trajectories (robots with >= 2 waypoints)
        entity_names = sorted(
            e for e in (self.known_robots & set(trajectories.keys()))
            if len(trajectories[e][0]) >= 2
        )
        if not entity_names:
            plt.close(fig)
            return

        # Pre-compute low-res positions for the moving marker
        marker_positions = self.get_entity_positions_at_times(
            frame_times, location_coords=location_coords,
        )

        # Location markers + labels (skip transient frontiers). Their content
        # never changes, but they sit above the navigation grid, so they are
        # kept in a list to be redrawn whenever that grid does.
        location_artists: list[Any] = []
        plotted_locs: set[str] = set()
        for positions in self._entity_positions.values():
            for _, loc_name, stored_coords in positions:
                if loc_name in plotted_locs:
                    continue
                plotted_locs.add(loc_name)
                if loc_name.startswith("frontier_"):
                    continue
                coord = stored_coords if stored_coords is not None else env_coords.get(loc_name)
                if coord is not None:
                    location_artists.extend(
                        ax.plot(coord[0], coord[1], "ks", markersize=4, zorder=4)
                    )
                    location_artists.append(ax.annotate(
                        loc_name, coord,
                        fontsize=5, color="brown",
                        xytext=(3, 3), textcoords="offset points",
                    ))

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

        # Single shared trail scatter artist. The per-element antialias list
        # is length 2 (rather than the default scalar) so that Collection.draw
        # never takes its single-element "stamp one marker" shortcut: that
        # shortcut antialiases slightly differently, which would show up
        # whenever a frame happens to add exactly one trail point.
        trail_scatter = ax.scatter([], [], s=[], zorder=5, alpha=1.0)
        trail_scatter.set_antialiased([True, True])

        legend_artist = ax.legend(fontsize=7, loc="upper right")
        # Fixed-width title so it doesn't jump during animation
        t_width = len(f"{t_end:.1f}")
        title_artist = ax.set_title(
            f"Entity Trajectories  (cost = {'0.0':>{t_width}} / {t_end:.1f})",
            fontfamily="monospace", fontsize=10,
        )

        if occupancy_grid is None:
            ax.autoscale()
            ax.set_aspect("equal", adjustable="datalim")

        # Compute label offset in data coords (fraction of y-axis range)
        y_range = ax.get_ylim()
        label_offset = (y_range[1] - y_range[0]) * 0.02

        # Snapshot time axes, so each frame can binary-search its state
        # instead of rescanning every snapshot.
        nav_times = np.array([t for t, _ in nav_grid_frames]) if nav_grid_frames else None
        overlay_times = (
            np.array([t for t, _ in frontier_overlay_frames])
            if frontier_overlay_frames else None
        )
        goal_times = np.array([t for t, _ in goal_snapshots]) if goal_snapshots else None
        action_times = np.array([t for _, t in action_texts]) if action_texts else None

        def _latest(times: Any, current_time: float) -> int:
            """Index of the last snapshot at or before *current_time* (>= 0)."""
            return max(int(np.searchsorted(times, current_time, side="right")) - 1, 0)

        # Which snapshot each image artist is currently holding, so repeated
        # frames do not re-push identical data.
        shown_idx = {"nav": -1, "overlay": -1}

        def _apply_frame(frame: int) -> tuple[tuple, tuple, tuple, int]:
            """Point every artist at its state for *frame*.

            Returns one key per cached layer plus the number of visible trail
            points, which is what the render loop uses to work out how much of
            the previous frame it can keep.
            """
            current_time = frame_times[frame]
            title_artist.set_text(
                f"Entity Trajectories  (cost = {current_time:>{t_width}.1f} / {t_end:.1f})"
            )
            # Animated navigation grid. set_data copies and rescans the array,
            # so only push it when the snapshot actually changes.
            nav_idx = -1
            if nav_grid_artist is not None and nav_times is not None:
                nav_idx = _latest(nav_times, current_time)
                if nav_idx != shown_idx["nav"]:
                    nav_grid_artist.set_data(nav_grid_frames[nav_idx][1])
                    shown_idx["nav"] = nav_idx
            # Frontier-probability overlay
            overlay_idx = -1
            if frontier_overlay_artist is not None and overlay_times is not None:
                overlay_idx = _latest(overlay_times, current_time)
                if overlay_idx != shown_idx["overlay"]:
                    frontier_overlay_artist.set_data(
                        frontier_overlay_frames[overlay_idx][1]
                    )
                    shown_idx["overlay"] = overlay_idx
            for idx, entity in enumerate(entity_names):
                # Update current position marker
                pos = marker_positions[entity]
                markers[idx].set_data([pos[frame, 0]], [pos[frame, 1]])
                # Update label position (just above the marker)
                labels[idx].set_position((pos[frame, 0], pos[frame, 1] + label_offset))
            # Trail points up to the current time. combined_times is sorted,
            # so the visible set is always a prefix.
            n_trail = int(np.searchsorted(combined_times, current_time, side="right"))
            # Update onboard images to the latest capture <= current_time
            for robot, artist in onboard_artists.items():
                idx = _latest(onboard_times[robot], current_time)
                if idx != onboard_frame_idx[robot]:
                    artist.set_data(onboard_by_robot[robot][idx].image)
                    onboard_frame_idx[robot] = idx
            # Show/hide actions based on whether their start time has passed.
            # Compared per action rather than by a sorted cutoff, so this does
            # not quietly depend on actions_taken being in time order.
            actions_shown: tuple[bool, ...] = ()
            if action_times is not None:
                actions_shown = tuple(action_times <= current_time)
                for shown, (txt, _act_time) in zip(actions_shown, action_texts):
                    txt.set_alpha(1.0 if shown else 0.0)
            # Update goal literal colors based on latest snapshot <= current_time
            goal_idx = -1
            if goal_times is not None:
                current_goals: dict[str, bool] = {}
                if goal_times[0] <= current_time:
                    goal_idx = _latest(goal_times, current_time)
                    current_goals = goal_snapshots[goal_idx][1]
                for txt, literal_str in goal_text_artists:
                    if literal_str is not None:
                        satisfied = current_goals.get(literal_str, False)
                        txt.set_color("green" if satisfied else "red")

            return (
                (goal_idx, actions_shown),            # chrome layer
                tuple(onboard_frame_idx.values()),    # onboard images
                (nav_idx, overlay_idx),               # main-axes stack
                n_trail,
            )

        # Frames are composited out of two cached rasters rather than redrawn:
        #
        #   chrome  axes frames, ticks, sidebar and the onboard images -- a
        #           full canvas draw, needed only when the sidebar changes
        #   stack   chrome plus the navigation grid, frontier overlay,
        #           location markers and the trail drawn so far
        #
        # and the handful of artists that move every frame are drawn on top.
        # Artists cached in a layer are marked animated so a plain canvas draw
        # skips them and leaves them to be composited in the right order.
        hot_artists: list[Any] = [*markers, *labels, title_artist, legend_artist]
        # Draw them in the same order a full figure draw would, so anything
        # that overlaps still stacks the way it does in the static plot.
        # Ordering only holds within a layer: a hot artist always lands on top
        # of the stack raster, so anything added here that could overlap the
        # trail or the location markers needs a zorder above them (the title
        # is exempt only because it sits outside the axes box).
        hot_artists.sort(key=lambda a: a.get_zorder())

        stack_artists: list[Any] = [
            a for a in (nav_grid_artist, frontier_overlay_artist) if a is not None
        ]
        if stack_artists:
            # The grid images cover the whole data area, and a real draw puts
            # the frame and tick marks on top of them, so those have to be
            # replayed above the images rather than left in the chrome layer.
            stack_artists += [*ax.spines.values(), ax.xaxis, ax.yaxis]
        stack_artists += location_artists
        stack_artists.sort(key=lambda a: a.get_zorder())

        onboard_list = list(onboard_artists.values())
        canvas = fig.canvas

        # Artists composited by hand each frame. Marking them animated keeps
        # them out of a plain canvas draw -- except AxesImages, which
        # Axes.draw deliberately keeps, so those are hidden for that pass
        # instead. (Hiding the rest would move the title, whose placement
        # consults the tick bboxes.)
        layered: list[Any] = [
            *stack_artists, *onboard_list, *hot_artists, trail_scatter,
        ]
        for artist in layered:
            artist.set_animated(True)
        layered_images = [
            a for a in layered if isinstance(a, mimage.AxesImage)
        ]

        def _onboard_sizes() -> tuple:
            return tuple(a.get_size() for a in onboard_list)

        drawn_onboard_sizes = _onboard_sizes()

        def _draw_chrome() -> Any:
            """Full draw of everything that is not composited per frame."""
            nonlocal drawn_onboard_sizes
            for artist in layered_images:
                artist.set_visible(False)
            try:
                canvas.draw()
            finally:
                for artist in layered_images:
                    artist.set_visible(True)
            for artist in onboard_list:
                fig.draw_artist(artist)
            drawn_onboard_sizes = _onboard_sizes()
            return canvas.copy_from_bbox(fig.bbox)

        def _set_trail(lo: int, hi: int) -> None:
            trail_scatter.set_offsets(combined_xy[lo:hi])
            trail_scatter.set_sizes(combined_sizes[lo:hi])
            trail_scatter.set_facecolors(combined_colors[lo:hi])

        chrome: Any = None
        stack: Any = None
        drawn_chrome: tuple | None = None
        drawn_onboard: tuple | None = None
        drawn_stack: tuple | None = None
        drawn_trail = 0

        def _render(frame: int) -> None:
            """Composite one frame into the canvas buffer."""
            nonlocal chrome, stack, drawn_chrome, drawn_onboard, drawn_stack, drawn_trail
            chrome_key, onboard_key, stack_key, n_trail = _apply_frame(frame)

            # True when the canvas already holds the chrome layer, so the
            # stack rebuild below can skip restoring what is already there.
            chrome_on_canvas = False
            if chrome is None or chrome_key != drawn_chrome:
                chrome = _draw_chrome()
                drawn_chrome, drawn_onboard, stack = chrome_key, onboard_key, None
                chrome_on_canvas = True
            elif onboard_key != drawn_onboard:
                # Patching an onboard image over the cached chrome only works
                # while it repaints every pixel the previous one covered; a
                # differently-sized frame needs the full redraw.
                if _onboard_sizes() == drawn_onboard_sizes:
                    canvas.restore_region(chrome)
                    for artist in onboard_list:
                        fig.draw_artist(artist)
                    chrome = canvas.copy_from_bbox(fig.bbox)
                else:
                    chrome = _draw_chrome()
                    drawn_chrome = chrome_key
                drawn_onboard, stack = onboard_key, None
                chrome_on_canvas = True

            if stack is None or stack_key != drawn_stack or n_trail < drawn_trail:
                # Something under the trail moved (or the trail rewound, which
                # happens after the poster frame): rebuild the stack on chrome.
                if not chrome_on_canvas:
                    canvas.restore_region(chrome)
                for artist in stack_artists:
                    fig.draw_artist(artist)
                _set_trail(0, n_trail)
                fig.draw_artist(trail_scatter)
                stack = canvas.copy_from_bbox(fig.bbox)
                drawn_stack, drawn_trail = stack_key, n_trail
            else:
                canvas.restore_region(stack)
                if n_trail > drawn_trail:
                    # The trail only ever grows here, so draw the new points
                    # and fold them into the stack for the next frame.
                    _set_trail(drawn_trail, n_trail)
                    fig.draw_artist(trail_scatter)
                    stack = canvas.copy_from_bbox(fig.bbox)
                    drawn_trail = n_trail

            for artist in hot_artists:
                fig.draw_artist(artist)

        def _frames() -> Any:
            """Yield frame indices, with a progress bar when interactive."""
            if _is_ci_environment():
                yield from range(n_frames)
                return
            from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
            with Progress(
                TextColumn("[bold blue]Saving video"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("frames", total=n_frames)
                for frame in range(n_frames):
                    yield frame
                    progress.update(task, completed=frame + 1)

        writer = _canvas_buffer_writer_cls()(fps=fps)

        interrupted = False
        try:
            with writer.saving(fig, path, dpi):
                for frame in _frames():
                    _render(frame)
                    writer.grab_frame()
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
        else:
            self._try_embed_video_in_notebook(path)

    def _render_static_plot(
        self: PlannerDashboard,
        figsize: tuple[float, float],
        *,
        location_coords: dict[str, tuple[float, float]] | None = None,
    ) -> Any | None:
        """Create a complete static trajectory figure with sidebar, or ``None``."""
        result = self._create_trajectory_figure(figsize, location_coords=location_coords)
        if result is None:
            return None
        fig, main_ax, sidebar_ax, trajectories, env_coords, t_end, _ = result

        self.plot_trajectories(ax=main_ax, location_coords=location_coords)

        goal_snapshots_at_end: dict[str, bool] = {}
        if self.history:
            goal_snapshots_at_end = self.history[-1].get("goals", {})

        self._render_sidebar(
            sidebar_ax, t_end,
            goal_snapshots_at_end=goal_snapshots_at_end,
            show_actions=True,
        )

        return fig

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

        fig = self._render_static_plot(figsize, location_coords=location_coords)
        if fig is None:
            return None

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
        video_fps: int = 60,
        video_dpi: int = 150,
        location_coords: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Convenience method that handles plot/video output based on CLI flags.

        Args:
            save_plot: If set, save the trajectory plot to this file path.
            show_plot: If True, display the trajectory plot interactively.
            save_video: If set, save a trajectory animation to this file path.
            video_fps: Frames per second for video (default: 60).
            video_dpi: Resolution in dots per inch for video (default: 150).
            location_coords: Optional explicit location->(x,y) mapping.
        """
        if not save_plot and not show_plot and not save_video:
            return

        if save_plot or show_plot:
            import matplotlib.pyplot as plt

            fig = self._render_static_plot(
                (12.8, 7.2), location_coords=location_coords,
            )
            if fig is not None:
                if save_plot:
                    fig.savefig(save_plot, dpi=300, bbox_inches='tight')
                    self.console.print(f"Saved plot to [yellow]{save_plot}[/yellow]")
                if show_plot:
                    plt.show()
                else:
                    plt.close(fig)

        if save_video:
            self.save_video(
                save_video, location_coords=location_coords,
                fps=video_fps, dpi=video_dpi,
            )
            self.console.print(f"Saved video to [yellow]{save_video}[/yellow]")
