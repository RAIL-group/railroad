# Renders a terrain graph: edges colored by failure hazard, goals as stars, start as a circle.
# Called during a run by scripts/playground.py and for the paper's topology panels by figures.py.

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from resilient_mrp.planning.core import ResilientGraph


class GraphVisualizer:
    def __init__(self, graph: ResilientGraph):
        self.graph = graph
        self.coords = graph.node_coords
        self.norm = Normalize(vmin=0.0, vmax=1.0)
        # Truncated YlOrRd so 0.0 lands on a visible amber rather than near-white.
        base = plt.get_cmap("YlOrRd")
        self.cmap = LinearSegmentedColormap.from_list("YlOrRd_bold", base(np.linspace(0.25, 1.0, 256)))

    # Render the graph; save to save_path and/or show it interactively.
    def render(self, title: str, save_path: Path | None = None, show: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')
        self._draw_edges(ax)
        self._draw_nodes(ax)
        self._style_axes(ax, title)
        self._add_colorbar(fig, ax)
        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            print(f"Graph saved: {save_path}")
        if show or not save_path:
            plt.show()
        plt.close()

    # Draw each undirected edge once, colored by its failure hazard and labeled with the value.
    def _draw_edges(self, ax) -> None:
        drawn: set = set()
        for (a, b), props in self.graph.edges.items():
            if (b, a) in drawn or a not in self.coords or b not in self.coords:
                continue
            drawn.add((a, b))
            hazard = props.get('hazard_severity', 0.0)
            (x1, y1), (x2, y2) = self.coords[a], self.coords[b]
            ax.plot([x1, x2], [y1, y2], color=self.cmap(self.norm(hazard)),
                    linewidth=2.2, alpha=0.9, zorder=1)
            # labelled so edges with similar colors stay distinguishable
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, f"{hazard:.2f}", ha='center', va='center', fontsize=6,
                    color='#222222', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

    # Monochrome nodes keep the edge colormap the only meaningful hue.
    def _draw_nodes(self, ax) -> None:
        for node, (x, y) in self.coords.items():
            if node == "start":
                ax.scatter([x], [y], c='white', s=200, marker='o', edgecolors='#1A1A1A', linewidth=2.0, zorder=6)
                ax.annotate('start', (x, y), textcoords='offset points', xytext=(0, -10),
                            ha='center', va='top', fontsize=9, color='#1A1A1A')
            elif node.startswith("g") and node[1:].isdigit():
                ax.scatter([x], [y], c='#1A1A1A', s=320, marker='*', edgecolors='white', linewidth=0.6, zorder=6)
                ax.annotate(node, (x, y), textcoords='offset points', xytext=(0, 11),
                            ha='center', va='bottom', fontsize=12, color='#1A1A1A', fontweight='bold')
            else:
                ax.scatter([x], [y], c='#9E9E9E', s=55, marker='s', edgecolors='#5A5A5A', linewidth=0.6, zorder=3)

    # Title, axis labels, spines, and grid styling.
    def _style_axes(self, ax, title: str) -> None:
        ax.set_title(title, fontsize=12, color='#333333', pad=10)
        ax.tick_params(colors='#666666', labelsize=8)
        ax.set_xlabel('X (m)', fontsize=9, color='#666666')
        ax.set_ylabel('Y (m)', fontsize=9, color='#666666')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('bottom', 'left'):
            ax.spines[side].set_color('#cccccc')
        ax.grid(True, alpha=0.25, color='#e0e0e0', linewidth=0.5)

    # Colorbar mapping edge color back to failure probability.
    def _add_colorbar(self, fig, ax) -> None:
        cbar = fig.colorbar(ScalarMappable(norm=self.norm, cmap=self.cmap), ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('Edge failure probability', fontsize=9, color='#333333')
        cbar.ax.tick_params(labelsize=8, colors='#666666')
