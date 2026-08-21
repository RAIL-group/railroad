"""Matplotlib artists for object sprites.

``AnnotationBbox`` around an ``OffsetImage`` is the only way to put a raster at a
data coordinate without stretching it with the axes. Two of its APIs do not do
what their names suggest, and both are load-bearing here:

* ``AnnotationBbox.set_alpha`` styles the (invisible) frame, not the image, and
  ``OffsetImage.set_alpha`` is never consulted by its draw. Fading means scaling
  the array's own alpha channel.
* ``AnnotationBbox.xy`` feeds the arrow target and the clip test; ``xybox`` is
  what actually positions the box.
"""

from __future__ import annotations

from typing import Any

SPRITE_POINTS = 18.0
"""Rendered sprite width, in points.

Sized in points rather than data units so a glyph stays legible whatever the map
covers -- the same reasoning behind the robot marker's fixed ``markersize``.
"""


def make_sprite(
    ax: Any,
    rgba: Any,
    xy: tuple[float, float],
    *,
    zorder: float,
    size_points: float = SPRITE_POINTS,
    animated: bool = False,
) -> tuple[Any, Any]:
    """Add a sprite to *ax*, returning ``(annotation_box, offset_image)``.

    No ``label`` is set: ``Axes.legend`` collects labelled artists, and a sprite
    in the robot legend would be nonsense.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    source_px = rgba.shape[0]
    # dpi_cor keeps the drawn size proportional to dpi, so `zoom` is a physical
    # size in points and the figure's dpi never has to be touched -- which the
    # video writer's frame-size guard depends on.
    image = OffsetImage(rgba, zoom=size_points / source_px, dpi_cor=True)
    box = AnnotationBbox(
        image, xy, frameon=False, pad=0.0, zorder=zorder, annotation_clip=True,
    )
    ax.add_artist(box)
    if animated:
        box.set_animated(True)
    return box, image


def update_sprite(
    box: Any, image: Any, rgba: Any, xy: tuple[float, float], alpha: float
) -> None:
    """Move and fade one sprite."""
    box.xy = box.xybox = (float(xy[0]), float(xy[1]))
    if alpha >= 1.0:
        image.set_data(rgba)
        return
    faded = rgba.astype(float) / 255.0
    faded[..., 3] *= alpha
    image.set_data(faded)
