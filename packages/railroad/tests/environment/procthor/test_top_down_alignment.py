"""Does the ProcTHOR overhead image actually land where we say it does?

One link in the chain cannot be unit-tested: that Unity's ``orthographicSize``
really is the camera's half-height in world meters, and that it honours the
rotation we pin. Everything else about the placement is arithmetic, covered in
``test_top_down_view.py``.

This checks the whole chain against a rendered scene, using AI2-THOR's own
reachable positions as ground truth: those are standable floor by construction,
so projecting them through the recorded extent must land on floor pixels. It
needs no live controller once the scene is cached, and skips on caches written
before the extent was recorded -- so it lies dormant until those are
regenerated, then becomes a real regression test.

Seed choice matters more than it looks. A square, centred house cannot tell a
correct placement from a transposed or vertically flipped one: seeds 4001 and
8611 both score 100% *however* the extent is transposed or flipped. These were
picked by measuring that deliberately wrong placements do drop:

    seed    correct   transposed   flipped   shifted 0.5m
    7005    100.0%       60.0%      83.4%       98.1%
    8610    100.0%       43.2%      93.4%       90.2%

Seed 1089 is deliberately *not* used. It measures 78.4% while being perfectly
aligned: a brightly lit pale floor blows out to pure white and joins the skybox
through a wall that is also white seen from above, so no amount of hole-filling
separates them. That is a limit of judging placement from pixels, not a
misalignment -- confirmed by overlaying its reachable positions, which tile its
floors exactly.
"""

import numpy as np
import pytest

from railroad.environment.procthor.thor_interface import ThorInterface

# 7005: 12.7 x 8.5 m, and contains a room the agent cannot reach.
# 8610:  3.8 x 7.7 m, the strongest discriminator of a transposed placement.
# 8618: 10.1 x 12.1 m, the strongest discriminator of a flipped one.
SEEDS = [7005, 8610, 8618]

SKYBOX_LUMINANCE = 250
"""At or above this is the white skybox the camera renders around the house.

Testing this per pixel is not enough: seed 1089 has a brightly lit pale floor
that blows out to pure white over a sixth of its reachable area, so "the pixel
is white" does not mean "outside the house". The silhouette below fixes that.
"""


def _house_silhouette(image: np.ndarray) -> np.ndarray:
    """Mask of the house as rendered, interior holes filled.

    The skybox is the only thing outside the house -- the camera renders it
    white by construction -- so everything darker is scene. Filling holes
    reclaims interior pixels that merely *look* like skybox: white floors,
    blown-out highlights, a white tabletop seen from above.
    """
    from scipy.ndimage import binary_fill_holes

    return binary_fill_holes(image.astype(int).mean(axis=2) < SKYBOX_LUMINANCE)


def _inside_house_rate(thor: ThorInterface) -> float:
    """Fraction of reachable positions landing inside the rendered house.

    Reachable positions are standable floor by construction, so under a correct
    placement essentially all of them fall within the silhouette. Under a
    transposed, flipped or shifted one they spill onto the skybox.
    """
    view = thor.get_top_down_view()
    assert view is not None
    silhouette = _house_silhouette(view.image)
    height, width = silhouette.shape

    positions = thor.get_reachable_positions()
    cells = np.array([
        thor.scale_to_grid_continuous((p["x"], p["z"])) for p in positions
    ])
    # Cell coordinates -> fractional position across the image -> pixel.
    columns = (cells[:, 0] - view.min_x) / (view.max_x - view.min_x) * width
    rows = (cells[:, 1] - view.min_y) / (view.max_y - view.min_y) * height

    inside_frame = (
        (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    )
    sampled = silhouette[
        np.clip(rows.astype(int), 0, height - 1),
        np.clip(columns.astype(int), 0, width - 1),
    ]
    return float((inside_frame & sampled).mean())


def _cached_scene_path(seed: int):
    """Where this seed's cache lives, without loading the scene to ask."""
    thor = object.__new__(ThorInterface)
    thor.seed = seed
    return thor._cache_dir() / f"scene_{seed}.pkl"


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_reachable_positions_land_on_floor(seed: int) -> None:
    # Check the cache before constructing: ThorInterface launches Unity on a
    # miss, so an unguarded construction turns "skip, no cache" into a hang or
    # a hard failure on any machine without a display.
    if not _cached_scene_path(seed).exists():
        pytest.skip(
            f"scene {seed} is not cached; generate it with a display attached "
            "to enable this test"
        )
    thor = ThorInterface(seed=seed)
    if thor.get_top_down_view() is None:
        pytest.skip(
            f"scene {seed} was cached before the top-down extent was recorded; "
            "delete the cache directory to regenerate and enable this test"
        )
    rate = _inside_house_rate(thor)
    # A correct placement measures exactly 100% on these seeds, and the nearest
    # wrong one (8610 flipped) measures 93.4%, so this sits in clear air rather
    # than being tuned to squeak past.
    assert rate > 0.99, (
        f"only {rate:.1%} of reachable positions land inside the rendered house"
    )
