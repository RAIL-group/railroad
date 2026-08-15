"""Does the ProcTHOR overhead image actually land where we say it does?

One link cannot be unit-tested: that Unity's ``orthographicSize`` really is the
camera's half-height in world meters, and that it honours the rotation we pin.
Everything else is arithmetic, covered in ``test_top_down_view.py``. This runs
the whole chain against a rendered scene, using AI2-THOR's reachable positions
as ground truth -- they are standable floor by construction, so a correct
placement puts them inside the rendered house.

Seed choice matters. A square, centred house cannot tell a correct placement
from a transposed or flipped one: 4001 and 8611 score 100% *whatever* you do to
the extent. These were picked by measuring that wrong placements drop:

    seed    correct   transposed   flipped   shifted 0.5m
    7005    100.0%       60.0%      83.4%       98.1%
    8610    100.0%       43.2%      93.4%       90.2%

Seed 1089 is deliberately unused: it measures 78.4% while being perfectly
aligned, because a lit pale floor blows out to pure white and joins the skybox
through a wall that is also white from above. A limit of judging placement from
pixels, not a misalignment.
"""

import numpy as np
import pytest

from railroad.environment.procthor.thor_interface import ThorInterface

# 7005 contains a room the agent cannot reach; 8610 is the strongest
# discriminator of a transposed placement, 8618 of a flipped one.
SEEDS = [7005, 8610, 8618]

SKYBOX_LUMINANCE = 250
"""At or above this is the white skybox rendered around the house."""


def _inside_house_rate(thor: ThorInterface) -> float:
    """Fraction of reachable positions landing inside the rendered house.

    Against the house *silhouette* -- non-skybox with interior holes filled --
    rather than per pixel, so a white floor inside the house does not read as
    being outside it.
    """
    from scipy.ndimage import binary_fill_holes

    view = thor.get_top_down_view()
    assert view is not None
    silhouette = binary_fill_holes(
        view.image.astype(int).mean(axis=2) < SKYBOX_LUMINANCE
    )
    height, width = silhouette.shape

    cells = np.array([
        thor.scale_to_grid_continuous((p["x"], p["z"]))
        for p in thor.get_reachable_positions()
    ])
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


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_reachable_positions_land_inside_the_rendered_house(seed: int) -> None:
    # Check the cache before constructing: ThorInterface launches Unity on a
    # miss, so an unguarded construction turns "skip" into a hang on any
    # machine without a display.
    probe = object.__new__(ThorInterface)
    probe.seed = seed
    if not (probe._cache_dir() / f"scene_{seed}.pkl").exists():
        pytest.skip(f"scene {seed} is not cached; generate it with a display")

    thor = ThorInterface(seed=seed)
    if thor.get_top_down_view() is None:
        pytest.skip(f"scene {seed} predates the recorded extent; regenerate it")

    rate = _inside_house_rate(thor)
    # Correct placement measures exactly 100% on these seeds and the nearest
    # wrong one 93.4%, so this sits in clear air rather than squeaking past.
    assert rate > 0.99, f"only {rate:.1%} landed inside the rendered house"
