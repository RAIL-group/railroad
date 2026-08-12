#!/usr/bin/env python3
"""Fast iteration loop for tuning molmospace_search.py's top-down camera.

Samples the same (seed, house, object) task molmospace_search.py uses, sets
up the top-down camera the same way, and saves a single PNG -- skipping
search planning, the physical search replay, and the fetch policy rollout
(the slow parts of the real pipeline, on the order of minutes). Editing
TOP_DOWN_MARGIN / _hide_ceilings in molmospace_search.py and re-running this
script is a several-seconds loop instead.

Run with: uv run python scripts/topdown_preview.py
          uv run python scripts/topdown_preview.py --out /tmp/preview.png
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio

from molmospace_search import (  # pyrefly: ignore [missing-import]
    DEFAULT_HOUSE_INDEX,
    DEFAULT_OBJECT_NAME,
    DEFAULT_SEED,
    build_task_sampler,
    render_top_down_frame,
    setup_top_down_camera,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--house-index", type=int, default=DEFAULT_HOUSE_INDEX)
    parser.add_argument("--object-name", type=str, default=DEFAULT_OBJECT_NAME)
    parser.add_argument("--out", type=str, default="topdown_preview.png")
    args = parser.parse_args()

    config, task_sampler = build_task_sampler(args.seed)
    config.task_config.pickup_obj_name = args.object_name
    task = task_sampler.sample_task(house_index=args.house_index)

    setup_top_down_camera(task.env)
    frame = render_top_down_frame(task.env)
    imageio.imwrite(args.out, frame)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
