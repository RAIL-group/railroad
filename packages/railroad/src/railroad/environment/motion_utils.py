"""Shared path-geometry helpers for movement skills."""

from __future__ import annotations

import numpy as np


def path_total_length(path: np.ndarray) -> float:
    """Return total Euclidean length for a 2xN path."""
    if path.size == 0 or path.shape[1] < 2:
        return 0.0
    diffs = np.diff(path, axis=1)
    return float(np.sum(np.linalg.norm(diffs, axis=0)))


def get_coordinates_at_distance(path: np.ndarray, distance: float) -> np.ndarray:
    """Interpolate coordinates along a 2xN path at cumulative distance."""
    if path.size == 0:
        return np.array([np.nan, np.nan])
    if path.shape[1] == 1:
        return path[:, 0].astype(float)

    diffs = np.diff(path, axis=1)
    segment_lengths = np.linalg.norm(diffs, axis=0)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    if distance <= 0.0:
        return path[:, 0].astype(float)
    if distance >= cumulative_lengths[-1]:
        return path[:, -1].astype(float)

    idx = int(np.searchsorted(cumulative_lengths, distance, side="right") - 1)
    seg_len = segment_lengths[idx]
    if seg_len <= 1e-9:
        return path[:, idx].astype(float)

    t = (distance - cumulative_lengths[idx]) / seg_len
    start = path[:, idx].astype(float)
    end = path[:, idx + 1].astype(float)
    return start + t * (end - start)
