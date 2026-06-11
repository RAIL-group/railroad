"""Camera matrix helpers (numpy, column-vector convention, OpenGL clip space)."""

from __future__ import annotations

import math

import numpy as np


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Standard OpenGL perspective projection; ``fov_y_deg`` is the
    *vertical* field of view (matching Unity's Camera.fieldOfView)."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """View matrix looking from ``eye`` toward ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, np.asarray(up, dtype=np.float64))
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)

    mat = np.eye(4, dtype=np.float32)
    mat[0, :3] = side
    mat[1, :3] = true_up
    mat[2, :3] = -forward
    mat[0, 3] = -np.dot(side, eye)
    mat[1, 3] = -np.dot(true_up, eye)
    mat[2, 3] = np.dot(forward, eye)
    return mat
