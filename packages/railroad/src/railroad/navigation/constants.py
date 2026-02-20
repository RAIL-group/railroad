"""Shared constants for grid navigation and mapping."""

COLLISION_VAL = 1.0
FREE_VAL = 0.0
UNOBSERVED_VAL = -1.0
OBSTACLE_THRESHOLD = 0.5 * (FREE_VAL + COLLISION_VAL)
