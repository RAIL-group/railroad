"""Skill protocols and concrete skill implementations."""

from typing import TYPE_CHECKING

from .protocols import ActiveSkill, MotionSkill, SupportsMovePathEnvironment

if TYPE_CHECKING:
    from .navigation import InterruptibleNavigationMoveSkill, NavigationMoveSkill

__all__ = [
    "ActiveSkill",
    "MotionSkill",
    "SupportsMovePathEnvironment",
    "NavigationMoveSkill",
    "InterruptibleNavigationMoveSkill",
]


def __getattr__(name: str):
    if name == "NavigationMoveSkill":
        from .navigation import NavigationMoveSkill
        return NavigationMoveSkill
    if name == "InterruptibleNavigationMoveSkill":
        from .navigation import InterruptibleNavigationMoveSkill
        return InterruptibleNavigationMoveSkill
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
