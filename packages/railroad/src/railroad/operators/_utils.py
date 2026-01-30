"""Utility functions for operator construction."""

from typing import Callable, Union

Num = Union[float, int]
OptCallable = Union[Num, Callable[..., float]]


def _make_callable(opt_expr: OptCallable) -> Callable[..., float]:
    """Convert an optional callable expression to a callable.

    Args:
        opt_expr: Either a number or a callable that returns a float.

    Returns:
        A callable that returns a float.
    """
    if isinstance(opt_expr, (int, float)):
        return lambda *args: float(opt_expr)
    else:
        return lambda *args: opt_expr(*args)


def _invert_prob(opt_expr: OptCallable) -> Callable[..., float]:
    """Invert a probability expression (return 1 - prob).

    Args:
        opt_expr: Either a number or a callable that returns a probability.

    Returns:
        A callable that returns 1 minus the probability.
    """
    if isinstance(opt_expr, (int, float)):
        return lambda *args: 1.0 - float(opt_expr)
    else:
        return lambda *args: 1.0 - opt_expr(*args)
