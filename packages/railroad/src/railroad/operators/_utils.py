"""Utility functions for operator construction."""

from __future__ import annotations

from typing import Callable, Union


class Numeric:
    """A callable that supports arithmetic operations.

    Wraps a function so that arithmetic produces new Numeric instances
    that compose the operations. This allows natural expressions like
    ``1 - prob_fn`` to create a callable that computes the complement.

    Can be used as a decorator via :func:`numeric`::

        @numeric
        def find_prob(robot, loc, obj):
            return 0.9

        complement = 1 - find_prob  # Numeric that returns 0.1
    """

    def __init__(self, fn: Callable[..., float]) -> None:
        self._fn = fn

    def __call__(self, *args: object) -> float:
        return self._fn(*args)

    def __repr__(self) -> str:
        return f"Numeric({self._fn!r})"

    # Numeric + Numeric, Numeric + float
    def __add__(self, other: object) -> Numeric:
        if isinstance(other, Numeric):
            lhs, rhs = self._fn, other._fn
            return Numeric(lambda *args: lhs(*args) + rhs(*args))
        if isinstance(other, (int, float)):
            fn, val = self._fn, float(other)
            return Numeric(lambda *args: fn(*args) + val)
        return NotImplemented

    # float + Numeric
    def __radd__(self, other: object) -> Numeric:
        if isinstance(other, (int, float)):
            fn, val = self._fn, float(other)
            return Numeric(lambda *args: val + fn(*args))
        return NotImplemented

    # Numeric - Numeric, Numeric - float
    def __sub__(self, other: object) -> Numeric:
        if isinstance(other, Numeric):
            lhs, rhs = self._fn, other._fn
            return Numeric(lambda *args: lhs(*args) - rhs(*args))
        if isinstance(other, (int, float)):
            fn, val = self._fn, float(other)
            return Numeric(lambda *args: fn(*args) - val)
        return NotImplemented

    # float - Numeric
    def __rsub__(self, other: object) -> Numeric:
        if isinstance(other, (int, float)):
            fn, val = self._fn, float(other)
            return Numeric(lambda *args: val - fn(*args))
        return NotImplemented

    # Numeric * Numeric, Numeric * float
    def __mul__(self, other: object) -> Numeric:
        if isinstance(other, Numeric):
            lhs, rhs = self._fn, other._fn
            return Numeric(lambda *args: lhs(*args) * rhs(*args))
        if isinstance(other, (int, float)):
            fn, val = self._fn, float(other)
            return Numeric(lambda *args: fn(*args) * val)
        return NotImplemented

    # float * Numeric
    def __rmul__(self, other: object) -> Numeric:
        if isinstance(other, (int, float)):
            fn, val = self._fn, float(other)
            return Numeric(lambda *args: val * fn(*args))
        return NotImplemented

    # -Numeric
    def __neg__(self) -> Numeric:
        fn = self._fn
        return Numeric(lambda *args: -fn(*args))


def numeric(fn: Callable[..., float]) -> Numeric:
    """Decorator that wraps a function as a :class:`Numeric`.

    The returned object is callable (same signature as *fn*) and supports
    arithmetic::

        @numeric
        def prob(robot, loc, obj):
            return 0.8

        complement = 1 - prob   # Numeric returning 0.2
        scaled = 0.5 * prob     # Numeric returning 0.4
    """
    return Numeric(fn)


Num = Union[float, int]
OptNumeric = Union[Num, Numeric, Callable[..., Num]]


def _to_numeric(value: OptNumeric) -> Numeric:
    """Normalize a numeric value, callable, or Numeric to a Numeric.

    Args:
        value: A number, callable, or Numeric instance.

    Returns:
        A Numeric wrapping the value.
    """
    if isinstance(value, Numeric):
        return value
    if isinstance(value, (int, float)):
        return Numeric(lambda *args: float(value))
    return Numeric(value)
