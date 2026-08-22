"""Helpers for constructing environments in tests.

A uniquely named module rather than `conftest` so static tooling can resolve
the import: several `conftest.py` files exist in this repo and `ty` binds the
name to the wrong one.
"""

from typing import Any, List, Type, TypeVar, cast

from railroad.core import Operator
from railroad.environment import Environment

E = TypeVar("E", bound=Environment)


def env_with_operators(env_cls: Type[E], /, **kwargs: Any) -> E:
    """Build `env_cls` with an explicit operator list.

    ``operators=`` on the constructor is deprecated in favour of a subclass
    overriding ``define_operators()``, and passing both raises. Tests need a
    different operator set per case, so synthesise the subclass here rather
    than writing one per test.

    Takes ``operators=`` exactly as the deprecated constructor did, so
    converting a call site is just wrapping it::

        SymbolicEnvironment(state=s, operators=[move_op])
        env_with_operators(SymbolicEnvironment, state=s, operators=[move_op])
    """
    ops: List[Operator] = list(kwargs.pop("operators"))
    # Pass None rather than dropping the kwarg: UnknownSpaceEnvironment takes
    # `operators` as a required positional, and None is the value that routes
    # resolution to define_operators() without tripping the deprecation.
    kwargs["operators"] = None
    subclass = type(
        env_cls.__name__,
        (env_cls,),
        {"define_operators": lambda self: ops},
    )
    # type() erases the parameter, so state the relationship rather than
    # silencing the checker: the synthesised class derives from env_cls.
    return cast(Type[E], subclass)(**kwargs)
