"""Helpers for constructing environments in tests.

A uniquely named module rather than `conftest` so static tooling can resolve
the import: several `conftest.py` files exist in this repo and `ty` binds the
name to the wrong one.
"""

from typing import Any, List, Sequence, Type, TypeVar, cast

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


# --- Dashboard scaffolding -------------------------------------------------- #
#
# Seven tests across three files built the same one-robot A -> B dashboard by
# hand, ~20 lines each. They differ in a handful of knobs, all exposed below.

#: Robot r1 leaves A at t=0 and arrives at B at t=10.
DEFAULT_TRAJECTORY = {"r1": [(0.0, "A", None), (10.0, "B", None)]}


def move_env(
    *,
    locations: Sequence[str] = ("A", "B"),
    start: str = "A",
    move_time: float = 10.0,
    with_no_op: bool = True,
    occupancy_grid: Any = None,
    scene: Any = None,
) -> Any:
    """One robot, `locations` to move between, nothing else."""
    from railroad import operators
    from railroad._bindings import State
    from railroad.core import Fluent as F
    from railroad.environment import ObjectSearchEnvironment

    ops = [operators.construct_move_operator_blocking(lambda r, a, b: move_time)]
    if with_no_op:
        ops.append(operators.construct_no_op_operator(no_op_time=1.0, extra_cost=10.0))

    env = env_with_operators(
        ObjectSearchEnvironment,
        state=State(0.0, {F(f"at r1 {start}"), F("free r1")}, []),
        objects_by_type={"robot": {"r1"}, "location": set(locations)},
        operators=ops,
    )
    # Set by the pathing/rendering mixins in production; the tests that draw a
    # map supply them directly.
    if occupancy_grid is not None:
        env.occupancy_grid = occupancy_grid  # ty: ignore[unresolved-attribute]
    if scene is not None:
        env.scene = scene  # ty: ignore[unresolved-attribute]
    return env


def move_dashboard(
    env: Any = None,
    *,
    goal_loc: str = "B",
    trajectory: Any = DEFAULT_TRAJECTORY,
    goal_time: float | None = 10.0,
    actions_taken: Any = None,
    **env_kwargs: Any,
) -> Any:
    """A non-interactive `PlannerDashboard` over `move_env()`.

    `trajectory=None` leaves `_entity_positions` empty, which is the case the
    "nothing to plot" tests need. Pass `env=` when the environment needs
    doctoring the factory does not cover.
    """
    from copy import deepcopy

    from railroad.core import Fluent as F
    from railroad.dashboard import PlannerDashboard

    if env is None:
        env = move_env(**env_kwargs)
    elif env_kwargs:
        raise TypeError("pass env= or move_env() keywords, not both")

    dashboard = PlannerDashboard(
        F(f"at r1 {goal_loc}"), env,
        force_interactive=False, print_on_exit=False,
    )
    if trajectory is not None:
        dashboard.known_robots = {"r1"}
        # Deep-copied: the default is module-level and dashboards mutate it.
        dashboard._entity_positions = deepcopy(trajectory)
        dashboard._goal_time = goal_time
    if actions_taken is not None:
        dashboard.actions_taken = list(actions_taken)
    return dashboard
