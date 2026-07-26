"""Object-find estimators: where search-action probabilities come from.

The object-search counterpart of :mod:`railroad.lsp.frontier_statistics`. Both
modules answer the same question — *what does the policy believe about this
subgoal?* — the same way: an estimator object with two methods, refreshed by the
environment whenever the map changes and queried per subgoal when actions are
grounded.

Search has two kinds of subgoal, so there are two accessors:

- :meth:`ObjectFindEstimator.frontier_probability` — "is the object beyond this
  frontier?" (unknown space only)
- :meth:`ObjectFindEstimator.container_probability` — "is the object inside this
  container?"

Both default to :meth:`ObjectFindEstimator.probability`, so a policy with one
flat belief implements a single method; only estimators that genuinely treat the
two kinds differently (an oracle, a learned served-vantage model) override them.

An object rather than a bare ``(robot, subgoal, object) -> float`` callable
because a callable has nowhere to put per-step work: a learned model wants **one
batched forward pass over all frontiers** per step, and an oracle wants to run
its grid searches once. Both need the ``refresh`` hook, and without it each
caller has to arrange its own. :func:`as_object_find_estimator` also accepts a
plain callable or a constant, for beliefs with no per-step work to do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

# (robot, subgoal, object) -> probability.
FindProbFn = Callable[[str, str, str], float]

#: Belief used when nothing else is specified: maximally uninformative.
DEFAULT_FIND_PROBABILITY = 0.5


class ObjectFindEstimator(ABC):
    """Maps each search subgoal to the probability of finding an object there."""

    def refresh(self, environment: Any) -> None:
        """Update internal state from the environment.

        Called whenever the observed map or frontier set changes, before any
        action grounding queries the probabilities. The default does nothing.
        """

    @abstractmethod
    def probability(self, robot: str, subgoal: str, obj: str) -> float:
        """Probability that searching *subgoal* with *robot* finds *obj*."""

    def frontier_probability(self, robot: str, frontier: str, obj: str) -> float:
        """Probability that *obj* lies beyond *frontier*. Defaults to :meth:`probability`."""
        return self.probability(robot, frontier, obj)

    def container_probability(self, robot: str, container: str, obj: str) -> float:
        """Probability that *obj* is inside *container*. Defaults to :meth:`probability`."""
        return self.probability(robot, container, obj)


class FixedObjectFind(ObjectFindEstimator):
    """The same find-probability for every subgoal and object."""

    def __init__(self, probability: float = DEFAULT_FIND_PROBABILITY) -> None:
        self._probability = float(probability)

    def probability(self, robot: str, subgoal: str, obj: str) -> float:
        del robot, subgoal, obj
        return self._probability


class CallableObjectFind(ObjectFindEstimator):
    """Adapts a plain ``(robot, subgoal, object) -> float`` callable.

    The shortest way to express a stateless belief inline, for cases with no
    per-step work to hang on :meth:`ObjectFindEstimator.refresh`.
    """

    def __init__(self, find_prob: FindProbFn) -> None:
        self._find_prob = find_prob

    def probability(self, robot: str, subgoal: str, obj: str) -> float:
        return float(self._find_prob(robot, subgoal, obj))


ObjectFindLike = Union[ObjectFindEstimator, FindProbFn, float, int, None]


def as_object_find_estimator(value: ObjectFindLike) -> ObjectFindEstimator:
    """Coerce an estimator / callable / constant / ``None`` to an estimator.

    ``None`` yields the neutral :data:`DEFAULT_FIND_PROBABILITY` prior, so an
    environment is always in a valid, policy-agnostic state before a policy is
    installed.
    """
    if value is None:
        return FixedObjectFind()
    if isinstance(value, ObjectFindEstimator):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return FixedObjectFind(float(value))
    if callable(value):
        return CallableObjectFind(value)
    raise TypeError(
        f"cannot use {value!r} as an object-find estimator; expected an "
        "ObjectFindEstimator, a (robot, subgoal, object) -> float callable, "
        "a constant probability, or None"
    )


def as_find_probability(value: ObjectFindLike, *, kind: str) -> Any:
    """What a search operator should use for its probability argument.

    An :class:`ObjectFindEstimator` becomes a late-binding callable onto the
    accessor for *kind* (``"frontier"`` or ``"container"``); a constant or a
    plain callable is passed straight through.
    """
    if not isinstance(value, ObjectFindEstimator):
        return value
    if kind == "frontier":
        return lambda robot, subgoal, obj: value.frontier_probability(
            robot, subgoal, obj
        )
    if kind == "container":
        return lambda robot, subgoal, obj: value.container_probability(
            robot, subgoal, obj
        )
    raise ValueError(f"unknown subgoal kind {kind!r}; expected frontier/container")


class LiveObjectFind(ObjectFindEstimator):
    """Indirection onto an environment's *current* estimator.

    Operators are built once (``Environment.__init__`` resolves them and
    ``get_actions`` re-grounds those same objects forever), so an operator that
    closed over an estimator would freeze the policy for the life of the
    environment. Handing it this instead makes
    ``env.object_find_statistics = ...`` take effect immediately — the search
    counterpart of :class:`railroad.lsp.env_mixin._LiveFrontierStatistics`.
    """

    def __init__(self, environment: Any, attribute: str = "_object_find_statistics"):
        self._environment = environment
        self._attribute = attribute

    def _live(self) -> ObjectFindEstimator:
        return getattr(self._environment, self._attribute)

    def refresh(self, environment: Any) -> None:
        self._live().refresh(environment)

    def probability(self, robot: str, subgoal: str, obj: str) -> float:
        return self._live().probability(robot, subgoal, obj)

    def frontier_probability(self, robot: str, frontier: str, obj: str) -> float:
        return self._live().frontier_probability(robot, frontier, obj)

    def container_probability(self, robot: str, container: str, obj: str) -> float:
        return self._live().container_probability(robot, container, obj)


def find_probability_of(
    estimator: Optional[ObjectFindEstimator], *, kind: str
) -> FindProbFn:
    """The *kind* accessor of *estimator* as a bare callable (neutral if ``None``).

    For call sites that still take a callable — notably the plain symbolic
    ``construct_search_operator`` used by the known-map flavor.
    """
    resolved = as_object_find_estimator(estimator)
    if kind == "frontier":
        return resolved.frontier_probability
    if kind == "container":
        return resolved.container_probability
    raise ValueError(f"unknown subgoal kind {kind!r}; expected frontier/container")
