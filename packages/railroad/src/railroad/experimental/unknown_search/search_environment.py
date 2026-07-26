"""Frontier + container object search in unknown space, with a swappable belief.

:class:`UnknownSpaceSearchEnvironment` is to object search what
:class:`~railroad.lsp.env_mixin.LSPEnvironmentMixin` is to point-goal
navigation: it owns its operator set and owns the estimator that parameterizes
them, so choosing a *policy* is one assignment on a built environment rather
than a reason to rebuild the world::

    env = UnknownSpaceSearchEnvironment(...)          # policy-agnostic
    env.object_find_statistics = my_estimator          # pick a policy
    env.object_find_statistics = a_different_estimator # ...or another

The environment refreshes the estimator on every frontier change, so a learned
or oracle belief stays current without the caller arranging it.
"""

from __future__ import annotations

from typing import Any, List

from railroad.core import Operator
from railroad.operators import construct_no_op_operator

from .environment import UnknownSpaceEnvironment
from .operators import (
    construct_move_navigable_operator,
    construct_search_at_site_operator,
    construct_search_frontier_operator,
)
from .statistics import (
    LiveObjectFind,
    ObjectFindEstimator,
    ObjectFindLike,
    as_object_find_estimator,
)

DEFAULT_SEARCH_TIME = 20.0
DEFAULT_NO_OP_TIME = 300.0
DEFAULT_NO_OP_EXTRA_COST = 100.0


class UnknownSpaceSearchEnvironment(UnknownSpaceEnvironment):
    """Unknown-space object search: navigable moves + frontier/container search.

    Separate from :class:`UnknownSpaceEnvironment` because that one is the
    operator-agnostic substrate, shared with exploration and — via
    :class:`~railroad.lsp.env_mixin.LSPEnvironmentMixin` — navigation.
    ``Environment`` forbids a class from both defining ``define_operators()`` and
    accepting ``operators=``, so owning an operator set means subclassing.

    The operator set is fixed (override :meth:`define_operators` to change it);
    the swappable part is :attr:`object_find_statistics`.
    """

    def __init__(
        self,
        *,
        object_find_statistics: ObjectFindLike = None,
        search_time: float = DEFAULT_SEARCH_TIME,
        no_op_time: float = DEFAULT_NO_OP_TIME,
        no_op_extra_cost: float = DEFAULT_NO_OP_EXTRA_COST,
        **kwargs: Any,
    ) -> None:
        # Must exist before super().__init__: it resolves operators via
        # define_operators() and then performs the initial frontier refresh,
        # which refreshes the estimator.
        self._object_find_statistics = as_object_find_estimator(object_find_statistics)
        self._search_time = float(search_time)
        self._no_op_time = float(no_op_time)
        self._no_op_extra_cost = float(no_op_extra_cost)
        kwargs.setdefault("operators", None)
        super().__init__(**kwargs)

    # -- the swappable policy -----------------------------------------

    @property
    def object_find_statistics(self) -> ObjectFindEstimator:
        """The estimator parameterizing the two search operators."""
        return self._object_find_statistics

    @object_find_statistics.setter
    def object_find_statistics(self, estimator: ObjectFindLike) -> None:
        """Swap the estimator on an already-constructed environment.

        Takes effect immediately — the operators read it through
        :class:`~railroad.experimental.unknown_search.statistics.LiveObjectFind`,
        so nothing is rebuilt — and the new estimator is refreshed at once so
        its predictions are available before the next planning step.
        """
        self._object_find_statistics = as_object_find_estimator(estimator)
        self._object_find_statistics.refresh(self)

    # -- operators ----------------------------------------------------

    def define_operators(self) -> List[Operator]:
        """Move + search-frontier + search-at-site + no-op."""
        # NOT the estimator itself: operators close over what they are given for
        # the life of the environment, so hand them the live indirection.
        live = LiveObjectFind(self)
        return [
            construct_move_navigable_operator(self.estimate_move_time_safe),
            construct_search_frontier_operator(live, self._search_time),
            construct_search_at_site_operator(
                live, self._search_time, container_type="container"
            ),
            construct_no_op_operator(
                no_op_time=self._no_op_time, extra_cost=self._no_op_extra_cost
            ),
        ]

    # -- keep the estimator current -----------------------------------

    def refresh_frontiers(self) -> None:
        super().refresh_frontiers()
        self._object_find_statistics.refresh(self)
