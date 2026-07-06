"""Unified replay driver over pluggable domains (design §9 precursor).

The three replay flavors — point-goal **navigation** in unknown space,
**object search** in an unknown map, and **object search** in a known map —
share one plan->act loop and differ only in three seams: how the arena is built
from a :class:`~railroad.replay.types.RolloutLog`, what the goal is, and how the
final env state reduces to a :class:`~railroad.replay.cost.Bounds`. A
:class:`ReplayDomain` bundles those three seams; :func:`replay` dispatches on
``log.problem_class`` and runs the shared loop.

This is the entry point the §9 selection layer builds on: ``for policy in
candidates: replay(log, policy)`` over one recording. The domain-specific
``run_*`` drivers (``run_replay`` etc.) are thin wrappers over :func:`replay`,
kept for their established signatures.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict

from railroad._bindings import Fluent, Goal
from railroad.core import get_action_by_name

from .cost import accumulate_bounds
from .policy import CandidatePolicy
from .types import ReplayResult, RolloutLog

# (env, actions, goal) -> action name (or "NONE"/"" to stop). Matches the
# selector every domain's driver used before unification.
ActionSelector = Callable[[Any, list, "Goal | Fluent"], str]


@dataclass(frozen=True)
class MctsParams:
    """MCTS knobs for the default selector (per-domain defaults below)."""

    iterations: int = 4000
    c: float = 10.0
    max_depth: int = 20
    heuristic_multiplier: float = 5.0


def mcts_selector(params: MctsParams) -> ActionSelector:
    """The default action selector: production MCTS with *params*."""
    from railroad.planner import MCTSPlanner

    def select(env: Any, actions: list, goal: "Goal | Fluent") -> str:
        return MCTSPlanner(actions)(
            env.state,
            goal,
            max_iterations=params.iterations,
            c=params.c,
            max_depth=params.max_depth,
            heuristic_multiplier=params.heuristic_multiplier,
        )

    return select


def run_plan_act_loop(
    env: Any, goal: "Goal | Fluent", select: ActionSelector, max_iterations: int
) -> str:
    """Drive the plan->act loop until the goal or a dead end; return why it stopped.

    Byte-for-byte the loop all three legacy drivers ran: check the goal, get
    applicable actions, ask *select* for one, act. Terminations:
    ``goal_reached`` / ``no_actions`` / ``planner_none`` / ``max_iterations``.
    """
    termination = "max_iterations"
    for _ in range(max_iterations):
        if goal.evaluate(env.state.fluents):
            termination = "goal_reached"
            break
        actions = env.get_actions()
        if not actions:
            termination = "no_actions"
            break
        name = select(env, actions, goal)
        if name in ("NONE", "", None):
            termination = "planner_none"
            break
        env.act(get_action_by_name(actions, name))
    if termination == "max_iterations" and goal.evaluate(env.state.fluents):
        termination = "goal_reached"
    return termination


# ----------------------------------------------------------------------
# Domains
# ----------------------------------------------------------------------


class ReplayDomain(ABC):
    """A replay flavor's three seams over the shared driver."""

    problem_class: str
    default_max_planning_iterations: int
    default_mcts: MctsParams

    @abstractmethod
    def build_arena(
        self, log: RolloutLog, policy: CandidatePolicy, *, config=None
    ) -> object:
        """Reconstruct the replay environment for *policy* from *log*."""

    @abstractmethod
    def goal(self, env: object, log: RolloutLog) -> "Goal | Fluent":
        """The planning goal for this replay."""

    @abstractmethod
    def finalize(
        self, env: object, log: RolloutLog, termination: str
    ) -> ReplayResult:
        """Reduce the terminal env state + log to a :class:`ReplayResult`."""


class NavigationDomain(ReplayDomain):
    """Point-goal navigation in unknown space (design §6): optimistic vs.
    simply-connected bounds from the alternative's frontier commitments."""

    problem_class = "navigation"
    default_max_planning_iterations = 300
    default_mcts = MctsParams(iterations=4000, c=10.0, max_depth=20, heuristic_multiplier=5.0)

    def build_arena(self, log, policy, *, config=None):
        from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics

        from .replay_env import ReplayEnvironment

        fs = policy.frontier_statistics or FixedPriorFrontierStatistics()
        return ReplayEnvironment.from_log(log, fs, config=config)

    def goal(self, env, log):
        from .replay_env import goal_fluent

        return goal_fluent(log.robots)

    def finalize(self, env, log, termination):
        total_cost = float(env.state.time)
        bounds = accumulate_bounds(env.replay_commits, total_cost)
        return ReplayResult(
            bounds=bounds,
            commits=list(env.replay_commits),
            termination=termination,
            total_cost=total_cost,
            sim_time=float(env.state.time),
            goal_reached=(termination == "goal_reached"),
        )


class UnknownSearchDomain(ReplayDomain):
    """Frontier-based object search in an unknown map (design §6/§7).

    Search outcomes resolve from the recorded ground truth. The optimistic bound
    is **commit-based**, exactly as in navigation: each not-found search is a
    commitment to a subgoal (frontier/container) the deployment never verified,
    logged as a :class:`Commit` with ``optimistic_to_goal = 0`` (the object could
    be immediately at/past it). ``optimistic_lb = min`` over those commits; the
    simply-connected bound is the candidate's actual replay makespan. (The old
    "straight to the true container" cost was a *policy-independent oracle*
    regret baseline, not the candidate's bound — see §7.1.)
    """

    problem_class = "object-search"
    default_max_planning_iterations = 80
    default_mcts = MctsParams(iterations=4000, c=300.0, max_depth=20, heuristic_multiplier=2.0)

    @staticmethod
    def _hidden_sites(log: RolloutLog) -> Dict[str, tuple]:
        return {
            s.signature: (int(s.centroid[0]), int(s.centroid[1]))
            for s in log.subgoals
        }

    @staticmethod
    def _recorded(log: RolloutLog) -> Dict[str, set]:
        return {s.signature: set(s.contents) for s in log.subgoals}

    def build_arena(self, log, policy, *, config=None):
        from .search_replay_env import build_search_replay_env

        target = _require_target(log)
        ffp = policy.frontier_find_prob or (lambda r, f, o: 0.5)
        cfp = policy.container_find_prob or (lambda r, l, o: 0.5)
        return build_search_replay_env(
            log,
            frontier_find_prob=ffp,
            container_find_prob=cfp,
            refresh_estimators=policy.refresh_estimators,
            hidden_sites=self._hidden_sites(log),
            target_object=target,
            recorded_object_locations=self._recorded(log),
            config=config,
        )

    def goal(self, env, log):
        return Fluent(f"found {_require_target(log)}")

    def finalize(self, env, log, termination):
        goal = Fluent(f"found {_require_target(log)}")
        total_cost = float(env.state.time)
        bounds = accumulate_bounds(env.replay_commits, total_cost)
        return ReplayResult(
            bounds=bounds,
            commits=list(env.replay_commits),
            termination=termination,
            total_cost=total_cost,
            sim_time=float(env.state.time),
            goal_reached=goal.evaluate(env.state.fluents),
            search_log=list(env.search_log),
        )


class KnownMapSearchDomain(ReplayDomain):
    """Object search in a fully known map (design §7.1, updated).

    Travel is exact (known map, no unobserved space → no frontier optimism), but
    object *presence* is known only where the deployment searched. Not assuming
    one-container-per-object, a revealed-but-unsearched container is an unverified
    subgoal: searching it forces not-found and logs a commit
    (``optimistic_to_goal = 0``). So the cost is a commit-based lower bound
    (``optimistic_lb`` vs. makespan), **not** an exact value.
    """

    problem_class = "known-map-search"
    default_max_planning_iterations = 60
    default_mcts = MctsParams(iterations=4000, c=300.0, max_depth=20, heuristic_multiplier=2.0)

    def build_arena(self, log, policy, *, config=None):
        from .known_map_search_replay_env import build_known_map_search_replay_env

        target = _require_target(log)
        cfp = policy.container_find_prob or (lambda r, l, o: 0.5)
        return build_known_map_search_replay_env(
            log, container_find_prob=cfp, target_object=target
        )

    def goal(self, env, log):
        return Fluent(f"found {_require_target(log)}")

    def finalize(self, env, log, termination):
        goal = Fluent(f"found {_require_target(log)}")
        total_cost = float(env.state.time)
        bounds = accumulate_bounds(env.replay_commits, total_cost)
        return ReplayResult(
            bounds=bounds,
            commits=list(env.replay_commits),
            termination=termination,
            total_cost=total_cost,
            sim_time=total_cost,
            goal_reached=goal.evaluate(env.state.fluents),
            search_log=list(env.search_log),
        )


def _require_target(log: RolloutLog) -> str:
    if not log.target_object:
        raise ValueError(
            f"{log.problem_class!r} replay needs a target object; set "
            "log.target_object (recorders capture it) or pass target_object= to "
            "replay()."
        )
    return log.target_object


# ----------------------------------------------------------------------
# Registry + unified entry
# ----------------------------------------------------------------------

DOMAINS: Dict[str, ReplayDomain] = {
    d.problem_class: d
    for d in (NavigationDomain(), UnknownSearchDomain(), KnownMapSearchDomain())
}


def get_domain(problem_class: str) -> ReplayDomain:
    """The :class:`ReplayDomain` for *problem_class* (raises if unregistered)."""
    try:
        return DOMAINS[problem_class]
    except KeyError:
        raise ValueError(
            f"no replay domain for problem_class={problem_class!r}; "
            f"known: {sorted(DOMAINS)}"
        ) from None


def replay(
    log: RolloutLog,
    policy: CandidatePolicy | None = None,
    *,
    domain: ReplayDomain | None = None,
    target_object: str | None = None,
    config=None,
    select_action: ActionSelector | None = None,
    max_planning_iterations: int | None = None,
    mcts: MctsParams | None = None,
) -> ReplayResult:
    """Replay one candidate *policy* over a recorded *log*; return its bounds.

    Dispatches on ``log.problem_class`` (override with *domain*) to the right
    arena + goal + cost reduction, then runs the shared plan->act loop. A fresh
    arena is built per call, so the same log replays many candidates. *policy*
    defaults to a policy-agnostic :class:`CandidatePolicy` (neutral priors).
    *target_object* overrides ``log.target_object`` for search domains.
    """
    if policy is None:
        policy = CandidatePolicy()
    if target_object is not None:
        log = dataclasses.replace(log, target_object=target_object)
    dom = domain if domain is not None else get_domain(log.problem_class)

    env = dom.build_arena(log, policy, config=config)
    goal = dom.goal(env, log)
    select = select_action or mcts_selector(mcts or dom.default_mcts)
    max_iters = (
        max_planning_iterations
        if max_planning_iterations is not None
        else dom.default_max_planning_iterations
    )
    termination = run_plan_act_loop(env, goal, select, max_iters)
    return dom.finalize(env, log, termination)
