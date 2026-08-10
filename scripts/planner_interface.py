import heapq as _heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from railroad._bindings import ff_heuristic as _ff_heuristic
from railroad.core import (
    AndGoal,
    Fluent,
    LiteralGoal,
    State,
    convert_goal_to_positive_preconditions,
    get_action_by_name,
    get_next_actions as _get_next_actions,
    transition as _transition,
)
from railroad.environment.symbolic import SymbolicEnvironment
from railroad.planner import AStarPlanner, MCTSPlanner

__all__ = [
    "SimpleAction",
    "SearchStats",
    "call_planner",
    "MCTSPlanner",
    "AStarPlanner",
    "CoordinationAStarPlanner",
    "ExtraCostFn",
]

# Signature for the extra-cost hook: (planner, action, successor_state) -> float.
# `planner` is the CoordinationAStarPlanner instance running the search —
# use `planner.heuristic_to_goal(successor, goal_str)` to estimate the
# FF-heuristic cost to any other goal from the successor state, reusing the
# actions/mapping already prepared for the current search.
# `action` is the C++ Action just taken (use action.name to identify it, e.g.
# to recover which robot is searching).
# `successor` is the C++ State resulting from the action.
# Return 0 to add no coordination cost (pure A* behaviour).
ExtraCostFn = Callable[..., float]


@dataclass
class SimpleAction:
    name: str
    duration: float
    preconditions: Set[str]
    add_effects: Set[str]
    del_effects: Set[str]


@dataclass
class SearchStats:
    """How much of the state space a single CoordinationAStarPlanner search
    actually explored -- set on the planner instance after plan_sequence()
    returns (success or failure), and surfaced by call_planner via its
    stats_out parameter.
    """

    states_expanded: int  # states popped off the open heap and processed (the closed set)
    states_generated: int  # distinct states ever pushed onto the open heap (closed + still-open)


def _fluent_to_str(f: Fluent) -> str:
    args_str = " ".join(f.args)
    if args_str:
        s = f"{f.name} {args_str}"
    else:
        s = f.name
    return f"not {s}" if f.negated else s


def _str_to_fluent(s: str) -> Fluent:
    negated = False
    if s.startswith("not "):
        negated = True
        s = s[4:]
    parts = s.split()
    name = parts[0]
    args = parts[1:]
    return Fluent(name, *args, negated=negated)


def _goal_from_str(goal_str: str):
    """Parse a '&'-joined fluent string into a LiteralGoal or AndGoal."""
    goal_parts = [p.strip() for p in goal_str.split("&")]
    if len(goal_parts) > 1:
        return AndGoal([LiteralGoal(_str_to_fluent(p)) for p in goal_parts])
    return LiteralGoal(_str_to_fluent(goal_str))


def _action_duration(action) -> float:
    """The operator's own stated duration: the time of its last effect."""
    duration = 0.0
    for eff in action.effects:
        if eff.time > duration:
            duration = eff.time
    return duration


def _action_to_simple(action) -> SimpleAction:
    """Convert a grounded Action to a SimpleAction for simulation."""
    preconds = {_fluent_to_str(f) for f in action.preconditions}
    add_eff: Set[str] = set()
    del_eff: Set[str] = set()
    duration = _action_duration(action)
    for eff in sorted(action.effects, key=lambda e: e.time):
        for f in eff.resulting_fluents:
            f_str = _fluent_to_str(~f if f.negated else f)
            if f.negated:
                del_eff.add(f_str)
                add_eff.discard(f_str)
            else:
                add_eff.add(f_str)
                del_eff.discard(f_str)
    return SimpleAction(
        name=action.name,
        duration=duration,
        preconditions=preconds,
        add_effects=add_eff,
        del_effects=del_eff,
    )


class CoordinationAStarPlanner(AStarPlanner):
    """Python A* with an injectable extra-cost term for coordination.

    Evaluation function:  f = g + h + w
      g  — time of the successor state (true cost so far)
      h  — FF heuristic (estimated remaining steps to goal)
      w  — extra_cost_fn(action, successor): caller-supplied coordination
           overhead, e.g. expected wait time other robots will incur because
           of the action just taken.  Defaults to 0 (pure A*).

    Pass extra_cost_fn to plan_sequence to bias the search away from plans
    that impose high coordination costs on other robots.
    """

    last_search_stats: Optional[SearchStats] = None

    def _search(
        self,
        converted_state: State,
        converted_goal,
        extra_cost_fn: Optional[ExtraCostFn] = None,
    ) -> tuple:
        """Core A* loop over already-converted state/goal.

        Returns (path, final_state), or ([], None) if the goal is unreachable.
        """
        actions = self._converted_actions

        def h(s: State) -> float:
            return _ff_heuristic(s, converted_goal, actions)

        # heap entries: (f, tie-break counter, state)
        counter = 0
        open_heap = [(h(converted_state), 0, converted_state)]
        # state.hash() -> best g seen so far
        best_g: Dict[int, float] = {hash(converted_state): converted_state.time}
        # state.hash() -> (parent_state, action_taken)
        came_from: Dict[int, tuple] = {}
        closed: Set[int] = set()

        while open_heap:
            _, _, current = _heapq.heappop(open_heap)
            c_hash = hash(current)
            if c_hash in closed:
                continue
            closed.add(c_hash)

            if converted_goal.evaluate(current.fluents):
                path = []
                cur_hash = c_hash
                while cur_hash in came_from:
                    parent, act = came_from[cur_hash]
                    path.append(act)
                    cur_hash = hash(parent)
                path.reverse()
                self.last_search_stats = SearchStats(
                    states_expanded=len(closed), states_generated=len(best_g),
                )
                return path, current

            for action in _get_next_actions(current, actions):
                for successor, prob in _transition(current, action):
                    if prob == 0.0:
                        continue
                    s_hash = hash(successor)
                    if s_hash in closed:
                        continue

                    g_new = successor.time
                    if best_g.get(s_hash, float("inf")) <= g_new:
                        continue

                    best_g[s_hash] = g_new
                    came_from[s_hash] = (current, action)

                    w = extra_cost_fn(self, action, successor) if extra_cost_fn else 0.0
                    counter += 1
                    _heapq.heappush(
                        open_heap, (g_new + h(successor) + w, counter, successor)
                    )

        self.last_search_stats = SearchStats(states_expanded=len(closed), states_generated=len(best_g))
        return [], None

    def heuristic_to_goal(self, state: State, goal_str: str) -> float:
        """FF-heuristic estimate of the remaining cost to reach `goal_str` from `state`.

        `state` is expected to already be in converted/positive-precondition
        form (e.g. a search successor produced by this planner). Reuses the
        actions/mapping already prepared for the current search rather than
        re-preparing from scratch, so this is cheap to call repeatedly during
        search (e.g. from an extra_cost_fn).
        """
        goal = _goal_from_str(goal_str)
        self._ensure_mapping_includes_goal(goal)
        converted_goal = convert_goal_to_positive_preconditions(goal, self._current_mapping)
        return _ff_heuristic(state, converted_goal, self._converted_actions)

    def simulate_time_to_goal(self, state: State, goal_str: str) -> Optional[float]:
        """Actually plan (full, uncoordinated A*) from `state` to `goal_str`, and
        return the total cost of that plan — the sum of each of its actions' own
        duration (move, pick, place, cut, ...) — i.e. simulate completing the
        task from here and add up what each step costs, rather than estimating it.

        `state` is expected to already be in converted/positive-precondition
        form (e.g. a search successor produced by this planner). Returns 0.0 if
        `goal_str` is already satisfied, or None if it's unreachable from `state`
        using this robot's actions alone.
        """
        goal = _goal_from_str(goal_str)
        self._ensure_mapping_includes_goal(goal)
        converted_goal = convert_goal_to_positive_preconditions(goal, self._current_mapping)

        if converted_goal.evaluate(state.fluents):
            return 0.0

        # extra_cost_fn=None: the lookahead plan itself is plain A*, not
        # recursively coordinated — it's just measuring "how long would this
        # task take from here", not re-optimizing for other robots.
        path, final_state = self._search(state, converted_goal)
        if final_state is None:
            return None
        return sum(_action_duration(a) for a in path)

    def plan_sequence(
        self,
        state: State,
        goal,
        extra_cost_fn: Optional[ExtraCostFn] = None,
    ) -> list:
        """Run A* with the extra-cost hook and return the full plan."""
        converted_state, converted_goal = self._prepare(state, goal)
        path, _ = self._search(converted_state, converted_goal, extra_cost_fn)
        return path

    def __call__(self, state, goal, extra_cost_fn=None, **kwargs) -> str:
        plan = self.plan_sequence(state, goal, extra_cost_fn=extra_cost_fn)
        return plan[0].name if plan else "NONE"


def call_planner(
    state_strs: Set[str],
    goal_str: str,
    objects_by_type: Dict,
    operators: List,
    robot: str,
    use_stub: bool = False,
    planner_cls: type = MCTSPlanner,
    extra_cost_fn: Optional[ExtraCostFn] = None,
    stats_out: Optional[Dict[str, int]] = None,
) -> Optional[List[SimpleAction]]:
    """
    Plan for a single robot from a snapshot of the world state.

    Each robot plans in isolation: only its own `free` fluent is kept
    (other robots' `free` fluents are stripped so they don't constrain
    the search), and only actions belonging to this robot are searched.

    extra_cost_fn is forwarded to CoordinationAStarPlanner.plan_sequence
    when planner_cls is CoordinationAStarPlanner; ignored otherwise.

    stats_out, if given, is populated in place with
    {"states_expanded": ..., "states_generated": ...} when planner_cls is
    CoordinationAStarPlanner (see SearchStats) -- left untouched for plain
    AStarPlanner or MCTSPlanner, and for the "goal already satisfied"/stub
    early-outs below, since no search actually ran.

    Returns:
        None  — goal already satisfied in snapshot
        []    — no plan found (failure)
        [...]  — ordered list of SimpleActions to execute
    """
    if use_stub:
        return [
            SimpleAction(
                name=f"move {robot} base pickup",
                duration=5.0,
                preconditions={f"at {robot} base", f"free {robot}"},
                add_effects={f"at {robot} pickup", f"free {robot}"},
                del_effects={f"at {robot} base", f"free {robot}"},
            )
        ]

    # Build isolated state snapshot for this robot
    fluents = set()
    for s in state_strs:
        if s.startswith("free ") and s != f"free {robot}":
            continue
        fluents.add(_str_to_fluent(s))
    state = State(0.0, fluents)

    goal = _goal_from_str(goal_str)

    if goal.evaluate(state.fluents):
        return None

    # Ground operators and restrict search to this robot's actions only
    env = SymbolicEnvironment(state, objects_by_type, operators)
    all_actions = env.get_actions()
    robot_actions = [a for a in all_actions if a.name.split()[1] == robot]
    planner = planner_cls(robot_actions)

    if isinstance(planner, CoordinationAStarPlanner):
        raw_plan = planner.plan_sequence(state, goal, extra_cost_fn=extra_cost_fn)
        if stats_out is not None and planner.last_search_stats is not None:
            stats_out["states_expanded"] = planner.last_search_stats.states_expanded
            stats_out["states_generated"] = planner.last_search_stats.states_generated
        if not raw_plan:
            return []
        return [_action_to_simple(get_action_by_name(all_actions, a.name)) for a in raw_plan]

    if isinstance(planner, AStarPlanner):
        raw_plan = planner.plan_sequence(state, goal)
        if not raw_plan:
            return []
        return [_action_to_simple(get_action_by_name(all_actions, a.name)) for a in raw_plan]

    # MCTS returns one action per call — step through the plan one action at a time
    plan: List[SimpleAction] = []
    iterations = 0
    while not goal.evaluate(env.state.fluents) and iterations < 50:
        action_name = planner(
            env.state, goal, max_iterations=2000, c=300, max_depth=20,
            heuristic_multiplier=3
        )
        if action_name == "NONE" or action_name.startswith("no_op"):
            break
        action = get_action_by_name(all_actions, action_name)
        plan.append(_action_to_simple(action))
        env.act(action)
        iterations += 1

    if not goal.evaluate(env.state.fluents):
        return []
    return plan
