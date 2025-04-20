from .core import Fluent, Action, State, transition
from typing import Callable, Dict, List, Set, Tuple


class RelaxedPlanningGraph:
    def __init__(self, actions: List[Action]):
        self.actions = actions
        self.known_fluents: Set[Fluent] = set()
        self.fact_to_action: Dict[Fluent, Action] = {}
        self.action_to_duration: Dict[Action, float] = {}
        self.visited_actions: Set[Action] = set()

    def extend(self, new_fluents):
        newly_added = new_fluents - self.known_fluents
        self.known_fluents |= new_fluents

        while newly_added:
            next_newly_added = set()
            state = State(time=0, fluents=self.known_fluents)
            for action in self.actions:
                if action in self.visited_actions:
                    continue
                if not state.satisfies_precondition(action, relax=True):
                    continue

                for successor, _ in transition(state, action, relax=True):
                    duration = successor.time  # assumes zeroed state
                    self.action_to_duration[action] = duration
                    self.visited_actions.add(action)

                    for f in successor.fluents:
                        if f not in self.known_fluents:
                            self.known_fluents.add(f)
                            next_newly_added.add(f)
                            self.fact_to_action[f] = action
                        elif f in self.fact_to_action.keys():
                            self.fact_to_action[f] = min(
                                action,
                                self.fact_to_action[f],
                                key=lambda a: a.effects[-1].time,
                            )
            newly_added = next_newly_added

    def compute_relaxed_plan_cost(
        self, initial_fluents: Set[Fluent], goal_fn: Callable[[Set[Fluent]], bool]
    ) -> float:
        if not goal_fn(self.known_fluents):
            return float("inf")

        # Identify required goal fluents
        required = set()
        for f in self.known_fluents:
            test_set = self.known_fluents - {f}
            if not goal_fn(test_set):
                required.add(f)

        # Extract plan
        needed = set(required)
        used_actions = set()
        total_duration = 0.0

        while needed:
            f = needed.pop()
            if f in initial_fluents:
                continue
            action = self.fact_to_action.get(f)
            if action and action not in used_actions:
                used_actions.add(action)
                total_duration += self.action_to_duration[action]
                for p in action._pos_precond:
                    if p not in initial_fluents:
                        needed.add(p)

        return total_duration


def make_ff_heuristic(
    actions: List[Action],
    goal_fn: Callable[[Set[Fluent]], bool],
    transition_fn: Callable[[State, Action, bool], List[Tuple[State, float]]],
):
    raise NotImplementedError()
    rpg = RelaxedPlanningGraph(actions)

    def ff_heuristic(state: State) -> float:
        ctime = state.time
        state = transition(state, None, relax=True)[0][0]
        dtime = state.time - ctime
        current_fluents = set(state.fluents)
        rpg.extend(current_fluents)
        return dtime + rpg.compute_relaxed_plan_cost(current_fluents, goal_fn)

    return ff_heuristic


def ff_heuristic(state: State, is_goal_fn, all_actions, ff_memory=None) -> float:
    # 1. Forward relaxed reachability
    t0 = state.time
    state = transition(state, None, relax=True)[0][0]
    dtime = state.time - t0
    initial_known_fluents = state.fluents
    known_fluents = set(state.fluents)
    if ff_memory and initial_known_fluents in ff_memory.keys():
        return dtime + ff_memory[initial_known_fluents]
    newly_added = set(known_fluents)
    fact_to_action = {}  # Fluent -> Action that added it
    action_to_duration = {}  # Action -> Relaxed duration
    all_actions = set(all_actions)
    visited_actions = set()

    while newly_added:
        next_newly_added = set()
        state = State(time=0, fluents=known_fluents)
        for action in all_actions:
            if action in visited_actions:
                continue
            if not state.satisfies_precondition(action, relax=True):
                continue

            for successor, _ in transition(state, action, relax=True):
                duration = successor.time - state.time
                action_to_duration[action] = duration
                visited_actions.add(action)

                for f in successor.fluents:
                    if f not in known_fluents:
                        known_fluents.add(f)
                        next_newly_added.add(f)
                        fact_to_action[f] = action
                    # elif f in fact_to_action.keys():
                    #     fact_to_action[f] = min(
                    #         action, fact_to_action[f], key=lambda a: a.effects[-1].time
                    #     )
        newly_added = next_newly_added
        all_actions -= visited_actions

    if not is_goal_fn(known_fluents):
        return float("inf")  # Relaxed goal not reachable

    # 2. Extract required goal fluents by ablation
    required_fluents = set()
    for f in known_fluents:
        test_set = known_fluents - {f}
        if not is_goal_fn(test_set):
            required_fluents.add(f)

    # 3. Backward relaxed plan extraction
    needed = set(required_fluents)
    used_actions = set()
    total_duration = 0.0

    while needed:
        f = needed.pop()
        if f in initial_known_fluents:
            continue
        action = fact_to_action.get(f)
        if action and action not in used_actions:
            used_actions.add(action)
            total_duration += action_to_duration[action]
            for p in action._pos_precond:
                if p not in initial_known_fluents:
                    needed.add(p)

    if ff_memory:
        ff_memory[initial_known_fluents] = total_duration
    return dtime + total_duration
