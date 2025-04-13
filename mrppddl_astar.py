import random
import heapq

from mrppddl.core import Fluent, Effect, Operator, State, get_next_actions, transition, get_action_by_name, ProbEffect, OptCallable
from mrppddl.helper import _make_callable

F = Fluent

# Move and Visit Operator
random.seed(8616)
move_time_fn = lambda *args: random.random() + 5.0  #noqa: E731
move_visit_op = Operator(
        name="move_visit",
        parameters=[("?robot", "robot"), ("?loc_from", "location"), ("?loc_to", "location")],
        preconditions=[F("at ?robot ?loc_from"),
                       # ~F("visited ?loc_to"),
                       F("free ?robot")],
        effects=[
            Effect(time=0,
                   resulting_fluents={~F("free ?robot"), ~F("at ?robot ?loc_from"),}),
            Effect(time=(move_time_fn, ["?robot", "?loc_from", "?loc_to"]),
                   resulting_fluents={F("free ?robot"), F("visited ?loc_to"), F("at ?robot ?loc_to")})
        ])

# Get all actions
objects_by_type = {
    "robot": ["r1", "r2"],
    # "location": ["start", "roomA", "roomB", "roomC", "roomD", "roomE", "roomF"],
    "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
}
all_actions = move_visit_op.instantiate(objects_by_type)

# Initial state
initial_state = State(
    time=0,
    active_fluents={
        F("at r1 start"), F("free r1"),
        F("at r2 start"), F("free r2"),
        F("visited start"),
    })



def astar(start_state, all_actions, is_goal_state, heuristic_fn=None):
    open_heap = []
    heapq.heappush(open_heap, (0, 0, start_state))  # (f = g + h, g, state)
    came_from = {}

    closed_set = set()
    counter = 0

    while open_heap:
        counter += 1
        _, current_g, current = heapq.heappop(open_heap)

        if is_goal_state(current.active_fluents):
            print(counter)
            return reconstruct_path(came_from, current)

        if current in closed_set:
            continue
        closed_set.add(current)

        for action in all_actions:
            if not current.satisfies_precondition(action):
                continue

            for successor, prob in transition(current, action):
                if prob == 0.0:
                    continue

                g = successor.time
                came_from[successor] = (current, action)

                h = heuristic_fn(successor) if heuristic_fn else 0
                f = g + h

                heapq.heappush(open_heap, (f, g, successor))

    return None  # No path found

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        prev, action = came_from[current]
        path.append(action)
        current = prev
    path.reverse()
    return path


def is_goal_state(active_fluents) -> bool:
    return len(objects_by_type['location']) == len([f for f in active_fluents
                                                    if f.name == 'visited'])

from mrppddl.core import Action
from typing import List, Set, Callable
def ff_heuristic(
    state: State,
    actions: List[Action],
    goal_fn: Callable[[Set[Fluent]], bool]
) -> float:
    # 1. Forward relaxed reachability
    known_fluents = set(state.active_fluents.fluents)
    newly_added = set(known_fluents)
    fact_to_action = {}       # Fluent -> Action that added it
    action_to_duration = {}   # Action -> Relaxed duration
    visited_actions = set()

    while newly_added:
        next_newly_added = set()
        for action in actions:
            if action in visited_actions:
                continue
            if not state.satisfies_precondition(action, relax=True):
                continue

            for successor, _ in transition(state, action, relax=True):
                duration = successor.time - state.time
                action_to_duration[action] = duration
                visited_actions.add(action)

                for f in successor.active_fluents.fluents:
                    if f not in known_fluents:
                        known_fluents.add(f)
                        next_newly_added.add(f)
                        fact_to_action[f] = action
        newly_added = next_newly_added

    if not goal_fn(known_fluents):
        return float('inf')  # Relaxed goal not reachable

    # 2. Extract required goal fluents by ablation
    required_fluents = set()
    for f in known_fluents:
        test_set = known_fluents - {f}
        if not goal_fn(test_set):
            required_fluents.add(f)

    # 3. Backward relaxed plan extraction
    needed = set(required_fluents)
    used_actions = set()
    total_duration = 0.0

    while needed:
        f = needed.pop()
        if f in state.active_fluents.fluents:
            continue
        action = fact_to_action.get(f)
        if action and action not in used_actions:
            used_actions.add(action)
            total_duration += action_to_duration.get(action, 0.0)
            for p in action._pos_precond:
                if p not in state.active_fluents.fluents:
                    needed.add(p)

    return total_duration


path = astar(initial_state, all_actions, is_goal_state,
             lambda state: ff_heuristic(state, all_actions, is_goal_state) / 2)
s = initial_state
for a in path:
    s = transition(s, a)[0][0]
    print(a.name)
    print(s)
