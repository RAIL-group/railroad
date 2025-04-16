from .core import (
    Fluent,
    Effect,
    Operator,
    State,
    get_next_actions,
    transition,
    get_action_by_name,
    ProbEffect,
    OptCallable,
)
import heapq


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        prev, action = came_from[current]
        path.append(action)
        current = prev
    path.reverse()
    return path


def astar(start_state, all_actions, is_goal_state, heuristic_fn=None):
    open_heap = []
    heapq.heappush(open_heap, (0, start_state))  # (f = g + h, g, state)
    came_from = {}

    closed_set = set()
    counter = 0

    while open_heap:
        counter += 1
        _, current = heapq.heappop(open_heap)

        if is_goal_state(current.fluents):
            return reconstruct_path(came_from, current)

        if current in closed_set:
            continue
        closed_set.add(current)

        for action in get_next_actions(current, all_actions):
            for successor, prob in transition(current, action):
                if prob == 0.0:
                    continue

                g = successor.time
                came_from[successor] = (current, action)

                h = heuristic_fn(successor) if heuristic_fn else 0
                f = g + h

                heapq.heappush(open_heap, (f, successor))

    return None  # No path found
