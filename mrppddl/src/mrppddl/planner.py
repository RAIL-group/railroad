from .core import (
    Fluent,
    Action,
    State,
    get_next_actions,
    transition,
)
from typing import Callable, List, Set, Dict, Optional
import heapq
import random
import math


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


def ucb_score(parent_visits: int, child, c: float = 1.414) -> float:
    """Upper Confidence Bound for Trees (UCT) score."""
    if child.visits == 0:
        return float('inf')
    exploitation = child.value / child.visits
    exploration = c * math.sqrt(math.log(parent_visits) / child.visits)
    return exploitation + exploration


class MCTSDecisionNode:
    def __init__(self, state: State, parent: Optional["MCTSChanceNode"] = None):
        self.state = state
        self.parent = parent
        self.children: Dict[Action, "MCTSChanceNode"] = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[Action] = []

class MCTSChanceNode:
    def __init__(self, action: Action, parent: MCTSDecisionNode):
        self.action = action
        self.parent = parent
        self.children: List[MCTSDecisionNode] = []
        self.outcome_weights: List[float] = []
        self.visits = 0
        self.value = 0.0

def mcts(
    root_state: State,
    # actions_fn: Callable[[State], List[Action]],
    all_actions: List[Action],
    goal_fn: Callable[[Set[Fluent]], bool],
    heuristic_fn: Callable[[State], float],
    max_iterations: int = 1000,
    max_depth: int = 20
) -> Dict[State, Action]:
    actions_fn = lambda state: get_next_actions(state, all_actions)  #noqa: E731
    root = MCTSDecisionNode(root_state)
    root.untried_actions = actions_fn(root_state)
    state_to_best_action: Dict[State, Action] = {}

    for _ in range(max_iterations):
        node = root
        depth = 0

        # Selection
        while depth < max_depth:
            if node.untried_actions:
                break
            if not node.children:
                break
            if goal_fn(node.state.fluents):
                break
            # Choose action with best UCB
            best_action, best_chance_node = max(
                node.children.items(),
                key=lambda item: ucb_score(node.visits, item[1])
            )
            if not best_chance_node.children:
                break
            # Sample an outcome node proportionally to visit count or randomly
            node = random.choices(best_chance_node.children)[0]
            # print(node.value/node.visits)
            depth += 1

        # Expansion
        if node.untried_actions:
            action = node.untried_actions.pop()
            outcomes = transition(node.state, action)
            if not outcomes:
                continue
            chance_node = MCTSChanceNode(action, node)
            node.children[action] = chance_node
            for successor, prob in outcomes:
                child_node = MCTSDecisionNode(successor, parent=chance_node)
                child_node.untried_actions = actions_fn(successor)
                chance_node.children.append(child_node)
                chance_node.outcome_weights.append(prob)
            # Expand to one of the children
            node = random.choices(chance_node.children, weights=chance_node.outcome_weights)[0]
            depth += 1

        # Greedy rollout using heuristic
        simulated_state = node.state
        # sim_depth = 0
        # while sim_depth < max_depth and not goal_fn(simulated_state.active_fluents.fluents):
        #     available_actions = actions_fn(simulated_state)
        #     if not available_actions:
        #         break
        #     best_action = min(available_actions, key=lambda a: heuristic_fn(transition_fn(simulated_state, a)[0][0]))
        #     outcomes = transition_fn(simulated_state, best_action)
        #     if not outcomes:
        #         break
        #     simulated_state, _ = max(outcomes, key=lambda x: x[1])  # most likely outcome
        #     sim_depth += 1

        if goal_fn(node.state.fluents):
            reward = -node.state.time
        else:
            hval = heuristic_fn(simulated_state)
            if hval > 1e10:
                hval = 100
            reward = -node.state.time - hval

        # Backpropagation
        while node:
            node.visits += 1
            node.value += reward
            if isinstance(node, MCTSDecisionNode):
                node = node.parent
            elif isinstance(node, MCTSChanceNode):
                node.visits += 1
                node.value += reward
                node = node.parent

    # Extract policy: best action from root
    if root.children:
        best_action = max(root.children.items(), key=lambda item: item[1].value / item[1].visits if item[1].visits > 0 else float('-inf'))[0]
        state_to_best_action[root.state] = best_action

    return state_to_best_action, root
