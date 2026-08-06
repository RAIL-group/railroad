import heapq
from collections.abc import Callable
from dataclasses import dataclass

from railroad.core import Action, Fluent, Goal, State, get_next_actions
from tqdm import tqdm

from .utilities import get_action_cost, get_next_state


# data structures for astar search
@dataclass
class Trajectory:
    """
    Data structure used to represent search tree trajectories (paths).
    """
    state_history: list[State]
    plan: list[Action]
    # used to avoid having to recompute the prob of no interruption for each child
    interruption_probs: list[float]
    cost: float = 0.0
    value: float = 0.0
    level: int = 0
    h_value: float = 0.0
    discounted_h_value: float = 0.0

    def create_child(
        self,
        goal: Goal,
        actions: list[Action],
        action: Action,
        interruption_value: float,
        interruption_prob: float,
        heuristic_fn: float | Callable[[State, Goal, list[Action]], float] = 0,
        current_task_reward: float = 0
    ) -> 'Trajectory':
        """
        Helper function for creation of trajectories on the frontier.
        """
        # compute f(n)
        accumulated_cost = discounted_accumulated_cost(
            self, action, interruption_value, interruption_prob
        )
        next_state, _ = get_next_state(self.state_history[-1], action)
        estimated_future_cost, q_value = h(
            self, next_state, goal, actions, heuristic_fn, interruption_prob,
            current_task_reward
        )

        return Trajectory(
            cost=accumulated_cost,
            value=accumulated_cost+estimated_future_cost,
            level=self.level+1,
            state_history=self.state_history + [next_state],
            plan=self.plan + [action],
            interruption_probs=self.interruption_probs + [interruption_prob],
            h_value=q_value,
            discounted_h_value=estimated_future_cost
        )

    def get_plan_cost(self):
        """
        Returns actual cost of a trajectory, without factoring the interruption probabilities.
        """
        plan_cost = 0
        for act in self.plan:
            plan_cost+=get_action_cost(act)
        return plan_cost

    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            raise NotImplementedError
        return self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, Trajectory):
            raise NotImplementedError
        return self.value < other.value

    def __le__(self, other):
        if not isinstance(other, Trajectory):
            raise NotImplementedError
        return self.value <= other.value

    def __gt__(self, other):
        if not isinstance(other, Trajectory):
            raise NotImplementedError
        return self.value > other.value

    def __ge__(self, other):
        if not isinstance(other, Trajectory):
            raise NotImplementedError
        return self.value >= other.value


def discounted_accumulated_cost(
    traj: Trajectory,
    action: Action,
    interruption_value: float,
    interruption_prob: float
) -> float:
    """
    Accumulated cost function of trajectory.
    """
    path_cost = traj.cost
    no_int_prob = get_no_int_prob(traj)

    # get reward
    r = get_action_cost(action)

    # discount the interruption value
    interruption_value*=interruption_prob

    return path_cost + no_int_prob * (r + interruption_value)


def h(
    traj: Trajectory,
    state: State,
    goal: Goal,
    all_actions: list[Action],
    heuristic_fn: float | Callable[[State, Goal, list[Action]], float],
    next_interruption_prob: float,
    reward: float
) -> tuple[float, float]:
    """
    Heuristic function used to estimate the cost remaining for the trajectory.
    Returns both the q_value and the discounted q_value, including any rewards
    for completing the current task.
    """
    if not isinstance(heuristic_fn, (int, float)):
        estimated_q_value = heuristic_fn(state, goal, all_actions)
    else:
        estimated_q_value = heuristic_fn

    discount = get_no_int_prob(traj) * (1 - next_interruption_prob)
    # reward term used to incentivize the completion of the current ask
    discounted_q_value = discount * (estimated_q_value + reward)

    return discounted_q_value, estimated_q_value


def astar_search(
    state: State,
    goal: Goal,
    actions: list[Action],
    interrupting_task_dist: tuple[list[Goal], list[float]] | None,
    heuristic_fn: float | Callable[[State, Goal, list[Action]], float] = 0,
    interruption_prob_fn: float | Callable[[float], float] = 0.1,
    interruption_value_fn: Callable[[frozenset[Fluent]], float] | None = None,
    current_task_reward: float = 0,
    num_steps: int = 20000,
    print_trace: bool = False
) -> tuple[list[Action], float, bool]:
    """
    Astar algorithm implementation.
    """
    value_cache: dict[frozenset[Fluent], float] = {}
    frontier = []
    expanded = set()
    insertion_order = 1

    initial_traj = Trajectory(state_history=[state], plan=[], interruption_probs=[])
    heapq.heappush(frontier, (initial_traj, -1, 0))
    num_expanded_nodes = 0

    # search loop
    print(f"Current Goal: {goal}")
    with tqdm(total=num_steps) as pbar:
        while num_expanded_nodes < num_steps:
            # some logging functionality for debugging
            if print_trace:
                print_frontier_trace(num_expanded_nodes, frontier)

            # find expansion node
            expand, _, _ = heapq.heappop(frontier)
            curr_state = expand.state_history[-1]

            # check for goal condition being met
            if goal.evaluate(curr_state.fluents):
                return expand.plan, expand.cost, True

            # check if we've already expanded this state
            if curr_state.fluents in expanded:
                continue
            # otherwise add it
            expanded.add(frozenset(curr_state.fluents))
            # expand search tree
            available_actions = get_next_actions(curr_state, actions)
            for action in available_actions:
                # probability of interruption after taking action from current state
                next_state, interruption_prob = get_next_state(
                    expand.state_history[-1],
                    action,
                    interruption_prob_fn
                )

                next_state_key = frozenset(next_state.fluents)
                # check if this state has already been expanded
                if next_state_key in expanded:
                    continue

                # value of next state for interrupting tasks needed and not found
                # compute and cache
                if interrupting_task_dist and not check_value_cache(next_state_key, value_cache):
                    # after an interrupting task has arrived once, we assume that another
                    # interrupting task cannot arrive. following that logic, the expected
                    # value of a state is not discounted.
                    if interruption_value_fn:
                        val = interruption_value_fn(next_state_key)
                    else:
                        val = compute_interruption_value(
                            next_state, actions, interrupting_task_dist, heuristic_fn, 0
                        )
                    value_cache[next_state_key] = val

                # construct new trajectory
                # use get method instead of directly indexing value_cache to account for case where
                # there are no interrupting tasks
                child_traj = expand.create_child(
                    goal, actions, action, value_cache.get(next_state_key, 0),
                    interruption_prob, heuristic_fn, current_task_reward
                )
                heapq.heappush(frontier, (child_traj, child_traj.h_value, insertion_order))
                insertion_order+=1

            pbar.update(1)

    # goal not reached, get best trajectory found
    best_found, _, _ = heapq.heappop(frontier)
    return best_found.plan, best_found.cost, False


def compute_interruption_value(
    state: State,
    actions: list[Action],
    interrupting_task_dist: tuple[list[Goal], list[float]],
    heuristic_fn: float | Callable[[State, Goal, list[Action]], float] = 0,
    interruption_prob_fn: float | Callable[[float], float] = 0
) -> float:
    """
    Computes the expected value of a state for a task distribution.
    Returns -1 if a successful plan for one of the tasks in the task distribution
    was unable to be found. Otherwise returns the expected cost.
    """
    expected_cost = 0.0
    for task, prob in zip(*interrupting_task_dist):
        plan, cost, success = astar_search(
            state, task, actions, None, heuristic_fn, interruption_prob_fn
        )
        if not success:
            return -1
        expected_cost += (prob * cost)
    return expected_cost


def check_value_cache(
    state: frozenset[Fluent], value_cache: dict[frozenset[Fluent], float]
) -> bool:
    """
    Checks if the value of a state is already cached.
    """
    return state in value_cache


def get_no_int_prob(traj: Trajectory) -> float:
    """
    Returns the probability of an interrupting task not arriving,
    based on the level of the search tree.
    """
    no_int_prob = 1
    for prob in traj.interruption_probs:
        no_int_prob*=(1 - prob)
    return no_int_prob

# debug helper functions
def print_frontier_trace(step: int, frontier: list[tuple[Trajectory, float]]) -> None:
    """
    Prints out a trace of the trajectories currently stored in the frontier.
    """
    print(f"Planning Step: {step}")
    print(f"Frontier: # of trajectories in frontier = {len(frontier)}\n")
    # sorted_frontier = sorted(frontier, key=lambda x: x[0])
    for j, traj_tuple in enumerate(frontier[:5]):
        traj = traj_tuple[0]
        print(f"Trajectory {j}: length - {traj.level}")
        print(f"Value: {traj.value}")
        print(f"Discounted Cost: {traj.cost}; Plan Cost: {traj.get_plan_cost()}")
        print(f"Discounted h-value: {traj.discounted_h_value}; h-value: {traj.h_value}")
        print(f"Last 5 actions in trajectory: {[a.name for a in traj.plan]}\n")
