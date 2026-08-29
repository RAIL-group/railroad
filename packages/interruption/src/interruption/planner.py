import heapq
from collections.abc import Callable
from dataclasses import dataclass
from tqdm import tqdm

from railroad.core import Action, Fluent, Goal, State, get_next_actions
from railroad.environment.procthor.scenegraph import SceneGraph

from .utilities import (
    get_action_cost,
    get_next_state,
    get_reward,
    get_discounted_value,
    get_updated_scene_graph
)

# constants
DEBUG = False

# data structures for astar search
@dataclass
class InterruptionSearchProblem:
    """
    Data structure used store inputs required to define a 
    search problem with continual task arrival.
    """
    goal: Goal
    actions: list[Action]
    interrupting_task_dist: tuple[list[Goal], list[float]] | None = None
    # environment side interruption probability function
    interruption_prob_fn: float | Callable[[float], float] = 0


@dataclass
class PlannerConfig:
    """
    Data stucture used to store user-specified hyperparameters for
    A* Search.
    """
    discount_fn: Callable[[list[float]], float]
    heuristic_fn: Callable[[State, Goal, list[Action], float], float] | float
    planner_interruption_prob_fn: float | Callable[[float], float] | None = None
    interruption_value_fn: Callable[[SceneGraph], float] | None = None
    current_task_reward: float = 0


@dataclass
class InterruptionTrajectory:
    """
    Data structure used to represent search tree trajectories (paths).
    """
    state_history: list[State]
    plan: list[Action]
    # used to avoid having to recompute the prob of no interruption for each child
    interruption_probs: list[float]
    scene_graph: SceneGraph | None
    level: int = 0
    cost: float = 0.0
    value: float = 0.0
    h_value: float = 0.0
    discounted_h_value: float = 0.0
    # # for debugging
    # v_ap: float = 0.0
    # ff_value: float = 0.0


    def create_child(
        self,
        search_problem: InterruptionSearchProblem,
        planner_params: PlannerConfig,
        action: Action,
        interruption_prob: float
    ) -> 'InterruptionTrajectory':
        """
        Helper function for creation of trajectories to add to the frontier.
        """
        next_state, _ = get_next_state(self.state_history[-1], action)

        # compute accumulated cost (g(traj))
        interrupting_task_ev = 0
        scene_graph = None
        if self.scene_graph is not None:
            scene_graph = self.scene_graph.copy()
            get_updated_scene_graph(scene_graph, next_state, action)
        if planner_params.interruption_value_fn is not None:
            assert scene_graph is not None
            interrupting_task_ev = planner_params.interruption_value_fn(scene_graph)

        accumulated_cost = self.cost + get_reward(
            action,
            planner_params.discount_fn(self.interruption_probs),
            interrupting_task_ev * interruption_prob
        )

        # compute estimated future cost (h(s'))
        if isinstance(planner_params.heuristic_fn, (int, float)):
            undiscounted_future_cost = planner_params.heuristic_fn
        else:
            v_ap = 0 if planner_params.interruption_value_fn is None else interrupting_task_ev
            undiscounted_future_cost = planner_params.heuristic_fn(
                next_state, search_problem.goal, search_problem.actions, v_ap
            )
        estimated_future_cost = get_discounted_value(
            undiscounted_future_cost,
            planner_params.discount_fn(self.interruption_probs) * (1 - interruption_prob),
            planner_params.current_task_reward
        )

        # # for debugging
        # if DEBUG:
        #     interested_plan = [
        #         'move robot1 start_loc countertop_3',
        #         'pick robot1 r1-right countertop_3 apple_8',
        #         'pick robot1 r1-left countertop_3 tomato_12'
        #     ]
        #     traj_plan_names = [act.name for act in self.plan + [action]]
        #     if all(act in traj_plan_names for act in interested_plan):
        #         print(v_ap)
        #         print(undiscounted_future_cost)


        return InterruptionTrajectory(
            cost=accumulated_cost,
            value=accumulated_cost+estimated_future_cost,
            level=self.level+1,
            state_history=self.state_history + [next_state],
            plan=self.plan + [action],
            interruption_probs=self.interruption_probs + [interruption_prob],
            h_value=undiscounted_future_cost,
            discounted_h_value=estimated_future_cost,
            scene_graph=scene_graph,
            # # for debugging
            # v_ap=v_ap,
            # ff_value=undiscounted_future_cost - v_ap
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
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value == other.value


    def __lt__(self, other):
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value < other.value


    def __le__(self, other):
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value <= other.value


    def __gt__(self, other):
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value > other.value


    def __ge__(self, other):
        if not isinstance(other, InterruptionTrajectory):
            raise NotImplementedError
        return self.value >= other.value


def astar_search(
    initial_state: tuple[State, SceneGraph | None],
    interruption_problem: InterruptionSearchProblem,
    search_params: PlannerConfig,
    num_steps: int = 20000,
) -> tuple[list[Action], float, bool, SceneGraph | None]:#, State]:
    """
    Astar algorithm implementation.
    """
    # TODO - may want to bring back ev caching for task distribution eventually
    # value_cache: dict[frozenset[Fluent], float] = {}

    # the heap stores tuple(traj, heuristic_value, insertion_order)
    frontier = []
    expanded = set()
    insertion_order = 1
    # keeps track of number of astar search iterations
    num_expanded_nodes = 0

    # push initial state onto heap
    heapq.heappush(
        frontier,
        (
            InterruptionTrajectory(
                state_history=[initial_state[0]],
                plan=[],
                interruption_probs=[],
                scene_graph=initial_state[1]
            ), -1, 0
        )
    )

    # search loop
    print(f"Current Goal: {interruption_problem.goal}")
    with tqdm(total=num_steps) as pbar:
        while num_expanded_nodes < num_steps:
            # some logging functionality for debugging
            if DEBUG:
                print_frontier_trace(num_expanded_nodes, frontier)

            # find expansion node
            expand, _, _ = heapq.heappop(frontier)
            curr_state = expand.state_history[-1]

            # check for goal condition being met
            if interruption_problem.goal.evaluate(curr_state.fluents):
                return expand.plan, expand.cost, True, expand.scene_graph#, curr_state

            # check if we've already expanded this state
            if curr_state.fluents in expanded:
                continue
            # otherwise add it
            expanded.add(frozenset(curr_state.fluents))
            # expand search tree
            num_expanded_nodes+=1
            for action in get_next_actions(curr_state, interruption_problem.actions):
                # probability of interruption after taking action from current state
                next_state, interruption_prob = get_next_state(
                    expand.state_history[-1],
                    action,
                    (
                        search_params.planner_interruption_prob_fn
                        if search_params.planner_interruption_prob_fn is not None
                        else 0
                    )
                )

                next_state_key = frozenset(next_state.fluents)
                # check if this state has already been expanded
                if next_state_key in expanded:
                    continue

                # construct new trajectory
                child_traj = expand.create_child(
                    interruption_problem,
                    search_params,
                    action,
                    interruption_prob
                )
                heapq.heappush(frontier, (child_traj, child_traj.h_value, insertion_order))
                insertion_order+=1

            pbar.update(1)

    # goal not reached, get best trajectory found
    best_found, _, _ = heapq.heappop(frontier)
    return best_found.plan, best_found.cost, False, best_found.scene_graph


def compute_interruption_value(
    state: State,
    actions: list[Action],
    interrupting_task_dist: tuple[list[Goal], list[float]],
    heuristic_fn: float | Callable[[State, Goal, list[Action], float], float] = 0,
) -> float:
    """
    Computes the expected value of a state for a task distribution.
    Returns -1 if a successful plan for one of the tasks in the task distribution
    was unable to be found. Otherwise returns the expected cost.
    """
    expected_cost = 0.0
    # setup for myopic planning approach
    search_params = PlannerConfig(
        discount_fn=lambda x: 1.0,
        heuristic_fn=heuristic_fn
    )

    for task, prob in zip(*interrupting_task_dist):
        # setup data structure for interruption problem where there is
        # no chance another interrupting task will arrive
        search_problem = InterruptionSearchProblem(
            task,
            actions
        )
        plan, cost, success, _ = astar_search(
            (state, None),
            search_problem,
            search_params
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


# debug helper functions
def print_frontier_trace(step: int, frontier: list[tuple[InterruptionTrajectory, float]]) -> None:
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
        # # added for debugging across machines
        # print(f"v_ap: {traj.v_ap}; ff-value: {traj.ff_value}")
        # print(f"Last 5 actions in trajectory: {[a.name for a in traj.plan]}\n")
