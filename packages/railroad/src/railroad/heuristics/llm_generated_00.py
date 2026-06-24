from railroad.core import Heuristic

class CustomHeuristic(Heuristic):
    """
    A custom heuristic that estimates the cost to reach the goal by summing the
    execution and wait costs of the actions in the cheapest relaxed plan.
    """

    def __call__(self, state, goal, rpg):
        """
        Estimates the cost to reach the goal from the current state.

        Args:
            state (State): The current state of the environment.
            goal (Goal): The goal to be achieved.
            rpg (dict): The Relaxed Planning Graph (RPG) information, including
                        the cheapest_relaxed_plan.

        Returns:
            float: The estimated cost to reach the goal. Returns 0.0 if the
                   goal is already satisfied, and float('inf') if the goal
                   is unreachable in the relaxed plan.
        """
        # If the goal is already satisfied in the current state, the cost is 0.
        if goal.evaluate(state.fluents):
            return 0.0

        total_relaxed_cost = 0.0
        # Sum the execution and wait costs of all actions in the cheapest relaxed plan.
        for step in rpg["cheapest_relaxed_plan"]:
            total_relaxed_cost += step["exec_cost"] + step["wait_cost"]

        # If the total relaxed cost is 0.0 but the goal is not satisfied,
        # it implies that the goal is unreachable even in the relaxed planning graph.
        # This state should be heavily penalized.
        if total_relaxed_cost == 0.0:
            return float('inf')

        return total_relaxed_cost