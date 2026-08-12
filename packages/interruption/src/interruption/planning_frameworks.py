"""
Implementations of the interruption-based, myopic, and anticipatory planning planners
for ProcTHOR environments. These planner implementations all utilize the astar_search
function from planner.py
"""
def get_no_int_prob(interruption_probs: list[float]) -> float:
    """
    Returns the probability of an interrupting task not arriving,
    based on the level of the search tree.
    """
    no_int_prob = 1
    for prob in interruption_probs:
        no_int_prob*=(1 - prob)
    return no_int_prob


def get_no_int_discount(interruption_probs: list[float]) -> float:
    """
    Helper function for the case where action costs/heuristic values
    are not discounted by the probility a task arriving during the
    execution of an action.
    """
    return 1
