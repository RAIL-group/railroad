"""
Cost formulations for the interruption-aware A* search.

A formulation decides how one search step (parent -> child, via an action) turns
into the child's accumulated cost g, its running discount, and the discounted
estimate of its future cost. Selected per planner through
PlannerConfig.cost_formulation; to try a new one, subclass DefaultFormulation
and override only the method being changed, e.g.

    @dataclass(frozen=True)
    class NoEVInAccumulatedCost(DefaultFormulation):
        def accumulated_cost(self, terms: StepTerms) -> float:
            return terms.parent_cost + terms.parent_discount * terms.action_cost
"""
from dataclasses import dataclass
from typing import Literal, Protocol

# the discount that, instead of a fixed factor, multiplies by the probability no
# interrupting task arrives during the step: gamma_{t+1} = gamma_t * (1 - p)
NO_INTERRUPTION = "no_interruption"
Discount = float | Literal["no_interruption"]


@dataclass(frozen=True)
class StepTerms:
    """
    Everything a cost formulation may read for one parent -> child step. It holds
    nothing derived from the heuristic: astar_search computes accumulated costs
    before the heuristic is evaluated, to prune candidate children.
    """
    parent_cost: float           # accumulated cost g of the parent
    parent_discount: float       # the parent's running discount (gamma_t)
    action_cost: float           # c(a)
    interruption_prob: float     # probability of an interrupting task arriving after a
    interrupting_task_ev: float  # learned value of the child state, 0 without a model


def _child_discount(discount: Discount, terms: StepTerms) -> float:
    if discount == NO_INTERRUPTION:
        return terms.parent_discount * (1 - terms.interruption_prob)
    return terms.parent_discount * discount


class CostFormulation(Protocol):
    def child_discount(self, terms: StepTerms) -> float:
        """The child's running discount (gamma_{t+1})."""
        ...

    def accumulated_cost(self, terms: StepTerms) -> float:
        """The child's accumulated cost g'."""
        ...

    def estimated_future_cost(
        self, terms: StepTerms, child_discount: float, undiscounted_future_cost: float
    ) -> float:
        """The discounted estimate of the child's future cost, from its heuristic value."""
        ...


@dataclass(frozen=True)
class DefaultFormulation:
    """
    The original formulation:
        gamma_{t+1} = gamma_t * (1 - p)        if discount is NO_INTERRUPTION
                    = gamma_t * discount       otherwise
        g'          = g + gamma_t * (c(a) + p * V(s'))
        future      = gamma_{t+1} * (h(s') + current_task_reward)
    """
    discount: Discount = 1.0
    current_task_reward: float = 0

    def child_discount(self, terms: StepTerms) -> float:
        return _child_discount(self.discount, terms)

    def accumulated_cost(self, terms: StepTerms) -> float:
        return terms.parent_cost + terms.parent_discount * (
            terms.action_cost + terms.interrupting_task_ev * terms.interruption_prob
        )

    def estimated_future_cost(
        self, terms: StepTerms, child_discount: float, undiscounted_future_cost: float
    ) -> float:
        return child_discount * (undiscounted_future_cost + self.current_task_reward)


@dataclass(frozen=True)
class TerminalValueFormulation:
    """
    The anticipatory planning objective (planners.AnticipatorySearchPlanner): no
    task arrives mid-task, so the learned value V_AP is charged once, for the
    state the task ends in, and never per step:
        gamma_{t+1} = 1
        g'          = g + c(a)                  V(s) = r(s, a) + V(s')
        future      = h(s')                     V(s_goal) = V_AP(s_goal)
    With FFHeuristic, h(s') = FF(s') + V_AP(s'), which is V_AP(s') once the task
    is done (FF = 0). It is DefaultFormulation(discount=1.0) with no task
    arrivals (p = 0): InterruptionAPPlanner's objective without its
    p * V_AP(s') step term.
    """
    def child_discount(self, terms: StepTerms) -> float:
        return 1.0

    def accumulated_cost(self, terms: StepTerms) -> float:
        return terms.parent_cost + terms.action_cost

    def estimated_future_cost(
        self, terms: StepTerms, child_discount: float, undiscounted_future_cost: float
    ) -> float:
        return undiscounted_future_cost
