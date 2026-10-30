"""
When tasks arrive: the per-action arrival probability, and the average time
between arrivals, set from the task-cost quantiles of a data-generation run.
"""
import json
import math
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from railroad.environment.procthor.resources import get_procthor_10k_dir


class RandomVariableType(Enum):
    """
    Enumeration for the valid types of a random variable. Ensures that the 
    user can only pass in a valid random variable type for get_interruption_prob function.
    """
    DISCRETE = 1
    CONTINUOUS = 2


def get_task_arrival_prob(
    rv_type: RandomVariableType,
    arrival_prob: float = -1,
    time_between_arrivals: float = -1,
    action_time: float = -1,
) -> float:
    """
    Helper function that returns the probability of a task arriving after the execution
    of an action. Supports both per-action (treating the random variable as discrete) and
    per-time-unit (treating the random variable as continuous) probabilities. In the later case,
    the arrival_prob argument is treated as the average time between task arrivals (tau).
    """
    # validation checks
    assert action_time >= 0
    if rv_type == RandomVariableType.DISCRETE:
        assert 0 <= arrival_prob <= 1
    if rv_type == RandomVariableType.CONTINUOUS:
        assert time_between_arrivals > 0

    if rv_type == RandomVariableType.DISCRETE:
        return arrival_prob
    return 1 - math.exp(-action_time / time_between_arrivals)


def calibrate_beta_parameter(prob: float, a_t: float) -> float:
    """
    Helper function for computing the value of the tau parameter for the
    CDF of the exponential distribution such the provided time to complete
    an action will have the specified probability.
    Reference: Equation for the CDF of the exponential distribution - 
    P(t <= X) = 1 - e^(-X/tau)
    Returns the computed tau parameter when valid inputs provided
    (Prob: [0, 1) and a_t >= 0). Otherwise returns -1 on invalid inputs.
    """
    if prob < 0 or prob >= 1 or a_t < 0:
        return -1
    if prob == 0:
        return float('inf')
    return -a_t / math.log(1 - prob)


def task_cost_quantiles_path(run_key: str | None) -> Path:
    """
    Location of the task-cost quantiles of a data-generation run, written by
    scripts/generate_task_cost_distribution.py --write-quantiles. run_key is the
    run's remapped_scenes/<hash>/ name (constants.REMAPPED_SCENES_HASH), which
    also names the run's other run_metadata files; None -> an unkeyed file.
    """
    name = "task_cost_quantiles.json" if run_key is None else f"task_cost_quantiles_{run_key}.json"
    return get_procthor_10k_dir() / "run_metadata" / name


@lru_cache(maxsize=None)
def load_task_cost_quantiles(path: Path) -> dict[str, Any]:
    """
    Loads a task-cost quantiles file: "costs"[i] is the "percentiles"[i]-th
    percentile of the pooled task-completion costs, alongside the provenance
    of the data they were computed from.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"no task-cost quantiles at {path}; generate them from the run's task costs with "
            "`uv run python scripts/generate_task_cost_distribution.py --write-quantiles`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_task_cost_quantile(quantiles: dict[str, Any], percentile: float) -> float:
    """
    Helper function that returns the task-completion cost at the given
    percentile ([0, 100]), linearly interpolating between the stored ones.
    """
    percentiles, costs = quantiles["percentiles"], quantiles["costs"]
    if not percentiles[0] <= percentile <= percentiles[-1]:
        raise ValueError(
            f"percentile {percentile} is outside the stored range "
            f"[{percentiles[0]}, {percentiles[-1]}]"
        )
    for idx in range(1, len(percentiles)):
        if percentile <= percentiles[idx]:
            lo, hi = percentiles[idx - 1], percentiles[idx]
            weight = (percentile - lo) / (hi - lo)
            return costs[idx - 1] + weight * (costs[idx] - costs[idx - 1])
    return costs[-1]


def get_expected_time_next_arrival(exceedance_fraction: float, run_key: str | None) -> float:
    """
    Helper function that returns the average time between task arrivals (tau)
    for which the given fraction of the run's training tasks take longer to
    complete than the median time to the next arrival, i.e. the median arrival
    time is the (1 - exceedance_fraction) quantile of the task-completion costs.
    A fraction of 0 means no interruptions (tau = inf) and needs no quantiles.
    """
    if not 0 <= exceedance_fraction < 1:
        raise ValueError(f"exceedance fraction must be in [0, 1), got {exceedance_fraction}")
    if exceedance_fraction == 0:
        return math.inf
    quantiles = load_task_cost_quantiles(task_cost_quantiles_path(run_key))
    a_t = get_task_cost_quantile(quantiles, 100 * (1 - exceedance_fraction))
    if a_t <= 0:
        raise ValueError(
            f"the {100 * (1 - exceedance_fraction):g}th percentile task cost is {a_t}, which gives "
            "no valid arrival rate; regenerate the quantiles with --drop-zero to exclude "
            "already-satisfied tasks"
        )
    return calibrate_beta_parameter(0.5, a_t)
