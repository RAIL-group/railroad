"""Execute the feature-example benchmarks once per registered case.

These benchmarks exist to demonstrate how the newer planning features are
defined (extra_cost, conditional effects, the PDDL converter), so this test
keeps the exemplars working end-to-end.
"""

import pytest

from railroad.bench.benchmarks import feature_examples
from railroad.bench.registry import BenchmarkCase

_BENCHMARKS = [
    feature_examples.bench_extra_cost_route_choice,
    feature_examples.bench_conditional_effects_briefcase,
    feature_examples.bench_pddl_converter_features,
]


def _all_cases():
    for bench in _BENCHMARKS:
        for case_idx, params in enumerate(bench.cases):
            case_id = f"{bench.name}[{case_idx}]"
            yield pytest.param(bench, case_idx, params, id=case_id)


@pytest.mark.parametrize("bench, case_idx, params", _all_cases())
def test_feature_example_case_succeeds(bench, case_idx, params):
    case = BenchmarkCase(
        benchmark_name=bench.name, case_idx=case_idx, repeat_idx=0, params=params
    )
    result = bench.fn(case)
    assert result["success"], f"{bench.name}{params} failed: {result}"
    assert result["actions_count"] == len(result["actions"])
    assert result["plan_cost"] > 0
