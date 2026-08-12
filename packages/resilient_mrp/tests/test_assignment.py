# What the two baselines should decide when handed a graph whose right answer is known by hand.
# Both are checked through .assign(), which is what the benchmark calls.

import math

import pytest

from resilient_mrp.planning.core import ResilientGraph
from resilient_mrp.planning.baselines import CautiousPolicy, OptimisticPolicy


# Two goals off to one side, so one robot walking both is short in total but long for that robot.
# Splitting them finishes at 10, one robot doing both finishes at 19.
@pytest.fixture
def side_by_side() -> ResilientGraph:
    g = ResilientGraph()
    g.add_edge("start", "gA", cost=10.0, terrain_type="clear", hazard_severity=0.0)
    g.add_edge("start", "gB", cost=10.0, terrain_type="clear", hazard_severity=0.0)
    g.add_edge("gA", "gB", cost=9.0, terrain_type="clear", hazard_severity=0.0)
    return g


# Each robot is good on one terrain and bad on the other, and the two routes cross over, so the
# safest split is not the one that just balances the risk between them.
@pytest.fixture
def crossed_terrain() -> ResilientGraph:
    g = ResilientGraph()
    g.add_edge("p1", "gA", cost=1.0, terrain_type="rock", hazard_severity=1.0)
    g.add_edge("p1", "gB", cost=1.0, terrain_type="sand", hazard_severity=1.0)
    g.add_edge("p2", "gA", cost=1.0, terrain_type="sand", hazard_severity=1.0)
    g.add_edge("p2", "gB", cost=1.0, terrain_type="rock", hazard_severity=1.0)
    return g


# hazard 1.0 means survival is exactly phi, so these set -log p to 2.0, 1.9, 1.0, 1.9
CROSSED_PROFILES = {
    "r1": {"rock": math.exp(-2.0), "sand": math.exp(-1.9)},
    "r2": {"rock": math.exp(-1.0), "sand": math.exp(-1.9)},
}

FLAT_PROFILES = {"r1": {"clear": 1.0}, "r2": {"clear": 1.0}}


@pytest.mark.parametrize("policy_cls", [OptimisticPolicy, CautiousPolicy])
def test_idle_robot_is_used(side_by_side, policy_cls):
    policy = policy_cls(side_by_side, ["gA", "gB"], FLAT_PROFILES)
    queues = policy.assign({"r1": "start", "r2": "start"}, set())
    assert sorted(len(q) for q in queues.values()) == [1, 1], (
        f"{policy_cls.__name__} left a robot idle: {queues}")


def test_cautious_takes_the_likeliest_split(crossed_terrain):
    policy = CautiousPolicy(crossed_terrain, ["gA", "gB"], CROSSED_PROFILES)
    queues = policy.assign({"r1": "p1", "r2": "p2"}, set())
    # r1 -> gA and r2 -> gB survives at exp(-3.0); the other split only at exp(-3.8)
    assert queues == {"r1": ["gA"], "r2": ["gB"]}, f"less likely split chosen: {queues}"


def test_optimistic_takes_the_quickest_split(side_by_side):
    policy = OptimisticPolicy(side_by_side, ["gA", "gB"], FLAT_PROFILES)
    queues = policy.assign({"r1": "start", "r2": "start"}, set())
    assert queues == {"r1": ["gA"], "r2": ["gB"]} or queues == {"r1": ["gB"], "r2": ["gA"]}
