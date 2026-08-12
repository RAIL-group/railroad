# Small-scale sample graph scenario.
# 8 nodes, 2 heterogeneous robots, 2 goal sites.
# Uses 4-type terrain taxonomy (clear, rocky, steep, deformable) based on research
# into wheeled robot stability and tipping failure modes.
#
# Graph topology:
#   start -> n1 (clear), n2 (rocky), n3 (steep), n4 (deformable), n5 (rocky)
#   n1 -> g1, g2
#   n2 -> g1, n5
#   n3 -> g1
#   n4 -> g1, g2
#   n5 -> g2
#
# Robots:
#   r1 -- high stability (steep terrain), good ground clearance (rocky terrain)
#   r2 -- high traction (deformable terrain), good ground clearance (rocky terrain)

import numpy as np

from resilient_mrp.scenarios.blackbox import TERRAIN_FEATURES, BlackBox, RobotSpecs
from resilient_mrp.planning.core import ResilientGraph, RobotProfile

ROBOT_SPECS: RobotSpecs = {
    "r1": {
        "terrain_affinity": {
            "clear":      0.95,
            "rocky":      0.88,
            "steep":      0.92,
            "deformable": 0.30,
        },
        "stability":        0.75,   # good at low CG for steep terrain
        "ground_clearance": 0.70,   # moderate clearance for rocky terrain
        "traction":         0.50,   # neutral on deformable terrain
        "baseline":         0.5,
    },
    "r2": {
        "terrain_affinity": {
            "clear":      0.95,
            "rocky":      0.90,
            "steep":      0.35,
            "deformable": 0.55,
        },
        "stability":        0.40,   # higher CG, less suited to steep
        "ground_clearance": 0.75,   # good ground clearance for rocky
        "traction":         0.80,   # wide tracks, good on deformable
        "baseline":         0.5,
    },
}

# Terrain feature vectors. These represent measurable properties:
# - slope: mean grade angle [0, 1] mapping to [0°, 45°]
# - roughness: surface irregularity (roughness index from point cloud) [0, 1]
# - deformability: inverse of soil cohesion; soft soil → high value [0, 1]
# - base_risk: baseline hazard before risk_scale applied
#
# References: arXiv 2507.12731, MDPI terrain characterization papers


BLACKBOX = BlackBox(ROBOT_SPECS, TERRAIN_FEATURES)
ROBOT_PROFILES: dict[str, RobotProfile] = BLACKBOX.build_all_profiles()

GOAL_SITES: list[str] = ["g1", "g2"]

LOCATION_COORDS: dict[str, np.ndarray] = {
    "start": np.array([ 0.0,  0.0]),
    "n1":    np.array([-3.0,  2.0]),
    "n2":    np.array([ 0.0,  3.5]),
    "n3":    np.array([ 3.0,  2.0]),
    "n4":    np.array([ 3.0, -2.0]),
    "n5":    np.array([-3.0, -2.0]),
    "g1":    np.array([ 0.0,  6.5]),
    "g2":    np.array([ 0.0, -6.5]),
}


# risk_scale multiplies all base hazard severities, allowing controlled
# variation in traversal failure probability across experiments.
def create_graph(risk_scale: float = 1.0) -> ResilientGraph:
    g = ResilientGraph()

    # Edge cost is the Euclidean distance between nodes (consistent with the sctp graphs).
    def edge(fr: str, to: str, terrain: str) -> None:
        cost = float(np.linalg.norm(LOCATION_COORDS[fr] - LOCATION_COORDS[to]))
        hazard = BLACKBOX.estimate_edge_hazard(terrain, risk_scale=risk_scale)
        g.add_edge(fr, to, cost=cost, terrain_type=terrain,
                   hazard_severity=hazard, bidirectional=True,
                   from_coord=tuple(LOCATION_COORDS[fr]), to_coord=tuple(LOCATION_COORDS[to]))

    edge("start", "n1", "clear")
    edge("start", "n2", "steep")
    edge("start", "n3", "rocky")
    edge("start", "n4", "deformable")
    edge("start", "n5", "deformable")

    # Routes to g1 — four distinct approaches
    edge("n1", "g1", "clear")
    edge("n2", "g1", "steep")
    edge("n3", "g1", "rocky")
    edge("n4", "g1", "deformable")

    # Routes to g2 — three approaches plus cross-link
    edge("n1", "g2", "clear")
    edge("n4", "g2", "deformable")
    edge("n5", "g2", "deformable")
    edge("n2", "n5", "rocky")

    return g
