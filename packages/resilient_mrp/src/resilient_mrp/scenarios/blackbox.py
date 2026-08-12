# Turns terrain features and robot specs into the two numbers the planner treats as opaque:
# edge hazard beta and robot-terrain compatibility phi. Values here are assumed, not measured;
# grounding them is what Verti-Bench is for.

from __future__ import annotations

from resilient_mrp.planning.core import RobotProfile

# Terrain feature vector: {base_risk, slope, roughness, deformability} normalized to [0,1]
TerrainFeatures = dict[str, float]

# slope is mean grade over [0deg, 45deg], roughness is surface irregularity, deformability is
# inverse soil cohesion, base_risk is the hazard before risk_scale. Shared by every scenario.
TERRAIN_FEATURES: dict[str, TerrainFeatures] = {
    "clear":      {"base_risk": 0.10, "slope": 0.04, "roughness": 0.05, "deformability": 0.0},
    "rocky":      {"base_risk": 0.30, "slope": 0.08, "roughness": 0.40, "deformability": 0.1},
    "steep":      {"base_risk": 0.40, "slope": 0.25, "roughness": 0.15, "deformability": 0.1},
    "deformable": {"base_risk": 0.35, "slope": 0.12, "roughness": 0.10, "deformability": 0.7},
}

RobotCapabilities = dict[str, float]   # capability name → value ∈ [0, 1]
RobotSpec = dict[str, RobotCapabilities | float]  # terrain_affinity dict + scalar capabilities
RobotSpecs = dict[str, RobotSpec]      # robot name → spec


def _clamp_to_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


class BlackBox:
    
    # Hazard weights per terrain feature, from UGV stability research. Slope drives rollover.
    _SLOPE_WEIGHT = 0.12
    _SLOPE_BASELINE = 0.08
    _ROUGHNESS_WEIGHT = 0.10
    _DEFORMABILITY_WEIGHT = 0.08

    # Which capability matters on each terrain, and how much it shifts compatibility.
    _TERRAIN_CAPABILITY_ADJUSTMENTS: dict[str, tuple[str, float]] = {
        "clear":      ("baseline",         0.0),
        "rocky":      ("ground_clearance", 0.12),
        "steep":      ("stability",        0.15),
        "deformable": ("traction",         0.10),
    }

    _DEFAULT_CAPABILITY = 0.5

    def __init__(
        self,
        robot_specs: RobotSpecs,
        terrain_features: dict[str, TerrainFeatures],
    ) -> None:
        self.robot_specs = robot_specs
        self.terrain_features = terrain_features

    # Compute edge hazard β from terrain features and risk scale
    def estimate_edge_hazard(
        self,
        terrain_type: str,
        risk_scale: float = 1.0,
    ) -> float:
        features = self.terrain_features.get(terrain_type, {})
        base_risk = features.get("base_risk", 0.0)
        slope = features.get("slope", 0.0)
        roughness = features.get("roughness", 0.0)
        deformability = features.get("deformability", 0.0)

        hazard = base_risk * risk_scale
        hazard += self._SLOPE_WEIGHT * max(0.0, slope - self._SLOPE_BASELINE)
        hazard += self._ROUGHNESS_WEIGHT * roughness
        hazard += self._DEFORMABILITY_WEIGHT * deformability

        return _clamp_to_unit_interval(hazard)

    # Compute robot compatibility φ: base affinity + capability adjustments
    def estimate_robot_compatibility(
        self,
        robot_name: str,
        terrain_type: str,
    ) -> float:
        robot = self.robot_specs.get(robot_name, {})
        terrain_affinity = robot.get("terrain_affinity", {})
        if not isinstance(terrain_affinity, dict):
            terrain_affinity = {}
        base_affinity = terrain_affinity.get(terrain_type, self._DEFAULT_CAPABILITY)

        compatibility_adjustment = 0.0
        if terrain_type in self._TERRAIN_CAPABILITY_ADJUSTMENTS:
            capability_attr, weight = self._TERRAIN_CAPABILITY_ADJUSTMENTS[terrain_type]

            capability_value = robot.get(capability_attr, self._DEFAULT_CAPABILITY)
            compatibility_adjustment = weight * (capability_value - self._DEFAULT_CAPABILITY)

        return _clamp_to_unit_interval(base_affinity + compatibility_adjustment)

    # Build φ profile for one robot: {terrain → φ}
    def build_robot_profile(self, robot_name: str) -> RobotProfile:
        robot = self.robot_specs.get(robot_name, {})
        terrain_affinity = robot.get("terrain_affinity", {})
        if not isinstance(terrain_affinity, dict):
            terrain_affinity = {}
        return {
            terrain: self.estimate_robot_compatibility(robot_name, terrain)
            for terrain in terrain_affinity.keys()
        }

    # Build φ profiles for all robots: {robot → {terrain → φ}}
    def build_all_profiles(self) -> dict[str, RobotProfile]:
        return {
            robot_name: self.build_robot_profile(robot_name)
            for robot_name in self.robot_specs.keys()
        }
