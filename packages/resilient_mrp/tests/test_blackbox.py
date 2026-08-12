from resilient_mrp.scenarios.blackbox import BlackBox


def test_blackbox_estimates_are_within_bounds() -> None:
    robot_specs = {
        "r1": {
            "terrain_affinity": {"clear": 0.9, "debris": 0.8},
            "stability": 0.7,
            "water_protection": 0.4,
            "debris_handling": 0.9,
            "clearance": 0.5,
        }
    }
    terrain_features = {
        "clear": {"base_risk": 0.1, "slope": 0.01, "roughness": 0.0, "water_depth": 0.0, "debris_density": 0.0},
        "debris": {"base_risk": 0.5, "slope": 0.1, "roughness": 0.4, "water_depth": 0.0, "debris_density": 0.7},
    }
    box = BlackBox(robot_specs, terrain_features)

    clear_hazard = box.estimate_edge_hazard("clear", risk_scale=1.0)
    debris_hazard = box.estimate_edge_hazard("debris", risk_scale=1.0)
    assert 0.0 <= clear_hazard <= 1.0
    assert 0.0 <= debris_hazard <= 1.0
    assert debris_hazard >= clear_hazard

    clear_phi = box.estimate_robot_compatibility("r1", "clear")
    debris_phi = box.estimate_robot_compatibility("r1", "debris")
    assert 0.0 <= clear_phi <= 1.0
    assert 0.0 <= debris_phi <= 1.0
    assert debris_phi <= clear_phi

    profile = box.build_all_profiles()
    assert "r1" in profile
    assert profile["r1"]["clear"] == clear_phi
    assert profile["r1"]["debris"] == debris_phi
