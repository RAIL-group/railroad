import json

from ai2thor.controller import Controller
from railroad.environment.procthor.resources import get_procthor_10k_dir


def load_scene(seed: int) -> dict:
    """
    Loads a raw ProcTHOR-10k scene dict by line index (mirrors
    ThorInterface._load_scene, bypassing ProcTHORScene/ThorInterface so we
    always get a live Controller regardless of whether a cache exists).
    """
    data_dir = get_procthor_10k_dir()
    with open(data_dir / "data.jsonl", "r") as f:
        line = f.readlines()[seed]
    return json.loads(line)


def positions_to_set(positions: list[dict], decimals: int = 3) -> set[tuple[float, float, float]]:
    """Converts GetReachablePositions output into a hashable set for diffing."""
    return {
        (round(p["x"], decimals), round(p["y"], decimals), round(p["z"], decimals))
        for p in positions
    }


def check_seed(seed: int, object_seed: int, resolution: float = 0.05) -> dict:
    """
    Compares GetReachablePositions before/after InitialRandomSpawn for a
    single ProcTHOR-10k house.
    """
    scene = load_scene(seed)
    with Controller(scene=scene, gridSize=resolution, width=300, height=300) as controller:
        before_event = controller.step(action="GetReachablePositions", raise_for_failure=True)
        before = positions_to_set(before_event.metadata["actionReturn"])

        event = controller.step(
            action="InitialRandomSpawn",
            randomSeed=object_seed,
            forceVisible=True,
            placeStationary=True,
            numPlacementAttempts=5,
            raise_for_failure=True,
        )

        after_event = controller.step(action="GetReachablePositions", raise_for_failure=True)
        after = positions_to_set(after_event.metadata["actionReturn"])

    removed = before - after
    added = after - before

    return {
        "seed": seed,
        "object_seed": object_seed,
        "num_before": len(before),
        "num_after": len(after),
        "num_removed": len(removed),
        "num_added": len(added),
        "changed": bool(removed or added),
    }


def main():
    seeds = [201]
    object_seed = 0
    resolution = 0.05

    results = []
    for seed in seeds:
        print(f"Checking seed {seed}...")
        try:
            result = check_seed(seed, object_seed, resolution)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        results.append(result)
        status = "CHANGED" if result["changed"] else "unchanged"
        print(
            f"  reachable positions {status}: "
            f"{result['num_before']} -> {result['num_after']} "
            f"(+{result['num_added']}/-{result['num_removed']})"
        )

    num_changed = sum(r["changed"] for r in results)
    print(
        f"\n{num_changed}/{len(results)} seeds had different reachable "
        "positions after InitialRandomSpawn."
    )


if __name__ == "__main__":
    main()
