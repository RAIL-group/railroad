# ProcTHOR Consolidation into Railroad

**Date:** 2026-02-04
**Status:** Approved

## Overview

Consolidate the ProcTHOR environment code from multiple packages (`procthor/`, `environments/`) into a single optional module at `railroad.environment.procthor`. This enables users to install ProcTHOR support via `pip install railroad[procthor]` and run examples with `uv run railroad example procthor-search`.

## Goals

1. Single installation path for ProcTHOR support (`railroad[procthor]`)
2. Clean 2-stage API: `ProcTHORScene` (data provider) + `ProcTHOREnvironment` (planning)
3. Remove redundant wrapper layers
4. Migrate core tests, drop obsolete ones

## Architecture

### Two-Stage Design

**Stage 1: ProcTHORScene** - Loads simulator, extracts domain data

```python
class ProcTHORScene:
    def __init__(self, seed: int, resolution: float = 0.05):
        self.thor = ThorInterface(seed=seed, resolution=resolution)

    locations: dict[str, tuple[float, float, float]]
    objects: set[str]
    object_locations: dict[str, set[str]]  # location -> objects at that location
    grid: ...
    scene_graph: ...

    def get_move_cost_fn(self) -> Callable[[str, str, str], float]: ...
    def get_intermediate_coordinates(self, action: Action, elapsed_time: float) -> tuple: ...
```

**Stage 2: ProcTHOREnvironment** - Planning environment with scene access and validation

```python
class ProcTHOREnvironment(SymbolicEnvironment):
    def __init__(
        self,
        scene: ProcTHORScene,
        state: State,
        objects_by_type: dict[str, set[str]],
        operators: list[Operator],
        validate: bool = True,
    ):
        self.scene = scene

        if validate:
            if "location" in objects_by_type:
                self._validate_locations_exist(objects_by_type["location"])
            if "object" in objects_by_type:
                self._validate_objects_exist(objects_by_type["object"])

        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            true_object_locations=scene.object_locations,
        )
```

### Usage Example

```python
from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
from railroad.core import Fluent as F, State
from railroad import operators

# Stage 1: Load scene
scene = ProcTHORScene(seed=4001)

# Stage 2: Create environment
move_op = operators.construct_move_operator_blocking(scene.get_move_cost_fn())
search_op = operators.construct_search_operator(...)
pick_op = operators.construct_pick_operator_blocking(...)
place_op = operators.construct_place_operator_blocking(...)

env = ProcTHOREnvironment(
    scene=scene,
    state=State(0.0, {F("at robot1 start_loc"), F("free robot1"), ...}),
    objects_by_type={
        "robot": {"robot1", "robot2"},
        "location": set(scene.locations.keys()),
        "object": {"teddybear_6", "pencil_17"},
    },
    operators=[search_op, pick_op, place_op, move_op],
)

# Access scene data through environment
env.scene.grid
env.scene.get_intermediate_coordinates(action, elapsed_time)
```

## Package Structure

### New Files in `railroad/environment/procthor/`

```
packages/railroad/src/railroad/environment/procthor/
├── __init__.py          # Lazy import guard, exports ProcTHORScene, ProcTHOREnvironment
├── scene.py             # ProcTHORScene class
├── environment.py       # ProcTHOREnvironment class
├── thor_interface.py    # From procthor/procthor.py
├── scenegraph.py        # From procthor/scenegraph.py
├── resources.py         # Resource downloading
├── utils.py             # Pathfinding, cost helpers
└── plotting.py          # Visualization
```

### Lazy Import Guard

```python
# railroad/environment/procthor/__init__.py
def __getattr__(name):
    if name in ("ProcTHORScene", "ProcTHOREnvironment"):
        try:
            if name == "ProcTHORScene":
                from .scene import ProcTHORScene
                return ProcTHORScene
            else:
                from .environment import ProcTHOREnvironment
                return ProcTHOREnvironment
        except ImportError as e:
            raise ImportError(
                "ProcTHOR dependencies not installed. "
                "Install with: pip install railroad[procthor]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

## Dependencies

### pyproject.toml Changes

```toml
[project.optional-dependencies]
procthor = [
    "ai2thor>=5.0.0",
    "sentence-transformers>=5.1.2",
    "prior>=1.0.3",
    "shapely",
    "networkx",
]
```

## Example CLI

Add `procthor_search.py` to the existing example system, accessible via:

```bash
uv run railroad example procthor-search
```

The example is adapted from `scripts/procthor_run.py`.

## Test Migration

### Test Location

```
packages/railroad/tests/environment/procthor/
├── test_scenegraph.py      # SceneGraph node/edge operations
├── test_thor_interface.py  # ThorInterface initialization, scene loading
└── test_environment.py     # ProcTHOREnvironment setup, validation
```

### Migration Plan

| Source | Destination | Action |
|--------|-------------|--------|
| `procthor/tests/test_procthor_scenegraph.py` | `test_scenegraph.py` | Migrate |
| `procthor/tests/test_procthor_simulator.py` | `test_thor_interface.py` | Migrate |
| `environments/tests/test_procthor_environment.py` | `test_environment.py` | Adapt core tests |

### Tests to Drop

- `add_object_at_location`, `remove_object_from_location` tests (methods removed)
- `get_objects_at_location` tests (fluents track state now)
- Robot skill execution/tracking tests (not in new design)

## Cleanup

### Packages to Remove

- `packages/procthor/` - fully absorbed into railroad
- `packages/environments/src/environments/procthor.py` - replaced by new module

### What Stays Unchanged

- `packages/environments/` package (other environments, benchmarks remain)
- `railroad.environment.symbolic` and core environment classes
- `railroad.environment.Environment` base class

## Implementation Order

1. Create `railroad/environment/procthor/` module structure
2. Move and adapt `thor_interface.py`, `scenegraph.py`, `resources.py`, `utils.py`, `plotting.py`
3. Implement `ProcTHORScene` class
4. Implement `ProcTHOREnvironment` class
5. Add `procthor` optional dependency to pyproject.toml
6. Create `procthor_search.py` example
7. Migrate tests
8. Remove old `procthor` package and `environments/procthor.py`
9. Update any remaining imports/references
