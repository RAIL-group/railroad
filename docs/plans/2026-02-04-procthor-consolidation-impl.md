# ProcTHOR Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate ProcTHOR environment code into `railroad.environment.procthor` as an optional module.

**Architecture:** Two-stage design with `ProcTHORScene` (data provider) and `ProcTHOREnvironment` (SymbolicEnvironment subclass). All procthor package code moves into railroad with lazy imports and helpful error messages for missing dependencies.

**Tech Stack:** Python 3.11+, ai2thor, sentence-transformers, prior, shapely, networkx, numpy

---

### Task 1: Create procthor module skeleton with lazy imports

**Files:**
- Create: `packages/railroad/src/railroad/environment/procthor/__init__.py`

**Step 1: Create the __init__.py with lazy import guard**

```python
"""ProcTHOR environment for AI2-THOR/ProcTHOR simulation.

This module provides integration with ProcTHOR 3D indoor environments.
Requires optional dependencies: pip install railroad[procthor]
"""

__all__ = ["ProcTHORScene", "ProcTHOREnvironment"]

_INSTALL_MSG = (
    "ProcTHOR dependencies not installed. "
    "Install with: pip install railroad[procthor]"
)


def __getattr__(name: str):
    if name == "ProcTHORScene":
        try:
            from .scene import ProcTHORScene
            return ProcTHORScene
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    elif name == "ProcTHOREnvironment":
        try:
            from .environment import ProcTHOREnvironment
            return ProcTHOREnvironment
        except ImportError as e:
            raise ImportError(_INSTALL_MSG) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Step 2: Verify the module is importable (will fail gracefully)**

Run: `uv run python -c "from railroad.environment import procthor; print('module exists')"`
Expected: PASS (module exists, even if dependencies missing)

**Step 3: Commit**

```bash
git add packages/railroad/src/railroad/environment/procthor/__init__.py
git commit -m "feat(procthor): add module skeleton with lazy imports"
```

---

### Task 2: Move SceneGraph class

**Files:**
- Create: `packages/railroad/src/railroad/environment/procthor/scenegraph.py`
- Reference: `packages/procthor/src/procthor/scenegraph.py`

**Step 1: Write test for SceneGraph**

Create: `packages/railroad/tests/environment/procthor/test_scenegraph.py`

```python
import pytest
from railroad.environment.procthor.scenegraph import SceneGraph


def test_scenegraph_basic_operations():
    """Test SceneGraph node and edge operations."""
    sg = SceneGraph()

    # Add apartment node
    idx_apt = sg.add_node({
        'id': 'apartment0',
        'type': [1, 0, 0, 0],
        'position': [0, 0],
        'name': 'apartment'
    })

    # Add room
    idx_room = sg.add_node({
        'id': 'bedroom0',
        'type': [0, 1, 0, 0],
        'position': [0, 1],
        'name': 'bedroom'
    })
    sg.add_edge(idx_apt, idx_room)

    # Add container
    idx_container = sg.add_node({
        'id': 'bed0',
        'type': [0, 0, 1, 0],
        'position': [0, 1],
        'name': 'bed'
    })
    sg.add_edge(idx_room, idx_container)

    # Add object
    idx_obj = sg.add_node({
        'id': 'pillow0',
        'type': [0, 0, 0, 1],
        'position': [0, 1],
        'name': 'pillow'
    })
    sg.add_edge(idx_container, idx_obj)

    assert len(sg.nodes) == 4
    assert len(sg.edges) == 3
    assert set(sg.room_indices) == {idx_room}
    assert set(sg.container_indices) == {idx_container}
    assert set(sg.object_indices) == {idx_obj}


def test_scenegraph_adjacency():
    """Test get_adjacent_nodes_idx and get_parent_node_idx."""
    sg = SceneGraph()
    idx_apt = sg.add_node({'id': 'apt', 'type': [1, 0, 0, 0], 'position': [0, 0], 'name': 'apt'})
    idx_room = sg.add_node({'id': 'room', 'type': [0, 1, 0, 0], 'position': [1, 0], 'name': 'room'})
    idx_cnt = sg.add_node({'id': 'cnt', 'type': [0, 0, 1, 0], 'position': [2, 0], 'name': 'cnt'})
    sg.add_edge(idx_apt, idx_room)
    sg.add_edge(idx_room, idx_cnt)

    assert set(sg.get_adjacent_nodes_idx(idx_room)) == {idx_apt, idx_cnt}
    assert sg.get_parent_node_idx(idx_room) == idx_apt
    assert sg.get_parent_node_idx(idx_cnt) == idx_room
    assert sg.get_parent_node_idx(idx_apt) is None


def test_scenegraph_object_free_copy():
    """Test get_object_free_graph removes objects."""
    sg = SceneGraph()
    idx_apt = sg.add_node({'id': 'apt', 'type': [1, 0, 0, 0], 'position': [0, 0], 'name': 'apt'})
    idx_cnt = sg.add_node({'id': 'cnt', 'type': [0, 0, 1, 0], 'position': [1, 0], 'name': 'cnt'})
    idx_obj = sg.add_node({'id': 'obj', 'type': [0, 0, 0, 1], 'position': [1, 0], 'name': 'obj'})
    sg.add_edge(idx_apt, idx_cnt)
    sg.add_edge(idx_cnt, idx_obj)

    sg_free = sg.get_object_free_graph()
    assert len(sg_free.nodes) == 2
    assert len(sg_free.object_indices) == 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/railroad/tests/environment/procthor/test_scenegraph.py -v`
Expected: FAIL (module not found)

**Step 3: Copy and adapt SceneGraph**

Create `packages/railroad/src/railroad/environment/procthor/scenegraph.py`:

```python
"""Scene graph data structure for ProcTHOR environments."""

import copy
from typing import Any, Dict, List, Optional, Tuple


class SceneGraph:
    """Hierarchical scene graph for indoor environments.

    Node types (one-hot encoded):
    - [1,0,0,0]: apartment
    - [0,1,0,0]: room
    - [0,0,1,0]: container
    - [0,0,0,1]: object
    """

    def __init__(self) -> None:
        """Initialize an empty scene graph."""
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.edges: List[Tuple[int, int]] = []
        self.asset_id_to_node_idx_map: Dict[str, int] = {}

    def add_node(self, node_dict: Dict[str, Any], node_idx: Optional[int] = None) -> int:
        """Add a new node to the graph.

        Args:
            node_dict: Node attributes (must include 'type', optionally 'id', 'name', 'position')
            node_idx: Optional specific index. If None, uses next available.

        Returns:
            The index of the added node.
        """
        node_idx = len(self.nodes) if node_idx is None else node_idx
        while node_idx in self.nodes:
            node_idx += 1
        self.nodes[node_idx] = node_dict
        if 'id' in node_dict:
            self.asset_id_to_node_idx_map[node_dict['id']] = node_idx
        return node_idx

    def add_edge(self, src_idx: int, dst_idx: int) -> None:
        """Add an edge between two nodes."""
        if src_idx not in self.nodes or dst_idx not in self.nodes:
            raise ValueError('Invalid node indices')
        self.edges.append((src_idx, dst_idx))

    def delete_node(self, node_idx: int) -> None:
        """Delete a node and its edges from the graph."""
        if node_idx not in self.nodes:
            raise ValueError('Invalid node index')
        del self.nodes[node_idx]
        self.edges = [(src, dst) for src, dst in self.edges
                      if src != node_idx and dst != node_idx]

    def delete_edge(self, src_idx: int, dst_idx: int) -> None:
        """Delete an edge between two nodes."""
        if (src_idx, dst_idx) not in self.edges:
            raise ValueError('Invalid edge')
        self.edges.remove((src_idx, dst_idx))

    def get_node_indices_by_type(self, type_idx: int) -> List[int]:
        """Get indices of all nodes of a given type."""
        return [idx for idx, node in self.nodes.items()
                if node['type'][type_idx] == 1]

    def get_node_indices_by_id(self, node_id: str) -> List[int]:
        """Get indices of all nodes with a given id."""
        return [idx for idx, node in self.nodes.items()
                if node.get('id') == node_id]

    def get_node_indices_by_name(self, name: str) -> List[int]:
        """Get indices of all nodes with a given name."""
        return [idx for idx, node in self.nodes.items()
                if node.get('name') == name]

    def check_if_node_exists_by_id(self, node_id: str) -> bool:
        """Check if a node with a given id exists."""
        return any(node.get('id') == node_id for node in self.nodes.values())

    def get_object_free_graph(self) -> "SceneGraph":
        """Get a copy of the graph with object nodes removed."""
        graph = self.copy()
        obj_indices = graph.object_indices
        for _, dst in list(self.edges):
            if dst in obj_indices:
                graph.delete_node(dst)
        return graph

    def get_node_name_by_idx(self, node_idx: int) -> str:
        """Get name of a node by its index."""
        return self.nodes[node_idx]['name']

    def get_node_position_by_idx(self, node_idx: int) -> Any:
        """Get position of a node by its index."""
        return self.nodes[node_idx]['position']

    def get_node_idx_by_position(self, position: Tuple[float, float]) -> Optional[int]:
        """Get index of a node by its position."""
        for idx, node in self.nodes.items():
            if node['position'][0] == position[0] and node['position'][1] == position[1]:
                return idx
        return None

    def __len__(self) -> int:
        return len(self.nodes)

    def copy(self) -> "SceneGraph":
        """Create a deep copy of the scene graph."""
        graph_copy = SceneGraph()
        graph_copy.nodes = copy.deepcopy(self.nodes)
        graph_copy.edges = copy.deepcopy(self.edges)
        graph_copy.asset_id_to_node_idx_map = copy.deepcopy(self.asset_id_to_node_idx_map)
        return graph_copy

    @property
    def room_indices(self) -> List[int]:
        """Get indices of all room nodes."""
        return self.get_node_indices_by_type(1)

    @property
    def container_indices(self) -> List[int]:
        """Get indices of all container nodes."""
        return self.get_node_indices_by_type(2)

    @property
    def object_indices(self) -> List[int]:
        """Get indices of all object nodes."""
        return self.get_node_indices_by_type(3)

    def get_adjacent_nodes_idx(
        self,
        node_idx: int,
        filter_by_type: Optional[int] = None
    ) -> List[int]:
        """Get indices of all adjacent nodes, optionally filtered by type."""
        adj_nodes_idx = set()
        for src, dst in self.edges:
            if src == node_idx:
                if filter_by_type is None or self.nodes[dst]['type'][filter_by_type] == 1:
                    adj_nodes_idx.add(dst)
            elif dst == node_idx:
                if filter_by_type is None or self.nodes[src]['type'][filter_by_type] == 1:
                    adj_nodes_idx.add(src)
        return list(adj_nodes_idx)

    def get_parent_node_idx(self, node_idx: int) -> Optional[int]:
        """Get the index of the parent node (one level up in hierarchy)."""
        node_type = self.nodes[node_idx]['type'].index(1)
        if node_type == 0:
            return None  # Apartment has no parent
        parent_nodes_idx = self.get_adjacent_nodes_idx(node_idx, filter_by_type=node_type - 1)
        return parent_nodes_idx[0] if parent_nodes_idx else None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/railroad/tests/environment/procthor/test_scenegraph.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/railroad/src/railroad/environment/procthor/scenegraph.py
git add packages/railroad/tests/environment/procthor/test_scenegraph.py
git commit -m "feat(procthor): add SceneGraph class"
```

---

### Task 3: Move utility functions

**Files:**
- Create: `packages/railroad/src/railroad/environment/procthor/utils.py`
- Reference: `packages/procthor/src/procthor/utils.py`

**Step 1: Create utils.py with graph connectivity and helper functions**

```python
"""Utility functions for ProcTHOR environments."""

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np


def get_nearest_free_point(
    point: Dict[str, float],
    free_points: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """Find the nearest free point to a given position.

    Args:
        point: Dict with 'x' and 'z' keys
        free_points: List of (x, z) tuples

    Returns:
        The closest free point as (x, z) tuple
    """
    min_dist = float('inf')
    nearest = free_points[0]
    for fp in free_points:
        dist = (fp[0] - point['x'])**2 + (fp[1] - point['z'])**2
        if dist < min_dist:
            min_dist = dist
            nearest = fp
    return nearest


def has_edge(doors: List[Dict], room_0: str, room_1: str) -> bool:
    """Check if two rooms are connected by a door."""
    for door in doors:
        if ((door['room0'] == room_0 and door['room1'] == room_1) or
            (door['room1'] == room_0 and door['room0'] == room_1)):
            return True
    return False


def get_generic_name(name: str) -> str:
    """Extract generic name from asset ID (e.g., 'table|1|2' -> 'table')."""
    return name.split('|')[0].lower()


def get_room_id(name: str) -> int:
    """Extract room ID from asset ID (e.g., 'table|1|2' -> 1)."""
    return int(name.split('|')[1])


def get_cost(grid: np.ndarray, robot_pose: Tuple[int, int], end: Tuple[int, int]) -> float:
    """Compute path cost between two grid positions.

    Args:
        grid: Occupancy grid (0=free, 1=occupied)
        robot_pose: Start position (x, y)
        end: End position (x, y)

    Returns:
        Path cost (distance)
    """
    import gridmap

    occ_grid = np.copy(grid)
    occ_grid[int(robot_pose[0])][int(robot_pose[1])] = 0
    occ_grid[end[0], end[1]] = 0

    cost_grid = gridmap.planning.compute_cost_grid_from_position(
        occ_grid,
        start=[robot_pose[0], robot_pose[1]],
        use_soft_cost=True,
        only_return_cost_grid=True
    )
    return cost_grid[end[0], end[1]]


def get_cost_and_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int]
) -> Tuple[float, np.ndarray]:
    """Compute path cost and path between two grid positions.

    Args:
        grid: Occupancy grid
        start: Start position (x, y)
        end: End position (x, y)

    Returns:
        Tuple of (cost, path) where path is 2xN array
    """
    import gridmap

    occ_grid = np.copy(grid)
    occ_grid[int(start[0])][int(start[1])] = 0
    occ_grid[end[0], end[1]] = 0

    cost_grid, get_path = gridmap.planning.compute_cost_grid_from_position(
        occ_grid,
        start=[start[0], start[1]],
        use_soft_cost=True
    )
    cost = cost_grid[end[0], end[1]]
    _, path = get_path(target=[end[0], end[1]])
    return cost, path


def get_coordinates_at_time(path: np.ndarray, time: float) -> np.ndarray:
    """Get coordinates along a path at a given time (distance).

    Args:
        path: 2xN array of path coordinates
        time: Distance along path

    Returns:
        Coordinates at that time
    """
    diffs = np.diff(path, axis=1)
    segment_lengths = np.linalg.norm(diffs, axis=0)
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    idx = np.searchsorted(cumulative_lengths, time, side='left')
    idx = min(idx, path.shape[1] - 1)
    return path[:, idx]


def get_edges_for_connected_graph(
    grid: np.ndarray,
    graph: Dict[str, Any],
    pos: str = 'position'
) -> List[Tuple[int, int]]:
    """Find edges needed to make graph connected.

    Args:
        grid: Occupancy grid for cost computation
        graph: Dict with 'nodes', 'edge_index', 'cnt_node_idx' keys
        pos: Key for position in node dict

    Returns:
        List of edges to add
    """
    edges_to_add = []

    # Find room node indices (between apartment and first container)
    room_node_idx = list(range(1, graph['cnt_node_idx'][0]))

    # Extract room-only edges
    filtered_edges = [
        edge for edge in graph['edge_index']
        if edge[1] in room_node_idx and edge[0] != 0
    ]

    sorted_dc = _get_disconnected_components(room_node_idx, filtered_edges)

    while len(sorted_dc) > 1:
        comps = sorted_dc[0]
        merged_set = set()
        for s in sorted_dc[1:]:
            merged_set |= s

        min_cost = float('inf')
        min_edge = None

        for comp in comps:
            for target in merged_set:
                cost = get_cost(
                    grid,
                    graph['nodes'][comp][pos],
                    graph['nodes'][target][pos]
                )
                if cost < min_cost:
                    min_cost = cost
                    min_edge = (comp, target)

        if min_edge:
            edges_to_add.append(min_edge)
            filtered_edges.append(min_edge)
            sorted_dc = _get_disconnected_components(room_node_idx, filtered_edges)

    return edges_to_add


def _get_disconnected_components(
    node_indices: List[int],
    edges: List[Tuple[int, int]]
) -> List[set]:
    """Get disconnected components sorted by size."""
    G = nx.Graph()
    G.add_nodes_from(node_indices)
    G.add_edges_from(edges)
    components = list(nx.connected_components(G))
    return sorted(components, key=len)
```

**Step 2: Verify by importing**

Run: `uv run python -c "from railroad.environment.procthor.utils import get_generic_name; print(get_generic_name('table|1|2'))"`
Expected: `table`

**Step 3: Commit**

```bash
git add packages/railroad/src/railroad/environment/procthor/utils.py
git commit -m "feat(procthor): add utility functions"
```

---

### Task 4: Move resources.py

**Files:**
- Create: `packages/railroad/src/railroad/environment/procthor/resources.py`
- Reference: `packages/procthor/src/procthor/resources.py`

**Step 1: Copy resources.py with minimal changes**

```python
"""Resource management for ProcTHOR environments.

Handles downloading and caching of:
- ProcTHOR-10k dataset
- SentenceTransformer model
- AI2-THOR simulator binaries
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# Base dir for all resources (overridable by env):
DEFAULT_RESOURCES_BASE = Path(
    os.environ.get("PROCTHOR_RESOURCES_DIR", Path.cwd() / "resources")
)

DEFAULT_PROCTHOR_10K_SUBDIR = os.environ.get("PROCTHOR_DATA_SUBDIR", "procthor-10k")
DEFAULT_SBERT_SUBDIR = os.environ.get("PROCTHOR_SBERT_SUBDIR", "sentence_transformers")
DEFAULT_AI2THOR_SUBDIR = os.environ.get("PROCTHOR_AI2THOR_SUBDIR", "ai2thor")
DEFAULT_SBERT_MODEL_NAME = os.environ.get(
    "PROCTHOR_SBERT_MODEL_NAME", "bert-base-nli-stsb-mean-tokens"
)


def get_procthor_10k_dir(base_dir: Optional[Path] = None) -> Path:
    """Get the ProcTHOR-10k data directory."""
    if base_dir is None:
        base_dir = DEFAULT_RESOURCES_BASE
    data_dir = Path(base_dir) / DEFAULT_PROCTHOR_10K_SUBDIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def ensure_procthor_10k(
    base_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> Path:
    """Ensure ProcTHOR-10k data is downloaded."""
    import prior

    data_dir = get_procthor_10k_dir(base_dir)
    data_path = data_dir / "data.jsonl"
    marker_path = data_dir / "download_complete.marker"
    tmp_path = data_dir / "data.jsonl.tmp"

    if not force and marker_path.exists() and data_path.exists():
        return data_dir

    print("Ensuring ProcTHOR-10k Dataset Downloaded.")
    dataset = prior.load_dataset("procthor-10k")
    train_data = dataset["train"]

    with tmp_path.open("w") as f:
        for entry in train_data:
            json.dump(entry, f)
            f.write("\n")

    tmp_path.replace(data_path)
    marker_path.touch()
    return data_dir


def get_sbert_dir(base_dir: Optional[Path] = None) -> Path:
    """Get the SBERT model directory."""
    if base_dir is None:
        base_dir = DEFAULT_RESOURCES_BASE
    model_dir = Path(base_dir) / DEFAULT_SBERT_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def ensure_sbert_model(
    base_dir: Optional[Path] = None,
    *,
    model_name: str = DEFAULT_SBERT_MODEL_NAME,
    force: bool = False,
) -> Path:
    """Ensure SBERT model is downloaded."""
    model_dir = get_sbert_dir(base_dir)
    marker_path = model_dir / "download_complete.marker"
    safetensor_path = model_dir / "model.safetensors"

    if not force and marker_path.exists() and safetensor_path.exists():
        return model_dir

    print("Ensuring SentenceTransformer Model Downloaded.")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    model.save(str(model_dir))
    marker_path.touch()
    return model_dir


def get_ai2thor_marker_dir(base_dir: Optional[Path] = None) -> Path:
    """Get the AI2-THOR marker directory."""
    if base_dir is None:
        base_dir = DEFAULT_RESOURCES_BASE
    marker_dir = Path(base_dir) / DEFAULT_AI2THOR_SUBDIR
    marker_dir.mkdir(parents=True, exist_ok=True)
    return marker_dir


def ensure_ai2thor_simulator(
    base_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> Path:
    """Ensure AI2-THOR simulator is available."""
    marker_dir = get_ai2thor_marker_dir(base_dir)
    marker_path = marker_dir / "download_complete.marker"

    if not force and marker_path.exists():
        return marker_dir

    print("Ensuring AI2THOR Simulator Downloaded.")
    from ai2thor.controller import Controller
    controller = Controller()
    try:
        controller.stop()
    except Exception:
        pass

    marker_path.touch()
    return marker_dir


def ensure_all_resources(
    base_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> None:
    """Ensure all ProcTHOR resources are available."""
    ensure_procthor_10k(base_dir=base_dir, force=force)
    ensure_sbert_model(base_dir=base_dir, force=force)
    ensure_ai2thor_simulator(base_dir=base_dir, force=force)
```

**Step 2: Commit**

```bash
git add packages/railroad/src/railroad/environment/procthor/resources.py
git commit -m "feat(procthor): add resource management"
```

---

### Task 5: Move ThorInterface as thor_interface.py

**Files:**
- Create: `packages/railroad/src/railroad/environment/procthor/thor_interface.py`
- Reference: `packages/procthor/src/procthor/procthor.py`

**Step 1: Create thor_interface.py**

This is a larger file. Key changes:
- Update imports to use local modules
- Keep the same API

```python
"""AI2-THOR interface for ProcTHOR environments."""

import copy
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely import geometry

from .scenegraph import SceneGraph
from . import utils
from .resources import get_procthor_10k_dir

IGNORE_CONTAINERS = [
    'baseballbat', 'basketball', 'boots', 'desklamp', 'painting',
    'floorlamp', 'houseplant', 'roomdecor', 'showercurtain',
    'showerhead', 'television', 'vacuumcleaner', 'photo', 'plunger',
    'box'
]


class ThorInterface:
    """Interface to AI2-THOR/ProcTHOR simulator.

    Handles scene loading, occupancy grid generation, and scene graph construction.

    Args:
        seed: Random seed for scene selection
        resolution: Grid resolution in meters
        preprocess: Whether to filter containers
        use_cache: Whether to use cached data
    """

    def __init__(
        self,
        seed: int,
        resolution: float = 0.05,
        preprocess: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.seed = seed
        self.grid_resolution = resolution
        random.seed(seed)

        self.scene = self._load_scene()
        self.rooms = self.scene['rooms']
        self.agent = self.scene['metadata']['agent']

        self.containers = self.scene['objects']
        if preprocess:
            self._preprocess_containers()

        self.cached_data = self._load_cache() if use_cache else None
        if self.cached_data is None:
            from ai2thor.controller import Controller
            self.controller = Controller(
                scene=self.scene,
                gridSize=self.grid_resolution,
                width=480,
                height=480
            )
            self.cached_data = self._save_and_get_cache()
        else:
            print("-----------Using cached procthor data-----------")
            self.controller = None

        self.occupancy_grid = self._get_occupancy_grid()
        self.scene_graph = self._get_scene_graph()
        self.robot_pose = self._get_robot_pose()
        self.known_cost = self._get_known_costs()

    def _preprocess_containers(self) -> None:
        """Filter containers and their children."""
        container_types = {c['id'].split('|')[0].lower() for c in self.containers}

        for container in self.containers:
            if 'children' in container:
                container['children'] = [
                    child for child in container['children']
                    if child['id'].split('|')[0].lower() not in container_types
                ]

        self.containers = [
            c for c in self.containers
            if c['id'].split('|')[0].lower() not in IGNORE_CONTAINERS
        ]

    def _load_scene(self) -> Dict[str, Any]:
        """Load scene from ProcTHOR-10k dataset."""
        data_dir = get_procthor_10k_dir()
        with open(data_dir / 'data.jsonl', 'r') as f:
            json_list = list(f)
        return json.loads(json_list[self.seed])

    def _save_and_get_cache(self, path: str = './resources/procthor-10k/cache') -> Dict:
        """Cache expensive computations."""
        cache = {
            'reachable_positions': self._get_reachable_positions_from_controller(),
            'image_ortho': self._get_top_down_image_from_controller(orthographic=True),
            'image_persp': self._get_top_down_image_from_controller(orthographic=False)
        }
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f'scene_{self.seed}.pkl', 'wb') as f:
            pickle.dump(cache, f)
        return cache

    def _load_cache(self, path: str = './resources/procthor-10k/cache') -> Optional[Dict]:
        """Load cached scene data."""
        cache_file = Path(path) / f'scene_{self.seed}.pkl'
        if not cache_file.exists():
            return None
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    def _get_reachable_positions_from_controller(self) -> List[Dict[str, float]]:
        """Get reachable positions from controller."""
        assert self.controller is not None
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def get_reachable_positions(self) -> List[Dict[str, float]]:
        """Get reachable positions (from cache or controller)."""
        if self.cached_data is not None:
            return self.cached_data['reachable_positions']
        return self._get_reachable_positions_from_controller()

    def _set_grid_offset(self, min_x: float, min_y: float) -> None:
        """Set grid coordinate offset."""
        self.grid_offset = np.array([min_x, min_y])

    def scale_to_grid(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        x = round((point[0] - self.grid_offset[0]) / self.grid_resolution)
        y = round((point[1] - self.grid_offset[1]) / self.grid_resolution)
        return x, y

    def _get_robot_pose(self) -> Tuple[int, int]:
        """Get initial robot pose in grid coordinates."""
        position = self.agent['position']
        position = np.array([position['x'], position['z']])
        return self.scale_to_grid(position)

    def _get_occupancy_grid(self) -> np.ndarray:
        """Build occupancy grid from reachable positions."""
        rps = self.get_reachable_positions()
        xs = [rp["x"] for rp in rps]
        zs = [rp["z"] for rp in rps]

        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        x_offset = min_x - self.grid_resolution if min_x < 0 else 0
        z_offset = min_z - self.grid_resolution if min_z < 0 else 0
        self._set_grid_offset(x_offset, z_offset)

        points = list(zip(xs, zs))
        self.g2p_map = {self.scale_to_grid(p): rps[i] for i, p in enumerate(points)}

        height, width = self.scale_to_grid([max_x, max_z])
        occupancy_grid = np.ones((height + 2, width + 2), dtype=int)
        for pos in self.g2p_map.keys():
            occupancy_grid[pos] = 0

        # Update container positions to nearest free grid cell
        for container in self.containers:
            position = container['position']
            if position is not None:
                nearest_fp = utils.get_nearest_free_point(position, points)
                scaled = self.scale_to_grid((nearest_fp[0], nearest_fp[1]))
                container['position'] = scaled
                container['id'] = container['id'].lower()

                if 'children' in container:
                    for child in container['children']:
                        child['position'] = container['position']
                        child['id'] = child['id'].lower()

        for room in self.rooms:
            floor = [(rp["x"], rp["z"]) for rp in room["floorPolygon"]]
            room_poly = geometry.Polygon(floor)
            point = room_poly.centroid
            nearest_fp = utils.get_nearest_free_point({'x': point.x, 'z': point.y}, points)
            room['position'] = self.scale_to_grid((nearest_fp[0], nearest_fp[1]))

        return occupancy_grid

    def _get_scene_graph(self) -> SceneGraph:
        """Build scene graph from scene data."""
        graph = SceneGraph()

        # Add apartment node
        apt_idx = graph.add_node({
            'id': 'Apartment|0',
            'name': 'apartment',
            'position': (0, 0),
            'type': [1, 0, 0, 0]
        })

        # Add room nodes
        for room in self.rooms:
            room_idx = graph.add_node({
                'id': room['id'],
                'name': room['roomType'].lower(),
                'position': room['position'],
                'type': [0, 1, 0, 0]
            })
            graph.add_edge(apt_idx, room_idx)

        # Add edges between connected rooms
        room_indices = graph.room_indices
        for i, src_idx in enumerate(room_indices):
            for dst_idx in room_indices[i + 1:]:
                src_node = graph.nodes[src_idx]
                dst_node = graph.nodes[dst_idx]
                if utils.has_edge(self.scene['doors'], src_node['id'], dst_node['id']):
                    graph.add_edge(src_idx, dst_idx)

        # Add container nodes
        for container in self.containers:
            room_id = utils.get_room_id(container['id'])
            room_node_idx = next(
                idx for idx, node in graph.nodes.items()
                if node['type'][1] == 1 and utils.get_room_id(node['id']) == room_id
            )
            cnt_idx = graph.add_node({
                'id': container['id'],
                'name': utils.get_generic_name(container['id']),
                'position': container['position'],
                'type': [0, 0, 1, 0]
            })
            graph.add_edge(room_node_idx, cnt_idx)

        # Add object nodes
        for container in self.containers:
            children = container.get('children', [])
            if children:
                cnt_idx = graph.asset_id_to_node_idx_map[container['id']]
                for obj in children:
                    obj_idx = graph.add_node({
                        'id': obj['id'],
                        'name': utils.get_generic_name(obj['id']),
                        'position': obj['position'],
                        'type': [0, 0, 0, 1]
                    })
                    graph.add_edge(cnt_idx, obj_idx)

        # Ensure connectivity
        graph.edges.extend(utils.get_edges_for_connected_graph(
            self.occupancy_grid,
            {
                'nodes': graph.nodes,
                'edge_index': graph.edges,
                'cnt_node_idx': graph.container_indices,
                'obj_node_idx': graph.object_indices,
                'idx_map': graph.asset_id_to_node_idx_map
            },
            pos='position'
        ))

        return graph

    def _get_known_costs(self) -> Dict[str, Dict[str, float]]:
        """Pre-compute costs between all containers."""
        known_cost: Dict[str, Dict[str, float]] = {'initial_robot_pose': {}}
        init_r = [self.robot_pose[0], self.robot_pose[1]]
        cnt_ids = ['initial_robot_pose'] + [c['id'] for c in self.containers]
        cnt_positions = [init_r] + [c['position'] for c in self.containers]

        for i, cnt1_id in enumerate(cnt_ids):
            known_cost[cnt1_id] = {}
            for j, cnt2_id in enumerate(cnt_ids):
                if cnt2_id not in known_cost:
                    known_cost[cnt2_id] = {}
                if cnt1_id == cnt2_id:
                    known_cost[cnt1_id][cnt2_id] = 0.0
                    continue
                cost = utils.get_cost(
                    self.occupancy_grid,
                    cnt_positions[i],
                    cnt_positions[j]
                )
                known_cost[cnt1_id][cnt2_id] = round(cost, 4)
                known_cost[cnt2_id][cnt1_id] = round(cost, 4)

        return known_cost

    def _get_top_down_image_from_controller(self, orthographic: bool = True) -> np.ndarray:
        """Get top-down image from controller."""
        assert self.controller is not None
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = orthographic
        pose["farClippingPlane"] = 50
        if orthographic:
            pose["orthographicSize"] = 0.5 * max_bound
        else:
            del pose["orthographicSize"]

        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        image = event.third_party_camera_frames[-1]
        return image[::-1, ...]

    def get_top_down_image(self, orthographic: bool = True) -> np.ndarray:
        """Get top-down image (from cache or controller)."""
        if self.cached_data is not None:
            key = 'image_ortho' if orthographic else 'image_persp'
            return self.cached_data[key]
        return self._get_top_down_image_from_controller(orthographic)

    def get_target_objs_info(self, num_objects: int = 1) -> Dict | List[Dict]:
        """Get info about target objects for search tasks."""
        object_name_to_idxs: Dict[str, List[int]] = {}
        for idx in self.scene_graph.object_indices:
            name = self.scene_graph.get_node_name_by_idx(idx)
            object_name_to_idxs.setdefault(name, []).append(idx)

        num_objects = min(num_objects, len(object_name_to_idxs))
        target_names = random.sample(list(object_name_to_idxs.keys()), num_objects)

        result = []
        for name in target_names:
            idxs = object_name_to_idxs[name]
            container_idxs = [self.scene_graph.get_parent_node_idx(idx) for idx in idxs]
            result.append({
                'name': name,
                'idxs': idxs,
                'type': self.scene_graph.nodes[idxs[0]]['type'],
                'container_idxs': container_idxs
            })

        return result[0] if num_objects == 1 else result
```

**Step 2: Commit**

```bash
git add packages/railroad/src/railroad/environment/procthor/thor_interface.py
git commit -m "feat(procthor): add ThorInterface"
```

---

### Task 6: Create ProcTHORScene class

**Files:**
- Create: `packages/railroad/src/railroad/environment/procthor/scene.py`

**Step 1: Create scene.py**

```python
"""ProcTHOR scene data provider."""

from typing import Any, Callable, Dict, Set, Tuple

import numpy as np

from railroad._bindings import Action

from .thor_interface import ThorInterface
from . import utils as procthor_utils


class ProcTHORScene:
    """Data provider for ProcTHOR environments.

    Loads a ProcTHOR scene and extracts all information needed for planning:
    - Location names and coordinates
    - Object names and their ground truth locations
    - Move cost function
    - Path interpolation for interrupted moves

    Example:
        scene = ProcTHORScene(seed=4001)
        print(scene.locations)  # All container locations
        print(scene.objects)    # All objects in scene

        # Create move operator with scene's cost function
        move_op = operators.construct_move_operator_blocking(scene.get_move_cost_fn())
    """

    def __init__(self, seed: int, resolution: float = 0.05) -> None:
        """Initialize ProcTHOR scene.

        Args:
            seed: Random seed for scene selection (0-9999 for ProcTHOR-10k)
            resolution: Grid resolution in meters
        """
        self._thor = ThorInterface(seed=seed, resolution=resolution)

        # Build location registry
        self._locations = self._build_locations()

        # Extract all objects
        self._objects = self._extract_objects()

        # Build ground truth object locations (location -> objects)
        self._object_locations = self._build_object_locations()

    @property
    def grid(self) -> np.ndarray:
        """Occupancy grid (0=free, 1=occupied)."""
        return self._thor.occupancy_grid

    @property
    def scene_graph(self):
        """Scene graph for visualization/debugging."""
        return self._thor.scene_graph

    @property
    def locations(self) -> Dict[str, Tuple[int, int]]:
        """Location names mapped to grid coordinates."""
        return self._locations

    @property
    def objects(self) -> Set[str]:
        """All objects in the scene."""
        return self._objects

    @property
    def object_locations(self) -> Dict[str, Set[str]]:
        """Ground truth: location name -> set of object names at that location."""
        return self._object_locations

    def _build_locations(self) -> Dict[str, Tuple[int, int]]:
        """Extract locations from scene graph containers."""
        locations = {'start_loc': self._thor.robot_pose}

        for idx in self._thor.scene_graph.container_indices:
            node = self._thor.scene_graph.nodes[idx]
            name = f"{node['name']}_{idx}"
            locations[name] = tuple(node['position'])

        return locations

    def _extract_objects(self) -> Set[str]:
        """Extract all object names from scene graph."""
        objects = set()
        for idx in self._thor.scene_graph.object_indices:
            name = self._thor.scene_graph.get_node_name_by_idx(idx)
            objects.add(f"{name}_{idx}")
        return objects

    def _build_object_locations(self) -> Dict[str, Set[str]]:
        """Build mapping of location -> objects at that location."""
        result: Dict[str, Set[str]] = {}

        for container_idx in self._thor.scene_graph.container_indices:
            container_node = self._thor.scene_graph.nodes[container_idx]
            location_name = f"{container_node['name']}_{container_idx}"

            object_idxs = self._thor.scene_graph.get_adjacent_nodes_idx(
                container_idx, filter_by_type=3
            )

            for obj_idx in object_idxs:
                obj_name = f"{self._thor.scene_graph.get_node_name_by_idx(obj_idx)}_{obj_idx}"
                result.setdefault(location_name, set()).add(obj_name)

        return result

    def get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        """Get move cost function for operator construction.

        Returns:
            Function (robot, loc_from, loc_to) -> cost
        """
        # Build lookup from location name to container ID
        loc_to_id: Dict[str, str] = {'start_loc': 'initial_robot_pose'}
        for container in self._thor.containers:
            idx = self._thor.scene_graph.asset_id_to_node_idx_map[container['id']]
            name = f"{procthor_utils.get_generic_name(container['id'])}_{idx}"
            loc_to_id[name] = container['id']

        def move_cost_fn(robot: str, loc_from: str, loc_to: str) -> float:
            id_from = loc_to_id.get(loc_from)
            id_to = loc_to_id.get(loc_to)

            if id_from and id_to and id_from in self._thor.known_cost:
                return self._thor.known_cost[id_from].get(id_to, float('inf'))

            # Fall back to grid-based cost
            coord_from = self._locations.get(loc_from)
            coord_to = self._locations.get(loc_to)
            if coord_from is None or coord_to is None:
                return float('inf')

            return procthor_utils.get_cost(self._thor.occupancy_grid, coord_from, coord_to)

        return move_cost_fn

    def get_intermediate_coordinates(
        self,
        action: Action,
        elapsed_time: float
    ) -> Tuple[int, int]:
        """Get coordinates along move path at elapsed time.

        Args:
            action: A move action with name "move robot loc_from loc_to"
            elapsed_time: Time elapsed since move started

        Returns:
            Grid coordinates at that time
        """
        parts = action.name.split()
        if len(parts) < 4 or parts[0] != 'move':
            raise ValueError(f"Expected move action, got: {action.name}")

        loc_from = parts[2]
        loc_to = parts[3]

        coord_from = self._locations.get(loc_from)
        coord_to = self._locations.get(loc_to)

        if coord_from is None or coord_to is None:
            raise ValueError(f"Unknown location: {loc_from} or {loc_to}")

        # Get full path
        _, path = procthor_utils.get_cost_and_path(
            self._thor.occupancy_grid,
            coord_from,
            coord_to
        )

        # Interpolate along path
        coords = procthor_utils.get_coordinates_at_time(path, elapsed_time)
        return int(coords[0]), int(coords[1])

    def get_top_down_image(self, orthographic: bool = True) -> np.ndarray:
        """Get top-down view image of the scene."""
        return self._thor.get_top_down_image(orthographic=orthographic)
```

**Step 2: Commit**

```bash
git add packages/railroad/src/railroad/environment/procthor/scene.py
git commit -m "feat(procthor): add ProcTHORScene data provider"
```

---

### Task 7: Create ProcTHOREnvironment class

**Files:**
- Create: `packages/railroad/src/railroad/environment/procthor/environment.py`

**Step 1: Create environment.py**

```python
"""ProcTHOR environment for PDDL planning."""

from typing import Dict, List, Set

from railroad._bindings import State
from railroad.core import Operator
from railroad.environment import SymbolicEnvironment

from .scene import ProcTHORScene


class ProcTHOREnvironment(SymbolicEnvironment):
    """Symbolic environment backed by a ProcTHOR scene.

    Subclass of SymbolicEnvironment that provides:
    - Direct access to the ProcTHOR scene
    - Optional validation that objects/locations exist
    - Convenience methods for scene data

    Example:
        scene = ProcTHORScene(seed=4001)
        move_op = operators.construct_move_operator_blocking(scene.get_move_cost_fn())

        env = ProcTHOREnvironment(
            scene=scene,
            state=State(0.0, {F("at robot1 start_loc"), F("free robot1")}),
            objects_by_type={
                "robot": {"robot1"},
                "location": set(scene.locations.keys()),
                "object": {"teddybear_6"},
            },
            operators=[move_op, search_op, pick_op, place_op],
        )
    """

    def __init__(
        self,
        scene: ProcTHORScene,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        validate: bool = True,
    ) -> None:
        """Initialize ProcTHOR environment.

        Args:
            scene: ProcTHORScene data provider
            state: Initial planning state
            objects_by_type: Objects organized by type
            operators: Planning operators
            validate: Whether to validate objects/locations exist in scene
        """
        self.scene = scene

        if validate:
            self._validate(objects_by_type)

        super().__init__(
            state=state,
            objects_by_type=objects_by_type,
            operators=operators,
            true_object_locations=scene.object_locations,
        )

    def _validate(self, objects_by_type: Dict[str, Set[str]]) -> None:
        """Validate that objects and locations exist in scene."""
        # Validate locations
        if "location" in objects_by_type:
            scene_locations = set(self.scene.locations.keys())
            for loc in objects_by_type["location"]:
                if loc not in scene_locations:
                    # Allow robot_loc intermediate locations
                    if not loc.endswith("_loc"):
                        raise ValueError(
                            f"Location '{loc}' not found in scene. "
                            f"Available: {sorted(scene_locations)[:5]}..."
                        )

        # Validate objects
        if "object" in objects_by_type:
            scene_objects = self.scene.objects
            for obj in objects_by_type["object"]:
                if obj not in scene_objects:
                    raise ValueError(
                        f"Object '{obj}' not found in scene. "
                        f"Available: {sorted(scene_objects)[:5]}..."
                    )
```

**Step 2: Commit**

```bash
git add packages/railroad/src/railroad/environment/procthor/environment.py
git commit -m "feat(procthor): add ProcTHOREnvironment"
```

---

### Task 8: Add optional dependency to pyproject.toml

**Files:**
- Modify: `packages/railroad/pyproject.toml`

**Step 1: Add procthor optional dependency**

Add after the existing `all` optional dependency:

```toml
procthor = [
    "ai2thor>=5.0.0",
    "sentence-transformers>=5.1.2",
    "prior>=1.0.3",
    "shapely",
    "networkx",
    "scikit-image",
]
```

And update `all` to include it:

```toml
all = [
    "railroad[test,bench,procthor]",
]
```

**Step 2: Verify dependencies**

Run: `uv run python -c "from railroad.environment.procthor import ProcTHORScene; print('success')"`
Expected: PASS (or helpful error if dependencies missing)

**Step 3: Commit**

```bash
git add packages/railroad/pyproject.toml
git commit -m "feat(procthor): add optional dependency group"
```

---

### Task 9: Add procthor-search example

**Files:**
- Create: `packages/railroad/src/railroad/examples/procthor_search.py`
- Modify: `packages/railroad/src/railroad/examples/__init__.py`

**Step 1: Create the example**

```python
"""ProcTHOR multi-robot search example.

Demonstrates using ProcTHOR environment with MCTS planning
for multi-robot object search and retrieval.
"""

from pathlib import Path

from railroad import operators
from railroad._bindings import ff_heuristic
from railroad.core import Fluent as F, State, get_action_by_name
from railroad.dashboard import PlannerDashboard
from railroad.planner import MCTSPlanner


def main() -> None:
    """Run ProcTHOR multi-robot search example."""
    # Lazy import to avoid loading heavy dependencies at startup
    try:
        from railroad.environment.procthor import ProcTHORScene, ProcTHOREnvironment
    except ImportError as e:
        print(f"Error: {e}")
        print("\nInstall ProcTHOR dependencies with: pip install railroad[procthor]")
        return

    # Configuration
    seed = 4001
    robot_names = ["robot1", "robot2"]
    target_objects = ["teddybear_6", "pencil_17"]
    target_location = "garbagecan_5"

    print(f"Loading ProcTHOR scene (seed={seed})...")
    scene = ProcTHORScene(seed=seed)

    # Build operators
    move_cost_fn = scene.get_move_cost_fn()
    search_time_fn = lambda r, l, o: 15.0 if r == "robot1" else 10.0
    pick_time_fn = lambda r, l, o: 15.0 if r == "robot1" else 10.0
    place_time_fn = lambda r, l, o: 15.0 if r == "robot1" else 10.0

    # Create probability function based on ground truth
    def object_find_prob(robot: str, location: str, obj: str) -> float:
        for loc, objs in scene.object_locations.items():
            if obj in objs:
                return 0.8 if loc == location else 0.1
        return 0.1

    move_op = operators.construct_move_operator_blocking(move_cost_fn)
    search_op = operators.construct_search_operator(object_find_prob, search_time_fn)
    pick_op = operators.construct_pick_operator_blocking(pick_time_fn)
    place_op = operators.construct_place_operator_blocking(place_time_fn)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Initial state
    initial_fluents = {
        F("revealed start_loc"),
        F("at robot1 start_loc"), F("free robot1"),
        F("at robot2 start_loc"), F("free robot2"),
    }
    initial_state = State(0.0, initial_fluents)

    # Goal: place both objects at target location
    goal = F(f"at {target_objects[0]} {target_location}") & F(f"at {target_objects[1]} {target_location}")

    # Create environment
    env = ProcTHOREnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": set(robot_names),
            "location": set(scene.locations.keys()),
            "object": set(target_objects),
        },
        operators=[no_op, pick_op, place_op, move_op, search_op],
    )

    print(f"Planning to place {target_objects} at {target_location}...")
    print(f"Scene has {len(scene.locations)} locations and {len(scene.objects)} objects")

    # Planning loop
    actions_taken = []
    max_iterations = 60

    h_value = ff_heuristic(env.state, goal, env.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        dashboard.update(state=env.state)

        for iteration in range(max_iterations):
            if goal.evaluate(env.state.fluents):
                dashboard.console.print("[green]Goal reached![/green]")
                break

            all_actions = env.get_actions()
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(
                env.state, goal,
                max_iterations=10000,
                c=300,
                max_depth=20,
                heuristic_multiplier=2
            )

            if action_name == 'NONE':
                dashboard.console.print("No more actions available.")
                break

            action = get_action_by_name(all_actions, action_name)
            env.act(action)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic(env.state, goal, env.get_actions())
            relevant_fluents = {
                f for f in env.state.fluents
                if any(kw in f.name for kw in ["at", "holding", "found", "searched"])
            }
            dashboard.update(
                state=env.state,
                relevant_fluents=relevant_fluents,
                tree_trace=tree_trace,
                step_index=iteration,
                last_action_name=action_name,
                heuristic_value=h_value,
            )

        dashboard.print_history(env.state, actions_taken)

    print(f"\nTotal actions: {len(actions_taken)}")
    print(f"Final time: {env.state.time:.1f}")


if __name__ == "__main__":
    main()
```

**Step 2: Register example in __init__.py**

Add to EXAMPLES dict:

```python
"procthor-search": {
    "main": _lazy_import("procthor_search"),
    "description": "Multi-robot search in ProcTHOR 3D environment (requires railroad[procthor])",
},
```

**Step 3: Verify example is registered**

Run: `uv run railroad example`
Expected: Should list `procthor-search` among available examples

**Step 4: Commit**

```bash
git add packages/railroad/src/railroad/examples/procthor_search.py
git add packages/railroad/src/railroad/examples/__init__.py
git commit -m "feat(procthor): add procthor-search example"
```

---

### Task 10: Add plotting module (optional)

**Files:**
- Create: `packages/railroad/src/railroad/environment/procthor/plotting.py`

**Step 1: Create plotting.py**

```python
"""Plotting utilities for ProcTHOR environments."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
    from skimage.morphology import erosion
    HAS_PLOTTING_DEPS = True
except ImportError:
    HAS_PLOTTING_DEPS = False

COLLISION_VAL = 1
FREE_VAL = 0
FOOT_PRINT = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


def make_plotting_grid(grid_map: np.ndarray) -> np.ndarray:
    """Convert occupancy grid to RGB plotting grid."""
    if not HAS_PLOTTING_DEPS:
        raise ImportError("Plotting requires scikit-image: pip install scikit-image")

    grid = np.ones([grid_map.shape[0], grid_map.shape[1], 3]) * 0.75
    collision = grid_map >= 0.5
    thinned = erosion(collision, footprint=FOOT_PRINT)
    boundary = np.logical_xor(collision, thinned)
    free = np.logical_and(grid_map < 0.5, grid_map >= FREE_VAL)

    grid[:, :, :][free] = 1
    grid[:, :, 0][boundary] = 0
    grid[:, :, 1][boundary] = 0
    grid[:, :, 2][boundary] = 0

    return grid


def plot_graph_on_grid(ax: Any, grid: np.ndarray, graph: Any) -> None:
    """Plot scene graph on occupancy grid."""
    plotting_grid = make_plotting_grid(grid.T)
    ax.imshow(plotting_grid)

    room_node_idx = graph.room_indices
    rc_idx = room_node_idx + graph.container_indices

    filtered_edges = [
        edge for edge in graph.edges
        if edge[1] in rc_idx and edge[0] != 0
    ]

    for (start, end) in filtered_edges:
        p1 = graph.nodes[start]['position']
        p2 = graph.nodes[end]['position']
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'c', linestyle="--", linewidth=0.3)

    for room in rc_idx:
        room_pos = graph.nodes[room]['position']
        room_name = graph.nodes[room]['name']
        ax.text(room_pos[0], room_pos[1], room_name, color='brown', size=6, rotation=40)


def plot_graph(
    ax: Any,
    nodes: Dict[int, Dict],
    edges: List[Tuple[int, int]],
    highlight_node: Optional[int] = None
) -> None:
    """Plot scene graph with nodes and edges."""
    if not HAS_PLOTTING_DEPS:
        raise ImportError("Plotting requires networkx")

    node_type_to_color = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
    G = nx.Graph()
    node_colors = []

    for k, v in nodes.items():
        G.add_node(k, label=f"{k}: {v['name']}")
        color = node_type_to_color.get(v['type'].index(1), 'violet')
        if k == highlight_node:
            color = 'cyan'
        node_colors.append(color)

    G.add_edges_from(edges)
    node_labels = nx.get_node_attributes(G, 'label')
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, ax,
        with_labels=True,
        labels=node_labels,
        node_color=node_colors,
        node_size=20,
        font_size=4,
        edge_color='black',
        width=0.5
    )
    ax.axis('off')
```

**Step 2: Commit**

```bash
git add packages/railroad/src/railroad/environment/procthor/plotting.py
git commit -m "feat(procthor): add plotting utilities"
```

---

### Task 11: Migrate ThorInterface test

**Files:**
- Create: `packages/railroad/tests/environment/procthor/test_thor_interface.py`

**Step 1: Create test file**

```python
"""Tests for ThorInterface."""

import pytest
import numpy as np


@pytest.fixture
def thor_interface():
    """Create ThorInterface for testing."""
    pytest.importorskip("ai2thor")
    from railroad.environment.procthor.thor_interface import ThorInterface
    return ThorInterface(seed=0, resolution=0.05)


@pytest.mark.timeout(30)
def test_thor_interface_initialization(thor_interface):
    """Test ThorInterface initializes correctly."""
    assert len(thor_interface.scene_graph.nodes) > 0
    assert len(thor_interface.scene_graph.edges) > 0
    assert thor_interface.occupancy_grid.size > 0

    # Check grid values
    unique_vals = np.unique(thor_interface.occupancy_grid)
    assert 1 in unique_vals and 0 in unique_vals


@pytest.mark.timeout(30)
def test_thor_interface_robot_pose(thor_interface):
    """Test robot pose is extracted."""
    pose = thor_interface.robot_pose
    assert isinstance(pose, tuple)
    assert len(pose) == 2


@pytest.mark.timeout(30)
def test_thor_interface_known_costs(thor_interface):
    """Test known costs are computed."""
    assert 'initial_robot_pose' in thor_interface.known_cost
    # Check symmetric
    for id1, costs in thor_interface.known_cost.items():
        for id2, cost in costs.items():
            if id1 != id2:
                assert thor_interface.known_cost[id2][id1] == cost


@pytest.mark.timeout(30)
def test_thor_interface_target_objects(thor_interface):
    """Test target object info extraction."""
    info = thor_interface.get_target_objs_info(num_objects=1)
    assert 'name' in info
    assert 'idxs' in info
    assert 'type' in info
    assert 'container_idxs' in info
```

**Step 2: Commit**

```bash
git add packages/railroad/tests/environment/procthor/test_thor_interface.py
git commit -m "test(procthor): add ThorInterface tests"
```

---

### Task 12: Create test __init__.py files

**Files:**
- Create: `packages/railroad/tests/environment/__init__.py`
- Create: `packages/railroad/tests/environment/procthor/__init__.py`

**Step 1: Create empty __init__.py files**

```python
# packages/railroad/tests/environment/__init__.py
# packages/railroad/tests/environment/procthor/__init__.py
```

Both files are empty (just for Python package structure).

**Step 2: Commit**

```bash
touch packages/railroad/tests/environment/__init__.py
touch packages/railroad/tests/environment/procthor/__init__.py
git add packages/railroad/tests/environment/__init__.py
git add packages/railroad/tests/environment/procthor/__init__.py
git commit -m "chore: add test package init files"
```

---

### Task 13: Add ProcTHOREnvironment integration test

**Files:**
- Create: `packages/railroad/tests/environment/procthor/test_environment.py`

**Step 1: Create test file**

```python
"""Tests for ProcTHOREnvironment."""

import pytest


@pytest.fixture
def scene():
    """Create ProcTHORScene for testing."""
    pytest.importorskip("ai2thor")
    from railroad.environment.procthor import ProcTHORScene
    return ProcTHORScene(seed=4001)


@pytest.mark.timeout(30)
def test_scene_locations(scene):
    """Test scene exposes locations."""
    assert 'start_loc' in scene.locations
    assert len(scene.locations) > 1


@pytest.mark.timeout(30)
def test_scene_objects(scene):
    """Test scene exposes objects."""
    assert len(scene.objects) > 0
    # Objects should be formatted as name_idx
    for obj in scene.objects:
        assert '_' in obj


@pytest.mark.timeout(30)
def test_scene_object_locations(scene):
    """Test scene provides ground truth object locations."""
    assert len(scene.object_locations) > 0
    for loc, objs in scene.object_locations.items():
        assert loc in scene.locations
        for obj in objs:
            assert obj in scene.objects


@pytest.mark.timeout(30)
def test_scene_move_cost_fn(scene):
    """Test move cost function."""
    cost_fn = scene.get_move_cost_fn()

    # Cost to self is 0
    cost = cost_fn("robot1", "start_loc", "start_loc")
    assert cost == 0.0

    # Cost to other location is positive
    other_loc = next(iter(scene.locations.keys() - {"start_loc"}))
    cost = cost_fn("robot1", "start_loc", other_loc)
    assert cost > 0


@pytest.mark.timeout(30)
def test_environment_creation(scene):
    """Test ProcTHOREnvironment can be created."""
    from railroad.environment.procthor import ProcTHOREnvironment
    from railroad.core import Fluent as F
    from railroad._bindings import State
    from railroad import operators

    move_op = operators.construct_move_operator_blocking(scene.get_move_cost_fn())

    initial_state = State(0.0, {
        F("at robot1 start_loc"),
        F("free robot1"),
        F("revealed start_loc"),
    })

    # Pick first available object
    target_obj = next(iter(scene.objects))

    env = ProcTHOREnvironment(
        scene=scene,
        state=initial_state,
        objects_by_type={
            "robot": {"robot1"},
            "location": set(scene.locations.keys()),
            "object": {target_obj},
        },
        operators=[move_op],
    )

    assert env.scene is scene
    assert len(env.get_actions()) > 0


@pytest.mark.timeout(30)
def test_environment_validation_invalid_location(scene):
    """Test validation catches invalid locations."""
    from railroad.environment.procthor import ProcTHOREnvironment
    from railroad.core import Fluent as F
    from railroad._bindings import State

    initial_state = State(0.0, {F("at robot1 start_loc")})

    with pytest.raises(ValueError, match="not found in scene"):
        ProcTHOREnvironment(
            scene=scene,
            state=initial_state,
            objects_by_type={
                "robot": {"robot1"},
                "location": {"nonexistent_location"},
                "object": set(),
            },
            operators=[],
            validate=True,
        )


@pytest.mark.timeout(30)
def test_environment_validation_invalid_object(scene):
    """Test validation catches invalid objects."""
    from railroad.environment.procthor import ProcTHOREnvironment
    from railroad.core import Fluent as F
    from railroad._bindings import State

    initial_state = State(0.0, {F("at robot1 start_loc")})

    with pytest.raises(ValueError, match="not found in scene"):
        ProcTHOREnvironment(
            scene=scene,
            state=initial_state,
            objects_by_type={
                "robot": {"robot1"},
                "location": set(scene.locations.keys()),
                "object": {"nonexistent_object"},
            },
            operators=[],
            validate=True,
        )
```

**Step 2: Run tests**

Run: `uv run pytest packages/railroad/tests/environment/procthor/ -v --timeout=60`
Expected: PASS (if procthor deps installed) or SKIP

**Step 3: Commit**

```bash
git add packages/railroad/tests/environment/procthor/test_environment.py
git commit -m "test(procthor): add ProcTHOREnvironment integration tests"
```

---

### Task 14: Update root pyproject.toml

**Files:**
- Modify: `pyproject.toml` (root)

**Step 1: Remove standalone procthor dependency if present**

Check if `procthor` is listed as a direct dependency. If the monorepo structure changes, ensure the root pyproject.toml knows about the new optional dependency path.

The railroad package now has `[procthor]` optional dependency, so users can install via:
- `uv pip install railroad[procthor]` or
- Adding `railroad[procthor]` to dependencies

**Step 2: Verify installation**

Run: `uv run railroad example`
Expected: Lists `procthor-search`

**Step 3: Commit if changes made**

```bash
git add pyproject.toml
git commit -m "chore: update root dependencies for procthor consolidation"
```

---

### Task 15: Final verification and cleanup

**Step 1: Run all procthor tests**

Run: `uv run pytest packages/railroad/tests/environment/procthor/ -v`
Expected: PASS

**Step 2: Run example (if deps installed)**

Run: `uv run railroad example procthor-search`
Expected: Runs planning demo

**Step 3: Verify lazy import error message**

In a fresh environment without procthor deps:
Run: `python -c "from railroad.environment.procthor import ProcTHORScene"`
Expected: `ImportError: ProcTHOR dependencies not installed. Install with: pip install railroad[procthor]`

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(procthor): complete consolidation into railroad.environment.procthor"
```

---

## Summary

This plan creates:
1. `railroad.environment.procthor` module with lazy imports
2. `ProcTHORScene` - data provider class
3. `ProcTHOREnvironment` - SymbolicEnvironment subclass
4. Supporting modules: `scenegraph.py`, `utils.py`, `resources.py`, `thor_interface.py`, `plotting.py`
5. `procthor-search` example accessible via CLI
6. Tests for core functionality

Total: 15 tasks with TDD approach for testable components.
