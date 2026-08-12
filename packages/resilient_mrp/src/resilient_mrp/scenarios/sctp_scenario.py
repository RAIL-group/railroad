# Generated graph topologies via Delaunay triangulation, in random and island shapes at any size.
# This is where a raw SCTP graph becomes the ResilientGraph everything else plans over.

import random

import numpy as np

from sctp.graph import generate_random_graph, generate_island_graph
from sctp.param import (MAX_EDGE_LENGTH, MIN_EDGE_LENGTH,
                        MAX_ISLAND_DISTANCE, MIN_ISLAND_DISTANCE)

from resilient_mrp.scenarios.blackbox import TERRAIN_FEATURES, BlackBox, RobotSpecs
from resilient_mrp.planning.core import ResilientGraph, RobotProfile
from resilient_mrp.scenarios.graph_analysis import (
    GraphStatistics, NodePlacement, TerrainInference, label_for, percentile_cuts)


# ── Robot Specifications ─────────────────────────────────────────────────────

ROBOT_SPECS: RobotSpecs = {
    # r1 — steep specialist: strong on steep/rocky, weak on soft (deformable) ground.
    "r1": {
        "terrain_affinity": {
            "clear":      0.95,
            "rocky":      0.88,
            "steep":      0.92,
            "deformable": 0.30,
        },
        "stability":        0.75,
        "ground_clearance": 0.70,
        "traction":         0.50,
        "baseline":         0.5,
    },
    # r2 — deformable specialist: strong on soft ground, weak on steep slopes.
    "r2": {
        "terrain_affinity": {
            "clear":      0.95,
            "rocky":      0.90,
            "steep":      0.35,
            "deformable": 0.92,
        },
        "stability":        0.40,
        "ground_clearance": 0.75,
        "traction":         0.80,
        "baseline":         0.5,
    },
    # r3 — rocky specialist: high-clearance rover, strong on rough/rocky, weak on soft soil.
    "r3": {
        "terrain_affinity": {
            "clear":      0.95,
            "rocky":      0.92,
            "steep":      0.55,
            "deformable": 0.45,
        },
        "stability":        0.55,
        "ground_clearance": 0.88,
        "traction":         0.55,
        "baseline":         0.5,
    },
    # r4 — balanced generalist: moderate on all terrain, master of none.
    "r4": {
        "terrain_affinity": {
            "clear":      0.95,
            "rocky":      0.70,
            "steep":      0.65,
            "deformable": 0.65,
        },
        "stability":        0.60,
        "ground_clearance": 0.60,
        "traction":         0.60,
        "baseline":         0.5,
    },
}



BLACKBOX = BlackBox(ROBOT_SPECS, TERRAIN_FEATURES)
ROBOT_PROFILES: dict[str, RobotProfile] = BLACKBOX.build_all_profiles()

GOAL_SITES: list[str] = ["g1", "g2"]

# island block_prob is near-binary, so anything at or below this counts as an open crossing
_ISLAND_CLEAR_BLOCK_PROB = 0.1


# ── SCTP Graph Conversion ────────────────────────────────────────────────────

class SCTPGraphConverter:
    # Consumes a raw SCTP graph and emits a self-contained ResilientGraph. The only place that
    # knows about SCTP's POIs. tradeoff_weight makes short edges tend to be riskier, so a quick
    # direct route trades against a safe long one; at 0 risk is unrelated to length.
    def __init__(self, risk_scale: float = 1.0, tradeoff_weight: float = 0.5):
        self.risk_scale = risk_scale
        self.tradeoff_weight = tradeoff_weight
        self.terrain_inference: TerrainInference | None = None
        self.goal_vertices: list = []

    @staticmethod
    def nav_edges(sctp_graph) -> list:
        # Collapse SCTP's vertex-POI-vertex chains into direct edges, carrying each POI's
        # block_prob onto its edge. Returns [(v1, v2, block_prob)].
        by_id = {v.id: v for v in sctp_graph.vertices}
        out = []
        for poi in sctp_graph.pois:
            nb = poi.neighbors
            if len(nb) >= 2 and nb[0] in by_id and nb[1] in by_id:
                out.append((by_id[nb[0]], by_id[nb[1]], poi.block_prob))
        return out

    @staticmethod
    def _island_terrain(nav: list, stats: GraphStatistics) -> dict:
        # Island block_prob is near-binary, so stand in a deterministic per-edge value and run it
        # through the same cuts as random graphs, leaving shape as the only difference.
        def key(v1, v2):
            return (min(v1.id, v2.id), max(v1.id, v2.id))

        def value(k):
            return (hash(k) % 10_000) / 10_000.0

        blocked = [(v1, v2) for v1, v2, bp in nav if bp > _ISLAND_CLEAR_BLOCK_PROB]
        p_clear, p_rocky = percentile_cuts(value(key(v1, v2)) for v1, v2 in blocked)

        terr: dict = {}
        for v1, v2, bp in nav:
            k = key(v1, v2)
            if bp <= _ISLAND_CLEAR_BLOCK_PROB:
                terr[k] = "clear"
            else:
                terr[k] = label_for(value(k), GraphStatistics.dist(v1, v2),
                                    p_clear, p_rocky, stats.median_edge_length)
        return terr

    def convert_random_graph_multi_goal(
        self, sctp_graph, start_vertex, n_goals: int = 2, min_hops: int = 4, max_hops: int = 5
    ) -> ResilientGraph:
        nav = self.nav_edges(sctp_graph)
        stats = GraphStatistics(nav)
        self.terrain_inference = TerrainInference(nav, stats, self.tradeoff_weight)
        adj = self._build_edge_adjacency(nav)
        self.goal_vertices = self._select_goal_vertices(
            adj, sctp_graph.vertices, start_vertex, n_goals, min_hops, max_hops
        )
        return self._build_graph(nav, start_vertex, island_terrain=None)

    def convert_island_graph_multi_goal(
        self, sctp_graph, islands, start_vertex,
        n_goals: int = 2, min_hops: int = 4, max_hops: int = 5,
    ) -> ResilientGraph:
        nav = self.nav_edges(sctp_graph)
        stats = GraphStatistics(nav)
        self.terrain_inference = TerrainInference(nav, stats, self.tradeoff_weight)
        adj = self._build_edge_adjacency(nav)
        island_vertices = [v for island in islands for v in island.vertices]
        self.goal_vertices = self._select_goal_vertices(
            adj, island_vertices, start_vertex, n_goals, min_hops, max_hops
        )
        return self._build_graph(nav, start_vertex, island_terrain=self._island_terrain(nav, stats))

    def _build_graph(self, nav: list, start_vertex, island_terrain: dict | None) -> ResilientGraph:
        # Rename start and goal nodes, assign terrain and hazard, and store coords so nothing
        # downstream ever needs the SCTP graph again.
        assert self.terrain_inference is not None
        graph = ResilientGraph()
        goal_ids = {v.id: f"g{i+1}" for i, v in enumerate(self.goal_vertices)}

        def name(v):
            return "start" if v.id == start_vertex.id else goal_ids.get(v.id, str(v.id))

        for v1, v2, bp in nav:
            dist = GraphStatistics.dist(v1, v2)
            if island_terrain is not None:
                terrain = island_terrain[(min(v1.id, v2.id), max(v1.id, v2.id))]
            else:
                terrain = self.terrain_inference.terrain_for(bp, dist)
            hazard = BLACKBOX.estimate_edge_hazard(terrain, risk_scale=self.risk_scale)
            graph.add_edge(name(v1), name(v2), cost=dist, terrain_type=terrain,
                           hazard_severity=hazard, bidirectional=True,
                           from_coord=tuple(v1.coord), to_coord=tuple(v2.coord))
        return graph

    # ── Goal Placement Helpers ────────────────────────────────────────────────

    def _build_edge_adjacency(self, nav: list) -> dict:
        # POI-free vertex connectivity over the collapsed nav edges.
        adj: dict = {}
        for v1, v2, _ in nav:
            adj.setdefault(v1.id, set()).add(v2.id)
            adj.setdefault(v2.id, set()).add(v1.id)
        return adj

    def _bfs_hops(self, adj: dict, start_id: int) -> dict:
        from collections import deque
        dist: dict = {start_id: 0}
        queue = deque([start_id])
        while queue:
            nid = queue.popleft()
            # sorted, because sctp hands out vertex ids from a process-global counter and raw set
            # order would make the BFS order depend on how many graphs were built before this one
            for nb in sorted(adj.get(nid, ())):
                if nb not in dist:
                    dist[nb] = dist[nid] + 1
                    queue.append(nb)
        return dist

    def _select_goal_vertices(
        self,
        adj: dict,
        vertices: list,
        start_vertex,
        n_goals: int,
        min_hops: int,
        max_hops: int = 5,
    ) -> list:
        # Goals prefer a fixed hop band so search stays tractable, but a small graph may not reach
        # that depth, so relax the floor until n_goals vertices are available.
        hops = self._bfs_hops(adj, start_vertex.id)
        by_id = {v.id: v for v in vertices}
        reachable = {vid: h for vid, h in hops.items() if vid in by_id and vid != start_vertex.id}

        band = [by_id[vid] for vid, h in reachable.items() if min_hops <= h <= max_hops]
        floor = min_hops
        while len(band) < n_goals and floor >= 1:
            band = [by_id[vid] for vid, h in reachable.items() if h >= floor]
            floor -= 1
        if len(band) < n_goals:  # last resort on a tiny graph: any non-start vertex
            band = [v for v in vertices if v.id != start_vertex.id]
        # canonical order before the shuffle, so the same seed draws the same goals no matter
        # where in the process this graph was built
        band.sort(key=lambda v: (tuple(v.coord), v.id))
        random.shuffle(band)

        selected: list = []
        for v in band:
            if len(selected) == n_goals:
                break
            if all(self._bfs_hops(adj, g.id).get(v.id, 0) >= min_hops for g in selected):
                selected.append(v)
        # fill any slots left when hop-separation between goals could not be met
        for v in band:
            if len(selected) >= n_goals:
                break
            if v not in selected:
                selected.append(v)
        return selected


# ── High-Level API ──────────────────────────────────────────────────────────

# The one way to build a graph: benchmark, playground and the figure scripts all come through here.
# graph_size is the vertex count for sctp_random and the island count for sctp_island.
def create_graph(
    graph_type: str,
    graph_size: int,
    risk_scale: float = 1.0,
    seed: int | None = None,
    n_goals: int = 2,
) -> tuple[ResilientGraph, list[str]]:
    # seeds the global RNGs on purpose: sctp's generators draw from them, and callers re-seed
    # afterwards for execution draws, so topology and draws stay separately reproducible
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    converter = SCTPGraphConverter(risk_scale)
    if graph_type == "sctp_random":
        _, _, sctp_graph = generate_random_graph(
            n_vertex=graph_size, xmin=0.0, ymin=0.0,
            max_edge_len=MAX_EDGE_LENGTH, min_edge_len=MIN_EDGE_LENGTH,
        )
        start = NodePlacement.random_graph(sctp_graph)
        graph = converter.convert_random_graph_multi_goal(sctp_graph, start, n_goals=n_goals)
    elif graph_type == "sctp_island":
        sctp_graph, islands, _ = generate_island_graph(
            xmin=0.0, ymin=0.0,
            max_edge_len=MAX_ISLAND_DISTANCE, min_edge_len=MIN_ISLAND_DISTANCE,
            n_islands=graph_size,
        )
        start = NodePlacement.island_graph(sctp_graph, islands)
        graph = converter.convert_island_graph_multi_goal(sctp_graph, islands, start, n_goals=n_goals)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    return graph, [f"g{i+1}" for i in range(len(converter.goal_vertices))]
