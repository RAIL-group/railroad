# Everything derived from a raw SCTP graph before it becomes a ResilientGraph: where a team starts,
# how long and connected its edges are, and what terrain each edge gets labelled.

import numpy as np


class NodePlacement:
    @staticmethod
    def random_graph(sctp_graph):
        return min(sctp_graph.vertices, key=lambda v: v.coord[0])

    @staticmethod
    def island_graph(sctp_graph, islands):
        return min((v for island in islands for v in island.vertices), key=lambda v: v.coord[0])


# ~20% of edges come out clear, ~20% rocky, the rest risky.
CLEAR_PERCENTILE = 20
ROCKY_PERCENTILE = 40


# The clear and rocky cutoffs for a set of risk keys.
def percentile_cuts(keys) -> tuple[float, float]:
    ks = sorted(keys)
    n = len(ks)
    if not n:
        return 0.20, 0.40
    return ks[n * CLEAR_PERCENTILE // 100], ks[n * ROCKY_PERCENTILE // 100]


# Label an edge from its risk key; the riskiest band splits by length into steep or deformable.
def label_for(key: float, length: float, p_clear: float, p_rocky: float, median_length: float) -> str:
    if key <= p_clear:
        return "clear"
    if key <= p_rocky:
        return "rocky"
    return "steep" if length < median_length else "deformable"


class GraphStatistics:
    # Median edge length and vertex degree over nav edges [(v1, v2, block_prob)].
    def __init__(self, nav_edges: list):
        self.median_edge_length: float = 15.0
        self.mean_edge_length: float = 15.0
        self.median_degree: float = 5
        self._compute(nav_edges)

    def _compute(self, nav_edges: list) -> None:
        lengths = [GraphStatistics.dist(v1, v2) for v1, v2, _ in nav_edges]
        degree: dict = {}
        for v1, v2, _ in nav_edges:
            degree[v1.id] = degree.get(v1.id, 0) + 1
            degree[v2.id] = degree.get(v2.id, 0) + 1
        if lengths:
            self.median_edge_length = float(np.median(lengths))
            self.mean_edge_length = float(np.mean(lengths))
        if degree:
            self.median_degree = float(np.median(list(degree.values())))

    @staticmethod
    def dist(v1, v2) -> float:
        return float(((v1.coord[0] - v2.coord[0]) ** 2 + (v1.coord[1] - v2.coord[1]) ** 2) ** 0.5)


class TerrainInference:
    # Labels each edge clear/rocky/steep/deformable. How risky an edge looks blends how blocked it is
    # with how short it is; w mixes the two, so short edges tend to be riskier but not always.

    def __init__(self, nav_edges: list, stats: GraphStatistics, tradeoff_weight: float = 0.5):
        self.stats = stats
        self.w = tradeoff_weight
        bps = [bp for _, _, bp in nav_edges]
        lengths = [GraphStatistics.dist(v1, v2) for v1, v2, _ in nav_edges]
        self._bp_lo, self._bp_hi = (min(bps), max(bps)) if bps else (0.0, 1.0)
        self._len_lo, self._len_hi = (min(lengths), max(lengths)) if lengths else (0.0, 1.0)
        self._p_clear, self._p_rocky = percentile_cuts(
            self._risk_key(bp, GraphStatistics.dist(v1, v2)) for v1, v2, bp in nav_edges)

    @staticmethod
    def _norm(x: float, lo: float, hi: float) -> float:
        return (x - lo) / (hi - lo) if hi > lo else 0.5

    # 0 = safe/long, 1 = risky/short: being more blocked and shorter both push it up
    def _risk_key(self, block_prob: float, length: float) -> float:
        blocked = self._norm(block_prob, self._bp_lo, self._bp_hi)
        shortness = 1.0 - self._norm(length, self._len_lo, self._len_hi)
        return (1.0 - self.w) * blocked + self.w * shortness

    def terrain_for(self, block_prob: float, length: float) -> str:
        return label_for(self._risk_key(block_prob, length), length,
                         self._p_clear, self._p_rocky, self.stats.median_edge_length)
