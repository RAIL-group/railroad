# Legacy homogeneous planning: flat p_success per edge, no terrain or robot compatibility.
# Ran the first proof-of-concept failure-aware tests; superseded by the terrain-aware
# ResilientGraph path in core.py. Kept so those POC tests still run.

from railroad.core import Fluent, Operator, Effect

F = Fluent


class SimpleGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}  # (from, to): {'cost': float, 'p_success': float}

    def add_edge(self, from_node, to_node, cost, p_success):
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges[(from_node, to_node)] = {'cost': cost, 'p_success': p_success}
        self.edges[(to_node, from_node)] = {'cost': cost, 'p_success': p_success}

    def get_edge_fluents(self):
        return {F(f"edge {fr} {to}") for (fr, to) in self.edges.keys()}


# Homogeneous move: flat per-edge p_success, failure just frees-and-strands the robot.
def create_move_operator(graph_instance: SimpleGraph, failure_penalty: float) -> Operator:
    graph = graph_instance

    def get_cost(from_: str, to_: str) -> float:
        return graph.edges.get((from_, to_), {}).get('cost', float('inf'))

    def get_p_success(robot: str, from_: str, to_: str) -> float:
        return graph.edges.get((from_, to_), {}).get('p_success', 0.0)

    def prob_fail(robot: str, from_: str, to_: str):
        return 1 - get_p_success(robot, from_, to_)

    return Operator(
        name="traverse_edge",
        parameters=[
            ("?robot", "robot"),
            ("?from", "location"),
            ("?to", "location")],
        preconditions=[
            F("at ?robot ?from"),
            F("edge ?from ?to"),
            F("free ?robot"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={
                ~F("at ?robot ?from"),
                ~F("free ?robot")}),

            Effect(
                time=(get_cost, ["?from", "?to"]),
                resulting_fluents=set(),
                prob_effects=[
                    (
                        (get_p_success, ["?robot", "?from", "?to"]),
                        [
                            Effect(time=0, resulting_fluents={
                                F("at ?robot ?to"),
                                F("free ?robot"),
                                F("visited ?to"),
                            })
                        ]
                    ),
                    (
                        (prob_fail, ["?robot", "?from", "?to"]),
                        [
                            Effect(time=failure_penalty, resulting_fluents={
                                ~F("free ?robot"),
                            })
                        ]
                    )
                ]
            )
        ]
    )
