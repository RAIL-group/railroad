# The map and the move for a mixed robot team on risky terrain: some robots handle some ground
# better than others. The older version, where every robot was the same, is in legacy.py.

from typing import Dict, Set
from railroad.core import Fluent, Operator, Effect

F = Fluent


# How well one robot handles each kind of ground, 0 to 1. Example: {"debris": 0.9, "clear": 0.95}
RobotProfile = Dict[str, float]


# Chance this robot gets across this edge: 1 - beta * (1 - phi), where beta is how dangerous the
# edge is and phi is how well the robot handles that ground.
def compute_p_success(profile: RobotProfile, terrain_type: str, hazard_severity: float) -> float:
    compatibility = profile.get(terrain_type, 0.5)
    susceptibility = 1.0 - compatibility
    return 1.0 - (hazard_severity * susceptibility)


# Every fluent this effect could set, including the ones inside its chance branches.
def effect_fluents(eff) -> list:
    out = list(eff.resulting_fluents)
    if eff.is_probabilistic:
        for branch in eff.prob_effects:
            for be in branch.effects:
                out.extend(effect_fluents(be))
    return out


# Which directed edges are still open. A robot failing on an edge closes it both ways, so a route
# table built once from the full graph goes stale and whoever is routing has to re-read this.
def parse_available_paths(state) -> set:
    return {(f.args[0], f.args[1]) for f in state.fluents
            if f.name == "path_available" and len(f.args) == 2}


# Where each robot is and which goals are done. A robot part-way across an edge has no location at
# all, since "at" is dropped on departure, so we use the place it is heading for.
def parse_state(state) -> tuple[dict, set]:
    pos, visited = {}, set()
    for f in state.fluents:
        if f.name == "at" and len(f.args) == 2:
            pos[f.args[0]] = f.args[1]
        elif f.name == "safely_visited" and len(f.args) == 1:
            visited.add(f.args[0])
    for _, eff in state.upcoming_effects:
        for rf in effect_fluents(eff):
            if not rf.negated and rf.name == "at" and len(rf.args) == 2 and rf.args[0] not in pos:
                pos[rf.args[0]] = rf.args[1]
    return pos, visited


# The map: places, and edges carrying a travel cost, a kind of ground, and how dangerous they are.
class ResilientGraph:

    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[tuple, dict] = {}  # (from, to): {cost, terrain_type, hazard_severity}
        self.node_coords: Dict[str, tuple] = {}  # node -> (x, y), so nothing else needs the sctp graph

    def add_edge(self, from_node: str, to_node: str, cost: float,
                 terrain_type: str = "normal", hazard_severity: float = 0.0,
                 bidirectional: bool = True,
                 from_coord: tuple | None = None, to_coord: tuple | None = None):
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        if from_coord is not None:
            self.node_coords[from_node] = from_coord
        if to_coord is not None:
            self.node_coords[to_node] = to_coord
        props = {'cost': cost, 'terrain_type': terrain_type, 'hazard_severity': hazard_severity}
        self.edges[(from_node, to_node)] = props
        if bidirectional:
            self.edges[(to_node, from_node)] = props

    def get_edge_fluents(self) -> set:
        return {F(f"edge {fr} {to}") for (fr, to) in self.edges.keys()}

    # One per direction of travel. Dropped when a robot fails there, which shuts the edge.
    def get_available_path_fluents(self) -> set:
        return {F(f"path_available {fr} {to}") for (fr, to) in self.edges.keys()}


# A robot crosses an edge and either arrives, or fails.
#
# blocks_on_failure decides what the wreck does to the map. True shuts that edge to everyone,
# which is what the terrain-hazard reading implies: whatever stopped this robot is still there.
# False leaves it open, so a failure costs the team the robot and nothing else. The two are
# genuinely different problems rather than a tuning knob, and the paired experiment runs both.
def create_risk_move_operator(
    graph_instance: ResilientGraph,
    robot_profiles: Dict[str, RobotProfile],
    blocks_on_failure: bool = True,
) -> Operator:
    graph = graph_instance
    profiles = robot_profiles

    def get_cost(robot: str, from_: str, to_: str) -> float:
        return graph.edges.get((from_, to_), {}).get('cost', float('inf'))

    def get_robot_compatibility(robot: str, from_: str, to_: str) -> float:
        if (from_, to_) not in graph.edges:
            return 0.0
        terrain_type = graph.edges[(from_, to_)]['terrain_type']
        return profiles.get(robot, {}).get(terrain_type, 0.5)

    def get_hazard(robot: str, from_: str, to_: str) -> float:
        return graph.edges.get((from_, to_), {}).get('hazard_severity', 1.0)

    def prob_reached(robot: str, from_: str, to_: str) -> float:
        phi = get_robot_compatibility(robot, from_, to_)
        beta = get_hazard(robot, from_, to_)
        return 1.0 - (beta * (1.0 - phi))

    def prob_fail(robot: str, from_: str, to_: str) -> float:
        return 1.0 - prob_reached(robot, from_, to_)

    failure_fluents = {~F("free ?robot"), ~F("operational ?robot")}
    if blocks_on_failure:
        failure_fluents |= {~F("path_available ?from ?to"), ~F("path_available ?to ?from")}

    return Operator(
        name="risk_move",
        parameters=[
            ("?robot", "robot"),
            ("?from", "location"),
            ("?to", "location")],
        preconditions=[
            F("at ?robot ?from"),
            F("edge ?from ?to"),
            F("free ?robot"),
            F("operational ?robot"),
            F("path_available ?from ?to"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={
                ~F("at ?robot ?from"),
                ~F("free ?robot")}),

            Effect(
                time=(get_cost, ["?robot", "?from", "?to"]),
                resulting_fluents=set(),
                prob_effects=[
                    (
                        (prob_reached, ["?robot", "?from", "?to"]),
                        [
                            Effect(time=0, resulting_fluents={
                                F("at ?robot ?to"),
                                F("free ?robot"),
                            })
                        ]
                    ),
                    (
                        (prob_fail, ["?robot", "?from", "?to"]),
                        [
                            # The robot is lost right when the crossing would have ended, and takes
                            # the edge with it when blocks_on_failure. No cost here: C_fail is
                            # charged once in the score.
                            Effect(time=0, resulting_fluents=failure_fluents)
                        ]
                    ),
                ]
            )
        ]
    )


# Marks a goal done, and nothing ever unmarks it, so it survives the robot moving on ("at" would not).
# The robot has to be standing there, idle and operational, so dying on arrival does not count.
def create_safely_visited_operator() -> Operator:
    return Operator(
        name="safely_visited",
        parameters=[
            ("?robot", "robot"),
            ("?location", "location"),
        ],
        preconditions=[
            F("at ?robot ?location"),
            F("free ?robot"),
            F("operational ?robot"),
            F("is_goal ?location"),
            ~F("safely_visited ?location"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={
                F("safely_visited ?location"),
            }),
        ],
    )
