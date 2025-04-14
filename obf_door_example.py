from mrppddl.core import Operator, Fluent, Effect, transition, OptCallable, get_next_actions, get_action_by_name
from mrppddl.core import State, Action
from mrppddl.helper import _make_callable
from typing import Callable, Union, List, Optional
import random
import hashlib
import matplotlib.pyplot as plt
import networkx as nx


F = Fluent

def _plot_graph(G, state_str=None):

    # Use spring layout for clarity
    # pos = nx.spring_layout(G, seed=8616)
    pos = nx.kamada_kawai_layout(G)

    # Extract edge labels (e.g., action name and weight)
    edge_labels = {
        (u, v): f"{data['action'].name.split()[0]}\n{data['weight']:.1f}s"
        for u, v, data in G.edges(data=True)
    }

    # Optional: label nodes by hash or small index
    if not state_str: 
        state_str = lambda state: '\n'.join([str(f) for f in state.active_fluents])

    node_labels = {node: f"{state_str(node)}" for idx, node in enumerate(G.nodes)}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=60)

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', width=1.5, edge_color="#a0a0a0f0")

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title("State-Action Graph")
    plt.axis('off')
    plt.tight_layout()


def _get_productive_edges_for_goal(
    G: nx.DiGraph,
    is_goal_fn: Callable,
) -> set[tuple]:
    # Find goal nodes
    goal_nodes = [n for n in G.nodes if is_goal_fn(n)]
    if not goal_nodes:
        return set()

    # Reverse graph to compute shortest-path *to* goal nodes
    reversed_G = G.reverse(copy=True)
    min_distance = {}

    for goal in goal_nodes:
        # lengths = nx.single_source_shortest_path_length(reversed_G, goal)
        costs, _ = nx.single_source_dijkstra(reversed_G, source=goal, weight='weight')  # pyright: ignore[reportArgumentType]
        for node, dist in costs.items():
            if node not in min_distance or dist < min_distance[node]:
                min_distance[node] = dist

    productive_edges = {
        (u, v)
        for u, v in G.edges
        if u in min_distance and v in min_distance and min_distance[v] < min_distance[u]
    }

    return productive_edges


def get_graph_subset_productive_edges(
    G: nx.DiGraph,
    goal_state_functions: List[Callable]
) -> nx.DiGraph:
    all_productive_edges = set()

    for is_goal_fn in goal_state_functions:
        productive_edges = _get_productive_edges_for_goal(G, is_goal_fn)
        all_productive_edges.update(productive_edges)

    # Create a subgraph with the union of productive edges
    productive_G = nx.DiGraph()
    productive_G.add_nodes_from(G.nodes(data=True))  # preserve node attributes
    productive_G.add_edges_from((u, v, G[u][v]) for u, v in all_productive_edges)  # preserve edge data

    return productive_G


def prune_disconnected_nodes(G: nx.DiGraph, initial_state: State) -> nx.DiGraph:
    pruned_G = G.copy()
    changed = True

    while changed:
        changed = False
        to_remove = [
            node for node in pruned_G.nodes
            if node != initial_state and pruned_G.in_degree(node) == 0
        ]
        if to_remove:
            pruned_G.remove_nodes_from(to_remove)
            changed = True

    return pruned_G


def build_full_graph(initial_state, all_actions, is_goal_state: Optional[Callable] = None):
    initial_state = initial_state.copy_and_zero_out_time()
    G = nx.DiGraph()
    G.add_node(initial_state)

    new_states = {initial_state}
    expanded_states = set()
    while new_states:
        state = new_states.pop()
        if state in expanded_states:
            continue
        expanded_states.add(state)

        available_actions = get_next_actions(state, all_actions)
        print("What's available:")
        for action in available_actions:
            print(action.name)
            outcomes = transition(state, action)
            for successor, prob in outcomes:
                if prob == 0.0:
                    continue

                # Normalize time for graph consistency
                state_zeroed = state.copy_and_zero_out_time()
                successor_zeroed = successor.copy_and_zero_out_time()

                # Add nodes and edge with duration (original un-zeroed delta)
                G.add_node(state_zeroed)
                G.add_node(successor_zeroed)

                duration = successor.time - state.time
                G.add_edge(state_zeroed, successor_zeroed, action=action, weight=duration, probability=prob)

                # Expand successor if not already seen
                if successor_zeroed not in expanded_states:
                    if not is_goal_state:
                        new_states.add(successor_zeroed)
                    elif is_goal_state(successor_zeroed):
                        new_states.add(successor_zeroed)
    
    return G


def construct_move_and_visit_operator(move_time: OptCallable) -> Operator:
    move_time = _make_callable(move_time)
    return Operator(
        name="move_visit",
        parameters=[("?robot", "robot"), ("?loc_from", "location"), ("?loc_to", "location")],
        preconditions=[F("at ?robot ?loc_from"), F("not visited ?loc_to")],
        effects=[Effect(time=(move_time, ["?robot", "?loc_from", "?loc_to"]),
                        resulting_fluents={F("visited ?loc_to"), F("at ?robot ?loc_to")})])

# Move and Visit Operator
def move_time_fn(robot, start, end):
    seed_input = start + '::' + end
    seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16)
    random.seed(seed)
    return 1.0

random.seed(8616)
move_visit_op = construct_move_and_visit_operator(move_time_fn)

# Ground actions
objects_by_type = {
    "robot": ["robot"],
    "location": ["start", "roomA", "roomB", "roomC", "roomD", "roomE", "roomF"],
    # "location": ["start", "roomA", "roomB", "roomC"],
}
actions = move_visit_op.instantiate(objects_by_type)
# Initial state
initial_state = State(
    time=0,
    active_fluents={
        Fluent("at robot start"), 
        Fluent("visited start"),
    }
)


def is_goal_vis_a(state: State) -> bool:
    return Fluent("visited roomA") in state.active_fluents

def is_goal_vis_b(state: State) -> bool:
    return Fluent("visited roomB") in state.active_fluents

def is_goal_vis_ab(state: State) -> bool:
    return (
        Fluent("visited roomA") in state.active_fluents
        and Fluent("visited roomB") in state.active_fluents
    )


## Door World
def build_door_world():
    objects_by_type = {
        "robot": ["robot"],
        "door": ["blue_door", "red_door"],
        "key": ["blue_key", "red_key"],
        "location": ["start", "rk_loc", "bk_loc", "doors_loc"]
    }
    def move_time_fn(robot, start, end):
        seed_input = start + '::' + end
        seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16)
        random.seed(seed)
        return 1.0
    operators = []
    operators.append(Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?loc_from", "location"), ("?loc_to", "location")],
        preconditions=[F("at ?loc_from ?robot")],
        effects=[Effect(time=1.0, resulting_fluents={F("at ?loc_to ?robot"), F("not at ?loc_from ?robot")})]))
    operators.append(Operator(
        name="pick_key",
        parameters=[("?robot", "robot"), ("?loc", "location"), ("?key", "key")],
        preconditions=[F("at ?loc ?robot"), F("at ?loc ?key")],
        effects=[Effect(time=1.0, resulting_fluents={F("holding ?robot ?key"), F("not at ?loc ?key")})]))
    operators.append(Operator(
        name="open_door",
        parameters=[("?robot", "robot"), ("?loc", "location"), ("?door", "door"), ("?key", "key")],
        preconditions=[F("at ?loc ?robot"), F("at ?loc ?door"), F("holding ?robot ?key"), F("fits ?door ?key")],
        effects=[Effect(time=1.0, resulting_fluents={F("open ?door")})]))
    all_actions = [action for operator in operators
                   for action in operator.instantiate(objects_by_type)]
   
    # Initial state
    initial_state = State(
        time=0,
        active_fluents={
            F("at start robot"),
            F("at rk_loc red_key"),
            F("at bk_loc blue_key"),
            F("at doors_loc red_door"),
            F("at doors_loc blue_door"),
            F("fits blue_door blue_key"),
            F("fits red_door red_key"),
        }
    )

    def is_goal_open_red(state: State) -> bool:
        return Fluent("open red_door") in state.active_fluents

    def is_goal_open_blue(state: State) -> bool:
        return Fluent("open blue_door") in state.active_fluents
    
    goal_functions = [is_goal_open_red, is_goal_open_blue]

    return initial_state, all_actions, goal_functions


initial_state, all_actions, goal_functions = build_door_world()
print(all_actions)
G = build_full_graph(initial_state, all_actions)
for s in G.nodes:
    print(s)
productive_G = get_graph_subset_productive_edges(G, goal_functions)
pruned_G = prune_disconnected_nodes(productive_G, initial_state)

# plt.subplot(221)
# _plot_graph(G)
# plt.subplot(222)
# _plot_graph(productive_G)
# plt.subplot(223)
_plot_graph(pruned_G,
        state_str=lambda state: '\n'.join([str(f) for f in state.active_fluents
                                           if 'holding' in f.name or 'open' in f.name])
)
plt.show()
