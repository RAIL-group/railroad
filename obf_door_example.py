from mrppddl.core import Operator, Fluent, Effect, transition, get_next_actions
from mrppddl.core import State
from typing import Callable, List, Optional, Dict
import matplotlib.pyplot as plt
import networkx as nx



F = Fluent

def _get_distance_from_goal(G: nx.DiGraph, is_goal_fn: Callable):
    goal_nodes = {n for n in G.nodes if is_goal_fn(n)}
    reversed_G = G.reverse(copy=True)
    costs, _ = nx.multi_source_dijkstra(reversed_G, sources=goal_nodes,
                                        weight='weight')
    return costs


def _plot_graph(G, state_str=None, 
                node_value_map: Optional[Dict[State, float]] = None,
                is_goal_fn: Optional[Callable] = None,
                pos=None):
    if not pos:
        pos = nx.kamada_kawai_layout(G)

    # Label the nodes and edges
    edge_labels = {
        (u, v): f"{data['action'].name.split()[0]}\n{data['weight']:.1f}s"
        for u, v, data in G.edges(data=True)
    }
    if not state_str:
        state_str = lambda state: '\n'.join([str(f) for f in state.active_fluents])  #noqa: E731
    node_labels = {node: f"{state_str(node)}" for node in G.nodes}

    norm = None
    cmap = plt.cm.viridis.reversed()
    color_map = {}

    # Goal node processing
    goal_nodes = set()
    if is_goal_fn:
        goal_nodes = {n for n in G.nodes if is_goal_fn(n)}
    else:
        goal_nodes = set()

    if node_value_map:
        norm = plt.Normalize(vmin=min(node_value_map.values()), vmax=max(node_value_map.values()))
        color_map = {node: cmap(norm(node_value_map[node])) for node in G.nodes}
    else:
        norm = None
        color_map = {n: 'lightblue' for n in G.nodes}

    # Separate goal and non-goal nodes
    non_goal_nodes = [n for n in G.nodes if n not in goal_nodes]
    goal_nodes = list(goal_nodes)  # Convert to list for indexing

    # Draw non-goal nodes (filled)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=non_goal_nodes,
        node_color=[color_map[n] for n in non_goal_nodes],
        node_size=60,
    )

    # Draw goal nodes (hollow)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=goal_nodes,
        node_color='lightgrey',
        node_size=60,
        edgecolors=[color_map[n] for n in goal_nodes],
        linewidths=1,
        node_shape='o',
    )

    # Edges and labels
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', width=1.5, edge_color="#a0a0a0f0")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Colorbar
    if node_value_map and norm:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.9, ax=plt.gca())  #noqa F841
        # cbar.set_label("Distance to Goal (seconds)")

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
        for action in available_actions:
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

## Door World
def build_door_world():
    objects_by_type = {
        "robot": ["robot"],
        "door": ["blue_door", "red_door"],
        "key": ["blue_key", "red_key"],
        "location": ["start", "rk_loc", "bk_loc", "doors_loc"]
    }
    operators = []
    operators.append(Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?from ?robot")],
        effects=[Effect(time=1.0, resulting_fluents={F("at ?to ?robot"), F("not at ?from ?robot")})]))
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
G = build_full_graph(initial_state, all_actions)
productive_G = get_graph_subset_productive_edges(G, goal_functions)
pruned_G = prune_disconnected_nodes(productive_G, initial_state)

pos = nx.kamada_kawai_layout(pruned_G)
plt.subplot(131)
node_value_map_0 = _get_distance_from_goal(pruned_G, goal_functions[0])
_plot_graph(pruned_G,
            state_str=lambda state: '\n'.join([str(f) for f in state.active_fluents if 'holding' in f.name or 'open' in f.name]),
            node_value_map=node_value_map_0,
            is_goal_fn=goal_functions[0],
            pos=pos)
plt.subplot(132)
node_value_map_1 = _get_distance_from_goal(pruned_G, goal_functions[1])
_plot_graph(pruned_G,
            state_str=lambda state: '\n'.join([str(f) for f in state.active_fluents if 'holding' in f.name or 'open' in f.name]),
            node_value_map=node_value_map_1,
            is_goal_fn=goal_functions[1],
            pos=pos)
plt.subplot(133)

def confusion_fn(values):
    c = (max(values)-min(values))/(max(values) + 0.0001)
    return c

confusion_map = {node: [] for node in pruned_G.nodes}
for goal_fn in goal_functions:
    distances = _get_distance_from_goal(pruned_G, goal_fn)
    for node in confusion_map.keys():
        confusion_map[node].append(distances[node])
        

confusion_map = {node: confusion_fn(values) for node, values in confusion_map.items()}
_plot_graph(pruned_G,
            state_str=lambda state: '\n'.join([str(f) for f in state.active_fluents if 'holding' in f.name or 'open' in f.name]),
            node_value_map=confusion_map,
            pos=pos)
plt.show()
