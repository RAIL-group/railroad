from mrppddl.core import Fluent, Effect, ActiveFluents, Operator, State, get_next_actions, transition, get_action_by_name, ProbEffect
from mrppddl.core import Operator, Fluent, Effect, State, transition, OptCallable
from mrppddl.helper import _make_callable
from typing import Callable, Union


def _plot_graph(G):
    import matplotlib.pyplot as plt
    import networkx as nx

    # Use spring layout for clarity
    # pos = nx.spring_layout(G, seed=8616)
    pos = nx.kamada_kawai_layout(G)

    # Extract edge labels (e.g., action name and weight)
    edge_labels = {
        (u, v): f"{data['action'].name}\n{data['weight']:.1f}s"
        for u, v, data in G.edges(data=True)
    }

    # Optional: label nodes by hash or small index
    def state_str(state):
        return '\n'.join([str(f) for f in state.active_fluents])

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
    plt.show()


def construct_move_and_visit_operator(move_time: OptCallable) -> Operator:
    move_time = _make_callable(move_time)
    return Operator(
        name="move_visit",
        parameters=[("?robot", "robot"), ("?loc_from", "location"), ("?loc_to", "location")],
        preconditions=[Fluent("at ?robot ?loc_from"),
                       ~Fluent("visited ?loc_to"),
                       Fluent("free ?robot")],
        effects=[
            Effect(time=0,
                   resulting_fluents={~Fluent("free ?robot"),
                                      ~Fluent("at ?robot ?loc_from"),}),
            Effect(time=(move_time, ["?robot", "?loc_from", "?loc_to"]),
                   resulting_fluents={Fluent("free ?robot"),
                                      Fluent("visited ?loc_to"),
                                      Fluent("at ?robot ?loc_to")})
        ])

# Move and Visit Operator
import random
random.seed(8616)
move_time_fn = lambda *args: random.random() + 5.0  #noqa: E731
move_visit_op = construct_move_and_visit_operator(move_time_fn)

# Ground actions
objects_by_type = {
    "robot": ["r1", "r2"],
    "location": ["start", "roomA", "roomB", "roomC", "roomD", "roomE", "roomF"],
}
actions = move_visit_op.instantiate(objects_by_type)
# Initial state
initial_state = State(
    time=0,
    active_fluents={
        Fluent("at r1 start"), Fluent("free r1"),
        Fluent("at r2 start"), Fluent("free r2"),
        Fluent("visited start"),
    }
)

import networkx as nx
G = nx.DiGraph()
expanded_states = set()
new_states = {initial_state.copy_and_zero_out_time()}
G.add_node(initial_state.copy_and_zero_out_time())

expanded_states = set()
new_states = {initial_state}
counter = 0

def is_goal_state(state: State) -> bool:
    return len(objects_by_type['location']) == len([f for f in state.active_fluents
                                                    if f.name == 'visited'])

while new_states:
    state = new_states.pop()
    if state in expanded_states:
        continue
    expanded_states.add(state)


    available_actions = [a for a in actions if state.satisfies_precondition(a)]
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
            if successor_zeroed not in expanded_states and not is_goal_state(successor_zeroed):
                new_states.add(successor_zeroed)

def find_lowest_cost_goal_path(
    G: nx.DiGraph,
    start_state: State,
    is_goal_state: Callable[[State], bool]
) -> tuple[float, list[State]]:
    # Step 1: Get shortest path lengths and paths from start_state
    costs, paths = nx.single_source_dijkstra(G, source=start_state, weight='weight')

    # Step 2: Find goal states among reachable nodes
    goal_states = [(state, cost) for state, cost in costs.items() if is_goal_state(state)]
    if not goal_states:
        raise ValueError("No reachable goal state found.")

    goal_state, min_cost = min(goal_states, key=lambda x: x[1])
    path = paths[goal_state]

    actions = []
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        actions.append(edge_data.get("action", None))

    return min_cost, path, actions

cost, path, actions = find_lowest_cost_goal_path(G, initial_state, is_goal_state)

state = initial_state
print("Computed path")
print(initial_state)
for action in actions:
    print(action)
    state, _ = transition(state, action)[0]
    print(state)

print(f"Start: {initial_state.active_fluents}")
for action in actions:
    print(f"({action.name})")

print(f"Total Time: {state.time}")
# for s in path:
#     print(s)
