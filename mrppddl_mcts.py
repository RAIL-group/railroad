from mrppddl.core import Operator, Fluent, Effect, transition
from mrppddl.core import State
from mrppddl.planner import astar, mcts
from mrppddl.heuristic import ff_heuristic
import time




F = Fluent

## Door World
def build_door_world():
    objects_by_type = {
        "robot": ["r1", "r2"],
        "door": ["blue_door", "red_door"],
        "key": ["blue_key", "red_key"],
        "location": ["start", "rk_loc", "bk_loc", "doors_loc"]
    }
    move_op = Operator(
        name="move",
        parameters=[("?robot", "robot"), ("?loc_from", "location"), ("?loc_to", "location")],
        preconditions=[F("at ?loc_from ?robot"), F("free ?robot")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?robot"), F("not at ?loc_from ?robot")}),
            Effect(time=1.0, resulting_fluents={F("free ?robot"), F("at ?loc_to ?robot")})
        ])
    pick_op = Operator(
        name="pick_key",
        parameters=[("?robot", "robot"), ("?loc", "location"), ("?key", "key")],
        preconditions=[F("free ?robot"), F("at ?loc ?robot"), F("at ?loc ?key")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?robot")}),
            Effect(time=1.0, resulting_fluents={F("holding ?robot ?key"), F("not at ?loc ?key"), F("free ?robot")})])
    open_door_op = Operator(
        name="open_door",
        parameters=[("?robot", "robot"), ("?loc", "location"), ("?door", "door"), ("?key", "key")],
        preconditions=[F("free ?robot"), F("at ?loc ?robot"), F("at ?loc ?door"), F("holding ?robot ?key"), F("fits ?door ?key")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?robot")}),
            Effect(time=1.0, resulting_fluents={F("open ?door"), F("free ?robot")})])
    all_actions = [action for operator in {move_op, pick_op, open_door_op}
                   for action in operator.instantiate(objects_by_type)]

    initial_state = State(
        time=0,
        fluents={
            F("free r1"), F("at start r1"),
            F("free r2"), F("at start r2"),
            F("at rk_loc red_key"),
            F("at bk_loc blue_key"),
            F("at doors_loc red_door"),
            F("at doors_loc blue_door"),
            F("fits blue_door blue_key"),
            F("fits red_door red_key"),
        }
    )

    def is_goal_open_red(fluents: frozenset[Fluent]) -> bool:
        return Fluent("open red_door") in fluents

    def is_goal_open_blue(fluents: frozenset[Fluent]) -> bool:
        return Fluent("open blue_door") in fluents

    def is_goal_open_all(fluents: frozenset[Fluent]) -> bool:
        return (
            Fluent("open blue_door") in fluents
            and Fluent("open red_door") in fluents
            and F("at start r1") in fluents
            and F("at start r2") in fluents
        )

    return initial_state, all_actions, is_goal_open_all


ff_memory = dict()
stime = time.time()
initial_state, all_actions, is_goal_fn = build_door_world()
path = astar(initial_state, all_actions, is_goal_fn,
             lambda state: ff_heuristic(state, is_goal_fn, all_actions, ff_memory=ff_memory))
s = initial_state
print(s)
for a in path:
    s = transition(s, a)[0][0]
    print(a.name, " Time: ", s.time)

## MCTS Solution
print("MCTS")
out, node = mcts(initial_state, all_actions, is_goal_fn,
           lambda state: 0.1 * ff_heuristic(state, is_goal_fn, all_actions, ff_memory=ff_memory))
state = initial_state
print(state)
while not is_goal_fn(node.state.fluents):
    best_action, chance_node = max(node.children.items(), key=lambda item: item[1].value / item[1].visits if item[1].visits > 0 else float('-inf'))
    node = chance_node.children[0]
    state = transition(state, best_action)[0][0]
    print(best_action.name, "Time: ", state.time)
    # print(state.fluents, is_goal_fn(state.fluents))
    # print(state)

print(f"Planning time: {time.time() - stime}")
s = initial_state
