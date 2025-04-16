from mrppddl.core import Operator, Fluent, Effect, transition
from mrppddl.core import State
from mrppddl.planner import astar
import time


ff_memory = dict()
def ff_heuristic(
        state: State,
        is_goal_fn
) -> float:
    # 1. Forward relaxed reachability
    t0 = state.time
    state = transition(state, None, relax=True)[0][0]
    dtime = state.time - t0
    initial_known_fluents = state.fluents
    known_fluents = set(state.fluents)
    if initial_known_fluents in ff_memory.keys():
        return dtime + ff_memory[initial_known_fluents]
    newly_added = set(known_fluents)
    fact_to_action = {}       # Fluent -> Action that added it
    action_to_duration = {}   # Action -> Relaxed duration
    visited_actions = set()

    while newly_added:
        next_newly_added = set()
        state = State(time=0, fluents=known_fluents)
        for action in all_actions:
            if action in visited_actions:
                continue
            if not state.satisfies_precondition(action, relax=True):
                continue

            for successor, _ in transition(state, action, relax=True):
                duration = successor.time - state.time
                action_to_duration[action] = duration
                visited_actions.add(action)

                for f in successor.fluents:
                    if f not in known_fluents:
                        known_fluents.add(f)
                        newly_added.add(f)
                        fact_to_action[f] = action
                    elif f in fact_to_action.keys():
                        fact_to_action[f] = min(action, fact_to_action[f],
                                                key=lambda a: a.effects[-1].time)
        newly_added = next_newly_added

    if not is_goal_fn(known_fluents):
        return float('inf')  # Relaxed goal not reachable

    # 2. Extract required goal fluents by ablation
    required_fluents = set()
    for f in known_fluents:
        test_set = known_fluents - {f}
        if not is_goal_fn(test_set):
            required_fluents.add(f)

    # 3. Backward relaxed plan extraction
    needed = set(required_fluents)
    used_actions = set()
    total_duration = 0.0

    while needed:
        f = needed.pop()
        if f in state.fluents:
            continue
        action = fact_to_action.get(f)
        if action and action not in used_actions:
            used_actions.add(action)
            total_duration += action_to_duration[action]
            for p in action._pos_precond:
                if p not in state.fluents:
                    needed.add(p)

    ff_memory[initial_known_fluents] = total_duration
    return dtime + total_duration


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


stime = time.time()
initial_state, all_actions, is_goal_fn = build_door_world()
path = astar(initial_state, all_actions, is_goal_fn,
             lambda state: ff_heuristic(state, is_goal_fn))
print(f"Planning time: {time.time() - stime}")
s = initial_state

print(s)
for a in path:
    s = transition(s, a)[0][0]
    print(a.name, " Time: ", s.time)
