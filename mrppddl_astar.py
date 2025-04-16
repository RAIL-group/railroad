import random
from mrppddl.core import Fluent, Effect, Operator, State, transition
from mrppddl.planner import astar
from mrppddl.heuristic import ff_heuristic

F = Fluent

# Move and Visit Operator
random.seed(8616)
move_time_fn = lambda *args: random.random() + 5.0  #noqa: E731
move_visit_op = Operator(
    name="move_visit",
    parameters=[("?robot", "robot"), ("?loc_from", "location"), ("?loc_to", "location")],
    preconditions=[F("at ?robot ?loc_from"), F("not visited ?loc_to"), F("free ?robot")],
    effects=[
        Effect(time=0,
               resulting_fluents={F("not free ?robot"), F("not at ?robot ?loc_from")}),
        Effect(time=(move_time_fn, ["?robot", "?loc_from", "?loc_to"]),
               resulting_fluents={F("free ?robot"), F("visited ?loc_to"), F("at ?robot ?loc_to")})
    ])
move_home_op = Operator(
    name="move_home",
    parameters=[("?robot", "robot"), ("?loc_from", "location")],
    preconditions=[F("at ?robot ?loc_from"), F("free ?robot"), F("not at ?robot start")],
    effects=[
        Effect(time=0,
               resulting_fluents={~F("free ?robot"), ~F("at ?robot ?loc_from"),}),
        Effect(time=(move_time_fn, ["?robot", "?loc_from", "start"]),
               resulting_fluents={F("free ?robot"), F("at ?robot start")})
    ])
wait_op = Operator(
    name="wait",
    parameters=[("?robot", "robot")],
    preconditions=[F("free ?robot"), F("not waited ?robot")],
    effects=[Effect(time=0, resulting_fluents={F("not free ?robot")}),
             Effect(time=1, resulting_fluents={F("free ?robot"), F("waited ?robot")})])

# Get all actions
objects_by_type = {
    "robot": ["r1", "r2", "r3", "r4"],
    "location": ["start", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
}
all_actions = (
    move_visit_op.instantiate(objects_by_type)
    + move_home_op.instantiate(objects_by_type)
    + wait_op.instantiate(objects_by_type)
)


# Initial state
initial_state = State(
    time=0,
    fluents={
        F("at r1 start"), F("free r1"),
        F("at r2 start"), F("free r2"),
        F("visited start"),
    })


def is_goal_state(fluents) -> bool:
    return (
        len(objects_by_type['location']) == len([f for f in fluents if f.name == 'visited'])
        and (F("at r1 start") in fluents)
        and (F("at r2 start") in fluents)
    )

import time
stime = time.time()
ff_memory = dict()
path = astar(initial_state, all_actions, is_goal_state,
             lambda state: ff_heuristic(state, is_goal_state, all_actions, ff_memory=ff_memory))
print(f"Planning time: {time.time() - stime}")
s = initial_state
for a in path:
    s = transition(s, a)[0][0]
    print(a.name)

print(s)

