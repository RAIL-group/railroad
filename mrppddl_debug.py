from .mrppddl import Fluent, Effect, ActiveFluents, Operator, State, get_next_actions, transition, get_action_by_name, ProbEffect
from .mrppddl_helper import specify_arguments

def get_success_prob(robot: str, obj: str) -> float:
    return 0.9 if robot == "r1" else 0.6


## Move (seems to work)

# Define the search operator again
move_op = Operator(
    name="move",
    parameters=[
        ("?robot", "robot"),
        ("?loc_from", "location"),
        ("?loc_to", "location"),
    ],
    preconditions=[
        Fluent("at", "?robot", "?loc_from"),
        Fluent("free", "?robot"),
    ],
    effects=[
        Effect(
            time=0,
            resulting_fluents={
                ~Fluent("free", "?robot"),
            }
        ),
        Effect(
            time=5,
            resulting_fluents={
                Fluent("free", "?robot"),
                ~Fluent("at", "?robot", "?loc_from"),
                Fluent("at", "?robot", "?loc_to"),
            }
        )]
)

# Ground a specific instance
objects_by_type = {
    "robot": ["r1", "r2"],
    "location": ["roomA", "roomB"],
    "object": ["cup", "bowl"]
}
move_actions = move_op.instantiate(objects_by_type)
for action in move_actions:
    print(action)

# Define the initial state
action = move_actions[0]
initial_state = State(
    time=0,
    active_fluents={
        Fluent("at", "r1", "roomA"),
        Fluent("at", "r2", "roomA"),
        Fluent("free", "r1"),
        Fluent("free", "r2")
    }
)


print("== Initial State")
print(initial_state)

# Execute transition
print("== All available next actions")
available_actions = get_next_actions(initial_state, move_actions)
for action in available_actions:
    print(action)

print("== State after one action")
outcomes = transition(initial_state, available_actions[0])
for state, prob in outcomes:
    print(state, prob)
    print(state.upcoming_effects.queue)
    updated_state = state

print("== All available next actions")
available_actions = get_next_actions(updated_state, move_actions)
for action in available_actions:
    print(action)

print("== State after another action")
outcomes = transition(updated_state, available_actions[0])
for state, prob in outcomes:
    print(state, prob)
    print(state.upcoming_effects.queue)
    updated_state = state
