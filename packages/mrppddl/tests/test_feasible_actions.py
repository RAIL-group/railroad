from mrppddl.core import Fluent as F, State, transition, get_action_by_name
from mrppddl.planner import MCTSPlanner
from mrppddl.helper import _make_callable
from mrppddl.core import OptCallable, Operator, Effect


def construct_move_visited_operator(move_time: OptCallable):
    move_time = _make_callable(move_time)
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), F("not visited ?to")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(
                time=(move_time, ["?r", "?from", "?to"]),
                resulting_fluents={
                    F("free ?r"),
                    F("not at ?r ?from"),
                    F("at ?r ?to"),
                    F("visited ?to"),
                },
            ),
        ],
    )


def test_feasible_actions_debug():
    state = State(
        time=0,
        fluents={
            F("free r1"),
            F("free r2"),
            F("visited r1_loc"),
            F("visited r2_loc"),
            F("at r1 r1_loc"),
            F("at r2 r2_loc"),
        },
    )

    objects_by_type = {
        "robot": ["r1", "r2"],
        "location": ["r1_loc", "r2_loc", "roomA"],
    }

    move_op = construct_move_visited_operator(lambda r, loc_from, loc_to: 5.0)
    all_actions = move_op.instantiate(objects_by_type)

    # There are only two feasible actions: move r1 to roomA, move r2 to roomA
    satisfies_precondition = [
        act.name for act in all_actions if state.satisfies_precondition(act)
    ]
    assert set(satisfies_precondition) == {
        "move r1 r1_loc roomA",
        "move r2 r2_loc roomA",
    }

    # Take action "move r1 r1_loc roomA"
    action_r1 = get_action_by_name(all_actions, "move r1 r1_loc roomA")
    next_state = transition(state, action_r1)[0][0]

    # r1 is now not free, r2 is still free, roomA not yet visited
    assert F("free r1") not in next_state.fluents
    assert F("free r2") in next_state.fluents
    assert F("visited roomA") not in next_state.fluents

    # Get feasible actions in the new state
    satisfies_precondition_after = [
        act.name for act in all_actions if next_state.satisfies_precondition(act)
    ]

    # "move r2 r2_loc roomA" must be the only feasible action now
    assert satisfies_precondition_after == ["move r2 r2_loc roomA"]

    # Use MCTS planner to get action with goal of visiting roomA
    goal = F("visited roomA")
    mcts = MCTSPlanner(all_actions)
    action_name = mcts(
        next_state,
        goal,
        max_iterations=1000,
        c=10,
        max_depth=10,
        heuristic_multiplier=2,
    )

    # Action should not be "NONE", it should be "move r2 r2_loc roomA"
    assert action_name != "NONE"
    assert action_name == "move r2 r2_loc roomA"
