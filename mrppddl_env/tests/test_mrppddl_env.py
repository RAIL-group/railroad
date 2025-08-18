import pytest
from mrppddl.core import Fluent, State, get_action_by_name
from mrppddl.core import transition
from mrppddl.helper import construct_move_operator

from mrppddl.core import OptCallable, Operator, Effect
from mrppddl.helper import _make_callable, _invert_prob

F = Fluent

def construct_search_operator(
    object_find_prob: OptCallable, move_time: OptCallable) -> Operator:
    object_find_prob = _make_callable(object_find_prob)
    inv_object_find_prob = _invert_prob(object_find_prob)
    move_time = _make_callable(move_time)
    return Operator(
        name="search",
        parameters=[("?robot", "robot"),
                    ("?loc_from", "location"),
                    ("?loc_to", "location"),
                    ("?object", "object")],
        preconditions=[F("free ?robot"),
                       F("at ?robot ?loc_from"),
                       F("not searched ?loc_to ?object"),
                       F("not revealed ?loc_to"),
                       F("not found ?object")],
        effects=[
            Effect(time=0, resulting_fluents={
                F("not free ?robot"),
                F("not at ?robot ?loc_from")}),
            Effect(time=(move_time, ["?robot", "?loc_from", "?loc_to"]),
                   resulting_fluents={F("at ?robot ?loc_to"),
                                      F("searched ?loc_to ?object"),
                                      F("free ?robot"),
},
                   prob_effects=[(
                       (object_find_prob, ["?robot", "?loc_to", "?object"]),
                       [Effect(time=0, resulting_fluents={F("found ?object")})]
                   ), (
                       (inv_object_find_prob, ["?robot", "?loc_to", "?object"]), [],
                   )])
        ],
    )



def test_search_reveal_simple():
    """This is a simple scenario showing that we can reveal locations in the
    simulator by searching them with a simple 'search' action from the robot. We
    show that, with a single robot, assigning the search action and then
    advancing time until something is revealed.

    We will need to handle that the robot thinks it could be in multiple states
    right now, but in fact that is not possible, fate determined by the
    environment.
    """

    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        """Robot 1 is faster than robot 2."""
        return 3.0 if robot == "r1" else 5.0

    def object_find_prob(robot: str, loc: str, obj: str):
        return 0.5

    objects_at_locations = {
        "start": [],
        "roomA": ["objA", "objC"],
        "roomB": ["objB"],
    }

    objects_by_type = {
        "robot": set(["r1"]),
        "location": set(["start", "roomA", "roomB"]),
        "object": set(["objA", "objB"]),
    }
    def _get_actions_from_objects(objects_by_type):
        search_op = construct_search_operator(object_find_prob, move_time)
        return search_op.instantiate(objects_by_type)

    def reveal(state, objects_by_type):
        # Look for locations 'searched' but not 'revealed'.
        # The set of objects should contain all objects in the revealed locations

        print(state)
        all_objects = objects_by_type["object"].union(*(
            objects_at_locations[location]
            for location in objects_by_type["location"]
            if F(f"revealed {location}") in state.fluents
        ))
        print(all_objects)
        
        return new_state, new_objects_by_type
        pass

    state = State(time=0, fluents={F("at r1 start"), F("free r1")})
    all_actions = _get_actions_from_objects(objects_by_type)
    print(all_actions)
    a1 = get_action_by_name(all_actions, "search r1 start roomA objA")
    print(a1)
    print(state)
    states = transition(state, a1)
    for resulting_state, prob in states:
        rstate_revealed = reveal(resulting_state, objects_by_type)
        print(rstate_revealed)
    raise NotImplementedError()

    # [This functionality should be inside the simulator eventually, but I'm
    # just developing for now.]
    # Execute the selected action:

    # Identify what locations have been revealed by the robot and update the state accordingly.

    

def test_robot_assignment():
    """
    We simulate two robots executing concurrent move skills with different durations.
    The simulator should:
      1) Expose an initial planner state with both robots free and at roomA.
      2) After assigning a move for r1, remove (free r1) and publish an upcoming effect for r1 finishing.
      3) After assigning a move for r2, remove (free r2) and publish both upcoming effects.
      4) step() advances to the earliest event, applies it, and returns which robot(s) are now free.
      5) step() again advances to the next event, finishing r2’s move.

    Environment-level fluents/upcoming events are empty here; robots supply the robot-related parts.
    """

    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        """Robot 1 is faster than robot 2."""
        return 3.0 if robot == "r1" else 5.0

    move_op = construct_move_operator(move_time)
    objects_by_type = {
        "robot": ["r1", "r2"],
        "location": ["roomA", "roomB"],
    }
    move_actions = move_op.instantiate(objects_by_type)

    # Pick the two moves we care about (A -> B) for each robot.
    a_r1 = get_action_by_name(move_actions, "move r1 roomA roomB")
    a_r2 = get_action_by_name(move_actions, "move r2 roomA roomB")
    assert a_r1 and a_r2, "Expected move actions to exist"

    # --- Build simulator with two robots starting at roomA
    r1 = Robot("r1", location="roomA")
    r2 = Robot("r2", location="roomA")
    sim = Simulator([r1, r2], env_fluents=set(), env_upcoming=[], start_time=0.0)

    # Initial planner-facing state
    s0 = sim.get_state()
    assert s0.time == 0.0
    assert F("at r1 roomA") in s0.fluents
    assert F("at r2 roomA") in s0.fluents
    assert F("free r1") in s0.fluents
    assert F("free r2") in s0.fluents
    assert len(s0.upcoming_effects) == 0

    # Assign move to r1 (roomA -> roomB). r1 should no longer be free; one upcoming effect appears.
    sim.assign(a_r1)
    s1 = sim.get_state()
    assert F("free r1") not in s1.fluents
    assert F("free r2") in s1.fluents
    assert len(s1.upcoming_effects) == 1
    t1, _ = s1.upcoming_effects[0]
    assert pytest.approx(t1) == 3.0

    # Assign move to r2 (roomA -> roomB). Now both robots are busy; two upcoming effects exist.
    sim.assign(a_r2)
    s2 = sim.get_state()
    assert F("free r1") not in s2.fluents
    assert F("free r2") not in s2.fluents
    assert len(s2.upcoming_effects) == 2
    times = sorted([t for (t, _) in s2.upcoming_effects])
    assert times == [3.0, 5.0]

    # Step once → advance to t=3.0, complete r1’s move.
    finished = sim.step()
    assert finished == ["r1"]  # r1 should now need a new action
    s3 = sim.get_state()
    assert pytest.approx(s3.time, rel=1e-7) == 3.0
    assert F("at r1 roomB") in s3.fluents
    assert F("free r1") in s3.fluents
    assert F("at r2 roomA") in s3.fluents  # r2 still traveling
    assert F("free r2") not in s3.fluents
    # Only r2’s completion remains, at absolute time 5.0
    assert len(s3.upcoming_effects) == 1
    t_next, _ = s3.upcoming_effects[0]
    assert pytest.approx(t_next, rel=1e-7) == 5.0
