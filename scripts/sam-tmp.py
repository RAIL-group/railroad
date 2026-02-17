import numpy as np
from railroad.core import Fluent as F, get_action_by_name, State, Operator, Effect
import railroad.operators
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner
from railroad.dashboard import PlannerDashboard

"""Basic Task Interruption Scenario.
    Predicates:
        at ?o - object ?l - location ; the object is at a location l
        hand-empty ?r - robot ; the robot's gripper is not holding anything
        holding ?r - robot ?o - object ; the robot is holding an object
                                            in one of its grippers
        free ?r ; robot is free to take an action
        is-turkey ?o ; is the object turkey
        is-bread ?o ; is the object bread
        is-mayo ?o ; is the object mayo
        sandwhich-made
        prep-station ?l
"""

# setup task planning problem
locations = {
    "refrigerator": np.array([0, 0]),
    "countertop1": np.array([1,1]),
    "countertop2": np.array([2,1]),
    "table": np.array([0,2])
}

objects_by_type = {
    "robot": {"robot1"},
    "location": set(locations),
    "objects": {"turkey", "bread", "mayo"},
    "prep-station": {"countertop1"},
}

# operators
def move_time(robot, loc_from, loc_to):
    return float(np.linalg.norm(locations[loc_from] - locations[loc_to]))

move = Operator(
    name="move",
    parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
    preconditions=[F("at ?r ?from"), F("free ?r")],
    effects=[  # not free at t=0, free again at destination after move_time
        Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
        Effect(time=(move_time, ["?r", "?from", "?to"]),
            resulting_fluents={F("free ?r"), F("at ?r ?to")}),
    ],
)

pick = Operator(
    name="pick",
    parameters=[("?r", "robot"), ("?o", "objects"), ("?l", "location")],
    preconditions=[F("free ?r"), F("at ?r ?l"), F("at ?o ?l"), F("hand-empty ?r")],
    effects=[
        Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?o ?l")}),
        Effect(time=1, resulting_fluents={
            F("not hand-empty ?r"), F("holding ?r ?o"), F("free ?r")
        })
    ]
)

place = Operator(
    name="place",
    parameters=[("?r", "robot"), ("?o", "objects"), ("?l", "location")],
    preconditions=[F("free ?r"), F("at ?r ?l"), F("holding ?r ?o")],
    effects=[
        Effect(time=0, resulting_fluents={F("not free ?r"), F("not holding ?r ?o")}),
        Effect(time=1, resulting_fluents={
            F("free ?r"), F("at ?o ?l"), F("hand-empty ?r")
        })
    ]
)

assemble = Operator(
    name="assemble",
    parameters=[
        ("?r", "robot"), ("?o1", "objects"), ("?o2", "objects"), ("?o3", "objects"),
        ("?l", "location")
    ],
    preconditions=[
        F("free ?r"), F("is-turkey ?o1"), F("is-bread ?o2"), F("is-mayo ?o3"),
        F("at ?o1 ?l"), F("at ?o2 ?l"), F("at ?o3 ?l"), F("at ?r ?l"),
        F("hand-empty ?r"), F("prep-station ?l")
    ],
    effects=[
        Effect(time=0, resulting_fluents={F("not free ?r"), F("not hand-empty ?r")}),
        Effect(time=3, resulting_fluents={
            F("free ?r"), F("not at ?o1 ?l"), F("not at ?o2 ?l"),
            F("not at ?o3 ?l"), F("sandwhich-made"), F("hand-empty ?r")
        })
    ]
)

# Both robots start free in the den
initial_state = State(0.0, {
    F("free robot1"), F("at robot1 table"), F("is-turkey turkey"),
    F("is-bread bread"), F("is-mayo mayo"), F("hand-empty robot1"),
    F("at turkey refrigerator"), F("at mayo countertop2"), F("at bread table"),
    F("prep-station table"), F("prep-station countertop1"), F("prep-station countertop2"),
    ~F("prep-station refrigerator"), ~F("sandwhich-made")
})

# goal = F("at turkey table") & F("at mayo table") & F("at bread table")
goal = F("sandwhich-made")

env = SymbolicEnvironment(
    state=initial_state, objects_by_type=objects_by_type,
    operators=[move, pick, place, assemble],
)

# def fluent_filter(f):
#     return any(kw in f.name for kw in ["at", "holding", "found"])

with PlannerDashboard(goal, env) as dashboard:
    # Plan-act loop: replan whenever a robot becomes free
    for _ in range(20): # changed from 20
        if goal.evaluate(env.state.fluents):
            break

        actions = env.get_actions()
        planner = MCTSPlanner(actions)
        action_name = planner(env.state, goal, max_iterations=10000, c=200, heuristic_multiplier=1, debug_heuristic=True)
        action = get_action_by_name(actions, action_name)
        env.act(action)
        dashboard.update(planner, action_name)
