from itertools import product

class Type:
    def __init__(self, name):
        self.name = name

class Object:
    def __init__(self, name, type_):
        self.name = name
        self.type = type_

class Predicate:
    def __init__(self, name, arg_names):
        self.name = name
        self.arg_names = arg_names  # e.g., ['?robot', '?loc']

    def __repr__(self):
        return f"{self.name}({', '.join(self.arg_names)})"

class GroundPredicate:
    def __init__(self, predicate, args):
        self.predicate = predicate
        self.args = args  # list of Object

    def __repr__(self):
        return f"{self.predicate.name}({', '.join(arg.name for arg in self.args)})"

    def __hash__(self):
        return hash((self.predicate.name, tuple(self.args)))

    def __eq__(self, other):
        return (self.predicate.name, self.args) == (other.predicate.name, other.args)

class State:
    def __init__(self, facts):
        self.facts = set(facts)

class Operator:
    def __init__(self, name, parameters, preconditions, effects):
        self.name = name
        self.parameters = parameters  # list of (var_name, type_name)
        self.preconditions = preconditions  # list of Predicate
        self.effects = effects  # list of Predicate

class ActionInstance:
    def __init__(self, name, preconditions, effects):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects

    def __repr__(self):
        return f"Action: {self.name}\n  Pre: {self.preconditions}\n  Eff: {self.effects}"

def all_args_distinct(binding):
    objs = list(binding.values())
    return len(objs) == len(set(objs))


def instantiate_operator(op, objects_by_type):
    domains = [objects_by_type[typ] for _, typ in op.parameters]
    for assignment in product(*domains):
        binding = {var: obj for (var, _), obj in zip(op.parameters, assignment)}
        
        if not all_args_distinct(binding):
            continue

        yield instantiate_action(op, binding)


def substitute(pred, binding):
    args = [binding[var] for var in pred.arg_names]
    return GroundPredicate(pred, args)

def instantiate_action(op, binding):
    grounded_preconds = [substitute(p, binding) for p in op.preconditions]
    grounded_effects = [substitute(p, binding) for p in op.effects]
    name = f"{op.name}({', '.join(binding[v].name for v, _ in op.parameters)})"
    return ActionInstance(name, grounded_preconds, grounded_effects)

# Define objects
robot_type = "robot"
location_type = "location"

objects = [
    Object("r1", robot_type),
    Object("r2", robot_type),
    Object("l1", location_type),
    Object("l2", location_type),
]

objects_by_type = {}
for obj in objects:
    objects_by_type.setdefault(obj.type, []).append(obj)

# Define predicates
at = Predicate("at", ["?robot", "?loc"])
connected = Predicate("connected", ["?from", "?to"])

# Define the move operator
move_op = Operator(
    "move",
    parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
    preconditions=[
        Predicate("at", ["?r", "?from"]),
        Predicate("connected", ["?from", "?to"]),
    ],
    effects=[
        Predicate("at", ["?r", "?to"]),
        Predicate("at", ["?r", "?from"]),  # assume this one gets *removed*; would need delete/add distinction later
    ]
)

actions = list(instantiate_operator(move_op, objects_by_type))
for act in actions:
    print(act)
