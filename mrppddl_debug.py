from mrppddl import ProbEffects, Fluent, Action, Effect

def get_success_prob(robot: str, obj: str) -> float:
    return 0.9 if robot == "r1" else 0.6


eff = ProbEffects(
    time=5,
    prob_effects=[
        (
            lambda b: get_success_prob(b["?robot"], b["?object"]),
            [
                Effect(
                    time=0,
                    resulting_fluents={Fluent("found", "?object"), Fluent("holding", "?robot", "?object")}
                )
            ]
        ),
        (
            lambda b: 1.0 - get_success_prob(b["?robot"], b["?object"]),
            [
                Effect(
                    time=0,
                    resulting_fluents={Fluent("free", "?robot")}
                )
            ]
        )
    ],
    resulting_fluents={
        Fluent("searched", "?loc_to", "?object"),
        ~Fluent("free", "?robot"),
        ~Fluent("at", "?robot", "?loc_from"),
        Fluent("at", "?robot", "?loc_to"),
    }
)

print(eff)
