"""The ordered steps of the guided tutorial.

Each step is a complete snapshot of ``demo.py``. Advancing does not append to a
growing file; it merges the *diff* between two snapshots into whatever the
presenter currently has on disk (see :mod:`railroad.tutorial._advance`), so a
live edit survives moving to the next step.

Keeping the snapshots whole rather than storing patches means a step can always
be inspected, run, and diffed on its own, and a botched merge is recoverable by
taking the snapshot verbatim.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class StepInfo(TypedDict):
    """One step of the tutorial."""

    id: str
    """Two-digit ordinal, also the CLI handle (``railroad tutorial goto 02``)."""

    title: str
    """Short label shown in the watch pane header."""

    filename: str
    """Snapshot filename under ``railroad/tutorial/steps/``."""

    point: str
    """One line: what this step demonstrates."""

    sweep: str
    """What pressing ``b`` sweeps over, or ``""`` when the step has no benchmark."""

    problem: str
    """Which problem this step solves.

    Plan costs are only comparable within one of these. Steps that change the
    world -- hiding the objects, moving to a bigger house -- start a new one, and
    ``compare`` drops the delta across the boundary rather than inviting the
    wrong reading.
    """

    notes: List[str]
    """Talking points, printed by ``peek`` so nothing has to be memorised."""


STEPS: List[StepInfo] = [
    {
        "id": "00",
        "title": "the language",
        "filename": "00_language.py",
        "problem": "",
        "point": "Fluents, states, timed effects, and what a transition actually does.",
        "sweep": "",
        "notes": [
            "Two spellings of a fluent are the same object; ~f is negation.",
            "An Action is a list of effects at times, not a single instant.",
            "Dispatch r1 and the clock stays at 0 -- r2 is still free to act.",
            "Dispatch r2 and time jumps to 5: transition() runs the world "
            "forward until somebody is free again.",
            "prob_effects makes transition() return a distribution, not a state.",
        ],
    },
    {
        "id": "01",
        "title": "clear the table",
        "filename": "01_clear_table.py",
        "problem": "clear-table",
        "point": "A whole problem: operators, a negated goal, and the plan-act loop.",
        "sweep": "mcts.iterations x c",
        "notes": [
            "move is written out by hand so both halves of a durative action "
            "are visible: lose free/at at t=0, regain them at t=d.",
            "pick and place come from railroad.operators -- same shape.",
            "The goal is a conjunction of *negated* literals: nothing on the table.",
            "The loop replans every time a robot frees up. That is the whole "
            "control structure; there is no plan to execute, only a next action.",
            "Press b: there is a search floor. Around 10-25 iterations the run "
            "fails outright, and where that floor sits depends on c -- a larger "
            "exploration constant wastes more of a small budget. Past ~100 "
            "iterations, more search buys nothing on a problem this small.",
        ],
    },
    {
        "id": "02",
        "title": "add a second robot",
        "filename": "02_two_robots.py",
        "problem": "clear-table",
        "point": "Concurrency for free: one more object of type robot, and time drops.",
        "sweep": "num_robots x mcts.iterations",
        "notes": [
            "The diff is four lines. No operator changes at all -- concurrency "
            "is a property of the state semantics, not of the actions.",
            "Watch the Braille timeline: the two rows overlap.",
            "Press b: 1 -> 2 -> 3 robots roughly halves and halves again. With "
            "three robots and three objects each takes one, and because the goal "
            "only asks that nothing *remain* on the table, the run ends at the "
            "last pick -- one trip plus one pick, nothing ever put away.",
        ],
    },
    {
        "id": "03",
        "title": "hide the objects",
        "filename": "03_hidden_objects.py",
        "problem": "house-search",
        "point": "Probabilistic search, a flat prior, and a failure mode to find.",
        "sweep": "num_robots x mcts.iterations",
        "notes": [
            "The objects leave the initial state; the robots have to look. That "
            "buys a probabilistic effect and a prior over where things are.",
            "ObjectSearchEnvironment resolves the branch against ground truth "
            "rather than sampling: a wrong prior costs time, it never invents a "
            "discovery. And searching *reveals* a room -- everything there "
            "becomes known and the room closes for good.",
            "Read the action list, not the number. Two robots search the same "
            "room at the same time: 'searched' only lands when a search "
            "finishes, so nothing rules it out, and a flat prior makes two draws "
            "at 0.5 look better than one.",
        ],
    },
    {
        "id": "04",
        "title": "one searcher per room",
        "filename": "04_search_lock.py",
        "problem": "house-search",
        "point": "A lock predicate, and the A/B that shows why it is not optional.",
        "sweep": "use_search_lock x num_robots",
        "notes": [
            "Three lines: a lock-search precondition, the lock taken at t=0, "
            "released when the search completes.",
            "Measured, 8 repeats: 1 robot 29.2 either way -- a robot cannot "
            "contend with itself. 2 robots 18.9 without vs 19.1 with, a wash. "
            "3 robots 18.7 -> 13.5.",
            "Cost hides it at two robots because the wasted search overlaps a "
            "useful one. Watch 'searches' instead: 3 / 4 / 5.6 without the lock "
            "as the team grows, and a flat 3 with it -- three rooms hold "
            "something, so 3 is the floor.",
        ],
    },
    {
        "id": "05",
        "title": "the value function",
        "filename": "05_heuristic.py",
        "problem": "big-house-search",
        "point": "h_add/h_ff mixing, the multiplier, and the probabilistic retry delta.",
        "sweep": "(lambda_add, lambda_ff) x mcts.h_mult",
        "notes": [
            "MCTS scores a state with lambda_add*h_add + lambda_max*h_max + "
            "lambda_ff*h_ff, plus a retry delta. Both relaxations assume an "
            "action does what it says; at a 0.5 find probability that is "
            "optimistic by a factor of two, and the delta is what pays for the "
            "searches you expect to repeat.",
            "The house is bigger here on purpose. In the four-room version every "
            "mix finds the same plan at any budget worth using.",
            "Measured, 8 repeats at 400 iterations: h_mult=1 wins for all three "
            "mixes (40.1 / 40.9 / 41.8); h_mult=5 costs 2-8s more. The "
            "multiplier matters more than the mix.",
            "Look at the violins, not the means: pure h_add has a standard "
            "deviation of 12-15 against 2.6 for the balanced mix. Cheap and "
            "noisy versus steady. This is what the distribution plots are for.",
        ],
    },
]


def step_ids() -> List[str]:
    """Every step id, in order."""
    return [step["id"] for step in STEPS]


def get_step(step_id: str) -> StepInfo:
    """Look up a step by id, accepting ``"2"`` for ``"02"``."""
    normalized = step_id.zfill(2)
    for step in STEPS:
        if step["id"] == normalized:
            return step
    raise KeyError(
        f"no tutorial step {step_id!r}; available: {', '.join(step_ids())}"
    )


def step_index(step_id: str) -> int:
    """Position of *step_id* in :data:`STEPS`."""
    return step_ids().index(get_step(step_id)["id"])


def neighbour(step_id: str, offset: int) -> Optional[StepInfo]:
    """The step *offset* places from *step_id*, or ``None`` past either end."""
    index = step_index(step_id) + offset
    if index < 0 or index >= len(STEPS):
        return None
    return STEPS[index]
