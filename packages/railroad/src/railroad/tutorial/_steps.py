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

from typing import List, NamedTuple, Optional, TypedDict

EXPERIMENT = "railroad-tutorial"
"""Every sweep accumulates here, so the dashboard is one page you refresh."""

DEMO_FILE = "demo.py"
"""The one file you edit."""

NOTEBOOK = "language.ipynb"
"""The primer that comes before the arc.

Fluents, states and transitions are small enough to hold in your head, and
they are the wrong shape for a script: you want to poke at one, print it, and
try the next thing. That is a notebook. The steps below are all *programs*,
which is why they are files you run.
"""

RUNNER = "uv run"
"""Everything in this repository runs through uv, the tutorial included.

Printed commands have to be commands: a card that says ``uv run python demo.py``
in a checkout where the interpreter lives in a uv-managed environment is
printing something that does not work.
"""

MEDIA_DIR = "media"
"""Where plots and videos go, and what the dashboard serves."""


class StepInfo(TypedDict):
    """One step of the tutorial."""

    id: str
    """Two-digit ordinal, also the CLI handle (``uv run railroad tutorial goto 02``)."""

    title: str
    """Short label, shown wherever the step is named."""

    filename: str
    """Snapshot filename under ``railroad/tutorial/steps/``."""

    point: str
    """One line: what this step demonstrates."""

    sweep: str
    """What the benchmark sweeps over. Every step has a sweep."""

    media: str
    """Stem for ``--save-plot`` / ``--save-video``, e.g. ``two-robots``."""

    problem: str
    """Which problem this step solves.

    Plan costs are only comparable within one of these. Steps that change the
    world -- hiding the objects, moving to a bigger house -- start a new one, and
    the step list drops the delta across the boundary rather than inviting the
    wrong reading.
    """

    requires: str
    """Optional extra this step needs (``"procthor"``), or ``""`` for none."""

    notes: List[str]
    """Talking points, printed by ``peek`` so nothing has to be memorised."""


STEPS: List[StepInfo] = [
    {
        "id": "01",
        "title": "one robot, two rooms",
        "filename": "01_two_rooms.py",
        "problem": "two-rooms",
        "requires": "",
        "point": "A whole problem in one file -- and what it does with the value function off.",
        "sweep": "mcts.h_mult x mcts.iterations",
        "media": "two-rooms",
        "notes": [
            "Two rooms, two objects, one robot. Everything after this step is "
            "these same three operators with more of something.",
            "move is written out by hand so both halves of a durative action "
            "are visible: lose free/at at t=0, regain them at t=d. pick and "
            "place are the same shape.",
            "The loop replans every time the robot frees up. That is the whole "
            "control structure; there is no plan to execute, only a next action.",
            "Then the experiment. MCTS scores a leaf by elapsed time plus "
            "h_mult times an estimate of the work remaining; h_mult=0 leaves "
            "only the clock. Run it: 'tutorial run --case 3'.",
            "Measured, 8 repeats: at h_mult=5, every budget in the grid finds "
            "the same 7-action, 38-second plan. At h_mult=0, 0 of 8 at every "
            "one of them -- 4000 iterations buys exactly what 400 does.",
            "Watch what it does rather than the number. It picks the mug up, "
            "puts it straight back down, and repeats that until the step limit: "
            "40 actions, every one of them a pick or a place, and it never "
            "leaves the table.",
            "That is not a search that needs a bigger budget. Scored by elapsed "
            "time alone, picking is the cheapest legal action and walking to "
            "the shelf is the most expensive, so there is nothing pointing at "
            "the goal to climb.",
        ],
    },
    {
        "id": "02",
        "title": "stop the robot undoing itself",
        "filename": "02_action_blocking.py",
        "problem": "two-rooms",
        "requires": "",
        "point": "A guard per action, and the same search at h_mult=0 starts working.",
        "sweep": "mcts.h_mult x mcts.iterations",
        "media": "action-blocking",
        "notes": [
            "One guard per action, each the same three lines: a precondition, a "
            "flag set when the action lands, an expiry a tenth of a second "
            "later. 'just-picked' is a fluent with a lifetime -- no timer, no "
            "special case, just an ordinary effect at an ordinary time.",
            "just-moved earns its place separately. Guard pick and place alone "
            "and the churn stops, but the robot only starts pacing between the "
            "two rooms instead -- equally cheap, equally pointless. All three, "
            "and every legal action left is one that makes progress.",
            "no_op is not decoration. A guard can leave a robot with nothing "
            "legal to do -- walk to the shelf empty-handed, and there is nothing "
            "to pick and no second move allowed -- and a search with no value "
            "function does not stumble into that state, it aims for it: a dead "
            "end is where the clock stops. Without no_op this step fails after "
            "a single move. extra_cost keeps waiting a last resort.",
            "The sweep is the same grid as step 01, which is the point: run the "
            "case that just failed, 'tutorial run --case 3', and compare the "
            "two sweeps side by side in the dashboard.",
            "Measured, 8 repeats: h_mult=0 goes from 0 of 8 at every budget to "
            "8 of 8 at every budget -- and on the *optimal* plan, the same 7 "
            "actions and 38 seconds the value function finds.",
            "There is no floor left to find: it sits below any budget worth "
            "running. The guards did not help the planner look harder; they "
            "removed everywhere wrong to look.",
            "The h_mult=5 half of the grid is the control, and it is identical "
            "to step 01 -- 38 seconds throughout. A decent estimate of "
            "work-remaining already knows that undoing yourself is not "
            "progress, so the guards buy it nothing.",
            "This is where the h_mult experiment ends. From step 03 the value "
            "function is back on and the guards stay, because that is the "
            "combination you would actually ship -- but it is worth knowing how "
            "far the guards alone get you: measured, the four-room house of step "
            "03 also solves 8 of 8 at h_mult=0, on the same 44.3-second plan.",
        ],
    },
    {
        "id": "03",
        "title": "clear the table",
        "filename": "03_clear_table.py",
        "problem": "clear-table",
        "requires": "",
        "point": "The same operators in a bigger world, and a goal made of negations.",
        "sweep": "mcts.iterations x c",
        "media": "clear-table",
        "notes": [
            "Four rooms and three objects now, and the same guards step 02 "
            "arrived at, carried forward unchanged for the rest of the arc.",
            "The goal is a conjunction of *negated* literals -- nothing on the "
            "table -- which names no destination at all. Anywhere else will do, "
            "and the planner picks.",
            "In the sweep: no floor. Measured, 8 repeats -- every budget in the "
            "grid, at either exploration constant, lands the same ~44 second "
            "plan in 10 actions. That is step 02's lesson at a bigger size: the "
            "guards removed everywhere wrong to look, so there is nothing left "
            "for a larger budget to buy.",
        ],
    },
    {
        "id": "04",
        "title": "add a second robot",
        "filename": "04_two_robots.py",
        "problem": "clear-table",
        "requires": "",
        "point": "Concurrency for free: one more object of type robot, and time drops.",
        "sweep": "num_robots x mcts.iterations",
        "media": "two-robots",
        "notes": [
            "The whole diff is the problem growing a robot: a count, a loop "
            "building their fluents, and the sweep's new axis. Not one "
            "operator is touched -- concurrency is a property of the state "
            "semantics, not of the actions.",
            "Watch the Braille timeline: the two rows overlap.",
            "In the sweep, 8 repeats: 44.3 -> 22.1 -> 8.6 seconds. The second "
            "robot halves it, the third more than halves it again. With three "
            "robots and three objects each takes one, and because the goal only "
            "asks that nothing *remain* on the table, the run ends at the last "
            "pick -- 6 actions, one trip plus one pick, nothing ever put away.",
        ],
    },
    {
        "id": "05",
        "title": "hide the objects",
        "filename": "05_hidden_objects.py",
        "problem": "house-search",
        "requires": "",
        "point": "Probabilistic search, a prior worth having, and the lock that stops it being wasted.",
        "sweep": "num_robots",
        "media": "hidden-objects",
        "notes": [
            "The objects leave the initial state; the robots have to look. That "
            "buys a probabilistic effect and a prior over where things are: "
            "find_prob, 0.8 where the object actually is and 0.1 everywhere "
            "else. It is an oracle wearing a probability's clothes, and it is "
            "the same function step 06 carries into a real house -- which is "
            "what makes step 07, where it stops reading the answer, a swap of "
            "one function rather than a rewrite.",
            "ObjectSearchEnvironment resolves the branch against ground truth "
            "rather than sampling: a wrong prior costs time, it never invents a "
            "discovery. And searching *reveals* a room -- everything there "
            "becomes known and the room closes for good.",
            "Which is what the lock is for. 'searched' only lands when a search "
            "finishes, so nothing would stop two robots searching one room at "
            "once, and two independent draws at 0.8 look better than one. They "
            "are not: one search reveals the room, so the second robot was "
            "never going to learn anything. Three lines settle it -- a "
            "lock-search precondition, the lock taken at t=0, released when the "
            "search completes -- and they are not optional. A search you can "
            "run unlocked is a search that can be bought twice, so the operator "
            "carries the lock the way it carries 'at ?r ?loc'.",
            "Measured, 8 repeats: about 28 -> 19.1 -> 13.5 seconds at one, two "
            "and three robots, 8 of 8 every time. Three robots repeat exactly, "
            "to the decimal; the smaller teams move by around a second, which is "
            "the search order changing rather than the plan.",
            "'searches' is the cleaner measure, and the one to put on the slide: "
            "3.0 at every team size. Three rooms hold something, so 3 is the "
            "floor, and the lock is what holds a bigger team to it. It does not "
            "make robots faster; it stops them buying the same information "
            "twice.",
            "What the prior does *not* buy here is worth saying, because it is "
            "the setup for step 06. Three objects in three rooms, and a goal "
            "that wants all three: every room gets searched whatever the belief "
            "says, so a better prior changes the order and barely moves the "
            "clock. Measured against the flat 0.5 this step used to carry, every "
            "team size lands inside the other's spread -- what tightens is the "
            "plan, two robots taking 7 actions every single time instead of 7 "
            "to 9. A prior earns "
            "its keep when there are more places than targets, which is the "
            "next step's house.",
        ],
    },
    {
        "id": "06",
        "title": "a real house",
        "filename": "06_procthor.py",
        "problem": "procthor-8613",
        "requires": "procthor",
        "point": "Same operators, real geometry: a ProcTHOR home and Theta* travel times.",
        "sweep": "scene_seed x num_robots",
        "media": "house",
        "notes": [
            "The operators barely change. Their *numbers* do: move times are "
            "Theta* paths over the scene's occupancy grid, and the locations are "
            "the containers of a generated home.",
            "This search operator is the one from step 05, lock and all -- and "
            "it is also, line for line, operators.construct_search_operator.",
            "'tutorial run --save-video house.mp4' renders the run over the scene's "
            "top-down view, into media/, which the dashboard serves. About 15 "
            "seconds on top of the run, so it is something you can do live.",
            "Its two-robot cases are also the oracle half of step 07's "
            "comparison -- same houses, same team, same budget -- so run this "
            "sweep before that one if the last slide is the point.",
            "Seed 8613 is chosen for pace: 8 containers ground out to ~440 "
            "actions with two robots, about 25 seconds. 8616 is prettier and "
            "does not finish inside MAX_STEPS at this budget.",
        ],
    },
    {
        "id": "07",
        "title": "stop cheating",
        "filename": "07_learned_prior.py",
        "problem": "procthor-8612",
        "requires": "procthor",
        "point": "Swap the oracle prior for the packaged network. One function changes.",
        "sweep": "scene_seed",
        "media": "learned",
        "notes": [
            "Step 06's prior read scene.object_locations -- an oracle wearing a "
            "probability's clothes. This one has never seen the answer: it is a "
            "small net over SBERT embeddings of the names, shipped with the "
            "package.",
            "The table prints before any planning, and it is the step. On 8612 "
            "it ranks the true container first for both targets (cellphone 0.454 "
            "on the dresser, desklamp 0.400 on the diningtable) and all but rules "
            "out the toilet. On 8613 a fork gets 0.825 on the countertop, while a "
            "pen spreads 0.41/0.37/0.36 over three flat surfaces -- which is the "
            "right answer for a pen.",
            "The oracle half of the comparison is step 06: the same problem, the "
            "same two houses, the same team. So it is two sweeps side by side in "
            "the dashboard rather than a flag inside this one -- and there is no "
            "way to run this step with the cheating prior, which is the point of "
            "the step.",
            "Measured at two robots, 32 repeats of each: on 8612 the oracle "
            "averages 333 seconds against this step's 385, and both finish every "
            "run. On 8613 the means are 336 against 510 -- but the *medians* are "
            "333 against 367, and that gap between the two summaries is the "
            "slide. The model's typical run is barely worse; its bad run is 891 "
            "seconds, and 9 runs in 32 never finish inside MAX_STEPS at all. A "
            "learned belief is not uniformly worse. It is occasionally wrong and "
            "expensive when it is, and a mean is exactly the statistic that "
            "hides that. Violins, not means.",
            "The sweep ships at 8 repeats and 8 cannot pin these down, which is "
            "its own lesson and this is the step to say it on. Two 8-repeat "
            "sweeps of 8612, same configuration, same afternoon, came back 318 "
            "and 480; the 32-repeat answer is 385. Read the shape of the violin, "
            "not the number under it, and treat any one sweep's mean as a "
            "sample. That is the sweep earning its keep on the last slide of its "
            "own tutorial.",
            "Nothing else changes: the find probability is a callable of "
            "(robot, location, object) either way.",
        ],
    },
]


class Command(NamedTuple):
    """One row of the card: what you would type, and why."""

    label: str
    """Left-hand goal (``run it``), or ``""`` to continue the row above."""

    command: str
    comment: str


def command_lines(step: StepInfo) -> List[Command]:
    """Every way to run *step*, as commands you could type yourself.

    This is the how-to half of the tutorial: ordered by how often you want
    them, phrased by goal rather than by tool. ``tutorial run`` and
    ``tutorial bench`` are short enough to type mid-sentence and each prints
    the longer command it expands to, so the card stays honest about what is
    actually happening underneath.
    """
    me = f"{RUNNER} railroad tutorial"
    stem = step["media"]
    rows = []
    if step["id"] == STEPS[0]["id"]:
        # Only on the first step: the primer is where you start, and after that
        # it is behind you.
        rows.append(Command("read it", f"{me} notebook",
                            "fluents, effects, transition -- in Jupyter"))
    rows += [
        Command("run it", f"{me} run", f"the same as: {RUNNER} python {DEMO_FILE}"),
        Command("", f"{me} run --list", "the parameter cases this step sweeps"),
        # Case 1 rather than a more interesting index: the card cannot see how
        # many cases a step has, and every step has at least two. Which case is
        # worth watching is a per-step question, so it lives in the notes.
        Command("", f"{me} run --case 1", "run one of them, live"),
        Command("", f"{me} run --save-plot {stem}.png",
                "trajectories and the action list, into media/"),
        Command("", f"{me} run --save-video {stem}.mp4",
                "or animate the whole run"),
        Command("sweep it", f"{me} bench", step["sweep"]),
        Command("see it", f"{me} dashboard",
                "start it; --status and --stop from here too"),
        Command("move on", f"{me} next",
                "shows the patch, then merges your edits -- 'next 05' jumps"),
    ]
    return rows


def step_ids() -> List[str]:
    """Every step id, in order."""
    return [step["id"] for step in STEPS]


class UnknownStep(KeyError):
    """No step by that id. A ``KeyError`` still, since it always was one.

    ``KeyError`` renders its argument with ``repr``, which puts a mistyped id
    inside two sets of quotes on the way to the terminal. There are seven steps
    and two commands that take an id, so this gets typed wrong live.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def get_step(step_id: str) -> StepInfo:
    """Look up a step by id, accepting ``"2"`` for ``"02"``."""
    normalized = step_id.zfill(2)
    for step in STEPS:
        if step["id"] == normalized:
            return step
    raise UnknownStep(
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
