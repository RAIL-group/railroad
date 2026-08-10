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

DEFAULT_PARALLEL = 12
"""Deliberately below ``cpu_count() - 2``: a sweep that eats every core makes
the machine you are presenting from unusable."""

DEMO_FILE = "demo.py"
"""The one file you edit."""

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
    """What the benchmark sweeps over, or ``""`` when the step has none."""

    media: str
    """Suggested ``--video`` filename, or ``""`` when the step draws nothing."""

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
        "id": "00",
        "title": "the language",
        "filename": "00_language.py",
        "problem": "",
        "requires": "",
        "point": "Fluents, states, timed effects, and what a transition actually does.",
        "sweep": "",
        "media": "",
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
        "requires": "",
        "point": "A whole problem: operators, a negated goal, and the plan-act loop.",
        "sweep": "mcts.iterations x c",
        "media": "",
        "notes": [
            "move is written out by hand so both halves of a durative action "
            "are visible: lose free/at at t=0, regain them at t=d.",
            "pick and place are written out too -- same shape, plus the "
            "just-picked / just-placed pair that stops a robot undoing itself.",
            "The goal is a conjunction of *negated* literals: nothing on the table.",
            "The loop replans every time a robot frees up. That is the whole "
            "control structure; there is no plan to execute, only a next action.",
            "In the sweep: there is a search floor. Around 10-25 iterations the run "
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
        "requires": "",
        "point": "Concurrency for free: one more object of type robot, and time drops.",
        "sweep": "num_robots x mcts.iterations",
        "media": "",
        "notes": [
            "Only one line of the problem changes: NUM_ROBOTS. Not one "
            "operator is touched -- concurrency is a property of the state "
            "semantics, not of the actions. The rest of the diff is prose and "
            "the sweep's new axis.",
            "Watch the Braille timeline: the two rows overlap.",
            "In the sweep: 1 -> 2 -> 3 robots roughly halves and halves again. With "
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
        "requires": "",
        "point": "Probabilistic search, a flat prior, and a failure mode to find.",
        "sweep": "num_robots x mcts.iterations",
        "media": "",
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
        "requires": "",
        "point": "A lock predicate, and the A/B that shows why it is not optional.",
        "sweep": "use_search_lock x num_robots",
        "media": "",
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
        "requires": "",
        "point": "h_add/h_ff mixing, the multiplier, and the probabilistic retry delta.",
        "sweep": "(lambda_add, lambda_ff) x mcts.h_mult",
        "media": "",
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
    {
        "id": "06",
        "title": "a real house",
        "filename": "06_procthor.py",
        "problem": "procthor-8613",
        "requires": "procthor",
        "point": "Same operators, real geometry: a ProcTHOR home and Theta* travel times.",
        "sweep": "scene_seed x num_robots",
        "media": "house.mp4",
        "notes": [
            "The operators barely change. Their *numbers* do: move times are "
            "Theta* paths over the scene's occupancy grid, and the locations are "
            "the containers of a generated home.",
            "This search operator is the one from step 04, lock and all -- and "
            "it is also, line for line, operators.construct_search_operator.",
            "'uv run python demo.py --video house.mp4' renders the run over the "
            "scene's top-down view, into media/, which the dashboard serves. "
            "Rendering costs minutes against the run's 30 seconds, so start it "
            "and keep talking -- or record one before the session.",
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
        "sweep": "use_learned_prior x scene_seed",
        "media": "learned.mp4",
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
            "On 8612 the two are inside the noise of each other, and which one "
            "looks better depends on the sample: 336.8 vs 339.4 at ten repeats, "
            "341.8 vs 402.6 at eight. Do not claim a winner there.",
            "On 8613 the model is clearly worse, and worse in the *tail*: about "
            "340 against 450-475, standard deviation 24 against 256, and it "
            "drops a run in eight or ten. A learned belief is not uniformly "
            "worse; it is occasionally wrong and expensive when it is -- the "
            "same reason step 05 wanted violins rather than means.",
            "Worth admitting out loud: at three repeats this first looked like a "
            "flat 35% penalty on both scenes. It is not. That is the sweep "
            "earning its keep on the last slide of its own tutorial.",
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


def command_lines(step: StepInfo, *, parallel: int = DEFAULT_PARALLEL) -> List[Command]:
    """Every way to run *step*, as commands you could have typed yourself.

    This is the how-to half of the tutorial and the reason there is no wrapper
    around any of it: watching somebody press a key teaches nothing about
    running a sweep. Ordered by how often you want them, and phrased by goal
    rather than by tool.
    """
    rows = [Command("run it", f"{RUNNER} python {DEMO_FILE}", "")]
    if step["sweep"]:
        rows += [
            Command("", f"{RUNNER} python {DEMO_FILE} --list",
                    "the parameter cases this step sweeps"),
            Command("", f"{RUNNER} python {DEMO_FILE} --case 4",
                    "run one of them, live"),
        ]
    if step["media"]:
        rows.append(Command("", f"{RUNNER} python {DEMO_FILE} --video {step['media']}",
                            "...and record it, into media/"))
    if step["sweep"]:
        rows += [
            # --tags is load-bearing: --include *adds* demo.py to the benchmarks
            # found through entry points, so without a filter the whole repo
            # comes along for the ride.
            Command("sweep it",
                    f"{RUNNER} railroad benchmarks run -i {DEMO_FILE} --tags tutorial "
                    f"--experiment {EXPERIMENT} --parallel {parallel}",
                    step["sweep"]),
            Command("see it", f"{RUNNER} railroad benchmarks dashboard",
                    "this playground's results only"),
        ]
    rows.append(Command("move on", f"{RUNNER} railroad tutorial next",
                        "shows the patch, then merges your edits"))
    return rows


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
