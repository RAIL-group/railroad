"""Tests for the guided tutorial.

The interesting surface is the advance machinery: it rewrites the one file a
presenter is live-editing, so "your edit survived" and "it refused rather than
mangling" are the properties that matter.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from railroad.tutorial import (
    STEPS,
    ENV_DIR,
    PlaygroundError,
    find_playground,
    get_step,
    init_playground,
    is_playground,
    neighbour,
    step_ids,
)
from railroad.tutorial import _advance as adv
from railroad.tutorial import _viewer as viewer
from railroad.tutorial import commands


def strip_ansi(text: str) -> str:
    """Rendered lines carry styling; length assertions want the text."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


SHIPPED_STEPS = Path(commands.__file__).parent / "steps"


def _procthor_available() -> bool:
    try:
        from railroad.environment.procthor import is_available
    except ImportError:
        return False
    return is_available()


def step_params():
    """One param per step, skipped or marked slow according to what it needs.

    The ProcTHOR steps load a scene and plan over several hundred grounded
    actions, so they are minutes rather than seconds.
    """
    params = []
    for step in STEPS:
        marks = []
        if step["requires"] == "procthor":
            marks.append(pytest.mark.slow)
            marks.append(
                pytest.mark.skipif(
                    not _procthor_available(),
                    reason="railroad[procthor] not installed",
                )
            )
        params.append(pytest.param(step, id=step["id"], marks=marks))
    return params


@pytest.fixture
def playground(tmp_path):
    return init_playground(tmp_path / "playground")


@pytest.fixture
def console():
    # record=True keeps the output out of the test log but lets us assert on it.
    return Console(record=True, width=100, force_terminal=False)


# -- the shipped snapshots ---------------------------------------------------


def test_every_step_snapshot_exists():
    for step in STEPS:
        assert (SHIPPED_STEPS / step["filename"]).is_file(), step["filename"]


def test_step_ids_are_ordered_and_unique():
    ids = step_ids()
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_get_step_accepts_unpadded_ids():
    assert get_step("2") == get_step("02")
    with pytest.raises(KeyError):
        get_step("99")


def test_neighbour_stops_at_both_ends():
    assert neighbour(step_ids()[0], -1) is None
    assert neighbour(step_ids()[-1], +1) is None
    following = neighbour(step_ids()[0], +1)
    assert following is not None and following["id"] == step_ids()[1]


@pytest.mark.parametrize("step", step_params())
def test_snapshot_registers_exactly_its_own_benchmark(step, monkeypatch):
    """Importing a snapshot must register its sweep and run nothing.

    Benchmark workers import these files via spec_from_file_location, so the
    ``__main__`` guard is what keeps a sweep from launching the demo.
    """
    from railroad.bench import registry
    from railroad.bench.discovery import load_benchmark_files

    monkeypatch.setattr(registry, "_BENCHMARKS", [])
    load_benchmark_files([str(SHIPPED_STEPS / step["filename"])])
    registered = list(registry._BENCHMARKS)

    assert len(registered) == 1
    assert registered[0].cases, "a sweep with no cases would silently run nothing"
    assert "tutorial" in registered[0].tags, (
        "the printed sweep command selects by the 'tutorial' tag; without it, "
        "--include would drag in every benchmark in the repo"
    )


# -- scaffolding -------------------------------------------------------------


def test_init_scaffolds_and_starts_on_the_first_step(playground):
    assert is_playground(playground.root)
    assert playground.current_step_id == STEPS[0]["id"]
    assert playground.demo.read_text() == playground.pristine_text(STEPS[0]["id"])
    for step in STEPS:
        assert (playground.steps_dir / step["filename"]).is_file()


def test_init_refuses_to_clobber_without_force(playground):
    with pytest.raises(PlaygroundError, match="already a tutorial playground"):
        init_playground(playground.root)


def test_find_playground_prefers_the_env_var(playground, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_DIR, str(playground.root))
    assert find_playground().root == playground.root


def test_a_playground_from_an_older_arc_says_so(playground):
    """Step 00 used to be a script; it is a notebook now, and its id is gone."""
    playground.write_state({"step": "00"})
    with pytest.raises(PlaygroundError, match="init --force"):
        playground.current_step_id


def test_find_playground_explains_itself_when_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ENV_DIR, raising=False)
    monkeypatch.setattr(sys, "argv", ["railroad"])
    with pytest.raises(PlaygroundError, match="railroad tutorial init"):
        find_playground()


# -- diffs -------------------------------------------------------------------


def test_consecutive_steps_actually_differ(playground):
    for step, following in zip(STEPS, STEPS[1:]):
        diff = adv.unified(
            playground.pristine_text(step["id"]),
            playground.pristine_text(following["id"]),
            "a", "b",
        )
        assert diff, f"step {following['id']} is identical to {step['id']}"


def test_first_changed_line_points_at_the_change():
    before = "a\nb\nc\n"
    after = "a\nB\nc\n"
    assert adv.first_changed_line(before, after) == 2
    assert adv.first_changed_line(before, before) == 1


def test_diff_stat_counts_both_directions():
    assert adv.diff_stat("a\nb\n", "a\nc\nd\n") == (2, 1)


def test_side_by_side_pairs_the_files_line_for_line():
    """Unchanged lines stay level; a changed run pads the shorter side."""
    rows = adv._aligned_rows(["a", "b", "c"], ["a", "B", "B2", "c"])
    assert rows == [
        ("equal", 0, 0),
        ("replace", 1, 1),
        ("replace", None, 2),
        ("equal", 2, 3),
    ]


def test_side_by_side_falls_back_on_a_narrow_terminal():
    """Half-width Python is worse to read than an ordinary unified diff."""
    from rich.table import Table

    wide = adv.side_by_side("a\n", "b\n", "before", "after", width=160)
    assert isinstance(wide, Table)
    narrow = adv.side_by_side("a\n", "b\n", "before", "after", width=60)
    assert not isinstance(narrow, Table)


def test_highlighting_keeps_one_line_per_line():
    """Alignment is computed from the plain text; the colours must agree."""
    source = '"""A docstring\nspanning lines."""\nx = 1\n\ndef f():\n    pass\n'
    assert len(adv._highlighted(source)) == len(source.splitlines())
    # A file the lexer cannot make sense of must not lose or gain lines either.
    broken = "def (((\nnot python at all\n"
    assert len(adv._highlighted(broken)) == len(broken.splitlines())


def test_highlighting_can_be_switched_off(monkeypatch):
    source = "def f():\n    return 1\n"
    coloured = adv._highlighted(source)
    assert any(line.spans for line in coloured), "expected pygments to colour this"

    monkeypatch.setenv(adv.SYNTAX_ENV, "off")
    plain = adv._highlighted(source)
    assert [line.plain for line in plain] == source.splitlines()
    assert not any(line.spans for line in plain)


def test_a_changed_line_keeps_its_syntax_colour():
    """The change is marked with a background, so the tokens survive it.

    Colouring the whole line red or green would throw the highlighting away
    exactly where the eye is being sent.
    """
    import io
    import re

    before = "x = 1\n"
    after = "def frobnicate():\n    return 1\n"
    buffer = io.StringIO()
    Console(file=buffer, width=160, force_terminal=True,
            color_system="truecolor").print(
        adv.side_by_side(before, after, "a", "b", width=160))
    rendered = buffer.getvalue()

    added = [line for line in rendered.splitlines() if "frobnicate" in line]
    assert added, "the added line should be on screen"
    codes = re.findall(r"\x1b\[([0-9;]*)m", added[0])
    background = adv.ADDED.removeprefix("on ").lstrip("#")
    tint = ";".join(str(int(background[i:i + 2], 16)) for i in (0, 2, 4))
    assert any(f"48;2;{tint}" in code for code in codes), "no added-line tint"
    assert any(re.match(r"^(3[0-7]|9[0-7])(;|$)", code) for code in codes), \
        "the line lost its syntax colour"


# -- the viewer -------------------------------------------------------------


def test_wheel_and_keys_map_to_scroll_actions():
    """The mouse wheel is the reason this exists rather than a call to less."""
    assert viewer.parse_key(b"\x1b[<64;10;20M") == "wheel-up"
    assert viewer.parse_key(b"\x1b[<65;10;20M") == "wheel-down"
    # Terminals that do not know the SGR encoding send the legacy form.
    assert viewer.parse_key(b"\x1b[M" + bytes([64 + 32, 33, 33])) == "wheel-up"
    assert viewer.parse_key(b"j") == "down"
    assert viewer.parse_key(b"\x1b[B") == "down"
    assert viewer.parse_key(b" ") == "page-down"
    assert viewer.parse_key(b"q") == "quit"
    assert viewer.parse_key(b"\x1b") == "quit"
    assert viewer.parse_key(b"z") == ""


def test_scrolling_stops_at_both_ends():
    """100 lines, a 10-line window: the last useful top is 90."""
    assert viewer.scroll_to(0, "up", 100, 10) == 0
    assert viewer.scroll_to(0, "down", 100, 10) == 1
    assert viewer.scroll_to(0, "wheel-down", 100, 10) == viewer.WHEEL_LINES
    assert viewer.scroll_to(0, "page-down", 100, 10) == 10
    assert viewer.scroll_to(95, "page-down", 100, 10) == 90
    assert viewer.scroll_to(50, "bottom", 100, 10) == 90
    assert viewer.scroll_to(50, "top", 100, 10) == 0
    # Shorter than the window: there is nowhere to go.
    assert viewer.scroll_to(0, "page-down", 5, 10) == 0


def test_scrolling_ignores_keys_it_does_not_know():
    assert viewer.scroll_to(7, "", 100, 10) == 7


def test_render_lines_lays_out_at_the_width_it_is_given():
    """Resizing re-renders, which is the whole reason this takes a callable."""
    console = Console(force_terminal=True, width=80)
    build = lambda width: Text("x" * (width - 1))
    assert len(strip_ansi(viewer.render_lines(console, build, 60)[0])) == 59
    assert len(strip_ansi(viewer.render_lines(console, build, 140)[0])) == 139


def test_rendered_rows_never_exceed_the_width_they_were_built_for(playground):
    """A row wider than the terminal wraps, and a wrapped row breaks alignment."""
    from rich.console import Group
    from rich.rule import Rule

    before = playground.pristine_text(STEPS[0]["id"])
    after = playground.pristine_text(STEPS[1]["id"])

    def build(width):
        return Group(Rule("patch"),
                     adv.side_by_side(before, after, "a", "b", width=width))

    console = Console(force_terminal=True, width=80)
    for width in (110, 150, 200):
        lines = viewer.render_lines(console, build, width)
        widest = max(len(strip_ansi(line)) for line in lines)
        assert widest <= width, f"{widest} > {width}"


def test_viewer_prints_inline_when_nobody_is_watching(console):
    """A recording console under test, or a pipe, must not take the screen."""
    assert not viewer.show(console, lambda width: Text(f"laid out at {width}"))
    assert f"laid out at {console.width}" in console.export_text()


def test_viewer_can_be_switched_off(monkeypatch):
    monkeypatch.setenv(viewer.VIEWER_ENV, "off")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    printing = Console(record=True, force_terminal=True, width=120)
    assert not viewer.show(printing, lambda _width: Text("printed inline"))
    assert "printed inline" in printing.export_text()


def test_content_that_already_fits_is_not_worth_a_takeover(monkeypatch):
    """It would vanish from the scrollback the moment you quit."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(viewer, "terminal_size",
                        lambda *a: os.terminal_size((120, 50)))
    printing = Console(record=True, force_terminal=True, width=120)
    assert not viewer.show(printing, lambda _width: Text("three\nshort\nlines"))
    assert "three" in printing.export_text()


# -- advancing ---------------------------------------------------------------


def test_advance_moves_the_file_and_the_recorded_step(playground, console):
    second = STEPS[1]["id"]
    assert commands.cmd_goto(console, second, playground=playground, force=True)
    assert playground.current_step_id == second
    assert playground.demo.read_text() == playground.pristine_text(second)


def test_merge_keeps_an_edit_the_patch_does_not_touch():
    # diff3 needs unchanged context between the two edits to merge them, which
    # is why this pads the middle: adjacent changes conflict even when they are
    # on different lines.
    filler = "".join(f"line {i}\n" for i in range(10))
    base = f"header = 1\n{filler}tuneme = 1\n"
    other = f"header = 2\n{filler}tuneme = 1\n"    # the step's patch
    current = f"header = 1\n{filler}tuneme = 99\n"  # the presenter's live edit
    merged, conflicts = adv.merge_three_way(
        current, base, other, labels=("mine", "base", "theirs")
    )
    assert conflicts == 0
    assert merged == f"header = 2\n{filler}tuneme = 99\n"


def test_merge_reports_a_conflict_on_the_same_line():
    base = "alpha\n"
    merged, conflicts = adv.merge_three_way(
        "mine\n", base, "theirs\n", labels=("mine", "base", "theirs")
    )
    assert conflicts > 0
    assert "<<<<<<<" in merged


def test_advance_preserves_a_live_edit(playground, console):
    """The whole point: a value tuned mid-talk survives moving on."""
    first, second = STEPS[0]["id"], STEPS[1]["id"]
    assert playground.current_step_id == first

    # A constant both snapshots share, well away from anything the patch edits.
    knob = "PICK_TIME = 5.0"
    assert knob in playground.pristine_text(first), "test needs a shared constant"
    assert knob in playground.pristine_text(second), "test needs a shared constant"
    playground.demo.write_text(
        playground.demo.read_text().replace(knob, "PICK_TIME = 7.0")
    )

    assert commands.cmd_goto(
        console, second, playground=playground,
        ask=lambda _prompt: "yes", editor_sync=False,
    )
    result = playground.demo.read_text()
    assert "PICK_TIME = 7.0" in result, "the live edit was lost"
    assert playground.current_step_id == second
    # The step's own change landed alongside it.
    only_in_second = '~F("just-picked ?r ?obj")'
    assert only_in_second not in playground.pristine_text(first)
    assert only_in_second in playground.pristine_text(second)
    assert only_in_second in result


def test_advance_refuses_a_conflict_rather_than_writing_markers(playground, console):
    """A file with conflict markers in it would fail to run mid-demo."""
    second = STEPS[1]["id"]
    # Overwrite the first line, which every step rewrites -- a guaranteed clash.
    original = playground.demo.read_text()
    playground.demo.write_text('"""Mine.\n' + original.split("\n", 1)[1])

    moved = commands.cmd_goto(
        console, second, playground=playground,
        ask=lambda _prompt: "yes", editor_sync=False,
    )
    assert not moved
    assert playground.current_step_id == STEPS[0]["id"]
    assert "<<<<<<<" not in playground.demo.read_text()
    assert "conflicting hunk" in console.export_text()


def test_force_takes_the_pristine_version_over_a_conflict(playground, console):
    second = STEPS[1]["id"]
    playground.demo.write_text('"""Mine.\n')
    assert commands.cmd_goto(
        console, second, playground=playground, force=True, editor_sync=False
    )
    assert playground.demo.read_text() == playground.pristine_text(second)


def test_declining_leaves_everything_alone(playground, console):
    before = playground.demo.read_text()
    moved = commands.cmd_goto(
        console, STEPS[1]["id"], playground=playground,
        ask=lambda _prompt: "no", editor_sync=False,
    )
    assert not moved
    assert playground.demo.read_text() == before
    assert playground.current_step_id == STEPS[0]["id"]


def test_undo_restores_the_file_and_the_step(playground, console):
    first, second = STEPS[0]["id"], STEPS[1]["id"]
    edited = playground.demo.read_text() + "\n# tuned live\n"
    playground.demo.write_text(edited)
    commands.cmd_goto(console, second, playground=playground, force=True,
                      editor_sync=False)

    commands.cmd_undo(console, playground)
    assert playground.current_step_id == first
    assert playground.demo.read_text() == edited


def test_undo_with_no_history_says_so(playground, console):
    commands.cmd_undo(console, playground)
    assert "nothing to undo" in console.export_text()


def test_clean_restores_this_step_without_moving_off_it(playground, console):
    """The way back from a live edit that went somewhere you did not mean."""
    first = STEPS[0]["id"]
    pristine = playground.demo.read_text()
    playground.demo.write_text(pristine.replace("MAX_STEPS = 40", "MAX_STEPS = 4"))

    assert commands.cmd_clean(console, playground, editor_sync=False)
    assert playground.demo.read_text() == pristine
    assert playground.current_step_id == first, "clean is not a step change"


def test_clean_lands_on_the_step_you_are_on_not_the_first_one(playground, console):
    second = STEPS[1]["id"]
    commands.cmd_goto(Console(quiet=True), second, playground=playground,
                      force=True, editor_sync=False)
    playground.demo.write_text(playground.demo.read_text() + "\n# tuned live\n")

    commands.cmd_clean(console, playground, editor_sync=False)
    assert playground.demo.read_text() == playground.pristine_text(second)
    assert playground.current_step_id == second


def test_clean_is_recoverable_with_undo(playground, console):
    """Throwing away an edit must itself be undoable, or nobody will type it."""
    edited = playground.demo.read_text() + "\n# tuned live\n"
    playground.demo.write_text(edited)

    commands.cmd_clean(console, playground, editor_sync=False)
    assert playground.demo.read_text() != edited

    commands.cmd_undo(console, playground)
    assert playground.demo.read_text() == edited


def _accumulate_results(playground, *, runs=2, media=1):
    """Whatever a sweep and a couple of live runs would have left behind."""
    playground.mlflow_db.write_text("not really sqlite")
    for index in range(runs):
        (playground.mlruns_dir / "1" / f"run{index}" / "artifacts").mkdir(parents=True)
    (playground.cache_dir / "railroad-tutorial").mkdir(parents=True)
    playground.media_dir.mkdir(exist_ok=True)
    for index in range(media):
        (playground.media_dir / f"plot{index}.png").write_text("png")
    playground.append_run({"step": "01", "cost": 38.0, "goal_reached": True})


def test_reset_clears_the_results_and_leaves_the_tutorial_alone(playground, console):
    _accumulate_results(playground)
    demo = playground.demo.read_text()

    assert commands.cmd_reset(console, playground, force=True)
    assert not playground.mlflow_db.exists()
    assert not playground.mlruns_dir.exists()
    assert not playground.cache_dir.exists()
    assert list(playground.media_dir.glob("*")) == []
    assert playground.read_runs() == []
    # The tutorial itself is not results, and is none of reset's business.
    assert playground.demo.read_text() == demo
    assert playground.current_step_id == STEPS[0]["id"]
    for step in STEPS:
        assert (playground.steps_dir / step["filename"]).is_file()


def test_reset_says_what_it_is_about_to_throw_away(playground, console):
    _accumulate_results(playground, runs=3, media=2)
    commands.cmd_reset(console, playground, force=True)
    text = console.export_text()
    assert "3 sweep runs" in text
    assert "1 recorded costs" in text
    assert "2 files in media/" in text


def test_reset_declined_changes_nothing(playground, console):
    _accumulate_results(playground)
    assert not commands.cmd_reset(
        console, playground, ask=lambda _prompt: "no"
    )
    assert playground.mlflow_db.exists()
    assert playground.read_runs()
    assert "nothing cleared" in console.export_text()


def test_reset_with_nothing_to_clear_says_so(playground, console):
    assert not commands.cmd_reset(console, playground, force=True)
    assert "no results to clear" in console.export_text()


def test_clean_on_an_untouched_file_is_a_message_not_a_write(playground, console):
    assert not commands.cmd_clean(console, playground, editor_sync=False)
    assert "already" in console.export_text()
    # Nothing was snapshotted, so 'undo' has nothing to walk back to.
    commands.cmd_undo(Console(quiet=True), playground)
    assert playground.demo.read_text() == playground.pristine_text(STEPS[0]["id"])


def test_stepping_past_the_end_is_a_message_not_a_crash(playground, console):
    commands.cmd_goto(console, STEPS[-1]["id"], playground=playground, force=True)
    assert not commands.cmd_step(console, +1, playground=playground, force=True)
    assert "last step" in console.export_text()


# -- running the demos -------------------------------------------------------


@pytest.mark.parametrize("step", step_params())
def test_step_runs_headless_and_files_a_record(step, playground, tmp_path):
    """Each snapshot must run to completion outside a terminal.

    CI=1 forces PlannerDashboard's non-interactive path, which is what a
    rehearsal over ssh or a piped run gets.
    """
    commands.cmd_goto(Console(quiet=True), step["id"], playground=playground,
                      force=True, editor_sync=False)
    env = dict(os.environ)
    env.update({ENV_DIR: str(playground.root), "CI": "1",
                "MPLCONFIGDIR": str(tmp_path / "mpl")})
    completed = subprocess.run(
        [sys.executable, str(playground.demo)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]

    records = playground.read_runs()
    assert records and records[-1]["step"] == step["id"]
    assert records[-1]["goal_reached"] is True
    assert records[-1]["cost"] > 0


def test_report_records_are_valid_json_lines(playground):
    playground.append_run({"step": "01", "cost": 1.0, "actions": []})
    playground.append_run({"step": "02", "cost": 2.0, "actions": []})
    lines = playground.runs_path.read_text().strip().splitlines()
    assert [json.loads(line)["step"] for line in lines] == ["01", "02"]


def test_read_runs_skips_a_truncated_line(playground):
    playground.append_run({"step": "01", "cost": 1.0, "actions": []})
    with playground.runs_path.open("a") as handle:
        handle.write('{"step": "02", "cos')  # interrupted mid-write
    assert [record["step"] for record in playground.read_runs()] == ["01"]


def test_doctor_does_not_let_rich_eat_the_extra_names(console, monkeypatch, tmp_path):
    """'railroad[bench]' reads as a style tag unless it is escaped."""
    monkeypatch.chdir(tmp_path)
    commands.cmd_doctor(console)
    text = console.export_text()
    assert "railroad[bench]" in text
    assert "railroad[procthor]" in text


def test_step_list_reports_the_change_between_steps(playground, console):
    playground.append_run({"step": "01", "cost": 44.3, "actions": 1,
                           "goal_reached": True, "wall": 0.6})
    playground.append_run({"step": "02", "cost": 22.1, "actions": 2,
                           "goal_reached": True, "wall": 1.4})
    commands.cmd_steps(console, playground)
    text = console.export_text()
    assert "44.3" in text and "22.1" in text and "-22.2" in text


def test_step_list_omits_the_delta_across_a_change_of_problem(playground, console):
    """Costs from different worlds are not comparable, so do not subtract them."""
    pairs = [(a, b) for a, b in zip(STEPS, STEPS[1:])
             if a["problem"] and a["problem"] != b["problem"]]
    assert pairs, "expected at least one problem boundary in the arc"
    before, after = pairs[0]
    playground.append_run({"step": before["id"], "cost": 10.0, "goal_reached": True})
    playground.append_run({"step": after["id"], "cost": 40.0, "goal_reached": True})
    commands.cmd_steps(console, playground)
    text = console.export_text()
    assert "40.0" in text
    assert "+30.0" not in text


# -- the card: real commands, not a wrapper ----------------------------------


def test_card_shows_every_way_to_run_this_step(playground, console):
    commands.cmd_goto(Console(quiet=True), "02", playground=playground,
                      force=True, editor_sync=False)
    commands.cmd_card(console, playground)
    text = console.export_text()
    for expected in ("tutorial run", "tutorial run --list", "tutorial run --case",
                     "tutorial run --save-plot", "tutorial run --save-video",
                     "tutorial bench", "tutorial dashboard", "tutorial next"):
        assert expected in text, expected
    assert "--parallel" not in text, "auto-detect is enough; do not pin it"


def test_card_flags_local_edits(playground, console):
    playground.demo.write_text(playground.demo.read_text() + "\n# tuned live\n")
    commands.cmd_card(console, playground)
    assert "local edits" in console.export_text()


def test_every_printed_command_runs_through_uv():
    """A card that prints something unrunnable is worse than no card.

    This repository's interpreter lives in a uv-managed environment, so a bare
    `python demo.py` is not a command anyone here can paste.
    """
    from railroad.tutorial import RUNNER, command_lines

    for step in STEPS:
        for row in command_lines(step):
            assert row.command.startswith(f"{RUNNER} "), row.command


@pytest.mark.parametrize("step", step_params())
def test_every_case_the_card_prints_actually_exists(step, monkeypatch):
    """A card that prints '--case 4' for a step with four cases is lying.

    demo.py exits with 'must be between 0 and N-1', mid-talk, on a command the
    tutorial itself told you to type.
    """
    import re

    from railroad.bench import registry
    from railroad.bench.discovery import load_benchmark_files
    from railroad.tutorial import command_lines

    monkeypatch.setattr(registry, "_BENCHMARKS", [])
    load_benchmark_files([str(SHIPPED_STEPS / step["filename"])])
    count = len(registry._BENCHMARKS[0].cases)

    for row in command_lines(step):
        found = re.search(r"--case (\d+)", row.command)
        if found:
            index = int(found.group(1))
            assert index < count, (
                f"step {step['id']} prints '{row.command}' but has {count} case(s)"
            )


def test_bench_fills_in_the_include_tag_and_experiment(playground):
    from railroad.tutorial import EXPERIMENT
    from railroad.tutorial import _launch as launch

    argv = launch.bench_argv(["--parallel", "8"])
    assert argv[argv.index("-i") + 1] == "demo.py"
    # --include *adds* to entry-point discovery, so the tag filter is what keeps
    # the whole repo's benchmarks out of a live sweep.
    assert argv[argv.index("--tags") + 1] == "tutorial"
    assert argv[argv.index("--experiment") + 1] == EXPERIMENT
    assert argv[-2:] == ["--parallel", "8"], "extra arguments pass through"


def test_printed_commands_are_typeable(playground):
    """A wrapper that hides what it runs teaches the wrapper, not the tool."""
    from railroad.tutorial import _launch as launch

    assert launch.pretty(launch.demo_argv(["--case", "4"])) == (
        "uv run python demo.py --case 4"
    )
    assert launch.pretty(launch.bench_argv()).startswith(
        "uv run railroad benchmarks run -i demo.py"
    )
    assert launch.pretty(launch.notebook_argv()).startswith(
        "uv run jupyter notebook language.ipynb"
    )
    assert sys.executable not in launch.pretty(launch.dashboard_argv())


def test_only_the_first_step_points_back_at_the_notebook(playground):
    """The primer is where you start, and after that it is behind you."""
    from railroad.tutorial import command_lines

    first = [row.command for row in command_lines(get_step(STEPS[0]["id"]))]
    assert any(command.endswith("tutorial notebook") for command in first)
    for step in STEPS[1:]:
        commands_here = [row.command for row in command_lines(step)]
        assert not any("notebook" in command for command in commands_here), step["id"]


def test_list_names_every_step_and_marks_the_one_you_are_on(playground, console):
    """The glance-at-it command: the whole arc, and where you are in it."""
    commands.cmd_goto(Console(quiet=True), "03", playground=playground,
                      force=True, editor_sync=False)
    commands.cmd_list(console, playground)
    text = console.export_text()

    for step in STEPS:
        assert step["title"] in text, step["id"]
    marked = [line for line in text.splitlines() if line.strip().startswith(">")]
    assert len(marked) == 1, "exactly one step is the current one"
    assert "clear the table" in marked[0]
    # Shorter than 'steps' on purpose: no costs, no deltas, no sweep axes.
    assert all(step["sweep"] not in text for step in STEPS)


def test_notes_are_separate_from_the_card(playground, console):
    """Diataxis, in one assertion: how-to and explanation are different pages."""
    commands.cmd_notes(console, "05", playground)
    text = console.export_text()
    assert "lock-search" in text
    assert "tutorial run" not in text


def test_every_step_sweeps_and_names_its_media():
    """Every step of the arc is a program with a sweep; the primer is a notebook.

    The card prints --save-plot and 'bench' lines unconditionally, which is
    only honest while this holds.
    """
    for step in STEPS:
        assert step["sweep"], f"step {step['id']} has nothing to sweep"
        assert step["media"], f"step {step['id']} has nothing to save as"


# -- the language primer, which is a notebook rather than a step -------------


SHIPPED_NOTEBOOK = Path(commands.__file__).parent / "language.ipynb"


def _code_cells(path):
    return [cell for cell in json.loads(path.read_text())["cells"]
            if cell["cell_type"] == "code"]


def test_the_notebook_runs_top_to_bottom():
    """It is the first thing anyone sees, so a stale cell is a broken talk.

    Executing the source is the whole check: the cells build states and
    transition them, and any drift in the core API shows up as an exception
    here rather than in front of a room.
    """
    namespace: dict = {}
    cells = _code_cells(SHIPPED_NOTEBOOK)
    for cell in cells:
        source = "".join(cell["source"])
        exec(compile(source, f"language.ipynb::{cell['id']}", "exec"), namespace)
    assert len(cells) >= 5

    # The prose around these two cells is the headline claim of the whole
    # tutorial: dispatching does not advance the clock, and the world only
    # moves once nobody is free. If that stops holding, the notebook lies.
    assert namespace["after_r1"].time == 0.0
    assert namespace["after_r2"].time == 5.0


def test_the_notebook_is_valid_against_the_nbformat_schema():
    """Structurally openable, not just parseable as JSON."""
    nbformat = pytest.importorskip("nbformat")  # only with railroad[tutorial]
    nbformat.validate(nbformat.read(str(SHIPPED_NOTEBOOK), as_version=4))


def test_the_notebook_ships_with_no_stored_output():
    """A talk starts from an empty notebook, not from last time's numbers."""
    for cell in _code_cells(SHIPPED_NOTEBOOK):
        assert cell["outputs"] == [], cell["id"]
        assert cell["execution_count"] is None, cell["id"]


def test_init_gives_you_your_own_copy_of_the_notebook(playground):
    assert playground.notebook.is_file()
    assert playground.notebook.read_bytes() == SHIPPED_NOTEBOOK.read_bytes()


def test_reinit_keeps_the_notebook_you_scribbled_in(playground):
    """--force resets demo.py, which has a step to be restored from. This has not."""
    playground.notebook.write_text('{"cells": [], "mine": true}')
    init_playground(playground.root, force=True)

    kept = list(playground.history_dir.glob("*-language.ipynb"))
    assert len(kept) == 1
    assert '"mine": true' in kept[0].read_text()
    assert playground.notebook.read_bytes() == SHIPPED_NOTEBOOK.read_bytes()


def test_the_card_starts_you_in_the_notebook(playground, console):
    commands.cmd_card(console, playground)
    assert "tutorial notebook" in console.export_text()


def test_notebook_command_shows_the_jupyter_command_it_runs(
    playground, console, monkeypatch
):
    from railroad.tutorial import _launch as launch

    seen = []
    monkeypatch.setattr(commands, "find_spec", lambda name: object())
    monkeypatch.setattr(
        launch, "run",
        lambda pg, argv: seen.append(list(argv)) or launch.RunResult(0),
    )
    assert commands.cmd_notebook(console, ["--ServerApp.terminals_enabled=False"],
                                 playground).ok
    assert seen[0][-1] == "--ServerApp.terminals_enabled=False", "arguments pass through"
    text = console.export_text()
    assert "uv run jupyter notebook language.ipynb" in text
    assert "http://127.0.0.1:8888/" in text, "the address to open is printed"


def test_the_notebook_is_started_for_the_machine_you_are_not_sitting_at():
    """Started over ssh, viewed from a laptop: no browser here, no token to copy."""
    from railroad.tutorial import _launch as launch

    argv = launch.notebook_argv()
    assert "--no-browser" in argv
    assert "--IdentityProvider.token=" in argv
    assert argv[argv.index("--ip") + 1] == "0.0.0.0"
    # Notebook 7's one-document interface, opened at the document.
    assert argv[:2] != ["jupyter", "lab"]
    # default_url is both what the interface opens and what '/' redirects to
    # (jupyter_server registers that 302 whenever it differs from base_url),
    # which is what lets the printed links stay bare addresses.
    assert f"--JupyterNotebookApp.default_url={launch.NOTEBOOK_URL_PATH}" in argv
    printed = [line.split()[0] for line in launch.urls(8888)]
    assert printed and all(link.endswith(":8888/") for link in printed), printed


def test_an_argument_you_pass_replaces_the_default_rather_than_repeating_it():
    """traitlets rejects a repeated argument outright; last-one-wins is not a thing."""
    from railroad.tutorial import _launch as launch

    argv = launch.notebook_argv(["--port", "8899"])
    assert argv.count("--port") == 1 and "8888" not in argv

    argv = launch.notebook_argv(["--ip=127.0.0.1"])
    assert "--ip" not in argv and "--ip=127.0.0.1" in argv

    argv = launch.notebook_argv(["--IdentityProvider.token=secret"])
    assert "--IdentityProvider.token=" not in argv


def test_notebook_command_says_how_to_get_jupyter(playground, console, monkeypatch):
    """jupyter is an extra, so its absence has to be a sentence, not a traceback."""
    monkeypatch.setattr(commands, "find_spec", lambda name: None)
    assert not commands.cmd_notebook(console, (), playground).ok
    text = console.export_text()
    assert "railroad[tutorial]" in text
    assert "uv run --with notebook" in text


# -- the dashboard, which outlives the command that starts it ----------------


def test_dashboard_reports_nothing_running_at_first(playground, console):
    assert not commands.cmd_dashboard(console, playground=playground, status=True)
    assert "no dashboard" in console.export_text()


def test_stopping_a_dashboard_that_is_not_running_is_not_an_error(playground, console):
    assert not commands.cmd_dashboard(console, playground=playground, stop=True)
    assert "no dashboard" in console.export_text()


def test_a_stale_record_is_cleared_rather_than_believed(playground):
    """A pid from a previous boot must not stop you starting a new dashboard."""
    from railroad.tutorial import _launch as launch

    # A pid that cannot be running: 0 is never a user process here.
    (playground.root / launch.DASHBOARD_STATE).write_text(
        json.dumps({"pid": 2 ** 31 - 1, "port": 8050, "host": "auto"})
    )
    assert launch.recorded(playground) is None
    assert not (playground.root / launch.DASHBOARD_STATE).exists()


def test_a_corrupt_record_is_cleared_too(playground):
    from railroad.tutorial import _launch as launch

    (playground.root / launch.DASHBOARD_STATE).write_text("{ half-written")
    assert launch.recorded(playground) is None


# -- running one case by hand ------------------------------------------------


def _fake_benchmark(cases, seen):
    from railroad.bench.registry import Benchmark

    def fn(case):
        seen.append(case)
        return {"success": True, "plan_cost": 12.5, "wall_time": 0.25,
                "actions_count": 7}

    bench = Benchmark(fn=fn, name="demo::s04_two_robots")
    bench.add_cases(cases)
    return bench


def test_main_runs_the_case_you_asked_for(playground, monkeypatch):
    from railroad import tutorial

    monkeypatch.chdir(playground.root)
    seen = []
    bench = _fake_benchmark([{"num_robots": 2}, {"num_robots": 1},
                             {"num_robots": 3}], seen)
    tutorial.main(bench, ["--case", "2"])

    assert len(seen) == 1
    assert seen[0].case_idx == 2
    assert seen[0].num_robots == 3
    # `live` is what tells the dashboard somebody is watching.
    assert seen[0].live is True


def test_main_defaults_to_case_zero(playground, monkeypatch):
    from railroad import tutorial

    monkeypatch.chdir(playground.root)
    seen = []
    tutorial.main(_fake_benchmark([{"num_robots": 2}, {"num_robots": 1}], seen), [])
    assert seen[0].num_robots == 2, "case 0 is the configuration the step is about"


def test_main_rejects_a_case_that_does_not_exist(playground, monkeypatch):
    from railroad import tutorial

    monkeypatch.chdir(playground.root)
    with pytest.raises(SystemExit):
        tutorial.main(_fake_benchmark([{"num_robots": 2}], []), ["--case", "9"])


def test_main_lists_the_cases_without_running_them(playground, monkeypatch, capsys):
    from railroad import tutorial

    monkeypatch.chdir(playground.root)
    seen = []
    tutorial.main(_fake_benchmark([{"num_robots": 2}, {"num_robots": 1}], seen),
                  ["--list"])
    assert seen == []
    out = capsys.readouterr().out
    assert "num_robots=2" in out and "num_robots=1" in out


def test_main_files_a_record_under_the_step_the_benchmark_names(playground, monkeypatch):
    from railroad import tutorial

    monkeypatch.chdir(playground.root)
    tutorial.main(_fake_benchmark([{"num_robots": 2}], []), [])
    records = playground.read_runs()
    assert records[-1]["step"] == "04", "the step id comes from the benchmark name"
    assert records[-1]["cost"] == 12.5
    assert records[-1]["goal_reached"] is True


def test_bare_media_names_land_where_the_dashboard_serves_them(playground, monkeypatch):
    from railroad import tutorial
    from railroad.bench.dashboard import media

    monkeypatch.chdir(playground.root)
    seen = []
    tutorial.main(_fake_benchmark([{"num_robots": 2}], seen),
                  ["--save-video", "house.mp4", "--save-plot", "/tmp/elsewhere.png"])
    assert seen[0].media["save_video"] == "media/house.mp4"
    assert seen[0].media["save_plot"] == "/tmp/elsewhere.png", "a path is left alone"
    assert media.media_dir() == playground.media_dir


def test_the_demo_and_the_examples_offer_the_same_media_flags():
    """One spelling across the tool: --save-video here is --save-video there.

    Both sides are checked against the shared declaration rather than against
    each other, so this fails if either stops borrowing it.
    """
    from railroad.cli import main as cli
    from railroad.dashboard import MEDIA_OPTION_NAMES
    from railroad.tutorial._harness import _parse

    example_group = cli.commands["example"]
    example = example_group.commands["clear-table"]  # ty: ignore[unresolved-attribute]
    example_flags = {flag for param in example.params for flag in param.opts}
    assert set(MEDIA_OPTION_NAMES) <= example_flags

    args = _parse(["--save-video", "x.mp4", "--save-plot", "x.png",
                   "--video-dpi", "80", "--show-plot"], [{}])
    assert args.save_video == "x.mp4"
    assert args.save_plot == "x.png"
    assert args.video_dpi == 80
    assert args.show_plot is True
    assert args.video_fps == 60, "the base default, inherited rather than restated"


def test_only_the_media_options_you_asked_for_are_passed_on():
    """An untouched flag must not override what a step passes to show_plots."""
    from railroad.dashboard import media_kwargs

    assert media_kwargs({"save_plot": None, "show_plot": False,
                         "save_video": None, "video_fps": 60,
                         "video_dpi": 150}) == {}
    asked = media_kwargs({"save_video": "out.mp4", "video_dpi": 80,
                          "video_fps": 60})
    assert asked == {"save_video": "out.mp4", "video_dpi": 80}


def test_media_paths_can_be_relocated_but_flags_cannot():
    from railroad.dashboard import media_kwargs

    got = media_kwargs({"save_video": "x.mp4", "video_dpi": 80},
                       relocate=lambda name: f"media/{name}")
    assert got == {"save_video": "media/x.mp4", "video_dpi": 80}


# -- isolation ---------------------------------------------------------------


def test_init_links_the_resources_and_makes_a_media_dir(tmp_path, monkeypatch):
    """The playground is self-contained; only the ProcTHOR scenes are borrowed."""
    home = tmp_path / "home"
    (home / "resources" / "procthor-10k").mkdir(parents=True)
    monkeypatch.chdir(home)

    playground = init_playground(home / "railroad-tutorial")
    assert playground.media_dir.is_dir()
    assert playground.resources_dir.is_symlink()
    assert (playground.resources_dir / "procthor-10k").is_dir()


def test_init_survives_having_no_resources_to_link(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    playground = init_playground(tmp_path / "railroad-tutorial")
    assert not playground.resources_dir.exists()


def test_every_step_declares_a_problem():
    for step in STEPS:
        assert step["problem"], f"step {step['id']} needs a problem tag"
