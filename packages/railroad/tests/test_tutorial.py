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
from railroad.tutorial import commands


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
    only_in_second = "NUM_ROBOTS = 2"
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


def test_notes_are_separate_from_the_card(playground, console):
    """Diataxis, in one assertion: how-to and explanation are different pages."""
    commands.cmd_notes(console, "04", playground)
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
    assert "/notebooks/language.ipynb" in text, "the link opens the notebook itself"


def test_the_notebook_is_started_for_the_machine_you_are_not_sitting_at():
    """Started over ssh, viewed from a laptop: no browser here, no token to copy."""
    from railroad.tutorial import _launch as launch

    argv = launch.notebook_argv()
    assert "--no-browser" in argv
    assert "--IdentityProvider.token=" in argv
    assert argv[argv.index("--ip") + 1] == "0.0.0.0"
    # Notebook 7's one-document interface, opened at the document.
    assert argv[:2] != ["jupyter", "lab"]
    assert f"--JupyterNotebookApp.default_url={launch.NOTEBOOK_URL_PATH}" in argv
    assert all(launch.NOTEBOOK_URL_PATH in line for line in launch.notebook_urls())


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

    bench = Benchmark(fn=fn, name="demo::s02_two_robots")
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
    assert records[-1]["step"] == "02", "the step id comes from the benchmark name"
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
