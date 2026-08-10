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

    if not step["sweep"]:
        assert registered == []
        return
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
    first, second = STEPS[1]["id"], STEPS[2]["id"]
    commands.cmd_goto(console, first, playground=playground, force=True,
                      editor_sync=False)

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
    if step["id"] == STEPS[0]["id"]:
        return  # the language primer has no dashboard, so nothing to file
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


def test_card_prints_commands_you_could_type_yourself(playground, console):
    commands.cmd_goto(Console(quiet=True), "02", playground=playground,
                      force=True, editor_sync=False)
    commands.cmd_card(console, playground)
    text = console.export_text()
    assert "uv run python demo.py" in text
    assert "uv run railroad benchmarks run -i demo.py" in text
    assert "uv run railroad benchmarks dashboard" in text
    assert "uv run railroad tutorial next" in text
    assert "railroad tutorial run" not in text, "the wrappers are gone"


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


def test_sweep_command_selects_only_the_tutorial(playground):
    from railroad.tutorial import EXPERIMENT, command_lines

    sweeps = [row.command for row in command_lines(get_step("02"))
              if "railroad benchmarks run" in row.command]
    assert len(sweeps) == 1
    # --include *adds* to entry-point discovery, so the tag filter is what keeps
    # the whole repo's benchmarks out of a live sweep.
    assert "--tags tutorial" in sweeps[0]
    assert f"--experiment {EXPERIMENT}" in sweeps[0]


def test_the_primer_offers_no_sweep(playground):
    from railroad.tutorial import command_lines

    commands_for_primer = command_lines(get_step(STEPS[0]["id"]))
    assert not any("benchmarks" in row.command for row in commands_for_primer)
    assert any(row.command == "uv run python demo.py"
               for row in commands_for_primer)


def test_notes_are_separate_from_the_card(playground, console):
    """Diataxis, in one assertion: how-to and explanation are different pages."""
    commands.cmd_notes(console, "04", playground)
    text = console.export_text()
    assert "lock-search" in text
    assert "demo.py" not in text


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
                  ["--video", "house.mp4", "--plot", "/tmp/elsewhere.png"])
    assert seen[0].video == "media/house.mp4"
    assert seen[0].plot == "/tmp/elsewhere.png", "an explicit path is left alone"
    assert media.media_dir() == playground.media_dir


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


def test_every_step_declares_a_problem_except_the_primer():
    for step in STEPS:
        if step["sweep"]:
            assert step["problem"], f"step {step['id']} needs a problem tag"
