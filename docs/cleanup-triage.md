# Test & Code Cleanup Triage

Working document for branch `gjstein/test-triage-cleanup`. Each item is a checkbox: tick to
approve, strike through to reject, annotate to amend. Nothing is executed until you mark it up.

**Verification legend**

| Tag | Meaning |
|---|---|
| **[V]** | Verified by hand on this branch — evidence quoted in the entry |
| **[R]** | Reported by exploration, **not yet re-verified**. Verify before acting |
| **[C]** | Correction — an exploration finding that was wrong; recorded so it is not re-proposed |

Baseline (this machine, `-n auto`, 12 workers): **765 collected / 763 pass, 1 skip, 1 xfail ·
112 warnings · 50–69 s full · 12.9 s `-m 'not slow'` · 19 `slow`-marked · ~18.6k test lines**.

---

## Tranche 0 — Tooling (done / decided)

- [x] **X-01 `--dist worksteal`** — applied to `pyproject.toml` addopts. **[V]**
  Measured over repeated full runs: `load` 50.4 s / 68.6 s (spread **18.2 s**); `worksteal`
  55.2 / 54.3 / 53.8 s (spread **1.4 s**). Slightly worse best case, far better worst case and
  far more predictable. Cause: the 25.6 s ProcTHOR test gets dealt to a worker late under
  `load`'s up-front slicing, leaving workers idle. Re-verified after the edit: 763 passed, 50.3 s.

- [x] **X-02 Rich terminal reporter — `pytest-richer` + warnings-as-errors.** **[V]**
  **DONE.** The requirement is the live progress *wall* during xdist runs, not a restyled
  summary. Only `pytest-richer` provides it, and it does work under xdist
  (`@main_process_only` throughout, `PYTEST_XDIST_WORKER` handled explicitly).

  Its one flaw is real and comes from its own source, `pytest_richer/terminal.py:759`:
  ```python
  def pytest_warning_recorded(self, warning_message, nodeid) -> None:
      """Note standard Python warning.

      Currently we are just dropping warnings.
      """
  ```
  **This is neutralised by `filterwarnings = ["error"]` (X-05).** A warning that is an error is
  never *just* a warning: it fails the run and is reported in full, with source context. Verified
  end to end — `--rich -W error` on a warning-emitting test prints
  `DeprecationWarning: THIS-WARNING-MUST-BE-VISIBLE` with a code frame and exits 1.

  `--rich` is safe in `addopts` because the plugin guards on `sys.stdout.isatty()`: in CI it is
  inert and the standard reporter (with its warnings summary) takes over. Confirmed both ways.
  `-v` was dropped from `addopts` — the per-test scroll it produced is exactly what the wall
  replaces.

  <details><summary>Two corrections to my own earlier testing</summary>

  1. My first "empirical" check of `pytest-richer` used `--richer`; the real flag is `--rich`, so
     the plugin never activated and the run died on an unrecognised argument. The "0 warnings" I
     reported was that usage error, not evidence. Re-tested in a clean venv: the conclusion holds,
     but it now rests on a valid measurement.
  2. A follow-up test was polluted by having `pytest-rich` *and* `pytest-richer` installed
     together — they both register `--rich` and collide with
     `ValueError: option names {'--rich'} already added`.

  `pytest-pretty` (my earlier recommendation) is removed: it preserves warnings but has no
  progress wall, so it did not meet the actual requirement. `pytest-rich` remains unusable —
  proof-of-concept, stubs the same hook, and crashes under xdist.
  </details>

- [x] **X-03 Two competing pytest configs.** **[V]**
  **DONE.** Removed `[tool.pytest.ini_options]` from `packages/railroad/pyproject.toml`; the root
  config is now the single source of truth. A subdirectory invocation that previously reported
  `rootdir: .../packages/railroad` and ran serially now reports the repo root and spins up 12
  workers. Checked that the one behaviour this could change — a bare `pytest` from inside
  `packages/railroad` — was already broken before the change (that directory's venv has no
  `railroad` installed), so nothing regressed.
  <details><summary>Original finding</summary>

  Root `pyproject.toml:38` and `packages/railroad/pyproject.toml:71` both define
  `[tool.pytest.ini_options]`. The nested one has **no `addopts`**, so any invocation whose
  rootdir resolves to `packages/railroad` silently loses `-n auto`, `--dist worksteal`, and the
  HTML report. Observed live during this investigation: running a single test file printed
  `rootdir: .../packages/railroad` and ran serially. Consolidate to one config.
  </details>

---

- [x] **X-04 An obsolete `filterwarnings` ignore was hiding SyntaxWarnings.** **[V]**
  **DONE.** `filterwarnings = ["ignore:invalid escape sequence:SyntaxWarning"]` sat in *both*
  pytest configs. Compiling all 266 tracked project files with warnings forced on produced **zero**
  invalid-escape warnings — the cause had been fixed at some point and the suppression outlived
  it. Removed. Without it the suite still reports 0 warnings, so "0" is now a real 0 rather than
  a filtered one.

- [x] **X-05 `filterwarnings = ["error"]`.** **[V]**
  **DONE.** Warnings are now errors. Verified the suite is clean under it across five runs, and
  that an injected warning fails with
  `E DeprecationWarning: A-NEW-WARNING-SLIPPED-IN`. This is what makes X-02's reporter choice
  safe, and it means a warning can no longer accumulate unnoticed the way the 105 `operators=`
  ones did. Silence an entry here only with a comment explaining why the cause cannot be fixed.

---

## Tranche A — Warnings (112 → 0)

**Two are genuine source bugs surfaced by test warnings. Fix these regardless of the rest.**

- [x] **A-01 `stacklevel` misattribution — source bug.** **[V]** (~2 lines)
  **DONE.** Fixed with a `_caller_stacklevel()` helper in `environment/environment.py` that walks
  out to the first frame outside the package, rather than a fixed number. A fixed bump could not
  work: probing confirmed `SymbolicEnvironment` is 3 frames from the caller and
  `ObjectSearchEnvironment` 4, yet both reported `symbolic.py:216`. Keyed on `__init__` frames rather than package path, so in-package callers (the examples) are named too. Portable to 3.11, so the
  `requires-python` mismatch below did not need resolving. **Result: the 105 warnings that all
  named one library line now name 63 distinct real call sites — that is the A-04 worklist.**
  <details><summary>Original finding</summary>

  `environment/environment.py:75` warns with `stacklevel=2`. The call chain is
  user code → `SymbolicEnvironment.__init__` (`symbolic.py:216`) → `Environment.__init__`, so
  level 2 lands on the *subclass*, not the caller. All 105 warnings named the same library line,
  which is why the report was useless for locating real callers.
  </details>

- [x] **A-02 File-descriptor leak — source bug.** **[V]** (~5 lines)
  **DONE.** Root cause located precisely: `run_point_goal_rollout` builds the writer at
  `rollout.py:233` but only enters the `try`/`finally` that closes it at `:243` — anything
  raising in that gap (including inside `build_point_goal_setup` after `:156`) leaks, and
  `bulk.py` swallows it and moves to the next seed. Fixed in the writer so *every* caller
  benefits: `index.jsonl` now opens lazily on first `write()`, with a `weakref.finalize` closing
  it if the writer is abandoned. Verified under `-W error::ResourceWarning` across four cases
  (never-wrote / abandoned-after-write / double-close / context manager).
  **Suite ResourceWarning count: 1 → 0.**
  <details><summary>Original finding</summary>

  `lsp/data.py:102` — `TrainingDataWriter.__init__` did
  `self._index_file = open(self.out_dir / "index.jsonl", "a")`, released only in
  `close()`/`__exit__`. When a rollout raises, `lsp/bulk.py:213` returns a failure `SeedResult`
  and abandons the writer, so a bulk sweep over failing seeds leaked one fd per failure and could
  reach `EMFILE`. Surfaced as the lone `ResourceWarning` from
  `lsp/test_bulk.py:187::test_failed_seed_leaves_no_final_dir_and_retries_cleanly`.
  </details>

- [x] **A-03 Dead deprecation branch.** **[V]** (~9 lines)
  **DONE.** Branch and the `_from_init` parameter removed; `_resolve_operators` is now a
  plain two-argument method.
  <details><summary>Original finding</summary>

  `environment/environment.py:115-121` — the `if not _from_init:` warning in
  `_resolve_operators` is unreachable. Verified: the method has exactly one caller repo-wide,
  `environment.py:78`, which always passes `_from_init=True`. Delete the branch and the
  `_from_init` parameter; consider inlining the method.
  </details>

- [x] **A-04 ~55 test call sites pass `operators=[...]`.** **[V]** (105 warnings)
  **DONE — 56 call sites across 11 files, plus 1 in *source*.** A-01's attribution fix revealed
  a production call site the audit had missed: `replay/environments/known_map_search.py:101`
  passed `operators=self._build_operators(...)` through `super().__init__`; converted to a
  `define_operators()` override. Test sites go through a new `tests/env_helpers.py`
  (`env_with_operators`), which synthesises the subclass so each call site is a one-token
  change and keeps `operators=` verbatim. Two things the plan did not anticipate:
  `UnknownSpaceEnvironment` takes `operators` as a *required positional*, so the helper passes
  `None` rather than dropping the kwarg; and the helper had to live in a uniquely-named module
  rather than `conftest`, because `ty` binds `conftest` to the wrong one of the repo's several.
  Convert to `define_operators()`. Concentrated in `test_symbolic_environment.py` (24),
  `test_object_search_environment.py` (16), `test_dashboard.py` (15), `test_dashboard_overhead.py`
  (12), `experimental/unknown_search/test_environment.py` (11). Many pass `operators=[]` purely
  to satisfy the signature — trivial. Do **after** A-01, so the warnings point at real callers.

- [x] **A-05 `test_environment_base.py:15` — keep, but assert it.** **[V]** (4 warnings)
  **DONE.** Split the test double: `_MinimalBody` holds the shared abstract-method bodies,
  `MinimalEnvironment` resolves via `define_operators()`, and `LegacyKwargEnvironment` keeps the
  deprecated path. New `test_deprecated_operators_kwarg_still_resolves_and_warns` asserts the
  kwarg still works, warns, **and blames the caller** — so A-01 cannot silently regress.
  This one is *intentional*: `MinimalEnvironment` exercises the base-class contract through the
  deprecated kwarg. Wrap in `pytest.warns(DeprecationWarning)` to turn noise into coverage.

- [x] **A-06 `fork()` in a threaded worker.** **[V]** (3 warnings)
  **DONE.** `mp.get_context("fork")` -> `"spawn"` at both sites; 8 tests pass, no warnings.
  `test_scene_lock.py:58,81` use `mp.get_context("fork")` inside an xdist worker; Python 3.12
  warns, 3.14 changes the default. The test already reports through append-only files
  (see its module docstring) so it does not need `fork` — switch to `"spawn"`.

- [x] **A-07 Preventive: unclosed figures — FALSE FINDING, no action.** **[C]**
  The report claimed 9 `plt.subplots()` with no `plt.close()`. Every one of the 9 is already
  paired with `plt.close("all")` (lines 32/42, 48/53, 57/62, 67/70, 77/80, 84/89, 96/99,
  109/113, 117/122). Adding the proposed autouse fixture would have been redundant.
  `dashboard/test_sprite_static_plot.py` calls `plt.subplots()` 9× with no `plt.close()`. Under
  matplotlib's 20-figure threshold now; one added test from warning. An autouse
  `plt.close("all")` fixture in a `tests/dashboard/conftest.py` forecloses it.

**Exit criterion:** `uv run pytest -W error::DeprecationWarning` clean, and
`uv run railroad example clear-table` warning-free (it currently emits A-01).

---

## Tranche B — Runtime (judged case by case)

Target: `not slow` 12.9 s → ~4–6 s. Note B-01/B-02/B-03 are pure accidents of test setup — they
buy their time back with **zero** coverage loss, so they are the best value in the document.

- [x] **B-01 Emoji SBERT reload — fix the cause, do not mark `slow`.** **[V]**
  **DONE, but the estimate below was wrong — correcting it.** I predicted 3.4 s → ~0.05 s. Actual:
  the `plotting/` directory went **5.30 s → 3.28 s** serial, and the second matching test went
  **0.60 s → 0.03 s** (20×). But `test_unknown_name_uses_fallback` still costs **3.27 s**, and
  running that file alone confirms this is a genuine one-time `SentenceTransformer` load paid by
  whichever matching test runs first — **not** an artifact of the fixture. Only the *repeated*
  reloads were removable. Cutting the remaining 3.27 s would mean mocking the model, which
  weakens the test; not recommended. Fixture is now non-autouse, with
  `pytestmark = pytest.mark.usefixtures("reset_emoji_caches")` scoping it to
  `test_emoji_glyphs.py` — verified the sole file in that directory that monkeypatches (16
  occurrences; the other four have zero).
  `plotting/test_emoji_matching.py:40` asserts one string equality but takes 3.4 s, entirely from
  a `SentenceTransformer` load at `plotting/emoji.py:232`. Cause: `plotting/conftest.py:14` is
  **`autouse`**, clearing `_MODELS` before and after *every* test in the directory. Verified by
  grep: only `test_emoji_glyphs.py` (`:51,52,59,60`) repoints `SYSTEM_FONT_PATHS` /
  `DEFAULT_RESOURCES_BASE`, which is the sole reason the fixture exists — `test_emoji_matching.py`
  never touches them. Make it non-autouse and request it explicitly in `test_emoji_glyphs.py`.

- [ ] **B-02 Artificial sleeps in bench timeout tests.** **[R]** (2.0 s → ~0.2 s)
  `bench/test_parallel_timeout.py:39` arms `timeout=1.0` against `time.sleep(5)`, twice
  (`:93`, `:108`). `timeout=0.1` / `sleep(1)` asserts the identical behaviour.

- [ ] **B-03 Lock-timeout hold is 6× longer than needed.** **[R]** (3.0 s → ~1.0 s)
  `test_scene_lock.py:79` holds a lock 3.0 s to prove a 0.5 s acquisition gives up
  (`assert elapsed < 3.0`). A 1.0 s hold with `timeout=0.2` and `assert elapsed < 0.9` proves the
  same property. Companion `:55` is intrinsically ~1 s — mark `slow` instead.

- [ ] **B-04 ProcTHOR "visualization" tests — keep, retarget, rename.** **[R]** (46.6 s → ~10 s?)
  `environment/procthor/test_visualization.py:99` (25.6 s), `:171` (18.7 s), `:254` (2.3 s).
  Despite the name these **never render** — plotting is gated behind `RAILROAD_TEST_PLOTS`
  (`:35,38-40`), and the file's own docstring concedes the image was never asserted. They are
  ProcTHOR *integration* smoke tests whose only assertion is `goal.evaluate(...)`. Three changes:
  make the `scene` fixture (`:50`) module-scoped rather than function-scoped (~2 s of triplicated
  setup), cut `max_iterations` 4000 → ~1000 (`:148,224,312`), rename the file to match what it
  tests. If it goes flaky at a lower budget, **that flakiness is itself a finding** — do not just
  restore the number.

- [ ] **B-05 `replay/test_search_replay_integration.py:103` — keep as-is, no change.** **[R]** (10.9 s)
  Earns its runtime. Sole end-to-end record → rebuild → replay coverage in the repo; asserts
  recorded truth, outcome resolution, and bound admissibility in deployment units. Already
  correctly `slow`. Listed only so it is not swept up by a "long test" pass.

- [ ] **B-06 `test_planner.py:112` — keep, mark `slow`, trim degenerate rows.** **[R]** (2.7 s)
  6 params × 20 attempts × 10,000 iterations = 1.2 M MCTS iterations, asserting roomA is chosen
  ≥80% of the time. The repetition **is** the test (a statistical claim) — do not cut the sample
  count for the non-degenerate rows. The `roomA_prob=1.0` rows are degenerate and can drop to ~5
  attempts.

- [ ] **B-07 `test_complex_goals.py:361,644` — remove the `slow` marker.** **[R]** (5 tests)
  Marked `slow` but run at 800/400 iterations on a 3-location toy and appear nowhere in the top
  40 durations (<0.1 s). The docstring at `:365-367` says they are marked because they are
  *stochastic*, not slow — so the marker deselects real coverage from the fast path for no
  runtime benefit. Either drop it, or add a distinct `stochastic` marker if that property is
  worth selecting on. Same question for the 6 `railsim` tests, whose `slow` marker stands in for
  "needs GL" — already handled properly by `railsim/conftest.py:7-8`.

- [ ] **B-08 Mark the remaining unmarked >1 s tests `slow`.** **[R]**
  `lsp/test_bulk.py:266` (2.2 s, real `ProcessPoolExecutor` — also the one test that fails under
  a sandbox, so a marker gives a clean escape hatch), `lsp/test_train.py:66` (1.3 s) and `:84`
  (0.6 s, real torch training loops), `test_wait.py:79` (1.1 s), `test_scene_lock.py:55` (1.1 s).

---

## Tranche C — Compaction

Ordered safest first. **Every [R] item gets verified before it is acted on** — see the two
corrections at the bottom for why.

### C1 — Parametrize merges

- [ ] **C1-01 Five search tests are one test.** **[V]** (~120 lines)
  `test_object_search_environment.py:316, 347, 517, 553, 590`. I diffed `:316` against `:553`:
  the entire delta is the `object_find_prob` literal (0.5 vs 0.0) and comment wording. The five
  vary only in `object_find_prob` × object-present × expected-found. One
  `@pytest.mark.parametrize("find_prob, true_loc, expect_found", [...])`. **The single largest
  parametrize win in the repo.**
- [ ] **C1-02** 8 `*_rejected` tests → one `(domain, problem, exc, reason)` table. **[R]** (~70)
  `pddl_converter/test_converter.py:111, 270, 317, 328, 406, 416, 592, 605`.
- [ ] **C1-03** 8 pickle round-trip tests → one parametrize. **[R]** (~45)
  `test_goal_pickle.py:12-60`; keep `test_goal_evaluate_after_pickle:62` separate.
- [ ] **C1-04** `TestComputeBestPathProgress` (7 tests) + `TestGetEntityPositionsAtTimes`
  (5 tests) → two parametrizes. **[R]** (~63) `test_dashboard.py:174-213, 286-334`.

### C2 — Shared fixtures

- [ ] **C2-01 The move operator is declared ~26 times.** **[R]** (~250 lines)
  ~14 lines each. **Do not write a new factory** — `test_grounding.py:33` already has the right
  one (`_move_op`); promote it to the root `conftest.py`. Heaviest users:
  `test_symbolic_environment.py` (5), `test_active_skill.py` (4), `test_environment_base.py` (3).
- [ ] **C2-02** `make_dashboard(...)` factory for the 5 copies of the
  `PlannerDashboard` + `ObjectSearchEnvironment` setup. **[R]** (~110)
  `test_dashboard.py:153,255,373,409`, `test_video_compositing.py:193,253`,
  `test_dashboard_overhead.py:_dashboard()`. Root `conftest.py` already hosts `fetch_dashboard`.
- [ ] **C2-03** `experimental/unknown_search/conftest.py` is an **empty file**; the grid builders
  and env constructor are inlined in 3 places. **[R]** (~60)
- [ ] **C2-04** `lsp/` has **no conftest**; `_FakeRecord` is copy-pasted in 4 files
  (`test_vantage.py:15`, `test_pano.py:90`, `test_generator.py:21`,
  `test_frontier_statistics.py:25`), plus `_square_polygon`, `_frontier`, `experiment_dir`. **[R]** (~70)

> Model all of these on `replay/conftest.py` — the best-factored conftest in the repo
> (ASCII-map DSL → `RolloutLog`).

### C3 — Strengthen weak assertions (do not delete)

- [ ] **C3-01 `TestGetSatisfiedBranch` never checks *which* branch.** **[V]** (~18 lines)
  `test_dashboard.py:76-105`. Read in full: `:81`, `:93`, `:99` assert only `result is not None`,
  so an implementation returning an arbitrary goal passes 3 of 5. (`:87` and `:105` assert
  `is None` and *are* meaningful — keep their intent.) Merge to a parametrize asserting branch
  **identity**.
- [ ] **C3-02 `TestGetBestBranch::test_or_picks_satisfied`.** **[V]** (~8 lines)
  `test_dashboard.py:115` asserts `isinstance(result, LiteralGoal)` — true of *either* child, so
  it cannot detect picking the wrong branch. Assert the specific goal.
- [ ] **C3-03** `pddl_converter/test_converter.py:617` asserts only `h < float("inf")`; any finite
  value passes. Pin an exact value. **[R]**

### C4 — Verified deletions

- [ ] **C4-01 `test_planner.py:177 test_basic_planning` — cannot fail.** **[V]** (~25 lines)
  Read in full: asserts only `isinstance(action_name, str)` and `len(action_name) > 0`. `"NONE"`
  is a valid planner return and satisfies both. Textbook non-enforcing test.
- [ ] **C4-02 `test_video_compositing.py:81`** — no assertion on its non-skip path; it renders two
  buffers and falls off the end. A reporting mechanism, not a test. **[V]** (~30)
  (One of only **2** tests in the whole repo with no assertion at all.)
- [ ] **C4-03 `test_core.py:338-361`** — deepcopies 264 actions into locals and asserts nothing. **[R]** (~24)
- [ ] **C4-04 `packages/environments/tests/test_pyrobosim_demo.py`** — permanent non-strict
  `xfail`, zero assertions, `except Exception: pass`. **[V]** (~50)
  It genuinely xfails (does not xpass), so the underlying path bug is live and nothing will alert
  you when it starts passing. Delete, or fix the resource path and make it `strict=True`.
- [ ] **C4-05** 55 stray `print()` calls in test bodies (39 in `test_wait.py`). **[R]** (~55)

### C5 — Domain rewrites

- [ ] **C5-01 `test_wait.py::test_couch_carry_with_wait`.** **[R]** (~120 of 212 lines)
  Hand-writes six `Action` objects (`:154-258`) that are the groundings of
  `construct_lift_couch_operator` / `construct_move_couch_operator` /
  `construct_put_down_couch_operator` — defined 3 lines below it in the same file (`:337-469`)
  and used by the next test. Two copies of one domain in one file. Rewrite on the helpers,
  keeping the fixed plan and timing assertions. `test_wait.py` is the worst lines-per-test in the
  repo (610 lines / 4 tests = 152).
- [ ] **C5-02 `test_feasible_actions.py`** — merge into `test_planner.py`, **do not delete**.
  See correction **[C-02]**. **[V]** (~40)

---

## Corrections — findings that were wrong

Recorded so they are not re-proposed. Both were "delete the whole file"; both would have cost
real coverage. This is why every **[R]** above is verified before action.

- **[C-01] `packages/environments/tests/test_simulator_operators.py` (214 lines) — NOT a duplicate.**
  Claimed to fully duplicate `test_simulator_actions.py`. Read both: the operators file drives
  `transition()` (pure planning-core semantics, no environment); the actions file drives
  `EnvironmentInterface.advance()` against a live `SimpleEnvironment`, asserting concurrency,
  `r2_loc` intermediate anchors, and `ongoing_actions`. **Different layers.** The narrower, real
  question — overlap with `railroad/tests/test_core.py` at the `transition()` layer — is
  unmeasured. Do not delete on the original rationale.

- **[C-02] `packages/railroad/tests/test_feasible_actions.py` (89 lines) — NOT redundant.**
  Claimed 100% redundant with `test_planner.py::test_pruning_unavailable_actions`. Read both: it
  asserts **exact sets** (`set(...) == {"move r1 r1_loc roomA", "move r2 r2_loc roomA"}` and
  `== ["move r2 r2_loc roomA"]`), which is strictly **stronger** than the other test's
  `len(after) < len(before)` — one pruned action passes that. Real defects: it redefines
  `construct_move_visited_operator` locally (`:7-25`) when `railroad.operators` exports it, and
  its MCTS tail (`:75-89`) is weak because only one action is applicable. Keep the set
  assertions, import the real operator, drop the tail.

---

## Running tally

| | Tests | Lines | Warnings | Full | `not slow` |
|---|---|---|---|---|---|
| Baseline | 763 pass | ~18.6k | 112 | 50–69 s | 12.9 s |
| After Tranche 0 + A + B-01 | **764 pass** | ~18.7k | **0, enforced** | ~69 s | ~14 s |
| Target | 763 pass¹ | ~16.3k | 0 | ~35 s | ~5 s |

² **Tranche A is complete: 112 warnings → 0, and now enforced** — `filterwarnings = ["error"]`
means a new warning fails the run rather than joining a backlog. Test count rose by one (A-05
added a deprecation-contract test). Lines rose slightly too: Tranche A *adds* structure, and the
line reduction all comes from Tranche C.

**On the timing column:** these numbers are not comparable across rows — the machine was under
much lighter load when the 50–69 s baseline was taken. Measured back to back under identical
conditions, the current config runs **69 s against the original config's 76–79 s**. Always
re-measure the comparison point rather than quoting an older row.

¹ **Test count is the wrong metric.** Parametrize merges *keep* the passing count while cutting
lines. Track lines and passing-count separately; any drop in passing-count must map to an
approved C4 deletion ID.

## Verification per tranche

```bash
uv run ty check                             # fast, run first
uv run pytest -m 'not slow'                 # iteration loop
uv run pytest                               # must stay 763 passed
uv run pytest -W error::DeprecationWarning  # Tranche A exit criterion
```

Plus, for anything touching source or operators — the examples are **not** under test:

```bash
uv run railroad example clear-table         # must be warning-free after A-01
uv run railroad example multi-object-search
uv run railroad example heterogeneous-robots
```

**CI note** (`.github/workflows/uv-run-tests.yml`): CI runs the **full** suite (not
`-m 'not slow'`) minus ProcTHOR, and does **not** run `packages/environments` at all. So tests
moved into `slow` still run in CI, but changes under `packages/environments/` are covered only
by local runs.

---

## Deferred — source consolidation (second pass)

Survey complete, ~2,500 lines (≈7% of source), **not** part of this branch:

| Item | Lines | Risk |
|---|---|---|
| Zero-reference dead code (~18 sites) | ~310 | safe |
| `plan→act` loop hand-rolled in 14 places; `replay/loop.py` already implements it | ~300 | needs care |
| 4 near-identical move-operator builders in `operators/core.py` | ~270 | needs care |
| `save_video` — 679 lines, 9 nested closures, inside the 1,918-line `dashboard/plotting.py` | ~110 | needs care |
| Goal-tree walkers — 7 recursive dispatches over the same 5-case enum, across 2 files | ~90 | needs care |
| `ProgressDisplay.__init__` — 13 parallel dicts keyed by benchmark name | ~90 | needs care |
| Legacy `experimental/environment/` stack (5 live callers) | ~858 | risky |

The compacted tests from this branch become the safety net for that work.
