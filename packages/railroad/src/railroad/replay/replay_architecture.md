# Replay Architecture

A reference for **how the offline-replay system is built** (the *as-implemented*
companion to `replay_design.md`, which captures the *why*). Code lives in
`packages/railroad/src/railroad/replay/`; demos in `scripts/replay/replay_*.py`.

---

## 1. What it does

From **one real deployment**, offline replay computes the cost an *alternative*
policy would have incurred — without redeploying it — by re-running that policy
over the recorded observations. This is the counterfactual evidence a
data-efficient **policy selector** consumes (selection itself is not yet built;
see §13).

Three problem domains share one design:

| Domain | Deployment task | Replay env | "Policy" |
| --- | --- | --- | --- |
| **Navigation** (point-goal) | `lsp-point-goal-nav` | `ReplayEnvironment` | a `FrontierStatisticsEstimator` (the `lsp-explore` knob) |
| **Object search** (unknown env) | `frontier-search` | `SearchReplayEnvironment` | `object_find_prob` callables (the `search` knobs) |
| **Object search** (known map) | `procthor-search` | `KnownMapSearchReplayEnvironment` | `container_find_prob` callable (the `search` belief knob) |

The first two confine the robot to a *recorded partial* map (frontier
exploration, served panoramas — `ReplayConfinementMixin`); the third is the §7.1
**known-map** case, where the whole floorplan is given and only object presence in
containers is unrecorded, so travel is exact and the replayed cost is the
alternative's *exact* counterfactual, not just a lower bound.

---

## 2. The pipeline (three stages)

```
  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐
  │ 1. DEPLOY   │   │ 2. RECORD    │   │ 3. REPLAY (per candidate)│
  │ (real run,  │──▶│ RolloutLog   │──▶│ alt policy over the      │
  │  records    │   │ (+panoramas) │   │ recording → ReplayResult │
  │  panoramas) │   │  ↕ disk      │   │ (rank by bound to select)│
  └─────────────┘   └──────────────┘   └──────────────────────────┘
   LSPVisualEnv      build_rollout_log    ReplayEnvironment /
   (GL, OpenGL)      save/load_rollout    SearchReplayEnvironment
```

1. **Deploy** — a normal planning run (the repo's standard env + operators +
   `MCTSPlanner` loop). If it is a *visual* env (`LSPVisualEnvironment` /
   `VisualUnknownSpaceEnvironment`, OpenGL) it accumulates a `pano_records`
   buffer — the only stage that needs GL.
2. **Record** — `recorder.build_rollout_log(env, …)` snapshots the final observed
   grid, frontiers/containers, and the panorama buffer into a `RolloutLog`,
   optionally persisted via `serialization.save_rollout_log`.
3. **Replay** — `run_replay` / `run_search_replay` / `run_known_map_search_replay`
   build a replay environment whose "world" is the recorded map and drive a
   candidate policy through the standard plan→act loop, producing a `ReplayResult`.
   To compare candidates, replay each over the same recording and rank by bound
   (e.g. `sorted(results, key=lambda r: r.bounds.simply_connected_lb)`); the
   cross-trial selection layer (§13) will build on this.

Stages 2–3 are **GL-free and torch-free**.

---

## 3. Core data model (`types.py`, `cost.py`)

- **`RolloutLog`** — everything a deployment hands to replay:
  `recorded_grid` (final observed map = the replay arena), `goal_cell`,
  `robot_starts`, `subgoals` (`SubgoalRecord`: signature + cells + `contents` for
  containers), `steps` (provenance), `config`, and **`pano_records`** (the
  accumulated panorama buffer; empty for non-visual logs).
  `pano_records` here is the **recorded deployment buffer** (the source the
  replay env serves observations from — see §7).
- **`Commit`** (`cost.py`) — one alternative-policy commitment to an unrecorded
  subgoal: `cost_accrued`, `optimistic_to_goal`, `robot`, `frontier_signature`.
- **`Bounds`** — `optimistic_lb` (C^{lb,opt}) and `simply_connected_lb` (C^{lb,s.c.}).
- **`ReplayResult`** — `bounds`, `commits`, `termination`, `total_cost`,
  `sim_time`, `goal_reached`, and `search_log` (object-search provenance:
  `(location, cost_accrued, found)` per search; empty for navigation).

`State` is never serialized — replay *re-runs* the policy rather than replaying
recorded states, so no binding types are pickled.

---

## 4. Module map

| File | Responsibility |
| --- | --- |
| `cost.py` | Pure bound math (no env/GL/torch): `optimistic_cost_to_goal` (admissible unseen-as-free Dijkstra), `accumulate_bounds`, `Commit`, `Bounds`. |
| `types.py` | Serializable dataclasses: `RolloutLog`, `SubgoalRecord`, `StepRecord`, `ReplayResult`. |
| `serialization.py` | `RolloutLog` ↔ disk: `grid.npz` + `panos.npz` + `meta.json`; `LoadedPanoRecord`. |
| `recorder.py` | `build_rollout_log(env, …)` — pull a `RolloutLog` from a live env (grid, panos, and frontiers *or* revealed containers+contents). |
| `base_env.py` | `ReplayConfinementMixin` (+ `ServedPano`) — confinement sensing, pristine correction, net-motion, and **per-pose pano serving** (`_serve_pano`). Shared base. |
| `replay_env.py` | `ReplayEnvironment` (nav) + `ReplayEnvironment.from_log` + `run_replay(arena, estimator)` + selectors. |
| `search_replay_env.py` | `SearchReplayEnvironment` (unknown-env object search) + `.from_log` + `run_search_replay(arena, …)` + `learned_frontier_search_prob`. |
| `known_map_search_replay_env.py` | `KnownMapSearchReplayEnvironment` (known-map object search, §7/§7.1) + `.from_log` + `build_known_map_search_log` (its own recorder) + `build_known_map_search_replay_env` + `run_known_map_search_replay`. |
| `stub_model.py` | Faked models: `PresetFrontierStatisticsModel` / `preset_model`, `PresetSearchModel`. The only "fake" in the learned path. |
| `__init__.py` | Public API. |

**Reused from the rest of the repo** (replay adds wiring, not new algorithms):
`LearnedFrontierStatistics` / `FixedPriorFrontierStatistics` /
`FrontierStatisticsModel` and `compute_frontier_views` / `select_best_vantage`
(`railroad.lsp`); `UnknownSpaceEnvironment`, `LSPEnvironmentMixin`, the
unknown-search operators, laser/mapping (`railroad.experimental.unknown_search`);
`SymbolicEnvironment` + `OccupancyGridPathingMixin` and the `move`/`search`
operators (known-map search); `LSPVisualEnvironment` + `build_point_goal_setup`
(`railroad.lsp.rollout`); `MCTSPlanner`; `load_frontier_statistics_model`
(`railroad.lsp.model`).

---

## 5. The replay environment layer

### 5.1 Shared base — `ReplayConfinementMixin` (`base_env.py`)

Mixed in **before** `UnknownSpaceEnvironment` in the MRO so its sensing overrides
take effect. It is the heart of "replay over a recorded map":

- **Two grids** (`_setup_replay_grids`, called before `super().__init__`):
  - **pristine grid** = the recorded map, unmodified — drives frontier detection,
    best-vantage perception, and the optimistic cost grid.
  - **confinement grid** = pristine with `UNOBSERVED → COLLISION` — used **only**
    for laser occlusion/motion, so the robot can never sense/move past known space.
- **`observe_from_pose`** — laser ranges are cast against the **confinement** grid
  (occlusion), but observed-cell *values* are corrected against the **pristine**
  map. Consequence (design §5.1.1): `_observed_grid` only ever holds an obstacle
  where the pristine map does — masked/behind-frontier cells are never painted as
  walls, keeping frontiers and the admissible bound intact.
- **`set_robot_pose`** — accumulates per-robot Euclidean travel into `_net_motion`
  (the replay cost), then delegates.
- **Per-pose pano serving** — the recorded deployment buffer is kept as
  `_recorded_panos`; `pano_records` starts **empty** and grows as the robot moves.
  At each `observe_from_pose`, `_serve_pano` retrieves the recorded panorama
  **nearest the robot's current pose** and appends it (de-duplicated, re-stamped
  with the replay time). This is replay's analogue of a visual env rendering a
  panorama at the robot's pose — it *retrieves* the closest recorded view instead
  of rendering. Both the learned estimator and the dashboard's onboard pane read
  this served buffer, so it tracks the replay trajectory (§7).

For type-checking it aliases its base to `UnknownSpaceEnvironment` under
`TYPE_CHECKING` (same trick as `LSPEnvironmentMixin`), while at runtime it is a
plain mixin whose concrete subclass supplies the base.

### 5.2 Navigation — `ReplayEnvironment` (`replay_env.py`)

`class ReplayEnvironment(LSPEnvironmentMixin, ReplayConfinementMixin, UnknownSpaceEnvironment)`.

- **Intercept** (`resolve_probabilistic_effect`): a committed `lsp-explore ?r ?f`
  always resolves to its **failure** branch (the deployment recorded no map beyond
  the frontier) → sets `explored ?f`; the planner's dead-frontier pruning retires
  it. Each commit logs `cost_accrued + optimistic_cost_to_goal(pristine, frontier,
  goal)` keyed by **frontier signature** (so a re-extracted frontier is never
  double-counted).
- `oracle_available = False` (replay resolves outcomes itself).

### 5.3 Object search — `SearchReplayEnvironment` (`search_replay_env.py`)

`class SearchReplayEnvironment(ReplayConfinementMixin, UnknownSpaceEnvironment)`.

- **Intercept** (`resolve_probabilistic_effect`): a `search ?r ?loc ?obj` /
  `search-frontier` outcome is resolved from the **recorded ground truth** —
  `found` iff `obj` is in the recorded contents of `?loc` (exact replay when the
  deployment revealed the truth). Logged in `_search_log`.
- Refreshes any registered served-vantage estimators in `refresh_frontiers`, and
  exposes `goal_cell` (a reference cell) + `oracle_labels = {}` to satisfy the
  `FrontierStatisticsEnvironment` protocol.

Both confinement envs rebuild the deployment's `NavigationConfig` from the log via
`navigation_config_from_log` (`base_env.py`) so they sense/map exactly as the
deployment did. There is **no default fallback** — a missing recorded config
raises `ValueError`, because a separately-maintained default could silently drift
(e.g. a mismatched `sensor_range` would re-sense a different observed map than was
recorded). Record it with `build_rollout_log` (which captures `env.config`) or
pass an explicit `config=`.

### 5.4 Known-map object search — `KnownMapSearchReplayEnvironment` (`known_map_search_replay_env.py`)

`class KnownMapSearchReplayEnvironment(OccupancyGridPathingMixin, SymbolicEnvironment)`.

The §7.1 case: the **whole floorplan is known** (e.g. a ProcTHOR scene), so there
is no confinement sensing, no panoramas, and no frontiers — it does **not** use
`ReplayConfinementMixin`. The env is a plain `move` + `search` symbolic
environment pathing on the known grid (`OccupancyGridPathingMixin`, exact travel).

- **No bespoke intercept.** `SymbolicEnvironment` already resolves a `search`
  deterministically from `_objects_at_locations`; replay just restricts that map
  to the **recorded** contents, so the existing resolution *is* exact replay from
  the recording. A container the deployment never inspected has empty recorded
  contents → `found` resolves false there (correct: the target was found
  elsewhere). `resolve_probabilistic_effect` is overridden only to **log**
  provenance into `_search_log`; resolution stays the base's job.
- `container_find_prob` is the swappable candidate policy, but it only drives the
  MCTS belief, **never** the outcome (the recording does).
- Its own recorder, `build_known_map_search_log`, snapshots the known grid +
  every searchable location's coords + the contents of inspected locations
  (uninspected → empty `contents`).

---

## 6. Policies — the swappable knob

A "policy" is **how subgoal probabilities/costs are produced**. Swapping the
policy = passing a different estimator/callable to the driver; the env, operators,
and loop are unchanged.

| Policy kind | Navigation | Object search (unknown / known map) |
| --- | --- | --- |
| Oracle (true map) | `OracleFrontierStatistics` | informed `object_find_prob` / `container_find_prob` |
| Fixed prior | `FixedPriorFrontierStatistics` | constant find-prob |
| **Learned** | `LearnedFrontierStatistics(model)` | `learned_frontier_search_prob(model)` (unknown env) |

For **known-map** search the policy (`container_find_prob`) drives only the MCTS
belief — the outcome comes from the recording — so any candidate's cost is exact.

### The "learned" path and the one faked piece

`LearnedFrontierStatistics(model)` maps each frontier's **best-vantage panorama**
(a `FrontierObservation`) to `FrontierStatistics` via `model`. During development
`model` is faked — `preset_model("optimistic"|"cautious"|"uniform")` returns
preset values but still *receives* the real observations, exercising the whole
pipeline. **Swap point** (identical call site):

```python
LearnedFrontierStatistics(preset_model("optimistic"))                 # faked
LearnedFrontierStatistics(load_frontier_statistics_model("net.pt"))   # trained
```

---

## 7. Served-vantage learned perception (how panos flow)

The design's load-bearing insight (§2.1): replay reuses the deployment's
panoramas instead of rendering. **As built**, the replay env serves the
observation *from the robot's pose* — at each sense it retrieves the recorded
panorama nearest the current pose (`_serve_pano`) and appends it to
`pano_records`, which grows along the replay trajectory (like a live visual env
accumulating onboard images, but by retrieval rather than rendering). Perception
runs best-vantage over that accumulated served buffer.

```
deployment (GL)            RolloutLog                  replay (GL-free)
─────────────────          ───────────                 ───────────────────────────────
VisualEnv.observe_from_pose                             from_log → env._recorded_panos
  → scene.get_pano_image()  pano_records  ──save/load──▶   (the recorded source buffer)
  → appends PanoRecord       (recorded)                 each observe_from_pose:
                                                          _serve_pano: nearest recorded
                                                          pano to current pose →append to
                                                          env.pano_records (replay-stamped)
                                                        LearnedFrontierStatistics.refresh
                                                          → compute_frontier_views(
                                                              frontiers, pano_records, goal)
                                                            → select_best_vantage per frontier
                                                            → model(observations) → stats
                                                          → drives lsp-explore / search-frontier
```

- `_serve_pano` makes the onboard observation track the robot's actual replay
  trajectory (the dashboard animates `pano_records` by capture time); the recorded
  buffer alone — deployment poses/timestamps — would mis-pose and freeze once the
  replay clock passed the deployment's last recorded time.
- Panoramas are heading-centered, so the served image is **rolled to the robot's
  current heading** (`roll_pano_to_bearing`) and re-stamped with that yaw —
  otherwise the onboard view would face wherever the deployment looked here, not
  where the replay robot now faces. Rolling image and yaw together leaves
  best-vantage perception unchanged (the roll-to-frontier step cancels the yaw);
  only the capture *position* stays recorded. De-dup keys on `(recorded-pano id,
  column shift)` so a turn-in-place still re-serves a re-rolled view.
- `select_best_vantage` scores each served pano by how many of a frontier's cells
  fall in its visibility polygon; frontiers no served pano covers fall back to the
  default prior.

> Note: this serves *nearest-by-pose* (the old `OfflineReplay.get_image`
> behavior). A pose-independent best-vantage over the *full* recorded buffer (strict
> design §2.1) is a possible variant; the current choice also keeps the onboard
> display pose-matched.

---

## 8. Bounds / cost (`cost.py`)

- **`optimistic_cost_to_goal(recorded_grid, point, goal_cell)`** — admissible
  lower bound: Dijkstra (`MCP_Geometric`) on the recorded map with `UNOBSERVED →
  FREE` (observed obstacles respected). Unreachable → `inf`. Kept **independent of
  the planner's heuristics** (design §10) so the one-sided guarantee holds.
- **`accumulate_bounds(commits, total_cost)`** — `optimistic_lb = min over commits
  of (cost_accrued + optimistic_to_goal)`; `simply_connected_lb = total_cost`.
  The two are independent lower bounds on the alternative's true cost.
- **Unknown-env** object search: travel is exact within the recorded map; the
  reported `simply_connected_lb` is the policy's replay makespan and `optimistic_lb`
  is the "straight to the true container" cost.
- **Known-map** object search (§7.1): the whole map is known and the deployment
  revealed the truth, so the reported `simply_connected_lb` is the alternative's
  **exact** counterfactual makespan (not a lower bound), and `optimistic_lb` is the
  optimal "straight to the true container" cost (exact travel + one search verify).

---

## 9. Drivers & comparison (the public API)

The API is three plain pieces — **record → build arena → replay candidates**:

```python
# 1. RECORD: a function pulls the log out of the deployment env (no class, no
#    file needed; serialization is optional for cross-process / bulk).
log = build_rollout_log(env, goal_cell=…, robot_starts=…)     # navigation
log = build_rollout_log(env, goal_cell=ref, robot_starts=…)   # search (captures
                                                              # containers+contents)

# 2. ARENA: each env has its own .from_log() → a policy-agnostic arena.
arena = ReplayEnvironment.from_log(log)                       # navigation
arena = SearchReplayEnvironment.from_log(log, target_object="book_9")  # search

# 3. REPLAY a candidate policy → Bounds (the candidate is given to run_replay,
#    not the arena, so one arena replays many candidates — option A: each call
#    rebuilds a fresh env from the arena's source log).
res = run_replay(arena, LearnedFrontierStatistics(model))                 # nav
res = run_search_replay(arena, frontier_find_prob=…, container_find_prob=…) # search
res.bounds.optimistic_lb, res.bounds.simply_connected_lb     # same type both domains
```

- **`run_replay(arena, frontier_statistics, *, select_action=None, …)`** — rebuilds
  a fresh `ReplayEnvironment` from `arena._source_log` configured with the
  candidate estimator (panos threaded from the log), runs the plan→act loop
  (default `MCTSPlanner`; tests inject `frontier_sweep_select`), returns a
  `ReplayResult`. Also accepts a raw `RolloutLog` for back-compat.
- **`run_search_replay(arena, *, frontier_find_prob, container_find_prob,
  refresh_estimators=…, …)`** — same shape for unknown-env object search; returns a
  `ReplayResult` (with `search_log`). Optimistic bound = cost straight to the
  container that truly holds the target; pessimistic = the policy's actual cost.
- **`run_known_map_search_replay(arena, *, container_find_prob, target_object=…,
  …)`** — known-map (§7.1) search; the candidate's `container_find_prob` drives
  MCTS belief while outcomes resolve from the recording, so `simply_connected_lb`
  is the alternative's **exact** counterfactual makespan and `optimistic_lb` the
  optimal straight-to-container cost. Accepts an arena handle or a raw `RolloutLog`.

To compare candidates, replay each over the same recording and rank by bound —
e.g. `sorted(results, key=lambda r: r.bounds.simply_connected_lb)` (see
`replay_learned_demo.py`); a dedicated comparison/selection helper is deferred to
the cross-trial selection layer (§13).

Per-domain `run_replay` signatures differ (nav takes an estimator; search takes
prob callables) — the shared contract is the `Bounds` return type. The accounting
is robot-count-agnostic (`total_cost = max over robots`; per-robot commits), so
single/multi-robot is handled by the planning side, not the API.

---

## 10. On-disk format (`serialization.py`)

A log directory contains:

- `grid.npz` — `recorded_grid`, plus subgoal cells stacked with a lengths vector.
- `panos.npz` (only if panos present) — `images` `(N,H,W,3)`, `pose_cells` `(N,3)`,
  `pose_meters` `(N,3)`, `times` `(N,)`, and visibility polygons stacked with a
  lengths vector (length 0 ⇒ `None`).
- `meta.json` — scalars, robot starts, subgoal signatures/centroids/contents,
  step provenance, and `pano_robots`.

Loaded panos become `LoadedPanoRecord`s (duck-typed; no railsim import needed).

---

## 11. GL / optional-dependency boundary

| Component | Needs |
| --- | --- |
| `cost.py`, `types.py`, `serialization.py`, `recorder.py`, `base_env.py`, `replay_env.py`, `search_replay_env.py`, `known_map_search_replay_env.py`, `stub_model.py` | nothing (GL-free, torch-free) |
| Recording **real** panoramas (deployment) | OpenGL (`VisualUnknownSpaceEnvironment` / `LSPVisualEnvironment`); verified headless on `egl` / `cpu` |
| A **trained** learned model | torch + `load_frontier_statistics_model` |

Replay with faked/oracle/fixed policies, and with *synthetic* panos in tests,
needs neither GL nor torch.

---

## 12. Testing architecture (`packages/railroad/tests/replay/`, 53 tests)

- `test_cost.py` — pure bound math on ASCII grids.
- `test_serialization.py` — `RolloutLog` (incl. panorama + config) round-trip.
- `test_replay_env.py` — navigation intercept, retirement, §5.1.1 no-corruption.
- `test_learned_replay.py` — replay serves panos to `LearnedFrontierStatistics`;
  faked model output reaches the planner; different presets → different stats; and
  `_serve_pano` returns the recorded pano **nearest each pose**, growing the buffer
  with replay timestamps (the onboard-tracking fix).
- `test_search_replay_env.py` — unknown-env search outcomes resolved from recorded
  truth; reusable arena across candidates.
- `test_known_map_search_replay.py` — known-map (§7.1) exact replay: outcomes from
  recorded contents, uninspected containers resolve not-found, exact counterfactual
  cost.
- `test_replay_golden.py`, `test_recorder.py` — golden e2e, recorder snapshotting.
- `test_search_replay_integration.py` — slow, procthor-gated end-to-end: deploy an
  informed search policy, record, rebuild the arena, replay a candidate; asserts
  the bounds are admissible and in deployment units.

Fixtures (`conftest.py`): `parse_grid` ASCII map parser, `build_log_from_ascii`,
deterministic selectors. Synthetic `PanoRecord`s (duck-typed) keep learned-path
tests GL-free.

---

## 13. Extension points & known gaps

- **Plug in a trained net** — replace `preset_model(...)` with
  `load_frontier_statistics_model(path)` (no other change). Object search: pass the
  trained model to `learned_frontier_search_prob`.
- **Selection layer (§9)** — comparison is currently just "replay each candidate,
  sort by bound" inline; the cross-trial aggregator + `bulk.py` (mirror
  `lsp/bulk.py`) + a `railroad replay` CLI are not
  built.
- **Multi-robot (§8)** — single-robot today; the env already supports concurrency,
  bounds generalize to makespan.
- **Visual object-search deployment** — recording *real* panos for object search
  needs a visual deployment env (ProcTHOR's `frontier-search` is non-visual); the
  served-vantage learned-search *code path* is built and tested with synthetic
  panos.
- **Known-map search is implemented** (§7.1) — `KnownMapSearchReplayEnvironment`
  yields exact counterfactual cost; no longer a gap.
- **Subgoal-agnostic observation source** — nav and search share
  `ReplayConfinementMixin` but still have separate intercepts; a single
  `observe(subgoals, buffer, goal)` interface is a possible refactor.

## 14. Entry points (demos)

- `scripts/replay/replay_learned_demo.py` — **the showcase** (navigation): visual
  deploy → record panos → save/reload → replay 3 candidate learned policies →
  compare bounds (`RAILSIM_GL_BACKEND=egl uv run python scripts/replay/replay_learned_demo.py`).
- `scripts/replay/replay_object_search.py` — unknown-env object search (informed vs naive).
- `scripts/replay/replay_known_map_search.py` — known-map (§7.1) object search on ProcTHOR:
  informed deployment → exact-replay a naive candidate on the same known map.
