# Offline Replay & Policy Selection — Design Doc

> **Status:** Implemented and tested in `packages/railroad/src/railroad/replay/`
> (53 tests; GL-free + torch-free *core*, with one optional GL deployment to
> record panoramas):
> - **Navigation lower-bound replay** (§13 step 1) — `ReplayEnvironment` +
>   `run_replay` + bounds.
> - **Object-search replay in unknown environments** — `SearchReplayEnvironment`
>   + `run_search_replay`; search outcomes resolve from the recorded ground
>   truth. (Note: this is *unknown-environment* frontier-based search, broader
>   than §7's original "known-map" framing — see §7.)
> - **Known-map object-search replay** (§7.1, now built) —
>   `KnownMapSearchReplayEnvironment` + `run_known_map_search_replay`: the whole
>   floorplan is given, travel is exact, the deployment revealed the truth, so the
>   replayed cost is the alternative's **exact** counterfactual, not a bound. No
>   bespoke intercept — it restricts `SymbolicEnvironment`'s deterministic search
>   resolution to the recorded contents.
> - **Learned-policy served-vantage replay** (the §2.1 insight, now real for both
>   domains): the deployment records panoramas (`VisualUnknownSpaceEnvironment` /
>   `LSPVisualEnvironment`, OpenGL — verified headless on `egl`/`cpu`); the
>   `RolloutLog` carries them; replay serves them to
>   `LearnedFrontierStatistics(model)` via `compute_frontier_views`. **Only the
>   model's numeric output is faked** (`replay/stub_model.py`,
>   `PresetFrontierStatisticsModel` / `preset_model`); a trained net drops in at
>   the same call site (`lsp.model.load_frontier_statistics_model`) with no other
>   change. Candidate policies are ranked by replaying each over one recording and
>   sorting by bound (`replay/replay_learned_demo.py`) — a selection precursor.
>
> **Modules:** `cost.py` (pure bounds), `types.py` + `serialization.py`
> (`RolloutLog` ↔ disk incl. panoramas), `base_env.py`
> (`ReplayConfinementMixin`: confinement sensing + net-motion + served panos,
> shared by the two confinement envs; `navigation_config_from_log`),
> `replay_env.py` (navigation), `search_replay_env.py` (unknown-env object search +
> `learned_frontier_search_prob`), `known_map_search_replay_env.py` (known-map
> object search §7.1, with its own `build_known_map_search_log` recorder),
> `stub_model.py`, `recorder.py`. Demos:
> `scripts/replay/replay_learned_demo.py` (point-goal navigation, learned policies +
> compare), `scripts/replay/replay_object_search.py` (unknown-env object search),
> `scripts/replay/replay_known_map_search.py` (known-map object search).
>
> **Remaining:** training real networks; the cross-trial **selection** layer
> (§9) + `bulk.py` + a `railroad replay` CLI; **multi-robot** (§8). The
> navigation bound uses the pristine recorded map for the admissible
> `optimistic_cost_to_goal`; scripted selectors must skip `explored` frontiers
> (the production MCTS planner does so via dead-frontier pruning).
>
> **Scope:** an offline-replay layer for the `railroad` LSP stack that, from a
> single real deployment, computes counterfactual cost evidence for alternative
> policies — to enable data-efficient *policy selection*. Targets, in order:
> (1) point-goal navigation in unknown space, (2) object search in partially
> known maps. Task planning is explicitly out of scope for now (it composes from
> the two; see §10).

---

## 1. Background & goal

### 1.1 What we are building

We have several candidate LSP **policies** — e.g. a learned `LSPFrontierNet`
trained on maze-A, another trained on maze-B, an `OracleFrontierStatistics`
baseline, a `FixedPriorFrontierStatistics` baseline. At deployment in a new
environment we want to choose the best one *without* paying to run all of them.

**Offline replay** extracts counterfactual evidence about *all* candidates from a
*single* real deployment of the chosen policy:

1. **Record** one deployment (the chosen policy runs for real), logging the
   observations it gathered and the cost it paid.
2. **Replay** each alternative policy over that recording — it re-decides actions
   while being served the recorded observations.
3. **Bound** each alternative's cost from below. The lower bound is *exact replay*
   while the alternative stays inside what the deployment observed, and switches
   to an *optimistic lower bound* the moment the alternative commits to a subgoal
   whose outcome the deployment never recorded.
4. **Select** across trials: if a candidate's lower bound consistently exceeds the
   deployed policy's *actual* cost, that is one-sided evidence it is worse — no
   deployment of it required. This is the data-efficiency.

This mirrors the IROS 2023 method (Paudel & Stein, *Data-Efficient Policy
Selection for Navigation in Partial Maps via Subgoal-Based Abstraction*) and its
multi-robot follow-up (`mrlsp_select`), reimplemented to the standards and
abstractions of this repository.

### 1.2 Reference implementations (old, deprecated repo)
Old repo: https://github.com/RAIL-group/RAIL-lsp-dev/tree/abpaudel/mrlsp-select/modules/lsp_select/lsp_select

`RAIL-lsp-dev @ abpaudel/mrlsp-select`:
- `modules/lsp_select/lsp_select/offline_replay.py` — single-robot `OfflineReplay`
  simulator + `get_lowerbound_planner_costs`.
- `modules/lsp_select/lsp_select/planners/policy_selection.py` —
  `PolicySelectionPlanner.get_costs`.
- `modules/mrlsp_select/mrlsp_select/offline_replay.py` — multi-robot variant
  (makespan cost, per-robot masking into a shared masked set).

We are **not** porting line-by-line; we are porting the *idea* onto this repo's
already-more-general abstractions, which let us drop several of the old hacks
(see §6).

---

## 2. The key enabling insight (why this is tractable here)

Two facts about the current `railroad` LSP stack make replay far cleaner than in
the old repo:

### 2.1 Perception is already a pure function of recorded data

During **live execution**, the learned estimator perceives a frontier *not* from
the robot's current camera but through an indirection:

- `env_mixin.py:254` → `self._lsp_frontier_statistics.refresh(self)` is called on
  every frontier change (live act→observe→plan loop).
- `LearnedFrontierStatistics.refresh` (`frontier_statistics.py:181`) calls
  `compute_frontier_views(frontiers, pano_records, goal_cell)` (`views.py`), which
  for each frontier runs `select_best_vantage` (`vantage.py:63`) — picking the
  stored panorama whose visibility polygon covers the most cells of that frontier —
  and builds a `FrontierObservation`.

So **live perception = `f(frontiers, pano_records, goal_cell)`**, a pure function
of the accumulated panorama buffer, independent of where the robot currently is.
The *same* `compute_frontier_views` builds the training data (`views.py` docstring).

**Consequence:** replay needs no bespoke "what would the robot see at pose X"
simulator (the old repo's `OfflineReplay.get_image` + nearest-pose alignment).
To replay an alternative estimator we feed it the recorded `pano_records` and call
the identical `refresh()`/`get()`. Train, live, and replay share one perception
path — which also means **zero train/live/replay skew**.

### 2.2 Subgoal commitment is already a typed, discrete action

Frontier exploration and object search are the *same shape* of probabilistic
achiever, parameterized by per-subgoal statistics:

- `lsp-explore ?r ?f` (`lsp/operators.py`) → probabilistic effect `reachable goal`,
  parameterized by `FrontierStatistics`.
- `search ?r ?loc ?obj` (`operators/core.py:152`) → probabilistic effect
  `found ?obj` / `at ?obj ?loc`, parameterized by `object_find_prob`.

Both expose a discrete **commit event** (the action selection) and a discrete
**outcome** (the probabilistic effect). The old continuous-motion planner had no
such event and had to *infer* "the robot is about to cross into the unknown"
geometrically (a 10-cell distance threshold). Here, crossing the boundary *is* a
typed action, so the replay intercept attaches to it directly (§6).

---

## 3. Core abstraction

A **subgoal** is a probabilistic-achiever action with:
- a **commit point** — the state/pose at which the robot commits to attempting it;
- **statistics** — `(prob_success, success_cost, failure/exploration_cost)` from an
  estimator (oracle / fixed prior / learned);
- an **outcome** — a probabilistic effect that, on success, advances toward the goal.

The **replay invariant** (identical across all problem classes):

> Replay executes any action whose outcome the deployment **recorded**. It
> intercepts the first probabilistic achiever whose outcome the deployment did
> **not** record, resolves it **pessimistically** (failure), and logs the
> **optimistic cost-to-goal** at that commit point. The alternative's true cost is
> ≥ this, so it is a valid lower bound.

| Problem class | Subgoal | Commit point | Unrecorded quantity | "Observed space" |
| --- | --- | --- | --- | --- |
| Unknown-space nav | frontier `?f` | frontier centroid | is goal reachable past `?f` | occupancy grid |
| Object search, known map | container `?loc` | inspection vantage of `?loc` | does `?obj` sit at `?loc` | set of inspected containers |
| Task planning *(out of scope)* | any prob. operator | its precondition state | the effect outcome | resolved fluents |

The navigation "distance-to-frontier" boundary becomes, in general, a **discrete
predicate over recorded outcomes** — which is why search needs no geometric
threshold at all (§7).

---

## 4. What gets recorded (the deployment log)

A successful real deployment produces a **`RolloutLog`** — serializable, mirroring
the existing `lsp/data.py` on-disk conventions (npz for arrays + jsonl index +
`meta.json`). `State` is a binding type and is **not** directly picklable;
serialize via string fluents + time + upcoming effects, reconstructable on load.

Run-level:
- `seed`, env name/config, `problem_class` tag, `start`, `goal` (cell coords).
- **final observed grid** (the union of everything observed across the deployment;
  this is the replay *arena* — see §5).
- **final subgoal set** (frontiers / searched containers) with geometric cells and
  a stable signature (`frontier_cells_hash` / `frontier_signature`).
- **pano buffer** (`pano_records`: `PanoRecord(robot, time, pose_cells,
  pose_meters, image, visibility_polygon)`), the whole thing, all robots, all times
  (§6 explains why time-of-capture is *not* filtered).
- actual total cost (makespan for multi-robot; see §8).

Per planning step (`StepRecord`): time, robot poses, chosen action name, the
subgoal set at that step, estimator outputs (provenance), net cost so far.

**Object-search extension (record now, use later — point 10):** when a container
is inspected, record its **full contents**, not just the binary found/not-found for
the deployed query. One deployment then becomes replayable for *arbitrary* future
object-search / task-planning queries (different target objects/goals). Keep the
recorded-observation schema general enough to carry per-container contents.

---

## 5. The replay arena: final map + two patches

We replay over the deployment's **final** observed map (we use *all* deployment
info regardless of when it was gathered — §6). Built from the old `OfflineReplay`,
two patches are essential; both primitives already exist in this repo.

### 5.1 Patch 1 — pessimistic sensing (mandatory)

The robot's **motion** must be confined to known free space. The old repo senses
against a **pessimistic grid**: the final known map with `UNOBSERVED_VAL →
COLLISION_VAL` (unknown treated as wall), used *only* to generate laser ranges:

```python
@property
def pessimistic_map(self):              # offline_replay.py:68
    m = self.known_map.copy()
    m[m == UNOBSERVED_VAL] = COLLISION_VAL
    return m
# get_laser_scan (offline_replay.py:75) feeds pessimistic_map to the laser ONLY
```

This **structurally confines the robot to known free space** — it can never sense
or move past observed space. Consequence (resolves an earlier worry): every pose
the robot can reach has panorama coverage, and every surviving frontier was seen
by ≥1 pano, so best-vantage perception never silently falls back to the prior for
reachable frontiers.

#### 5.1.1 CRITICAL nuance — masking is a *motion/occlusion* block, not a *map fact*

(Confirmed by reading the full reference `offline_replay.py`, all 264 lines.)
Two separate grids must be kept distinct, or the representation gets corrupted:

- **Pristine grid** — the recorded final partial map, **unmodified**. This drives
  frontier detection, best-vantage perception, and the optimistic cost grid.
  Frontier cells stay `FREE` and the space behind them stays `UNOBSERVED`.
- **Confinement grid** — pessimistic (`UNOBSERVED → COLLISION`) plus retired
  frontiers walled off. Used **only** for pathing/occlusion to keep the robot in
  known space.

In the reference, masked-frontier `COLLISION` is written into `self.known_map`
(the sensing grid, `offline_replay.py:123`) but the cost computation rebuilds its
grids from the **pristine** `partial_map` (`inflate_grid(partial_map, ...)`,
lines 172–173) and re-applies masking *explicitly and per-purpose* (lines
181–195). The two never mix. As a result **the lidar never paints masked
frontiers (or the unknown behind active frontiers) into the map representation
the planner reasons over** — masking blocks motion, nothing else.

**We DO build the observed map incrementally with live lidar sensing during
replay** — same as real execution, just that the "world" the laser reads is the
recorded final map instead of a live simulator. `self._observed_grid` grows as the
robot moves (it is *not* the static final map). The quirk is purely about *what
values the lidar is allowed to write*: the confinement grid's `COLLISION` values
(unknown space, and walled-off masked frontiers) exist **only to bound occlusion
and motion** and must **not** be recorded as obstacles in the observed map.

**Why this is a trap.** If `observe_from_pose` writes raw confinement-grid hits
into `self._observed_grid`:

1. the cell *behind* an active frontier becomes `COLLISION` instead of
   `UNOBSERVED` → the frontier (FREE-adjacent-to-`UNOBSERVED`) **vanishes** from
   the frontier set; and
2. `_optimistic_goal_cost_grid` (env_mixin.py:170–172) copies `_observed_grid`
   and sets `UNOBSERVED → FREE` — if `COLLISION` got baked in, the unseen-as-free
   bound treats openings as walls → **the admissible lower bound is wrong.**

**How the live-sensing build stays correct — known-grid correction against the
recorded map.** The lidar uses the confinement grid for **range/occlusion only**
(how far each ray travels, so the robot can't see past known space); the **values
written** into `_observed_grid` come from **correcting against the pristine
recorded final map** (this repo already has known-grid correction in mapping —
`observe_from_pose` / mapping `insert_scan`). Because the recorded map holds
`UNOBSERVED` behind active frontiers and `FREE` at masked-frontier cells:

- behind active frontiers stays `UNOBSERVED` → **frontiers survive** the scan;
- masked-frontier cells stay `FREE` → **the lidar never records them as
  obstacles**, even though they are walled in the confinement grid for motion.

So the incrementally-built `_observed_grid` converges toward the recorded map as
the robot explores, never accumulating confinement artifacts. This is the
deliberate divergence from the old repo: it retired frontiers by baking
`COLLISION` into `known_map` (map-level retirement); we keep masked cells as
geometric openings and retire at the **planning level** (the `explored` fluent,
§5.2 / §6) — which is exactly why "the lidar must not sense masked frontiers as
obstacles" here.

New repo wiring: override sensing
(`experimental/unknown_search/environment.py:196`) so the laser's range query runs
against the confinement grid while the scan is corrected against the recorded map;
retirement walls a committed frontier in the confinement grid (motion) and sets
`explored` (planning), but never edits `_observed_grid`.

### 5.2 Patch 2 — frontier retirement

Pessimistic sensing never dissolves a frontier on its own (nothing is ever
revealed beyond it), so the robot would oscillate at a frontier forever without an
explicit **retirement trigger**. The old repo used a 10-cell distance mask
(`mask_grid_with_frontiers` when `dist_to_frontier <= dist_mask_frontiers`,
`offline_replay.py:120–126`), walling the frontier in the *confinement* grid only.

**This repo replaces the distance heuristic with the typed-action commit** (§6):
selecting `lsp-explore ?r ?f` *is* the commit; resolving it as failure sets
`explored ?f`; the existing dead-frontier pruning retires it. No threshold to tune,
and — crucially — retirement is a *planning-level* fact, so it never touches the
pristine observed grid (§5.1.1). (`mask_grid_with_frontiers` from `lsp/oracle.py:47`
remains available for the confinement grid if we ever want the geometric fallback.)

---

## 6. The intercept (navigation)

`ReplayEnvironment(LSPEnvironmentMixin, ...)` overrides
`resolve_probabilistic_effect` (the live seam at `env_mixin.py:321`, base at
`environment/environment.py:167` / `symbolic.py:343`). The live LSP environment
resolves explore outcomes from the oracle; replay resolves them from *recorded
data*:

1. Robot selects `lsp-explore ?r ?f`. It navigates to the frontier centroid along
   a path in observed free space (pessimistic sensing keeps it in-bounds) →
   **real travel cost accrues, exactly**.
2. At the outcome, `resolve_probabilistic_effect` matches the explore branch
   (`_match_lsp_explore_branches`, `env_mixin.py:289`) and **forces the failure
   branch** — because the deployment recorded no map beyond `?f`. This sets
   `explored ?f`.
3. The existing `prune_probabilistic_achievers` **dead-frontier removal** retires
   `?f`; the planner naturally reselects. (Reselection is emergent — we write no
   reselection logic.)
4. We **log the optimistic bound at this commit**:
   `cost_accrued + optimistic_goal_cost(robot, ?f)`, where
   `optimistic_goal_cost` (`env_mixin.py:209`) is the **admissible** Dijkstra
   frontier→goal cost on the unseen-as-free grid (`_optimistic_goal_cost_grid`,
   `env_mixin.py:162`). This is already implemented and already wired into
   `lsp-explore`.

**Outputs** (per alternative policy), matching the reference's two bounds:
- `optimistic_lb = min over commits of (cost_accrued + optimistic_goal_cost)` —
  the cheapest "if this frontier had led to the goal" total.
- `simply_connected_lb = total cost_accrued` at termination — the "explore
  everything, no frontier shortcuts" cost.

### 6.1 Frontier identity across re-extraction (the §5-of-discussion point)

Frontiers are re-extracted every step and get **fresh ids**; ids are *not* stable.
Retirement must be tracked by **geometric signature** (`frontier_cells_hash` /
`frontier_signature`), not id — otherwise a retired frontier reappears under a new
id and is re-offered (oscillation / double-count). Use the signature as the
replay-stable frontier key that the `explored ?f` fluent refers to.

---

## 7. The intercept (object search)

> **Implemented (unknown environment), `replay/search_replay_env.py`.** What was
> built is object search in an *unknown* environment (frontier exploration +
> container search, the `examples/frontier_search.py` task), not the known-map
> case this section originally described. `SearchReplayEnvironment` confines the
> robot to the recorded map (shared `ReplayConfinementMixin`) and resolves each
> `search`/`search-frontier` outcome from the **recorded ground truth** — exact
> when the deployment revealed it (the §7.1 idea below). The frontier-search
> probability is the learned, served-vantage knob (`learned_frontier_search_prob`
> + a served `LearnedFrontierStatistics`), so a trained net drops in there too.
>
> **The known-map specialization below is now also built** —
> `replay/known_map_search_replay_env.py` (`KnownMapSearchReplayEnvironment` +
> `run_known_map_search_replay`, demo `scripts/replay/replay_known_map_search.py`). It
> needs **no bespoke intercept**: `SymbolicEnvironment` already resolves `search`
> deterministically from `_objects_at_locations`, so replay just restricts that map
> to the **recorded** contents and the existing resolution becomes exact replay
> (§7.1). Travel is exact on the known grid (`OccupancyGridPathingMixin`), and
> `container_find_prob` is the swappable candidate policy that drives only the MCTS
> belief, never the outcome.

The original known-map analysis still holds and is **cleaner and strictly more
informative** than navigation:

- The map is fully known → **all travel is exact and replayable**; there is no
  unseen-as-free optimism in the *travel* term.
- The only unrecorded thing is **object presence**. The "boundary" is the discrete
  fluent `searched ?loc ?obj` — no geometric threshold whatsoever.

Intercept: when the alternative commits to `search ?r ?loc ?obj` at a container the
deployment **did inspect**, replay the recorded outcome exactly. At a container it
**did not** inspect, force the **not-found** branch and log the optimistic bound
(if it *were* there: travel + verify).

### 7.1 Exact replay when the deployment revealed the truth (point 10)

A successful search deployment reveals the object's true container. With the truth
known *and* the map fully known, every alternative policy's cost is computable
**exactly** (it searches in its own order until it hits the true container; every
other container's emptiness is also known). No optimism, no looseness.

**Decision (DONE, as recommended):** when the map is fully known and the deployment
revealed the truth → **exact replay** (exact counterfactual cost), not a lower
bound. Built as `KnownMapSearchReplayEnvironment`: it restricts the recorded
contents and lets the base deterministic search resolution do exact replay; an
uninspected container has empty recorded contents and resolves not-found (correct,
since the truth was found elsewhere). `run_known_map_search_replay` returns
`simply_connected_lb` = the alternative's exact makespan and `optimistic_lb` = the
optimal straight-to-container cost.

---

## 8. Multi-robot

**No special casing is needed**, for two reasons:

1. The deployment-time environment already handles concurrency (timestamped
   effects, concurrent skills, C++ free-robot serialization). Replay reconstructs
   that same environment with *served* observations instead of *live* sensing, so
   it **inherits** concurrency. Cost accrual (makespan) is whatever the env already
   produces.
2. The reference confirms the multi-robot variant is "single-robot per robot":
   per-robot masking accumulated into a **shared** retired-frontier set, cost =
   **makespan** (`max(net_motion over team)`, or the winner's motion on success).
   Both behaviors are how the env already shares state and accrues time.

So: get the single-robot `ReplayEnvironment` right and multi-robot follows. The
bound formula generalizes directly — per commit, `cost_accrued_for_that_robot +
optimistic_goal_cost(frontier)`, min across all commits; terminal makespan for the
simply-connected bound.

---

## 9. Policy selection layer

`PolicySelectionPlanner`-equivalent:
- Holds N candidate estimators (the swappable knob — `lsp-explore` is parameterized
  by the estimator via `construct_lsp_explore_operator`; search by
  `object_find_prob`).
- Deploys the chosen estimator for real, records the `RolloutLog`.
- Offline-replays each other candidate → `(optimistic_lb, simply_connected_lb)` or
  exact cost.
- Aggregates across trials to recommend a policy (regret-style; the deployed policy
  gets its actual cost, others their bounds). Cross-trial aggregation detail to be
  pulled from the reference `scripts/` when we build this layer.

---

## 10. Out of scope (for now) & future work

- **Task planning under uncertainty.** Composes from navigation + search; revisit
  once both work. The replay invariant (§3) already covers it in principle.
- **Heuristic-based bounds.** The bound must stay **independent of the planner's
  search heuristics**. Navigation uses the admissible Dijkstra
  `optimistic_goal_cost`; do **not** substitute the FF / relaxed-plan heuristic
  (generally inadmissible → would silently break the one-sided guarantee). Treat
  the bound as its own admissible computation, not a heuristic call.
- **Container-contents logging** (§4) — recorded now, exploited later for arbitrary
  replayed search/task queries.
- **MCTS stochasticity.** The replay policy is MCTS for now; counterfactual cost is
  therefore a random variable. We will observe the variance empirically before
  deciding whether to switch to a deterministic greedy selector. No mitigation yet.
- **railsim exact re-render** as a ground-truth validation oracle for bound
  tightness — not now.
- **No hardcoded termination caps.** Rely on frontiers exhausting + goal reached.
  Ensure "all frontiers retired, goal unreachable in observed space" resolves to a
  clean terminal (= `simply_connected_lb`), not a hang.

---

## 11. Open decisions to confirm before coding

1. **Frontier retirement:** typed `lsp-explore` failure (recommended, tuning-free)
   vs. porting the 10-cell distance mask. *Default: typed action.*
2. **Known-map search:** exact replay when truth is revealed (recommended) vs.
   lower bound everywhere for uniformity. *Default: exact when available.*
   ✅ **Resolved & built** — `KnownMapSearchReplayEnvironment` does exact replay.
3. **Doc/code location:** new `packages/railroad/src/railroad/replay/` subpackage
   (sibling to `lsp/`), GL-free/torch-free core with optional deps gated as in
   `lsp/`. *Default: yes.*

---

## 12. Proposed module layout

New `packages/railroad/src/railroad/replay/` (mirrors `lsp/` conventions —
`__init__.py` import map + `README.md`):

| File | Responsibility |
| --- | --- |
| `types.py` | `RolloutLog`, `StepRecord`, `ReplayCost`, `PolicyVerdict` (serializable dataclasses). |
| `recorder.py` | Hook the live act→observe loop; write the `RolloutLog`. |
| `serialization.py` | `State` ↔ string-fluents; npz + jsonl + `meta.json` (as `lsp/data.py`). |
| `replay_env.py` | `ReplayEnvironment` — pessimistic sensing, served-vantage perception, `resolve_probabilistic_effect` intercept. |
| `search_replay_env.py` | `SearchReplayEnvironment` — unknown-env object search; outcomes from recorded ground truth; `learned_frontier_search_prob`. |
| `known_map_search_replay_env.py` | `KnownMapSearchReplayEnvironment` — known-map (§7.1) exact replay; its own `build_known_map_search_log` recorder. |
| `base_env.py` | `ReplayConfinementMixin` — confinement sensing + served panos + net-motion, shared by the two confinement envs. |
| `cost.py` | Bound accumulator → `(optimistic_lb, simply_connected_lb)` or exact cost. |
| `selection.py` *(not built)* | `PolicySelectionPlanner`-equivalent + cross-trial aggregation. |
| `bulk.py` *(not built)* | Parallel offline-replay cost generation (mirror `lsp/bulk.py`). |

CLI under the existing group: `railroad replay record …`,
`railroad replay costs <log-dir> --policies …`, `railroad replay select <logs>`.

### Reuse map (already in the repo — replay is mostly *wiring*, not new algorithms)

| Need | Existing primitive |
| --- | --- |
| Served-vantage perception | `compute_frontier_views` (`views.py`), `select_best_vantage` (`vantage.py:63`) |
| Estimator swap (the "policy") | `FrontierStatisticsEstimator` via `construct_lsp_explore_operator`; `object_find_prob` for search |
| Resolve outcome from recording | override `resolve_probabilistic_effect` (`env_mixin.py:321`) |
| Admissible optimistic bound | `optimistic_goal_cost` / `_optimistic_goal_cost_grid` (`env_mixin.py:209` / `:162`) |
| Frontier retirement → reselection | `explored ?f` + `prune_probabilistic_achievers` dead-frontier removal |
| Stable frontier identity | `frontier_cells_hash` / `frontier_signature` (`oracle.py:32` / `data.py:41`) |
| Geometric masking fallback | `mask_grid_with_frontiers` (`oracle.py:47`) |
| Pano buffer | `VisualUnknownSpaceEnvironment.pano_records` (`visual_environment.py:106`) |

---

## 13. Sequencing

1. ✅ **Single-robot navigation lower-bound replay.** `RolloutLog` + recorder;
   `ReplayEnvironment`; bound accumulator. (Validation against the *reference's
   numbers* not yet done — soundness/structure tested instead.)
2. ✅ **Object-search replay** — `SearchReplayEnvironment` (unknown env) and
   `KnownMapSearchReplayEnvironment` (known map, §7.1 exact replay), outcomes from
   recorded ground truth; container contents recorded (§4).
3. ✅ **Learned-policy served-vantage replay** — record panoramas in the log, serve
   them in replay to `LearnedFrontierStatistics`; model output faked
   (`stub_model.py`), trained net is a drop-in. Candidates ranked by replaying
   each and sorting by bound (selection precursor).
   (Shared `ReplayConfinementMixin` unifies the two replay envs; a fully
   subgoal-agnostic observation source remains a possible refactor.)
4. **Multi-robot** — should require little beyond confirming makespan accrual and
   shared retirement (§8).
5. **Selection + cross-trial aggregation** (`selection.py`, `bulk.py`, CLI, bench).
6. **Train real networks** and run end-to-end policy selection among them
   (replaces `preset_model(...)` with `load_frontier_statistics_model(...)`).

---

## 14. Testing strategy

`PolicySelectionPlanner` is **out of scope for now** (§9 is the next milestone).
The immediate goal is: get offline replay working and compute the **optimistic**
and **simply-connected** ("pessimistic") bounds. Tests are designed around three
seams that keep them fast, deterministic, GL-free, and torch-free.

### 14.1 Design-for-testability seams

1. **Bounds are pure geometry.** `cost.py` exposes pure functions over
   `(recorded_grid, point, goal_cell, commits)` — no environment, no GL, no torch.
   Unit-testable with hand-verified numbers.
2. **The replay policy is injectable.** `run_replay(log, policy, estimator)`.
   Production passes `MCTSPlanner`; tests pass a **scripted policy** (fixed
   frontier-commit order) → deterministic, sidestepping MCTS stochasticity (§10).
3. **Perception is already covered and bound-irrelevant.** Best-vantage/pano/image
   handling has its own tests (`test_vantage`/`test_pano`/`test_views`). With an
   `Oracle`/`FixedPrior`/stub estimator the bound tests need **no images**.

### 14.2 Test pyramid (`packages/railroad/tests/replay/`)

| Level | File | What it pins down |
| --- | --- | --- |
| **L0** pure math | `test_cost.py` | `optimistic_cost_to_goal` on ASCII grids (straight-shot, routes-through-unobserved, unreachable→`inf`); `accumulate_bounds` (min lands on right commit, empty→`inf`, inf-commit tolerated). Exact assertions. |
| **L1** intercept | `test_replay_env.py` | committing `lsp-explore ?r ?f` on an unrecorded frontier → failure branch (`explored ?f` set, `reachable goal` *not* set); retired signature never reappears (§6.1). |
| **L1** §5.1.1 quirk | `test_replay_env.py` | after sensing near a masked frontier, `_observed_grid` keeps masked cells `FREE` / behind-active `UNOBSERVED`; `_optimistic_goal_cost_grid` still routes through. Regression guard for the pristine/confinement split. |
| **L1** serialization | `test_serialization.py` | `RolloutLog → disk → load` identity, incl. `State` string-fluent round-trip. |
| **L2** driven replay | `test_replay_env.py` | `run_replay` + scripted policy → asserts commit sequence, per-commit accrued cost, retirement order, clean termination (no hang). |
| **L3** golden e2e | `test_replay_golden.py` | tiny ASCII maps + scripted policy → exact `optimistic_lb` and structure; `optimistic_lb ≤ simply_connected_lb`. Tolerance on any-angle travel. Parametrized topologies. |
| **L4** real recording | `tests/environment/railsim/test_replay_integration.py` | record one short rollout, replay deployed policy on its own log, assert property `optimistic_lb ≤ actual_cost`. `pytest.mark.slow` + railsim `conftest` gate. One or two only. |

### 14.3 Invariant / property tests (cheap, high-signal)

- **Lower-bound soundness:** when the replayed policy reaches the goal,
  `optimistic_lb ≤ total_cost`. (Each commit contributes
  `accrued + optimistic_to_goal`; since `optimistic_to_goal` is admissible from the
  frontier, `accrued + optimistic_to_goal ≤ total_cost`, so their min is too.)
  Note the two bounds are *independent* lower bounds on the alternative's true
  cost — there is **no** general ordering between `optimistic_lb` and
  `simply_connected_lb`, so we do not assert one.
- **Determinism:** same log + same scripted policy → identical bounds.
- **No-corruption:** across a full replay, `_observed_grid` never holds `COLLISION`
  at a recorded-`FREE`/`UNOBSERVED` cell — by construction the pristine-correction
  makes `_observed_grid` a value-subset of the recorded map (the §5.1.1 guard).

### 14.4 Fixtures & gating

- `tests/replay/conftest.py`: `parse_grid` fixture — ASCII map parser
  (`#`=collision, `.`=free, `?`=unobserved; any other char = a `FREE` marker cell,
  returned as `markers[char] -> [(row, col)]`). Plus reuse `_frontier(fid, cells)`
  from `test_frontier_statistics.py`, a `make_replay_log(...)` builder (no images by
  default), and a `ScriptedPolicy(order)`.
- Default suite: fast, GL-free, torch-free, inside the `-m 'not slow'` ~10s budget.
- Learned estimator → `torch = pytest.importorskip("torch")`. GL → railsim
  `conftest` skip + `pytest.mark.slow` (L4 only).

### 14.5 `cost.py` contract (the L0 anchor — implemented first)

Pure, env-free, GL-free, torch-free:
- `optimistic_cost_grid_from_goal(recorded_grid, goal_cell) -> np.ndarray` —
  Dijkstra (`compute_cost_grid_from_position`, `MCP_Geometric`) on the recorded map
  with `UNOBSERVED → FREE` (admissible unseen-as-free; mirrors
  `_optimistic_goal_cost_grid`). Unreachable → `inf`.
- `optimistic_cost_to_goal(recorded_grid, point, goal_cell) -> float` — lookup at
  `point` (out-of-bounds / unreachable → `inf`).
- `Commit(cost_accrued, optimistic_to_goal, robot="", frontier_signature="")`,
  `Bounds(optimistic_lb, simply_connected_lb)`.
- `accumulate_bounds(commits, total_cost) -> Bounds` —
  `optimistic_lb = min(c.cost_accrued + c.optimistic_to_goal)` over commits (empty →
  `inf`); `simply_connected_lb = total_cost`.

---

## 15. Glossary

- **C^{lb,opt} / `optimistic_lb`** — optimistic lower bound: min over commit points
  of `cost_accrued + admissible optimistic cost-to-goal`.
- **C^{lb,s.c.} / `simply_connected_lb`** — simply-connected lower bound: total cost
  accrued when every frontier is treated as a dead end (explore-everything cost).
- **Commit point** — the discrete moment an alternative policy selects a
  probabilistic-achiever action (an `lsp-explore` / `search`).
- **Pessimistic / confinement grid** — final map with unobserved cells (and
  retired frontiers) as obstacles; used **only** for pathing/occlusion to confine
  the replayed robot to known free space. Kept strictly separate from the pristine
  grid (§5.1.1).
- **Pristine grid** — the recorded final partial map, left unmodified; drives
  frontier detection, best-vantage perception, and the optimistic cost grid.
  Masking and retirement must never write into it.
- **Served vantage** — the best recorded panorama for a frontier, chosen by
  visibility-polygon cell coverage (`select_best_vantage`).
- **Retirement** — marking a committed frontier `explored` so dead-frontier pruning
  removes it and the planner reselects.
</content>
</invoke>
