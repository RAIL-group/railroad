# Offline Replay

Counterfactual cost evidence for **navigation** and **object-search** policies,
computed from a *single* recorded deployment. Given one real rollout, offline
replay re-runs an *alternative* policy over the recorded map — without deploying
it — so policies can be compared data-efficiently.

Three flavors share one plan→act loop, differing only in how the arena is built
from a `RolloutLog`, what the goal is, and how the terminal state reduces to a
`Bounds`:

| Flavor | `problem_class` | Environment | Cost meaning |
| --- | --- | --- | --- |
| Point-goal navigation, unknown space | `navigation` | `ReplayPointGoalNavEnvironment` | optimistic vs. simply-connected **lower bounds** |
| Object search, unknown map | `object-search` | `ReplayUnknownSearchEnvironment` | commit-based **lower bound** |
| Object search, known map | `known-map-search` | `ReplayKnownMapSearchEnvironment` | commit-based **lower bound** (travel exact) |

The replay **core** is GL-free and torch-free (the dashboard/render path imports
lazily). Recording panoramas (for learned-policy replay) needs a visual
deployment (OpenGL).

## Quickstart

Three calls, one per stage — record, reconstruct, replay:

```python
from railroad.replay import build_rollout_log, build_replay_env, run_replay, CandidatePolicy

log    = build_rollout_log(deployment_env, goal_cell=..., robot_starts=..., goal=deployment_goal)
env    = build_replay_env(log)             # policy-agnostic arena (dispatch on problem_class)
result = run_replay(env, CandidatePolicy())  # neutral priors; returns cost bounds

result.bounds.optimistic_lb        # C^{lb,opt}
result.bounds.simply_connected_lb  # C^{lb,s.c.}
```

The recorded `goal` is the deployment's actual planning goal — a full `Goal`, so
it may be compound (`found book & found plate`), not a single target. Replay plans
toward that same goal, deriving the searchable objects from its literals. Search
logs require a goal; navigation derives the point-goal from the robots when none
is recorded. Change the policy, hold the goal (and everything else) constant.

Change the policy, hold everything else constant — that is the whole idea. To
compare candidates over one recording, build a fresh arena per candidate and
replay each:

```python
from railroad.lsp.frontier_statistics import LearnedFrontierStatistics
from railroad.replay import CandidatePolicy, build_replay_env, preset_model, run_replay

for policy in candidates:
    env    = build_replay_env(log)
    result = run_replay(env, policy)          # silent; pass dashboard=True to render
```

A `CandidatePolicy` carries whatever the target flavor consumes (a
frontier-statistics estimator for navigation; find-probability callables for
search); the env reads the fields it needs via `apply_policy` and ignores the
rest. `run_replay` holds the planner fixed at the env's per-flavor defaults;
pass `mcts=MctsConfig(...)` / `max_planning_iterations=...` to match a specific
deployment, and `dashboard=True` (with `scene=`/`save_video=`) to render.

## Learned-policy replay (served-vantage panoramas)

The deployment records panoramas; the `RolloutLog` carries them; replay serves
them to a learned estimator via best-vantage selection — the same path used live
and at training time. During development only the *model output* is faked
(`preset_model(...)`); a trained network is a drop-in at the same call site:

```python
# SWAP preset_model(...) for a trained net:
from railroad.lsp.model import load_frontier_statistics_model
estimator = LearnedFrontierStatistics(load_frontier_statistics_model("LSPFrontierNet.pt"))
policy = CandidatePolicy(name="learned", frontier_statistics=estimator)
```

See `scripts/replay/point_goal_nav.py` for the full record → serialize →
serve → compare pipeline (with videos).

## How it works

`ReplayPointGoalNavEnvironment` is an LSP environment whose "world" is the
recorded final map instead of a live simulator:

- **Confinement sensing.** The laser is cast against a *confinement* grid
  (recorded map with `UNOBSERVED → COLLISION`) so the robot stays in known free
  space, while observed-cell *values* are corrected against the **pristine**
  recorded map — so masked / behind-frontier cells are never recorded as
  obstacles. By construction `_observed_grid` only holds an obstacle where the
  pristine map does.
- **Intercept.** `lsp-explore` always resolves to its *failure* branch (the
  deployment recorded no map beyond a frontier); this sets `explored ?f`, and the
  planner retires the frontier. Each commitment logs `cost_accrued +
  optimistic_cost_to_goal`, keyed by a frontier *signature* so a re-extracted
  frontier is never double-counted. Search flavors commit with
  `optimistic_to_goal = 0` (the object could be immediately past the subgoal).

`accumulate_bounds` reduces the recorded commits + final makespan to the bounds.

## Soundness: the log carries no ground truth

Replay is a valid lower bound only if the log holds *only what the deployment
observed*. The recorder enforces this: the recorded grid is the observed (not
true) map, and a revealed-but-unsearched container's true contents are withheld
(`_site_subgoals` records contents only for containers the deployment actually
searched). This is checked by dedicated leakage tests in
`tests/replay/test_recorder.py`, `test_known_map_search_replay.py`, and the
ProcTHOR integration test.

## Module map

| File | Responsibility |
| --- | --- |
| `driver.py` | `build_replay_env` (dispatch on `problem_class`) + `run_replay` (apply policy, drive loop, return bounds). |
| `loop.py` | The shared plan→act loop (silent or dashboard) + `MctsConfig` / `mcts_selector` (shared by deployment and replay). |
| `cost.py` | Pure bound math: `optimistic_cost_to_goal`, `accumulate_bounds`, `Commit`, `Bounds`. |
| `types.py` | Serializable `RolloutLog` (incl. `pano_records`) / `StepRecord` / `SubgoalRecord` / `ReplayResult`. |
| `serialization.py` | `RolloutLog` ↔ disk (`grid.npz` + `panos.npz` + `meta.json`). |
| `recorder.py` | `build_rollout_log` — the one recorder; snapshots a live env (map, frontiers/containers, panos) into a log for any flavor. |
| `policy.py` | `CandidatePolicy` — the flavor-agnostic candidate container + neutral-prior resolvers. |
| `environments/base.py` | `ReplayConfinementMixin` (confinement sensing + net-motion + served panos) and `ReplayArenaMixin` (policy/goal/finalize contract); `navigation_config_from_log`. |
| `environments/point_goal_nav.py` | `ReplayPointGoalNavEnvironment` (navigation) + `goal_fluent`, `frontier_sweep_select`. |
| `environments/unknown_search.py` | `ReplayUnknownSearchEnvironment` (unknown-map object search) + `learned_frontier_search_prob`. |
| `environments/known_map_search.py` | `ReplayKnownMapSearchEnvironment` (known-map object search). |
| `stub_model.py` | Faked `FrontierStatisticsModel` (preset output); drop-in for a trained net. |
| `selection.py` | Cross-trial policy-selection layer — **deferred stub** (see below). |

## Status

Implemented: navigation replay, unknown-map and known-map object-search replay,
and learned-policy served-vantage replay (panoramas recorded, serialized, served;
model output faked). Multi-robot deployments are supported (arenas and goals span
all robots in the log).

Deferred: the cross-trial policy-**selection** layer (`selection.py` is an
explicit stub) and training real networks. Single-recording candidate comparison
is available today by calling `run_replay(build_replay_env(log), policy)` per
candidate and ranking by bound (see `point_goal_nav.py`).
