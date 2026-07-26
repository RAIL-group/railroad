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
from railroad.replay import build_rollout_log, build_replay_env, run_replay

log    = build_rollout_log(deployment_env, goal_cell=..., robot_starts=..., goal=deployment_goal)
env    = build_replay_env(log)             # policy-agnostic arena (dispatch on problem_class)
result = run_replay(env, policy)            # policy = the estimator; returns bounds

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
from railroad.replay import build_replay_env, run_replay

for policy in candidates:
    env    = build_replay_env(log)
    result = run_replay(env, policy)          # silent; pass dashboard=True to render
```

A *policy* is an estimator, and which kind depends on the problem class: a
`FrontierStatisticsEstimator` for point-goal navigation, an `ObjectFindEstimator`
for object search (containers only, on a known map). `apply_policy` installs it;
passing the wrong family fails rather than silently degrading to a neutral prior.
Build them with the per-family helpers — `oracle_frontier_statistics(scene)` /
`oracle_object_find(scene, ...)`, `constant_frontier_statistics(p)` /
`FixedObjectFind(p)`, `learned_frontier_statistics(path)` (navigation, from a
trained `LSPFrontierNet`) / `learned_container_find(scene)` (known-map search,
from ProcTHOR's packaged `FCNNforObjectSearch`). *Which* policies a study compares is an experiment
choice and lives in `scripts/replay/*.py`, not here. `run_replay` holds the planner fixed at the env's per-flavor defaults;
pass `mcts=MctsConfig(...)` / `max_planning_iterations=...` to match a specific
deployment, and `dashboard=True` (with `scene=`/`save_video=`) to render.

## Learned-policy replay (served-vantage panoramas)

The deployment records panoramas; the `RolloutLog` carries them; replay serves
them to a learned estimator via best-vantage selection — the same path used live
and at training time. `constant_frontier_statistics(p)` fakes only the *model
output*, via a `ConstantFrontierStatisticsModel`; the real observations still
flow through it, so a trained network is a drop-in:

```python
policy = constant_frontier_statistics(0.9)             # fakes only the numbers
policy = learned_frontier_statistics("LSPFrontierNet.pt")  # same pipeline, trained net
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
| `deployment.py` | `run_deployment` — drives the loop over a live env and returns the `RolloutLog` replay consumes (+ `DeploymentResult`); the mirror image of `driver.py`. |
| `driver.py` | `build_replay_env` (dispatch on `problem_class`) + `run_replay` (apply policy, drive loop, return bounds). |
| `loop.py` | The shared plan→act loop (silent or dashboard) + `MctsConfig` / `mcts_selector` (shared by deployment and replay). |
| `cost.py` | Pure bound math: `optimistic_cost_to_goal`, `accumulate_bounds`, `Commit`, `Bounds`. |
| `types.py` | Serializable `RolloutLog` (incl. `pano_records`) / `StepRecord` / `SubgoalRecord` / `ReplayResult`. |
| `serialization.py` | `RolloutLog` ↔ disk (`grid.npz` + `panos.npz` + `meta.json`). |
| `recorder.py` | `build_rollout_log` — the one recorder; snapshots a live env (map, frontiers/containers, panos) into a log for any flavor. |
| `policy.py` | Per-family policy builders (`oracle_*`, `constant_frontier_statistics`, `learned_*`), the object-find estimators, and `ConstantFrontierStatisticsModel`. |
| `environments/base.py` | `ReplayConfinementMixin` (confinement sensing + net-motion + served panos) and `ReplayArenaMixin` (policy/goal/finalize contract); `navigation_config_from_log`. |
| `environments/point_goal_nav.py` | `ReplayPointGoalNavEnvironment` (navigation) + `goal_fluent`, `frontier_sweep_select`. |
| `environments/unknown_search.py` | `ReplayUnknownSearchEnvironment` (unknown-map object search). |
| `environments/known_map_search.py` | `ReplayKnownMapSearchEnvironment` (known-map object search). |
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
