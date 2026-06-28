# Offline Replay

Counterfactual cost evidence for **navigation** and **object-search** policies,
computed from a *single* recorded deployment. Given one real rollout, offline
replay re-runs an *alternative* policy over the recorded map — without deploying
it — so policies can be compared data-efficiently. The full design lives in
[`replay_design.md`](replay_design.md); the as-built reference is
[`replay_architecture.md`](replay_architecture.md).

Three domains share one design: point-goal **navigation** (`ReplayEnvironment`),
**object search in an unknown env** (`SearchReplayEnvironment`), and **object
search on a known map** (`KnownMapSearchReplayEnvironment` — §7.1, where travel is
exact and the replayed cost is the alternative's *exact* counterfactual, not just a
lower bound).

The replay **core** is GL-free and torch-free. Recording panoramas (for
learned-policy replay) needs a visual deployment (OpenGL).

```python
from railroad.lsp.frontier_statistics import FixedPriorFrontierStatistics
from railroad.replay import RolloutLog, run_replay

result = run_replay(log, FixedPriorFrontierStatistics(prob_feasible=0.8))
result.bounds.optimistic_lb        # C^{lb,opt}
result.bounds.simply_connected_lb  # C^{lb,s.c.}
```

## Learned-policy replay (served-vantage panoramas)

The deployment records panoramas; the `RolloutLog` carries them; replay serves
them to a learned estimator via best-vantage selection — the same path used live
and at training time. Only the *model output* is faked during development; a
trained network is a drop-in:

```python
from railroad.lsp.frontier_statistics import LearnedFrontierStatistics
from railroad.replay import ReplayEnvironment, build_rollout_log, preset_model, run_replay

log   = build_rollout_log(env, goal_cell=…, robot_starts=…)  # pull log from deployment env
arena = ReplayEnvironment.from_log(log)                      # policy-agnostic arena (serves panos)

policy = LearnedFrontierStatistics(preset_model("optimistic"))   # faked model
# SWAP: LearnedFrontierStatistics(load_frontier_statistics_model("LSPFrontierNet.pt"))
result = run_replay(arena, policy)           # candidate policy → Bounds
result.bounds.optimistic_lb, result.bounds.simply_connected_lb
```

Object search (unknown environment) is symmetric — `run_search_replay` +
`learned_frontier_search_prob`. See `scripts/replay/replay_learned_demo.py` for the full
record → serialize → serve → compare pipeline.

Known-map object search (§7.1) is exact rather than bounded — record with
`build_known_map_search_log`, then `run_known_map_search_replay(arena,
container_find_prob=…, target_object=…)`. The candidate's `container_find_prob`
only drives the MCTS belief; outcomes resolve from the recorded contents, so the
realized makespan is the alternative's exact counterfactual cost. Demo:
`scripts/replay/replay_known_map_search.py`.

## How it works

A `ReplayEnvironment` is an LSP point-goal environment whose "world" is the
recorded final map instead of a live simulator:

- **Confinement sensing.** The laser is cast against a *confinement* grid
  (recorded map with `UNOBSERVED → COLLISION`) so the robot stays in known
  free space, while observed-cell *values* are corrected against the
  **pristine** recorded map — so masked / behind-frontier cells are never
  recorded as obstacles. By construction `_observed_grid` only holds an
  obstacle where the pristine map does.
- **Intercept.** `lsp-explore` always resolves to its *failure* branch (the
  deployment recorded no map beyond a frontier); this sets `explored ?f`, and
  the planner retires the frontier. Each commitment logs
  `cost_accrued + optimistic_cost_to_goal` — keyed by a frontier *signature*
  so a re-extracted frontier is never double-counted.

`run_replay` drives the plan→act loop with an **injectable** action selector
(production: MCTS; deterministic tests: `frontier_sweep_select`) and reduces
the recorded commits + final makespan to two lower bounds.

## Module map

| File | Responsibility |
| --- | --- |
| `cost.py` | Pure bound math: `optimistic_cost_to_goal`, `accumulate_bounds`, `Commit`, `Bounds`. |
| `types.py` | Serializable `RolloutLog` (incl. `pano_records`) / `StepRecord` / `SubgoalRecord` / `ReplayResult`. |
| `serialization.py` | `RolloutLog` ↔ disk (`grid.npz` + `panos.npz` + `meta.json`). |
| `recorder.py` | `build_rollout_log` — snapshot a live env (map, frontiers, panos) into a log. |
| `base_env.py` | `ReplayConfinementMixin` — confinement sensing + net-motion + served panos (shared by the two confinement envs); `navigation_config_from_log`. |
| `replay_env.py` | `ReplayEnvironment` (navigation) + `run_replay` + selectors. |
| `search_replay_env.py` | `SearchReplayEnvironment` (unknown-env object search) + `run_search_replay` + `learned_frontier_search_prob`. |
| `known_map_search_replay_env.py` | `KnownMapSearchReplayEnvironment` (known-map object search, §7.1) + `build_known_map_search_log` + `run_known_map_search_replay`. |
| `stub_model.py` | Faked `FrontierStatisticsModel` (preset output); drop-in for a trained net. |

## Status

Implemented: navigation replay, unknown-env and known-map (§7.1) object-search
replay, and learned-policy served-vantage replay (panoramas recorded, serialized,
served; model output faked). Not yet: the cross-trial policy-selection layer (§9),
`bulk.py` + CLI, multi-robot (§8), and training real networks.
