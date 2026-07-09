# Offline Replay

Counterfactual cost evidence for **navigation** and **object-search** policies,
computed from a *single* recorded deployment. Given one real rollout, offline
replay re-runs an *alternative* policy over the recorded map — without deploying
it — so policies can be compared data-efficiently.

Three domains share one plan→act loop, differing only in how the arena is built
from a `RolloutLog`, what the goal is, and how the terminal state reduces to a
`Bounds`:

| Domain | `problem_class` | Environment | Cost meaning |
| --- | --- | --- | --- |
| Point-goal navigation, unknown space | `navigation` | `ReplayEnvironment` | optimistic vs. simply-connected **lower bounds** |
| Object search, unknown map | `object-search` | `SearchReplayEnvironment` | commit-based **lower bound** |
| Object search, known map | `known-map-search` | `KnownMapSearchReplayEnvironment` | commit-based **lower bound** (travel exact) |

The replay **core** is GL-free and torch-free. Recording panoramas (for
learned-policy replay) needs a visual deployment (OpenGL).

## Quickstart

`replay()` is the single entry point: it dispatches on `log.problem_class`,
builds a fresh arena per call (so one log replays many candidates), runs the
shared loop, and returns a `ReplayResult`.

```python
from railroad.replay import replay, CandidatePolicy

result = replay(log)                                  # policy-agnostic (neutral priors)
result.bounds.optimistic_lb        # C^{lb,opt}
result.bounds.simply_connected_lb  # C^{lb,s.c.}
```

A `CandidatePolicy` carries whatever the target domain consumes (a
frontier-statistics estimator for navigation; find-probability callables for
search); each domain reads the fields it needs and ignores the rest.

```python
from railroad.lsp.frontier_statistics import LearnedFrontierStatistics
from railroad.replay import CandidatePolicy, preset_model, replay

policy = CandidatePolicy(name="learned", frontier_statistics=LearnedFrontierStatistics(preset_model("optimistic")))
result = replay(log, policy)
```

## Learned-policy replay (served-vantage panoramas)

The deployment records panoramas; the `RolloutLog` carries them; replay serves
them to a learned estimator via best-vantage selection — the same path used live
and at training time. During development only the *model output* is faked
(`preset_model(...)`); a trained network is a drop-in at the same call site:

```python
# SWAP preset_model(...) for a trained net:
from railroad.lsp.model import load_frontier_statistics_model
estimator = LearnedFrontierStatistics(load_frontier_statistics_model("LSPFrontierNet.pt"))
```

See `scripts/replay/replay_learned_demo.py` for the full record → serialize →
serve → compare pipeline (with videos).

## How it works

A `ReplayEnvironment` is an LSP environment whose "world" is the recorded final
map instead of a live simulator:

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
  frontier is never double-counted. Search domains commit with
  `optimistic_to_goal = 0` (the object could be immediately past the subgoal).

`accumulate_bounds` reduces the recorded commits + final makespan to the bounds.

## Module map

| File | Responsibility |
| --- | --- |
| `domains.py` | `replay()` unified entry + per-domain seams (`ReplayDomain`, `CandidatePolicy` dispatch). |
| `cost.py` | Pure bound math: `optimistic_cost_to_goal`, `accumulate_bounds`, `Commit`, `Bounds`. |
| `types.py` | Serializable `RolloutLog` (incl. `pano_records`) / `StepRecord` / `SubgoalRecord` / `ReplayResult`. |
| `serialization.py` | `RolloutLog` ↔ disk (`grid.npz` + `panos.npz` + `meta.json`). |
| `recorder.py` | `build_rollout_log` — snapshot a live env (map, frontiers, panos) into a log. |
| `policy.py` | `CandidatePolicy` — the domain-agnostic candidate container. |
| `base_env.py` | `ReplayConfinementMixin` — confinement sensing + net-motion + served panos (shared by the two confinement envs); `navigation_config_from_log`. |
| `replay_env.py` | `ReplayEnvironment` (navigation) + `run_replay` + selectors. |
| `search_replay_env.py` | `SearchReplayEnvironment` (unknown-map object search) + `run_search_replay`. |
| `known_map_search_replay_env.py` | `KnownMapSearchReplayEnvironment` (known-map object search) + drivers. |
| `stub_model.py` | Faked `FrontierStatisticsModel` (preset output); drop-in for a trained net. |
| `selection.py` | Cross-trial policy-selection layer — **deferred stub** (see below). |

## Status

Implemented: navigation replay, unknown-map and known-map object-search replay,
and learned-policy served-vantage replay (panoramas recorded, serialized, served;
model output faked). Multi-robot deployments are supported (arenas and goals span
all robots in the log).

Deferred: the cross-trial policy-**selection** layer (`selection.py` is an
explicit stub) and training real networks. Single-recording candidate comparison
is available today by calling `replay()` per candidate and ranking by bound (see
`replay_learned_demo.py`).
