# Learning over Subgoals Planning (LSP)

Point-goal navigation in **partially-known** environments. The robot knows
*where* the goal is (its coordinates) but not *whether* a path there exists
through the unobserved parts of the map. LSP turns the boundaries of observed
space — **frontiers** — into probabilistic subgoals: exploring a frontier may
reveal a route to the goal, with a learned (or oracle) estimate of how likely
that is and how much it costs. The planner reasons over these subgoals to reach
the goal in as little expected travel as possible.

This package provides the planning operators, the frontier-statistics
estimators that parameterize them, action pruning to keep the branching factor
tractable, and the full training pipeline (data generation → dataset → network).

```bash
uv run railroad example lsp-point-goal-nav                       # maze, oracle stats
uv run railroad example lsp-point-goal-nav --env office --num-robots 3
uv run railroad example lsp-point-goal-nav --frontier-statistics learned \
    --network-file data/maze/training/LSPFrontierNet.pt
```

The planning core is GL-free and torch-free; only the *visual* environment
(`environment.py`, `rollout.py`) needs railsim, and only `model.py` / `train.py`
need torch. See [`__init__.py`](__init__.py) for the import map.

## The planning model

### Operators

`LSPEnvironmentMixin.define_operators()` ([`env_mixin.py`](env_mixin.py)) owns
four operators:

| Operator | Role |
| --- | --- |
| `move` (navigable) | Drive between *observed* locations along a real occupancy-grid path. |
| `lsp-explore ?r ?f` | Push robot `?r` through frontier `?f`; **probabilistically** establishes that the goal is reachable beyond it. |
| `move-to-goal ?r ?from goal` | The pre-observation **landmark**: head for the goal once exploration shows it is reachable but before its cell has been sensed. |
| `no_op ?r` | Wait. Offered only when a robot has no other applicable action (see *Serialization & waiting*). |

### The goal lifecycle: `reachable` vs `revealed`

The goal is registered as a `goal`-typed object with known coordinates from the
start, but it is **not** a `location` and cannot be moved to directly. Two
deliberately-distinct fluents track its status:

```
                 lsp-explore success                direct observation
  (unknown)  ─────────────────────────►  reachable ─────────────────────────►  revealed
                                          goal                                   goal
   no plan to goal exists                a route is known to exist            goal cell is in the
   in the relaxed graph                  beyond an explored frontier;         observed map; goal is
                                          robot has NOT sensed the cell        now a real `location`
                                                    │                                  │
                                                    ▼                                  ▼
                                          move-to-goal active                 plain `move` takes over;
                                          (optimistic known-coord path)       move-to-goal deactivates
```

- **`reachable goal`** is set by an `lsp-explore` *success* branch. It means a
  path to the goal is known to exist through the explored frontier — but the
  goal cell itself is usually still beyond sensor range, so the goal is not yet
  a `location` and the navigable `move` cannot target it.
- **`revealed goal`** is reserved for **direct observation**: when the goal cell
  enters the observed map, `LSPEnvironmentMixin._reveal_goal_if_observed`
  promotes the goal to a real `location` and sets `revealed goal`. (`revealed`
  has the same "this cell has been observed" meaning here as everywhere else in
  the codebase, e.g. the `start` location.)

### Why `move-to-goal` exists (and why it is gated the way it is)

`move-to-goal` is gated on **`reachable goal` ∧ ¬`revealed goal`**. This is the
crux of the design, so it is worth spelling out:

- **It is the planning landmark.** Before the goal is observed it is not a
  `location`, so the navigable `move` has no grounding that targets it. Without
  `move-to-goal` there would be *no* action achieving `at ?r goal` in the
  relaxed planning graph, the heuristic to the goal would be infinite, and MCTS
  would have nothing pulling it to explore. `move-to-goal` (with an
  *optimistic*, unseen-as-free travel-time estimate, see
  `estimate_goal_move_time`) is the action that lets the planner reason
  "explore the right frontier → goal becomes reachable → drive to it."
- **It really executes**, in the window after `lsp-explore` reveals a route but
  before the robot has driven close enough to sense the goal cell. There the
  navigable `move` still has no goal grounding, so `move-to-goal` is the only
  action that advances the robot toward the goal.
- **It hands off cleanly.** The instant the goal cell is observed, the goal
  becomes a `location` and the ordinary `move ?r ?from goal` grounds. The
  `¬revealed` gate switches `move-to-goal` *off* at exactly that point, so the
  two operators are never both live for the same destination — which previously
  produced a `move … goal` name collision and a planner crash. The plain `move`
  uses the real observed path from then on.

In practice the goal is often reached via direct observation + plain `move`
(the robot sees the goal cell while exploring), and `move-to-goal` never
dispatches — but it remains essential as the **planning landmark** that makes
the relaxed heuristic finite and guides exploration in the first place.

## Action pruning

Every reachable frontier yields an `lsp-explore` action (all achievers of the
single probabilistic `reachable goal` fluent) plus the `move`s that reach it, so
the grounded action set grows into the thousands and MCTS cannot branch over it
tractably. [`_action_pruning.py`](../_action_pruning.py)
(`prune_probabilistic_achievers`, wired into `MCTSPlanner` via
`prune_achievers=True`) bounds it, mirroring the original LSP heuristic:

1. **Per-robot achiever ranking.** For each probabilistic fluent on the relaxed
   path to the goal, keep per robot only the `top_n` achievers by success
   probability and the `cheapest_m` by time-to-attempt (reach-then-explore),
   take the union, discard the rest. Prefer high-probability subgoals, but keep
   a few nearby ones cheap enough to be worth a look.
2. **Dead-frontier removal.** A frontier with no surviving achiever and nothing
   located at it is purposeless — *all* actions referencing it (including
   transit `move`s in densely-connected location graphs) are dropped. This is
   what collapses the action count once the goal is revealed: every frontier
   becomes dead and only the goal-reaching moves remain.
3. **Orphaned-support pruning** (opt-in): drop support actions that achiever
   pruning *newly* orphaned, via a before/after diff of the relaxed backward
   closure — leaving always-irrelevant actions (e.g. `no_op`) untouched.

The dashboard shows the *considered* count next to a dim *total*, so the effect
of pruning is visible live.

### Serialization & waiting

Two `get_next_actions` filters (C++, [`planner.hpp`](../../../include/railroad/planner.hpp))
further bound branching at each decision: when several robots are free, only the
first (deterministically-ordered) free robot's actions are offered (multi-robot
actions involving it still count); and `no_op` is offered only when a robot has
no other applicable action, so the planner never chooses to wait.

## Frontier statistics

`lsp-explore` is parameterized by a `FrontierStatisticsEstimator`
([`frontier_statistics.py`](frontier_statistics.py)) returning, per frontier, a
`prob_feasible` and `delta_success_cost` / `exploration_cost`:

- **`OracleFrontierStatistics`** — exact values from the true map (simulation
  only; also used to *label* training data).
- **`FixedPriorFrontierStatistics`** — fixed constants, no oracle needed (the
  deployment-safe default).
- **`LearnedFrontierStatistics`** — an `LSPFrontierNet` predicting statistics
  from the same panorama observations the training data stores.

## Training pipeline

```bash
# 1. Generate training data: one resumable rollout per seed, in parallel,
#    one directory per seed. Uses the oracle to label each frontier.
uv run railroad lsp generate-data data/maze/training --env maze --num-seeds 200

# 2. (optional) Summarize / visualize what was written.
uv run railroad lsp inspect-data data/maze/training

# 3. Train the frontier-statistics network.
uv run railroad lsp train-network data/maze/training
```

The pieces: [`bulk.py`](bulk.py) (parallel generation) → [`generator.py`](generator.py)
/ [`data.py`](data.py) (per-rollout writing) → [`dataset.py`](dataset.py)
(`LSPFrontierDataset`, PyTorch) → [`train.py`](train.py) / [`model.py`](model.py)
(`LSPFrontierNet`). Panorama handling and best-vantage selection live in
[`pano.py`](pano.py), [`vantage.py`](vantage.py), and [`views.py`](views.py);
oracle labeling in [`oracle.py`](oracle.py).

## Module map

| File | Responsibility |
| --- | --- |
| `operators.py` | `lsp-explore`, `move-to-goal` constructors (the planning model above). |
| `env_mixin.py` | `LSPEnvironmentMixin`: operators, oracle labels, goal revelation, explore resolution. |
| `environment.py` | `LSPVisualEnvironment` — the railsim-backed concrete environment. |
| `rollout.py` | Shared setup (`build_point_goal_setup`) + headless run loop. |
| `frontier_statistics.py` | Estimator protocol and the three estimators. |
| `oracle.py` | True-map frontier labeling and goal-observation test. |
| `bulk.py`, `generator.py`, `data.py` | Training-data generation and on-disk format. |
| `dataset.py`, `model.py`, `train.py` | PyTorch dataset, network, training loop. |
| `pano.py`, `vantage.py`, `views.py` | Panorama framing and best-vantage selection. |
| `types.py` | Shared dataclasses (`FrontierStatistics`, `TrainingDatum`, …). |
