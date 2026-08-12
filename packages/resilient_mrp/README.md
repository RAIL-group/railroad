# Resilient MRP: Failure-Aware Multi-Robot Planning

Multi-robot planning over graphs where traversing an edge can render a robot non-operational.
Robot loss is terminal and irreversible, so a plan has to account for failure before it happens
rather than replan after it.

## Problem

A team of robots must safely visit every goal site. Each traversal either arrives or the robot
becomes non-operational, closing that edge for everyone. A mission fails when a goal can no longer
be reached by any operational robot.

**Failure model:** `P_fail(r, e) = β(e) × (1 − φ(r, τ(e)))`

- `β(e)`: hazard severity of edge `e`, scaled globally by `risk_scale`
- `τ(e)`: terrain type of `e`, one of `clear`, `rocky`, `steep`, `deformable`
- `φ(r, τ)`: robot `r`'s compatibility with terrain `τ`

**Objective.** A trial that reaches every goal costs its makespan, the clock reading when the last
robot stops. A trial that does not costs `C_fail`, one flat number for the whole experiment, so
every failed trial scores the same regardless of when or how it failed. Expected cost is the mean
over all trials of an instance, failures included, and is how a planner is judged. Total travel,
the sum of every edge any robot walked, is reported beside it as a secondary number.

Losing a robot is not itself penalised. It only costs when it strands a goal.

---

## Package Structure

```
src/resilient_mrp/
├── planning/               the planner
│   ├── core.py             ResilientGraph, RobotProfile, the risk_move and safely_visited operators
│   ├── baselines.py        route tables, the two edge weights, best_assignment, and the two policies
│   ├── value_function.py   RiskAwareCostToGo, the MCTS leaf estimate our planner uses
│   └── legacy.py           superseded pre-terrain model, kept only for the legacy tests
├── scenarios/              environment representation
│   ├── blackbox.py         turns terrain features and robot specs into β and φ
│   ├── sctp_scenario.py    create_graph: generated random and island topologies (the live path)
│   ├── small_scale.py      hand-built 8-node graph, useful for debugging
│   └── graph_analysis.py   start placement, edge statistics, terrain labelling
├── experiments/            what defines and runs a trial
│   ├── instance.py         Spec, Instance, TrialOutcome, ALL_PLANNERS, planner_setup, run_trial
│   └── mission.py          run_episode: one mission, plan-execute-replan
└── analysis/
    └── graph_viz.py        the terrain graph render the playground demo draws

scripts/
└── playground.py           single missions, the video recorder, and the benchmark
```

---

## Operators

Defined in `planning/core.py`. The environment always executes `risk_move`; a planner may *search*
a different model by passing its own operator list as `planning_operators`.

| Factory | Action name | Model it encodes |
|---|---|---|
| `create_risk_move_operator(graph, profiles)` | `risk_move` | Per-edge, per-robot failure from the model above |
| `create_risk_move_operator(graph, profiles, flat_survival=p)` | `risk_move` | One average survival probability on every edge |
| `create_safely_visited_operator()` | `safely_visited` | Deterministic; marks a goal visited |

On failure three fluents change at once, which is what makes loss terminal and visible to the
search:

```
~operational ?robot        the robot can no longer act
~free ?robot               it never becomes free again
~path_available ?from ?to  the edge closes for every robot
```

`safely_visited` is a separate operator so the FF heuristic has a finite-cost path to the goal
condition. Without it `safely_visited` appears only inside probabilistic branches, which FF
ignores, and every state scores `h = ∞`.

---

## Planners

Four strategies (`ALL_PLANNERS`), all executing against the same true model and differing in what
they plan with.

| Key | Search | What it optimises |
|---|---|---|
| `optimistic` | none, Dijkstra route table | least-cost path, failure ignored |
| `cautious` | none, Dijkstra route table | most survivable path, cost ignored |
| `failure_aware_ff` | MCTS + FF | the true model, guided by cost alone |
| `failure_aware_split` | MCTS + `RiskAwareCostToGo` | the true model, guided by cost and risk |

The two baselines are deterministic replanning relaxations, not stochastic planners. Each builds
Dijkstra route tables over the terrain graph under one of two edge weights, in `baselines.py`:
`optimistic_weight` counts travel only, `cautious_weight` counts negative log survival.

`RiskAwareCostToGo` hands the outstanding goals out once, then reads that single assignment twice.
A goal goes to whichever robot would finish it soonest counting what it already carries, so two
free robots split rather than stack up on the nearest goal. Cost comes off the optimistic route
table, survival off the cautious one, and the estimate is
`max(load) + (1 − survival) × C_fail`. Treating any robot's failure as the mission failing
over-states the risk, which is the relaxation the leaf trades for being cheap to evaluate.

---

## Usage

```python
from resilient_mrp.planning.core import create_risk_move_operator, create_safely_visited_operator
from resilient_mrp.scenarios.sctp_scenario import ROBOT_PROFILES, create_graph

graph, goal_sites = create_graph("sctp_random", 20, risk_scale=1.0, seed=43, n_goals=2)
profiles = {r: dict(ROBOT_PROFILES[r]) for r in ("r1", "r2")}

move_op = create_risk_move_operator(graph, profiles)
visited_op = create_safely_visited_operator()
```

`create_graph` is the single entry point for graphs; `graph_size` is the vertex count for
`sctp_random` and the island count for `sctp_island`.

## Running

Everything runs from `scripts/playground.py`. Edit the block under `__main__` to pick a demo.

```bash
# single mission with the live dashboard
uv run python packages/resilient_mrp/scripts/playground.py

# the full risk sweep, all four planners
uv run railroad benchmarks run --tags resilient_mrp \
  --include packages/resilient_mrp/scripts/playground.py --parallel 2
uv run railroad benchmarks dashboard
```

## Testing

```bash
uv run ty check
uv run pytest packages/resilient_mrp/tests/

# WSL: suppress the display backend warning
export MPLBACKEND=Agg
```
