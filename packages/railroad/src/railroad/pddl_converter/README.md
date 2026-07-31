# `railroad.pddl_converter` — run IPC PDDL/PPDDL problems in railroad

Downloads International Planning Competition benchmarks, converts them into
railroad planning problems (or raises a typed error explaining why a domain
cannot be represented), and runs them with railroad's planners.

## Quickstart

```bash
uv run railroad pddl list ipc-2000
uv run railroad pddl check ipc-2000             # per-domain compatibility report
uv run railroad pddl check ippc-2008 --markdown

uv run railroad pddl run --collection ipc-2000 --domain blocks-strips-typed --instance 1
uv run railroad pddl run --domain-file dom.pddl --problem-file prob.pddl
```

```python
from railroad import pddl_converter as pc

fetched = pc.fetch_domain("ipc-2000", "logistics-strips-typed", max_instances=1)
problem = pc.load_problem(fetched.domain_for(fetched.instances[0]), fetched.instances[0])
result = pc.solve(problem, seed=0)          # replans with MCTSPlanner each step
print(result.success, result.plan, result.sim_time)
```

Conversion failures raise `pc.UnsupportedPDDLError` with a machine-readable
`.reason` slug (e.g. `conditional-effects`); `pddl check` aggregates them per
domain.

## Problem sources

| Collection | Source | Contents |
|---|---|---|
| `ipc-1998` … `ipc-2014` | `potassco/pddl-instances` | deterministic IPC tracks |
| `ippc-2006`, `ippc-2008` | `probfd/ppddl-benchmarks` | PPDDL probabilistic tracks |

Only `api.github.com` needs to be reachable; set `GITHUB_TOKEN` to lift the
60/hour anonymous rate limit. Downloads cache permanently under
`$RAILROAD_PDDL_CACHE_DIR` (default `~/.cache/railroad/pddl`), so everything
works offline once fetched. Directory listings cache too — run
`railroad pddl clear-cache` to pick up upstream additions or renames.

## How PDDL maps to railroad

| PDDL | railroad |
|---|---|
| single implicit agent | a synthetic serializing agent: every action takes and releases `free agent`, which is what the core's concurrency machinery keys on |
| `minimize (total-cost)` | each action's cost becomes its **duration**, so minimizing time is exactly minimizing cost (zero-cost actions get ε=1e-3 so time advances) |
| no metric / `minimize (total-time)` | duration 1 per action — PDDL's implicit plan-length objective |
| `maximize (reward)` with `(:goal-reward …)` | reach-the-goal, minimize expected plan length (IPPC convention) |
| `(probabilistic p e …)` | probabilistic effect branches; a total short of 1 gets an implicit "nothing happens" remainder |
| `(when c e)` | native conditional branches, evaluated against the pre-action state |
| `forall` | expanded over the object universe at conversion time |
| `exists` (preconditions) | lifted into an extra operator parameter; grounding enumerates witnesses |
| type hierarchies | flattened — `objects_by_type[t]` holds `t` and all its subtypes |
| `(= ?x ?y)` | evaluated while grounding, filtering bindings |
| `free`, `waiting`, `at`, `found`, `not-*` | renamed with a `pddl-` prefix. These are keywords to the core (`railroad.core.RESERVED_PLANNING_PREDICATES`); gripper's `(free ?g)` would otherwise make every hand a schedulable agent |

Grounding runs through `railroad.core.ground_operators`. Its backtracking
static-precondition pruning is what keeps large domains tractable — IPC-2000
freecell grounds to ~34k actions instead of blowing up — and it compiles
static facts out of the initial state entirely. That docstring is the
reference for the behaviour.

Two ordering rules are worth knowing, because PDDL leaves them loose: deletes
apply before adds *within* one effect (so a self-referential binding like
`fly apt apt` keeps the fluent), and branch sub-effects apply *after* their
parent effect rather than simultaneously with it. Domains relying on
whole-effect simultaneity will diverge.

## Supported features

- `:strips`, `:typing` (hierarchies), `:negative-preconditions`, `:equality`
- `:universal-preconditions`, `:existential-preconditions`,
  `:quantified-preconditions`, `forall` effects
- `:conditional-effects` — conjunctive `when` conditions, `forall`+`when`,
  and `when` nested inside probabilistic branches
- `:action-costs` (`total-cost`, including per-binding function values)
- `:probabilistic-effects` (PPDDL), rational probabilities (`1/2`), nested
  and multiple probabilistic groups, `(:goal-reward …)` metrics
- Files bundling domain and problem in one `(define …)` pair (IPPC-2008 style)

## Unsupported features

Each raises `UnsupportedPDDLError` with the slug shown.

| Slug | Feature |
|---|---|
| `disjunctive-preconditions` | `(or …)` in preconditions |
| `imply-conditions` | `(imply …)` |
| `conditional-effect-condition` | `or`/`exists`/`imply` inside a `when` condition |
| `durative-actions` | PDDL 2.1 temporal actions |
| `numeric-conditions` / `numeric-effects` | numeric fluents beyond `total-cost` |
| `rewards` | `(increase (reward) …)` in effects |
| `metric:*` | any other objective |
| `derived-predicates` | `(:derived …)` axioms |
| `either-types` | `(either …)` types |
| `oneof-nondeterminism` | FOND `(oneof …)` |
| `negated-compound-condition` | `(not (and …))` — only literal negation |
| `preferences` / `constraints` | PDDL 3 soft goals, trajectory constraints |
| `probabilistic-cost` / `conditional-cost` | cost increase inside a branch |
| `timed-initial-literals` | `(at 5 (p))` in `:init` |
| `object-fluents` | non-boolean fluents |

## Compatibility

Conversion coverage, first instance per domain:

| Collection | Converts | What fails |
|---|---|---|
| ipc-2000 | 11/12 | `elevator-adl-full-typed` (`imply-conditions`) |
| ippc-2006 | 9/10 | `drive` (`disjunctive-preconditions`) |
| ippc-2008 | 5/10 | `numeric-effects` ×2, `imply-conditions`, `rewards`, and one domain the source repo ships without a domain file |

Later deterministic years spot-check clean: ipc-2011
`elevator-sequential-optimal` grounds to 1793 actions with function-valued
costs mapped to durations.

The authoritative per-domain list lives in
`tests/pddl_converter/test_ipc_sweep.py`, which re-derives every status from
the source repos and fails if one moves. It needs the GitHub API, so it is
gated:

```bash
RAILROAD_PDDL_NETWORK_TESTS=1 uv run pytest -q -k ipc_sweep
```

## Solving, and what it does not tell you

The sweep pins *conversion*, not solving — whether MCTS reaches a goal is a
planner property. With the shipped defaults (4000 iterations, seeds 0-2, a
300-step cap) the converted domains solve 11/11, 6/9, and 3/5 respectively.
A 10x iteration budget moved no domain in either direction, so the failures
are heuristic and dead-end problems rather than search-effort ones.

Two known limitations account for most of them:

- **Dead ends are not avoided.** The FF heuristic is a delete-relaxation, so
  it cannot see that driving without a spare risks an unrecoverable flat.
  Separately, where the relaxation *does* prove a state hopeless, MCTS clamps
  that to a reward of 0 — making doomed states look goal-adjacent.
  `solve(..., dead_end_penalty=...)` replaces the clamp with a flat failure
  cost, fixing the second problem (ippc-2008 triangle-tireworld 1/3 → 3/3,
  ex-blocksworld 0/3 → 2/3) but not the first. It stays opt-in because it
  perturbs multi-robot search-ordering ties.
- **Disjunctive goals** compile to an `AndGoal` of `OrGoal`s whose DNF is a
  product. `simplify_static_goal` folds statically-decidable disjuncts first
  (ippc-2008 boxworld: 5^10 = 9.7M branches → 1), and above 1024 branches the
  heuristic picks one conjunction greedily instead of materialising the DNF.
  After folding, no domain above still exceeds the cap.

Solve rates come from one fixed and fairly small budget. Read them as "what
the current defaults do", not as a limit of the domains.
