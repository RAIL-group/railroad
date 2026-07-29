# `railroad.pddl_converter` — run IPC PDDL/PPDDL problems in railroad

Downloads International Planning Competition benchmark problems, converts
them into railroad planning problems (or raises a typed error explaining why
a domain cannot be represented), and runs them with railroad's planners.

## Quickstart

```bash
# List / scan a collection (downloads are cached; see Caching below)
uv run railroad pddl list ipc-2000
uv run railroad pddl check ipc-2000            # per-domain compatibility report
uv run railroad pddl check ippc-2008 --markdown

# Convert + solve one instance
uv run railroad pddl run --collection ipc-2000 --domain blocks-strips-typed --instance 1
uv run railroad pddl run --collection ippc-2006 --domain blocksworld --instance 1 --planner greedy
uv run railroad pddl run --domain-file dom.pddl --problem-file prob.pddl
```

```python
from railroad import pddl_converter as pc

fetched = pc.fetch_domain("ipc-2000", "logistics-strips-typed", max_instances=1)
problem = pc.load_problem(fetched.domain_for(fetched.instances[0]), fetched.instances[0])
result = pc.solve(problem, seed=0)          # planner="mcts" (default) or "greedy"
print(result.success, result.plan, result.sim_time)
```

Conversion failures raise `pc.UnsupportedPDDLError` with a machine-readable
`.reason` slug (e.g. `conditional-effects`, `metric:maximize (reward)`); the
`check` command aggregates these per domain.

## Problem sources

| Collection | Source (via GitHub API) | Contents |
|---|---|---|
| `ipc-1998` … `ipc-2014` | `potassco/pddl-instances` | deterministic IPC tracks |
| `ippc-2006`, `ippc-2008` | `probfd/ppddl-benchmarks` | PPDDL probabilistic tracks |

Only `api.github.com` needs to be reachable. Unauthenticated requests are
rate-limited to 60/hour; set `GITHUB_TOKEN` to raise the limit.

**Caching**: files land in `$RAILROAD_PDDL_CACHE_DIR` (default
`~/.cache/railroad/pddl`) and are never re-downloaded; everything works
offline once cached. Directory listings are cached too, so upstream
additions or renames stay invisible until you run
`railroad pddl clear-cache`.

## Mapping semantics

| PDDL concept | railroad mapping |
|---|---|
| single implicit agent | synthetic serializing agent: every action requires `free agent`, consumes it at t=0, releases it on completion (satisfies the C++ core's hardcoded `free` semantics) |
| `(:metric minimize (total-cost))` | each action's `(increase (total-cost) …)` amount (constant or `:init` function value) becomes its **duration**, so minimizing completion time is exactly minimizing total cost; zero-cost actions get an ε=1e-3 duration so time advances |
| no metric / `minimize (total-time)` | duration 1 per action (PDDL's implicit objective is plan length) |
| `(:metric maximize (reward))` **with** `(:goal-reward …)` and no reward-bearing effects | reinterpreted as reach-the-goal, minimize expected plan length (IPPC goal-directed convention) |
| `(probabilistic p₁ e₁ …)` | railroad probabilistic effect branches; probabilities < 1 total get an implicit "nothing happens" remainder branch; nesting and multiple independent `probabilistic` groups per action are supported |
| `(when c e)` conditional effects | native railroad conditional branches (state-selected analogue of probabilistic branches): the branch fires iff its condition fluents hold when the action's completion effect fires, evaluated **before** the effect's own fluents apply — so conditions see the pre-action state, matching PDDL. `forall`+`when` expands per object; `when` inside probabilistic branches is supported (its condition then sees the state at branch resolution); `(= …)` conditions are evaluated away at grounding time (a false equality drops that grounding's branch; a true one is removed from the condition). Triggered sub-effects apply **after** the parent effect's own fluents — see the branching-effect ordering note under "Grounding notes" |
| objective overall | railroad's MCTS maximizes `−(expected completion time + Σ extra_cost)`, so converted problems minimize expected cost/time to goal |
| `forall` (preconditions, unconditional effects, goals) | expanded over the finite object universe at conversion time |
| `exists` (preconditions) | the quantified variable is lifted into an extra operator parameter (grounding enumerates witnesses) |
| goals: `and`/`or`/`not`/`forall`/`exists` over literals | railroad `AndGoal`/`OrGoal`/negated-literal goal trees |
| type hierarchy | flattened: `objects_by_type[t]` contains objects of `t` and all subtypes |
| `(= ?x ?y)` / `(not (= ?x ?y))` | evaluated while grounding (bindings filtered) |
| predicates named `free`, `waiting`, or `not-*` | transparently renamed with a `pddl-` prefix. `free` and `waiting` are keywords to the railroad core: `free X` means "agent X is available" and drives the concurrency machinery (a free agent makes the state a decision point; the synthetic serializing agent relies on exactly this). A PDDL domain's `free` is an ordinary predicate with unrelated meaning — gripper's `(free ?gripper)` says a hand is empty — and left unrenamed it would make the core treat every gripper hand as a schedulable agent. `not-*` is reserved as the bookkeeping prefix for compiled negative preconditions. A domain that already contains both a reserved name and its `pddl-`-prefixed form is rejected (`predicate-rename-collision`) rather than silently merged |

Grounding notes:

- Grounding uses `railroad.core.ground_operators`, a backtracking enumerator
  with **static-precondition pruning** (preconditions on predicates no effect
  ever touches are checked against `:init` as soon as their variables are
  bound). This keeps large domains tractable — e.g. IPC-2000 freecell grounds
  to ~34k actions instead of a combinatorial blowup.
- **Static material is compiled away** (see `docs/design/static-grounding.md`):
  verified static preconditions are stripped from grounded actions, static
  conjuncts of `when` conditions are evaluated per grounding, and static
  facts nothing references at runtime (only goal-referenced ones survive)
  are dropped from the initial state. Plans and reachability are unchanged —
  states just stop carrying immutable facts through every hash and copy.
- Unlike `Operator.instantiate`, the same object may bind several parameters
  (PDDL permits it; domains that need distinctness say `(not (= ?x ?y))`).
  Same-fluent add/delete pairs created by such bindings (e.g. `fly apt apt`)
  behave the PDDL way: the railroad core applies deletes before adds, so the
  add wins and the fluent survives.

  This holds **within a single effect**. Branching sub-effects (`when` and
  `probabilistic`) are separate effects applied **after** their parent, even
  at the same timestamp, so a triggered branch that deletes a fluent the
  parent adds leaves it absent. PDDL leaves add/delete conflicts across an
  `and`-effect's components undefined in practice (planners disagree);
  railroad resolves them by sequencing branches after their predecessors,
  which keeps effect application local to one effect at a time. Domains that
  rely on whole-effect simultaneity (an unconditional add "winning" over a
  conditional delete of the same fluent) will diverge here.
- Groundings whose cost function value is missing from `:init` are treated
  as undefined and skipped.

## Supported PDDL features

- `:strips`, `:typing` (hierarchies), `:negative-preconditions`, `:equality`
- `:universal-preconditions`, `:existential-preconditions`,
  `:quantified-preconditions`, `forall` effects
- `:conditional-effects` — `(when c e)` with conditions that flatten to a
  conjunction of (possibly negated) literals, `forall`/`and` in conditions,
  static `(= …)`/`(not (= …))` conditions, `forall`+`when`, and `when`
  nested inside probabilistic branches
- `:action-costs` (`total-cost` only, including per-binding function values)
- `:probabilistic-effects` (PPDDL), rational probabilities (`1/2`), nested
  and multiple probabilistic groups, `(:goal-reward …)` goal-directed reward
  metrics
- Files that bundle the domain and problem in one `(define …)` pair
  (IPPC-2008 style)
- Metrics: none, `minimize (total-cost)`, `minimize (total-time)`,
  `maximize (reward)` (goal-reward-only)

## Unsupported PDDL features (error slugs)

| Slug | Feature | Notes |
|---|---|---|
| `disjunctive-preconditions` | `(or …)` in preconditions | intended future work: DNF operator-splitting |
| `imply-conditions` | `(imply …)` | same compilation as disjunctions |
| `conditional-effect-condition` | `or`/`exists`/`imply` inside a `when` condition | conjunctive conditions (incl. quantified/equality) are supported |
| `durative-actions` | PDDL 2.1 temporal actions | ironic for a temporal planner — needs numeric durations + `at start`/`at end` effects; a natural follow-up |
| `numeric-conditions` / `numeric-effects` | numeric fluents beyond `total-cost` | includes `decrease`, `assign`, arithmetic cost expressions |
| `rewards` | `(increase (reward) …)` in effects | reward-shaping MDPs don't map to reach-goal planning |
| `metric:*` | any other objective | e.g. `maximize (reward)` without a goal-reward |
| `derived-predicates` | `(:derived …)` axioms | |
| `either-types` | `(either …)` types | |
| `oneof-nondeterminism` | FOND `(oneof …)` | non-probabilistic nondeterminism |
| `negated-compound-condition` | `(not (and …))` etc. | only literal negation supported |
| `preferences` / `constraints` | PDDL 3 soft goals / trajectory constraints | |
| `probabilistic-cost` / `conditional-cost` | `total-cost` increase inside a probabilistic or conditional branch | cannot map to a fixed duration |
| `timed-initial-literals` | `(at 5 (p))` in `:init` | |
| `object-fluents` | non-boolean fluents | |

## Compatibility scan results

`railroad pddl check <collection>` (first instance per domain), 2026-07-07.

These tables are the converter's contract with the IPC collections and are
checked by `tests/pddl_converter/test_ipc_sweep.py`, which re-derives every
status and compares it here. It needs the GitHub API, so it is gated:

```bash
RAILROAD_PDDL_NETWORK_TESTS=1 uv run pytest -q -k ipc_sweep
RAILROAD_PDDL_NETWORK_TESTS=1 uv run pytest -q -k ipc_solves -m slow
```

A full sweep needs ~130 API calls against the 60/hour anonymous limit, so set
`GITHUB_TOKEN` or let the (permanent) cache fill over several runs; an
incomplete run skips rather than reporting a green tick it did not earn.

**ipc-2000 — 11/12 domains convert**

| Domain | Status | Notes |
|---|---|---|
| blocks-strips-{typed,untyped} | ok | 40 actions; solved optimally |
| elevator-strips-simple-{typed,untyped} | ok | |
| elevator-adl-simple-typed | ok | forall+when boarding; instance 1 solved optimally |
| schedule-adl-{typed,untyped} | ok | conditional effects with equality conditions |
| freecell-strips-{typed,untyped} | ok | 8.4k/34k actions |
| logistics-strips-{typed,untyped} | ok | instance 1 solved in 21 steps |
| elevator-adl-full-typed | unsupported | imply-conditions |

**ippc-2006 (PPDDL) — 9/10 domains convert**

| Domain | Status | Notes |
|---|---|---|
| blocksworld | ok | solved 5/5 seeds with `--planner greedy` |
| drive-unrolled, elevators, pitchcatch, random, schedule, zenotravel | ok | conditional effects (some with equality conditions) |
| ex-blocksworld | ok | converts + executes correctly; solving blocked by dead ends (detonations) |
| tireworld | ok | converts, but unsolved: dead ends (see below) |
| drive | unsupported | disjunctive-preconditions |

**ippc-2008 (PPDDL) — 5/10 domains convert**

| Domain | Status | Notes |
|---|---|---|
| blocksworld | ok | bundled domain+problem files |
| boxworld, ex-blocksworld, schedule | ok | conditional effects |
| triangle-tireworld | ok | instance 1 solved |
| search-and-rescue | unsupported | imply-conditions |
| rectangle-tireworld, zenotravel | unsupported | numeric-effects |
| sysAdmin-SLP | unsupported | rewards |
| 2-tireworlds | parse-error | source repo ships no domain file for it |

Later deterministic years (spot-checked): `:action-costs` domains convert —
e.g. ipc-2011 `elevator-sequential-optimal` grounds to 1793 actions with
function-valued costs mapped to durations.

## Planner notes

- `solve(..., planner="mcts")` replans with `MCTSPlanner` each step;
  `planner="greedy"` picks the applicable action minimizing the expected
  (time + FF heuristic) over the outcome distribution. Greedy is much more
  robust on IPPC domains full of degenerate-but-legal actions (e.g.
  probabilistic blocksworld), while MCTS gives better plans on deterministic
  domains.
- **Dead ends are not avoided by MCTS.** Two separate reasons: (a) the FF
  heuristic is a delete-relaxation, so it cannot see that driving without a
  spare risks an unrecoverable flat (ippc-2006 tireworld); (b) even when the
  relaxation *does* flag a state as hopeless (h = inf, e.g. after breaking
  an object a negated goal needs intact), MCTS currently clamps that to
  `HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 0`, which makes such states look
  goal-adjacent. Raising the penalty perturbs multi-robot search-ordering
  behavior, so it stays 0 for now; dead-end-aware planning is future planner
  work. The `greedy` planner uses the unclamped heuristic and does steer
  around type-(b) dead ends (see the `fragile-delivery` feature benchmark).
- **Conditional effects and the heuristic**: the relaxed-plan heuristic
  optimistically assumes `when` conditions hold (relaxation fires every
  conditional branch), so conditionally-achievable goals get finite h values
  but the heuristic may underestimate when a condition is hard to establish.
- Competition-scale instances (e.g. IPC-2011 elevators) convert correctly
  but may need far more MCTS iterations than the defaults to solve.
