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
uv run railroad pddl run --collection ippc-2006 --domain blocksworld --instance 1
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
- **Static material is compiled away** (see `railroad.core.ground_operators`):
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
```

The sweep checks *conversion* only. Whether a planner reaches the goal on a
given instance is a planner property and is deliberately not pinned here —
see the planner notes below for what is known, and the
`pddl_converter_features` benchmark for optional end-to-end demos.

A full sweep needs ~130 API calls against the 60/hour anonymous limit, so set
`GITHUB_TOKEN` or let the (permanent) cache fill over several runs; an
incomplete run skips rather than reporting a green tick it did not earn.

### Methodology for the solve columns

`solve()` on the **first instance** of each domain, seeds 0-2, 4000 MCTS
iterations, at most 300 steps. `default` is the shipped configuration;
`penalty` adds `dead_end_penalty=1e4`. Re-measured 2026-07-29.

Solve rates are **not** part of the checked contract — `test_ipc_sweep.py`
verifies conversion only. They are recorded because "converts" and "is
usable" are different claims and the difference is worth knowing.

A 10x budget (40,000 iterations) was also measured and **moved no domain in
either direction**: every failure below is a heuristic or dead-end problem,
not a search-effort one. The one thing it improved was plan *quality* —
logistics finds 20 steps instead of 21.

**ipc-2000 — 11/12 convert, 11/11 solved**

| Domain | Actions | default | penalty | Notes |
|---|---:|---|---|---|
| blocks-strips-{typed,untyped} | 40 | 3/3 | 3/3 | 6 steps (optimal) |
| elevator-adl-simple-typed | 4 | 3/3 | 3/3 | 4 steps; forall+when boarding |
| elevator-strips-simple-{typed,untyped} | 4 | 3/3 | 3/3 | 4 steps |
| freecell-strips-typed | 8408 | 3/3 | 3/3 | 9 steps |
| freecell-strips-untyped | 34436 | 3/3 | 3/3 | 9 steps |
| logistics-strips-{typed,untyped} | 164 | 3/3 | 3/3 | 21 steps (20 at 40k iterations) |
| schedule-adl-{typed,untyped} | 49 | 3/3 | 3/3 | 2 steps; equality conditions |
| elevator-adl-full-typed | — | — | — | unsupported: imply-conditions |

**ippc-2006 (PPDDL) — 9/10 convert, 5/9 solved**

| Domain | Actions | default | penalty | Notes |
|---|---:|---|---|---|
| blocksworld | 330 | 3/3 | 3/3 | 18-42 steps; MCTS wanders |
| elevators | 74 | 3/3 | 3/3 | 13 steps |
| schedule | 11 | 3/3 | 3/3 | 3-81 steps |
| zenotravel | 740 | 3/3 | 3/3 | **goal already true at t=0** — proves nothing |
| drive-unrolled | 44 | 2/3 | 2/3 | 6 steps; one seed hits the 300-step ceiling |
| ex-blocksworld | 60 | 0/3 | 0/3 | dead ends (detonations) |
| pitchcatch | 6 | 0/3 | 0/3 | 300-step ceiling |
| tireworld | 62 | 0/3 | 0/3 | dead ends |
| random | 2302 | — | — | not measured: >1200 s |
| drive | — | — | — | unsupported: disjunctive-preconditions |

**ippc-2008 (PPDDL) — 5/10 convert, 3/5 solved**

| Domain | Actions | default | penalty | Notes |
|---|---:|---|---|---|
| schedule | 14 | 3/3 | 3/3 | 3-81 steps |
| triangle-tireworld | 18 | 1/3 | **3/3** | penalty converts it |
| ex-blocksworld | 60 | 0/3 | **2/3** | penalty converts it |
| blocksworld | 305 | 0/3 | 0/3 | 300-step ceiling; bundled domain+problem |
| boxworld | 750 | — | — | not measured: >1200 s. Its goal has 9.7M DNF branches (see planner notes) |
| 2-tireworlds | — | — | — | parse-error: source repo ships no domain file |
| rectangle-tireworld, zenotravel | — | — | — | unsupported: numeric-effects |
| search-and-rescue | — | — | — | unsupported: imply-conditions |
| sysAdmin-SLP | — | — | — | unsupported: rewards |

Later deterministic years (spot-checked): `:action-costs` domains convert —
e.g. ipc-2011 `elevator-sequential-optimal` grounds to 1793 actions with
function-valued costs mapped to durations.

## Planner notes

- `solve()` replans with `MCTSPlanner` at every step, applying the chosen
  action to a sampled successor. It is the only planner this package uses;
  action selection is deliberately not a knob here, because the point of the
  converter is to validate the *conversion*, not to benchmark planners.
- **Dead ends are not avoided.** Two separate reasons: (a) the FF heuristic
  is a delete-relaxation, so it cannot see that driving without a spare risks
  an unrecoverable flat (ippc-2006 tireworld); (b) even when the relaxation
  *does* flag a state as hopeless (h = inf, e.g. after breaking an object a
  negated goal needs intact), MCTS clamps that to
  `HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 0`, which makes such states look
  goal-adjacent — the clamped value is the *best* available reward, so the
  search is actively drawn to them.

  `solve(..., dead_end_penalty=...)` / `MCTSPlanner(dead_end_penalty=...)`
  replaces that clamp with a **flat** failure cost: the branch's elapsed time
  and accumulated `extra_cost` are *not* added, so a doomed branch is worth
  exactly `-penalty` however long it took to fail, and failing slowly is not
  ranked below failing fast.

  It fixes class (b), and only nibbles at (a) — as expected, since (a) is the
  relaxation being blind rather than the reward being wrong. In the tables
  above it converts ippc-2008 triangle-tireworld (1/3 -> 3/3) and
  ex-blocksworld (0/3 -> 2/3), and it is what makes the `fragile-delivery`
  feature benchmark solvable at all (0/3 -> 3/3). It does not rescue the
  domains whose dead ends the delete-relaxation cannot see (ippc-2006
  tireworld, ex-blocksworld).

  It stays opt-in (default `None`) because it also perturbs multi-robot
  search-ordering ties (`test_mcts_search_picks_more_likely_location`,
  `test_procthor_object_search_replay_end_to_end`); making it the default is
  planner work, tracked separately from the converter.
- **The solve rates above are not domain limits.** They come from a fixed,
  fairly small budget (4000 iterations, `c=100`, unchanged exploration
  parameters). Raising the iteration count or tuning exploration has not been
  explored, so read these as "what the current defaults do", not as evidence
  about what the domains require.
- **Disjunctive goals and the heuristic**: `(forall ?x (exists ?y ...))` goals
  compile to an `AndGoal` of `OrGoal`s, whose DNF is a *product* — ippc-2008
  boxworld (10 boxes, 5 cities) is 5^10 = 9.7M branches. The FF heuristic used
  to materialise and walk all of them, which exhausted memory rather than
  merely being slow. It now checks `Goal.dnf_branch_count()` first and, above
  1024 branches, picks one conjunction greedily (cheapest disjunct per `OR` by
  optimistic cost) and runs the ordinary backward pass on it. Below the cap
  the exhaustive minimum is unchanged, and every other IPC domain's goal has a
  DNF of exactly 1 branch. Above it the result is an upper bound on the true
  minimum; unreachability stays exact.
- **Conditional effects and the heuristic**: the relaxed-plan heuristic
  optimistically assumes `when` conditions hold (relaxation fires every
  conditional branch), so conditionally-achievable goals get finite h values
  but the heuristic may underestimate when a condition is hard to establish.
- Competition-scale instances (e.g. IPC-2011 elevators) convert correctly
  but may need far more MCTS iterations than the defaults to solve.
