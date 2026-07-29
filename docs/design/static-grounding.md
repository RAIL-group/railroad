# Design: First-Class Static Preconditions and a Shared Grounder

**Status:** implemented (phases 1a/1b/2; phases 3–4 remain per-setting
follow-ups). Notable deltas from this document as written:

- The `goal=` parameter became `runtime_referenced=`, a predicate set. (The
  original rationale — "Goal objects are not introspectable from Python" — was
  wrong; `Goal.get_all_literals()` and `extract_negative_goal_fluents` both
  walk goal trees. The predicate set is simply the smaller interface.)
- `free`/`waiting` are always treated dynamic by `Environment` (the C++ core
  mutates them outside operator effects).
- `invalidate_grounding()` was reinstated for the one case compare-on-call
  keys cannot see: mutable state captured inside operator callables (e.g.
  replay policy swaps).
- **§4.4's state-level elimination was withdrawn on the `Environment` path.**
  Grounding still strips verified static preconditions from grounded actions,
  but it no longer removes facts from the environment's fluent set.
  `eliminable_fluents` answers "what do *these operators* reference", which is
  not the same question as "what does this environment reference": goals,
  branch conditions on effects a state is carrying, and subclass code all read
  fluents that grounding cannot see. A regression test pins the case that
  caught this — a `when` condition on a state-carried effect silently stopped
  firing. `Environment.static_facts` survives as a read-only view.
  Dropping search-irrelevant fluents moved to the planner, where the problem
  *is* closed (§9). The converter path keeps its filtering: it owns the whole
  problem and passes `runtime_referenced=` for the goal.

**Scope:** `railroad.core` (grounding), `railroad.environment` (integration),
`railroad.pddl_converter` (adoption)

---

## 1. Problem

Railroad currently has three mechanisms that decide which grounded actions
exist, and two of them are invisible to the person reading an operator
definition:

1. **`Operator.instantiate()`** enumerates the full typed cross-product of
   parameters and silently enforces an *implicit all-distinct rule*: a
   binding is dropped if any object binds two parameters, regardless of type
   (`len(set(binding.values())) != len(binding)`). Neither the enumeration
   cost nor the distinctness rule is visible in the operator.
2. **Environment-side filtering** hides constraints after the fact:
   `move_time` callables return `inf` for unreachable location pairs (then
   the action filter drops the action), and `ObjectSearchEnvironment` applies
   name-keyed hygiene rules (`move`/`place`/`search` conventions). A reader
   of the operator cannot tell which groundings will actually exist.
3. **The PDDL converter's private grounder** does this properly — backtracking
   enumeration with static-precondition pruning, duplicate bindings, and
   equality constraints — but it is unavailable to native railroad domains
   and duplicates grounding logic that ought to live in one place.

The costs are concrete:

- **Clarity.** Constraints like "you can only move between connected
  locations" are enforced by machinery the user never declared and cannot
  see in the operator.
- **Performance.** `instantiate()` materializes every binding before any
  filtering. On IPC-2000 freecell instance 1 the nominal cross-product is
  **46.6 M bindings**; the converter's backtracking grounder produces the
  **8,408** real actions in **0.5 s**. Native domains pay the same tax at
  smaller scale: moves ground as |locations|² and are then hidden by `inf`
  durations — significant in large frontier-search / LSP settings.
- **Downstream cost.** Grounded-but-dead actions are scanned by
  `get_next_actions` every step, relaxed over by the FF heuristic, and
  branched on by MCTS. Actions that never exist cost nothing anywhere.

## 2. Design principles

1. **Constraints are ordinary preconditions.** A static constraint is written
   as a normal precondition on a predicate no effect ever touches
   (`F("connected ?from ?to")`), exactly as PDDL domains write `road ?a ?b`.
   The only new construct is equality (§3.2), which PDDL also treats
   specially.
2. **One source of truth for facts.** Static facts live in the initial state
   like every other fluent, as in PDDL's `:init`. No separate facts channel.
3. **Staticness is inferred, never declared.** A predicate is static iff it
   appears in no effect of any operator in the set — the same inference the
   converter's `_find_static_predicates` performs today (and the same as
   Fast Downward's translator). An optional `assert_static=` exists purely
   for validation.
4. **Semantics first, optimization second — and separately.** Pruning during
   grounding never changes what a grounded action *means*. The optimizations
   that do rewrite actions/states (stripping verified preconditions,
   eliminating static fluents from the runtime state, simplifying
   conditional-branch conditions) are **individually flagged and default
   off**, adopted only after the IPC parity evidence is in (§6, §7).
5. **PDDL's duplicate-binding convention, made explicit.** Duplicate bindings
   are legal in PDDL; domains opt out visibly with `(not (= ?x ?y))`. The
   grounder exposes this as a boolean; no existing call site changes behavior
   silently.
6. **Nothing breaks.** `Operator.instantiate()` is untouched. Every phase is
   independently shippable and behavior-preserving until a call site
   explicitly opts in. The converter's current grounding behavior is treated
   as a **compatibility contract**, verified by a dual-run parity harness
   before the private grounder is deleted (§7).

## 3. User-facing API

### 3.1 Writing a constrained operator (the 90% case)

Nothing new to learn — the constraint is a precondition; the facts are state:

```python
move = Operator(
    name="move",
    parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
    preconditions=[
        F("at ?r ?from"), F("free ?r"),
        F("connected ?from ?to"),          # static: no effect touches it
    ],
    effects=[
        Effect(time=0, resulting_fluents={~F("free ?r"), ~F("at ?r ?from")}),
        Effect(time=(move_time, ["?r", "?from", "?to"]),
               resulting_fluents={F("free ?r"), F("at ?r ?to")}),
    ],
)

initial_state = State(0.0, {
    F("free r1"), F("at r1 kitchen"),
    F("connected kitchen hall"), F("connected hall kitchen"),
    F("connected hall office"),  F("connected office hall"),
})
```

`connected` is detected as static automatically. Groundings of `move` that
violate it are never created. Nothing about this operator requires the
environment to know anything.

### 3.2 Equality: the one new construct

PDDL's built-in `=` is the only constraint that is not an ordinary predicate
(its extension is identity, it needs no facts, and it may never appear in an
effect). Native operators get a direct equivalent:

```python
from railroad.core import Neq   # and Eq, for the rare positive case

swap = Operator(
    name="swap",
    parameters=[("?a", "item"), ("?b", "item")],
    preconditions=[F("held ?a"), F("held ?b"), Neq("?a", "?b")],
    effects=[...],
)
```

`Neq`/`Eq` are grounding-time constraints: they appear in the
`preconditions` list (so they read like PDDL), are evaluated the moment both
variables bind, and never appear on the grounded `Action`.
`Operator.instantiate()` raises if it encounters one — accidental use with
the legacy path is loud, not silent.

### 3.3 The grounding entry point

```python
from railroad.core import ground_operators

result = ground_operators(
    operators,                        # full operator set (staticness is inferred from it)
    objects_by_type,
    initial_fluents,                  # facts checked during grounding
    allow_duplicate_bindings=False,   # False = instantiate's all-distinct rule
    skip_on=(),                       # exception types that skip a single grounding
    assert_static=None,               # optional {"connected", ...} — validation only
    eliminate_static=False,           # opt-in optimization, see §4.4
)

result.actions              # list[Action]
result.static_predicates    # frozenset[str] — what was inferred static
result.eliminable_fluents   # set[Fluent] — safe to drop from runtime state
                            #   (empty unless eliminate_static=True)
result.stats                # bindings visited / nominal product / actions kept
```

Notes on the signature:

- `allow_duplicate_bindings: bool = False`. The default reproduces
  `instantiate()`'s all-distinct rule **exactly** (no object binds two
  parameters, regardless of type), so native call sites migrate with zero
  behavior change. The converter passes `True` (PDDL semantics). `Neq` works
  in both modes; under `False` it is simply always satisfied.
- `skip_on` supports the converter's undefined-cost behavior: a grounding
  whose duration callable raises one of these exception types is skipped,
  not fatal. The converter passes `(_UndefinedFunctionValue,)`, preserving
  today's "cost value missing from `:init` → skip that grounding" rule
  without the core knowing anything about PDDL functions.
- `assert_static` never *makes* anything static. It raises if a listed
  predicate is touched by some effect — catching "I thought this was static"
  at grounding time instead of as a silent mis-prune.
- Actions with non-finite effect times are filtered here (centralizing the
  existing base-`Environment` rule), so `inf`-duration sentinels keep
  working.

### 3.4 Environment integration

```python
class Environment:
    def get_actions(self) -> List[Action]:
        # grounds via ground_operators(self._operators, self._grounding_objects(),
        #                              self._static_fact_snapshot(), ...)
        # cached; recomputed only when the cache key changes (see below)
```

- Users put static facts in the initial `State` — no new hook.
- The grounding is **cached**, keyed on
  `(self._grounding_objects() snapshot, static-fact snapshot)`. The key is
  recomputed and compared on every call (cheap — object sets are small
  relative to grounding), so the two legitimate change vectors are handled
  automatically, with no explicit invalidation API:
  - `ObjectSearchEnvironment` revelation adds objects to `objects_by_type`
    → objects snapshot changes → reground.
  - Registering `robot1_loc` after an interrupted move changes the location
    universe (and, in the later phase 4, adds its `connected` facts) → key
    changes → reground.
- With `eliminate_static=True` (a later, opt-in phase), the environment
  strips `eliminable_fluents` from its runtime fluent set at construction and
  keeps them in a `static_facts` property — still facts about the world for
  ground-truth queries and re-grounding, just not carried in every planner
  state.

### 3.5 What stays the same

- `Operator.instantiate()` keeps its exact current behavior (full
  enumeration, implicit all-distinct, no static awareness). Its docstring
  gains a paragraph saying precisely that, with a pointer to
  `ground_operators`. Deprecation is not part of this proposal.
- `move_time`-returns-`inf` continues to work — it remains the right tool for
  *dynamic* cost infeasibility. It stops being the idiom for *static*
  connectivity.
- Grounding and static inference operate on **original operators**, before
  the planner's negative-precondition (`not-*`) conversion. That conversion
  introduces effects on `not-P` bookkeeping fluents, which must not affect
  staticness — the existing pipeline ordering (ground → plan) already
  guarantees this; it is now stated as a contract.

## 4. Semantics (precise)

### 4.1 Static inference

A predicate is **static** with respect to an operator set iff it does not
appear in any `resulting_fluents`, probabilistic branch, conditional branch
*effect* (conditions only read), or `ForallEffect` sub-effect, recursively,
of any operator in the set. This matches the converter's
`_find_static_predicates` exactly. `Neq`/`Eq` are always static.

Environment contract: code outside the operator set (revelation, skill
hooks) must not mutate fluents of a static predicate, except through the
static-fact snapshot that participates in the grounding cache key (§3.4).
`ObjectSearchEnvironment` today mutates only dynamic predicates
(`at`/`found`/`revealed`/`searched` in any domain that searches), so the
contract holds; a debug assertion is cheap to add.

### 4.2 Constraint evaluation during grounding

- Every **positive** static precondition and every `Neq`/`Eq` is evaluated as
  soon as all of its variables are bound (constants count as bound); failure
  prunes the whole binding subtree.
- **Negated** static preconditions are **not** checked at grounding and
  remain runtime preconditions. This matches the converter's current,
  deliberate behavior (`converter.py`: "a negative static precondition
  rarely prunes; skip"). Checking them at grounding would be semantically
  sound and would *shrink* action sets (removing never-applicable actions),
  but that is an observable change and therefore belongs to the flagged
  optimizations (§4.4), not the parity baseline.

### 4.3 What grounding does *not* change (the parity baseline)

In the baseline (all optimization flags off), `ground_operators` is a pure
pruning device. Matching the converter today:

- Grounded actions **keep** their static preconditions.
- Static facts **stay** in the initial/runtime state.
- Equality conditions inside `when` branches continue to compile via the
  seeded `pddl-eq` mechanism, unchanged.

A grounded action produced by the baseline is structurally identical to one
produced today — same name, same preconditions, same effects. This is what
makes the parity harness (§7) meaningful.

### 4.4 Flagged optimizations (each individually opt-in, default off)

1. **`eliminate_static=True`** — strip verified positive static
   preconditions from grounded actions, and report as `eliminable_fluents`
   every static fluent whose predicate is, after stripping, referenced by
   **none** of: any grounded action's preconditions, any conditional-branch
   condition, the goal, or a caller-declared runtime-reader list. Anything
   still referenced simply stays in the state — correctness never depends on
   elimination. (Deliberately, there is **no goal-tree simplification**: a
   goal-referenced static fluent just stays in the state. Rewriting goals
   buys little and risks much.)

   Elimination is semantically neutral by construction (it removes only
   grounding-verified, immutable material), so **default-off is a
   verification stance, not the end state**: it exists so the phase-1a
   parity harness can compare old and new grounder output for *identity* —
   the strongest available check, and one we only get while both grounders
   coexist. Once phase 1b's evidence passes with the flag on, the default
   flips to `True` everywhere; `eliminate_static=False` survives as the
   debugging/compat mode ("show me everything"). One interop contract comes
   with the flip: actions built through legacy `instantiate()` keep their
   static preconditions, so mixing them with an eliminated state fails
   precondition checks — the environment raises a descriptive error if a
   foreign action references an eliminated static predicate, rather than
   failing obscurely.
2. **Grounding-time conditional simplification** — evaluate static conjuncts
   of `when` conditions per grounding (false → drop the branch from that
   grounding; true → remove the conjunct), which lets the converter retire
   `pddl-eq` seeding. Changes `GroundedEffect` structure (hashes, pickles,
   the `test_equality_in_when_condition` assertions), so it ships behind its
   own flag with its own IPC re-verification.
3. **Negated-static grounding checks** — prune bindings whose negated static
   preconditions are violated (see §4.2). Shrinks action sets; flagged.

### 4.5 Duplicate bindings

`allow_duplicate_bindings=True` matches PDDL: all type-consistent bindings
exist; distinctness is expressed per-operator with `Neq`.
`False` matches `instantiate()`'s legacy rule verbatim. The flag is
per-call; there is no global mode.

## 5. Implementation

### 5.1 The grounder (hoisted, not rewritten)

The converter's backtracking enumerator (`_ground_operator`,
`_order_parameters`, the check-scheduling logic) moves from
`pddl_converter/converter.py` into `railroad/core.py`, generalized only
where the converter's types are PDDL-specific (its `Literal`/`Equals` nodes
become `Fluent`/`Neq`/`Eq`). Structure, unchanged from what the converter
does today:

1. **Compile checks per operator.** Positive static preconditions + equality
   constraints; each check is assigned the earliest parameter depth at which
   all its variables are bound.
2. **Order parameters greedily** so checks fire as early as possible (the
   converter's existing heuristic) — this is what prevents the "static check
   on parameters 6 and 7 of 7" case from walking the prefix product.
3. **Backtrack** over bindings; run each depth's checks; prune subtrees on
   failure. Fact lookup via a predicate-name index
   (`dict[str, set[tuple[str, ...]]]`).
4. **Emit** each surviving binding through the existing
   `Operator._ground(binding, objects_by_type)` path — cond effects, forall
   expansion, duration callables all work as today. A `skip_on` exception
   during `_ground` skips that single binding.
5. **Post-filter** non-finite effect times.

**v2 optimization (not required for parity):** candidate-driven enumeration —
when a check has exactly one unbound variable and it is the next parameter,
enumerate its candidates from the fact index (e.g. `connected ?from ?to`
walks neighbors: O(|edges|) visited instead of O(|locations|²)
visited-then-pruned). Plain early checking already delivers the freecell
result; this matters for very large sparse relations.

### 5.2 Converter adoption

`pddl_converter` deletes its private grounder and calls
`ground_operators(..., allow_duplicate_bindings=True,
skip_on=(_UndefinedFunctionValue,))`. PDDL `(not (= ...))` / `(= ...)` map
to `Neq`/`Eq`. `pddl-eq` seeding is untouched in this phase (§4.3).

### 5.3 Environment adoption

`Environment.get_actions()` switches to the cached path with
`allow_duplicate_bindings=False`. Because no existing native operator has
static preconditions or `Neq`, the grounder degenerates to exactly today's
enumeration + all-distinct rule — behavior-preserving by construction, now
cached. The surrounding logic (`_grounding_objects()` hook, `_is_valid_action`
filtering, the duplicate-action-name check) is unchanged.

## 6. Migration plan

Each phase ships independently and leaves the system consistent:

- **Phase 1a — the hoist (parity-critical).** `Neq`/`Eq`,
  `ground_operators` with the baseline semantics of §4.3, static inference,
  `skip_on`; converter adopts it and deletes its grounder **only after** the
  §7 harness passes. No observable behavior changes anywhere.
- **Phase 1b — flagged optimizations, then flipped defaults.**
  `eliminate_static`, conditional simplification, negated-static checks —
  each lands behind its flag with the §7 evidence re-run before adoption.
  The converter is the first adopter (its problems benefit most and are the
  best-instrumented). The phase *ends* by flipping `eliminate_static` (and,
  evidence permitting, the other flags) to default-on everywhere: the flags
  are a verification schedule, not a permanent posture (§4.4).
- **Phase 2 — environment.** `get_actions()` grounds through the new path
  (`allow_duplicate_bindings=False`, no optimization flags) with caching.
  Behavior-identical for all existing domains; strictly faster.
- **Phase 3 — native domains opt in.** Examples and `railroad.operators`
  helpers gain optional connectivity (`connected` facts + precondition
  instead of `inf` move times) where each setting wants it; settings that
  relied on implicit distinctness write `Neq` and may flip
  `allow_duplicate_bindings=True`. This is where constraints currently
  enforced inside `ObjectSearchEnvironment` move into operator definitions,
  one setting at a time.
- **Phase 4 — convention cleanup (optional).** With `connected` facts
  standing in for reachability (including `robot_loc` registration adding
  its facts), revisit `ObjectSearchEnvironment`'s `_loc` hygiene filters and
  consider retiring the name-keyed rules.

## 7. Verification: the IPC compatibility contract

The converter's grounding behavior on the problems we have studied is the
contract. Concretely:

1. **Dual-run parity harness (pre-deletion).** While both grounders exist in
   the tree, a test grounds every vendored offline problem **and** the first
   instance of every convertible domain in ipc-2000 (11), ippc-2006 (9), and
   ippc-2008 (5) through both paths and asserts *structural* equality —
   action name sets, per-action preconditions, effects (times, fluents,
   prob/cond branches), `extra_cost`. Name-set equality alone is not enough;
   §4.3 exists precisely so this comparison can be exact. (The IPC sweep
   needs the download cache; it runs behind the existing
   `RAILROAD_PDDL_NETWORK_TESTS` gate, with the vendored problems covering
   the feature matrix offline: duplicate bindings, inequality, static
   pruning, quantifiers, `pddl-eq`, undefined costs, probabilistic effects.)
2. **`railroad pddl check` table unchanged** for all three collections —
   same convert/unsupported/error status and notes per domain.
3. **Solved-instance regression** — the end-to-end runs recorded in the
   converter README stay green with identical plans under fixed seeds:
   IPC-2000 blocks and logistics solved optimally, miconic (`forall`+`when`)
   solved optimally, IPPC-2008 triangle-tireworld and IPPC-2006 blocksworld
   solved.
4. **Native regression** — the full existing test suite, plus new unit tests
   for: static inference over prob/cond/forall effects, early-bound pruning,
   `Neq`/`Eq` in both duplicate modes, `skip_on`, the all-distinct parity of
   `allow_duplicate_bindings=False` against `instantiate()` on identical
   inputs, grounding-cache key sensitivity (revelation's `objects_by_type`
   mutation; `robot_loc` registration), and a freecell-shaped synthetic
   domain asserting visited-bindings ≪ nominal product via `result.stats`.
5. **Per-flag re-verification.** Each §4.4 flag repeats items 1–3 with the
   flag on before any call site adopts it (item 1 compares against the
   flag's *specified* transform rather than identity, e.g. "identical except
   static preconditions absent").

### 7.1 Test conventions: elimination-agnostic by default

Since `eliminate_static` ends up default-on (§4.4), tests are written so the
flip is a non-event. The convention, applied to new tests immediately and to
existing tests as part of the flip:

- **Assert behavior, not structure.** Prefer "the plan reaches the goal",
  "this action is applicable / not applicable"
  (`state.satisfies_precondition`, `get_next_actions`), and membership of
  **dynamic** fluents (`F("at r1 kitchen") in state.fluents`) — none of
  which elimination can affect.
- **Never assert full state sets or full precondition sets.** Whole-set
  equality (`state.fluents == {...}`) and static-fluent membership
  (`F("connected a b") in action.preconditions`) are the two assertion
  shapes elimination breaks. If a test needs "these dynamic facts and
  nothing else dynamic", it filters by predicate rather than comparing the
  raw set.
- **Tests about grounding structure pin the flag explicitly.** The parity
  harness passes `eliminate_static=False` by definition; the elimination
  unit tests pass `True`; neither depends on the ambient default, so
  flipping it changes nothing in the suite.
- **Flip-time audit.** The default flip includes a sweep of existing tests
  for the two forbidden assertion shapes. Known offenders today are few and
  converter-local (e.g. `test_reserved_predicates_renamed` asserts
  `pddl-waiting x1` — a static predicate — in the initial state; it gets
  rewritten to assert renaming behavior through a grounded action instead).
  The sweep converts each to a behavioral assertion where possible and pins
  the flag only where structure is genuinely the subject.

## 8. Revision notes (v1 → v2)

Code review of the actual converter grounder corrected four v1 claims that
would have violated the "don't break what we have" requirement:

1. v1 stripped static preconditions and eliminated static state fluents *by
   default*. The converter does neither — grounded actions keep static
   preconditions and static facts stay in the state — so both are now
   flagged optimizations (§4.4) with the baseline defined as today's
   behavior (§4.3).
2. v1 claimed negated static preconditions are checked at grounding via
   negation-as-absence. The converter deliberately skips them; the baseline
   now matches, and grounding-time negated checks are a flagged optimization.
3. v1 replaced `pddl-eq` seeding with grounding-time conditional
   simplification as part of the initial hoist. That changes
   `GroundedEffect` structure and existing test assertions, so it moves
   behind its own flag in phase 1b.
4. v1 omitted the undefined-cost skip mechanism (`_UndefinedFunctionValue`)
   entirely; `skip_on` now carries it.

Also per review: the duplicate-bindings mode is a plain boolean
(`allow_duplicate_bindings`) rather than a string enum, and the explicit
`invalidate_grounding()` API is dropped in favor of compare-on-call cache
keys, which automatically handle revelation's `objects_by_type` mutation —
a case v1's explicit-invalidation design would have missed.

---

## 9. Relevance projection (supersedes §4.4's state-level elimination)

The elimination §4.4 specified — dropping static facts nothing references from
the runtime state — is correct in principle and was implemented on both
adopters. It is sound on the converter path and unsound on the environment
path, for a reason worth recording.

**The question grounding can answer** is "which predicates do *these operators*
read?" — precondition literals and conditional-branch conditions, both visible
in the operator set. **The question elimination needs answered** is "which
predicates does *anything* read?" On the converter path these coincide: the
converter builds the initial state (with no pre-existing upcoming effects),
knows the goal, and is the only reader. On the environment path they diverge:

- goal literals (§4.4 anticipated this; `runtime_referenced=` covers it only
  if the caller supplies them, and `Environment` has no goal),
- conditional-branch conditions on effects carried in a `State` or queued by a
  skill — grounding never sees these, and this is what broke in practice,
- subclass code reading `env.fluents` directly,
- foreign `Action`s built outside `get_actions()` (§4.4's own interop caveat).

The planner has the closed world the environment lacks: at call time it holds
every action, the goal, and the state including its queued effects. So the
projection moved there, and got stronger in the process:

```python
relevant = relevant_predicates(actions, goal, state.upcoming_effects)
```

A fluent is *read* only through an action precondition, a branch condition
(in an action or a queued effect), the goal, or the core's name-keyed
machinery — `free`/`waiting` drive the concurrency logic in `transition()`,
and the FF heuristic's `at_implies_found` rule keys on `at`/`found`. Fluents
of every other predicate are written and never consulted, so two states
differing only in those are bisimilar: projecting them away preserves the
policy and merges more search nodes.

Two steps are required, not one. Projecting the root state alone leaks the
benefit back, because effects re-add the irrelevant fluents at each rollout
step; `project_action` therefore strips irrelevant adds from action effects
(recursively through probabilistic and conditional branches) once per relevance
set. Branch *conditions* are left alone — their predicates are relevant by
construction.

This subsumes static-fact elimination and extends it: it also removes
**dynamic write-only** predicates, which no staticness analysis can touch. On
a ring-connected search domain (40 locations, 280 actions, 242 fluents) MCTS
runs ~5.8× faster with projection on, choosing the same action.

`MCTSPlanner(project_irrelevant=False)` / `GreedyPlanner(project_irrelevant=False)`
search the full fluent set for debugging.
