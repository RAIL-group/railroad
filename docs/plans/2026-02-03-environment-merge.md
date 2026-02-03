# Environment Merge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge `EnvironmentInterfaceV2` and `Environment` protocol into a single unified class hierarchy with base `Environment` class and `SymbolicEnvironment` subclass.

**Architecture:** Base `Environment` class provides concrete implementations for skill tracking, time management, state assembly, `get_actions()`, `act()`, and `is_goal_reached()`. Subclasses implement abstract methods: `fluents`, `objects_by_type`, `create_skill()`, `apply_effect()`, `resolve_probabilistic_effect()`.

**Tech Stack:** Python with ABC (abstract base classes), existing railroad core bindings

**Design Doc:** `docs/plans/2026-02-03-environment-merge-design.md`

---

## Task 1: Create Base Environment Class (Abstract Structure)

**Files:**
- Create: `packages/railroad/src/railroad/environment/environment.py`
- Test: `packages/railroad/tests/test_environment_base.py`

**Step 1: Write the failing test**

```python
"""Tests for base Environment class."""
import pytest
from abc import ABC


def test_environment_is_abstract():
    """Test that Environment cannot be instantiated directly."""
    from railroad.environment.environment import Environment

    assert issubclass(Environment, ABC)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Environment(state=None, operators=[])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/railroad/tests/test_environment_base.py::test_environment_is_abstract -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
"""Base Environment class for planning environments."""

import itertools
import math
from abc import ABC, abstractmethod
from typing import Callable, Collection, Dict, List, Optional, Set, Tuple

from railroad._bindings import Action, Fluent, GroundedEffect, State
from railroad.core import Operator

from .skill import ActiveSkill, SymbolicSkill


class Environment(ABC):
    """Base class for planning environments.

    Provides concrete implementations for:
    - Active skill tracking and time management
    - State assembly (fluents + upcoming effects)
    - Action instantiation from operators
    - The act() loop (execute until a robot is free)

    Subclasses implement environment-specific behavior:
    - How fluents are stored/retrieved
    - How skills are created for actions
    - How effects are applied
    - How probabilistic effects are resolved
    """

    def __init__(
        self,
        state: State,
        operators: List[Operator],
    ) -> None:
        self._operators = operators
        self._time: float = state.time
        self._active_skills: List[ActiveSkill] = []

        # Convert initial upcoming effects to a skill
        if state.upcoming_effects:
            initial_skill = self._create_initial_effects_skill(
                state.time, list(state.upcoming_effects)
            )
            self._active_skills.append(initial_skill)

    def _create_initial_effects_skill(
        self,
        start_time: float,
        upcoming_effects: List[Tuple[float, GroundedEffect]],
    ) -> SymbolicSkill:
        """Create a SymbolicSkill from initial upcoming effects."""
        relative_effects = [
            GroundedEffect(abs_time - start_time, effect.resulting_fluents)
            for abs_time, effect in upcoming_effects
        ]
        action = Action([], relative_effects, name="_initial_effects")
        return SymbolicSkill(action=action, start_time=start_time)

    @property
    def time(self) -> float:
        return self._time

    @property
    @abstractmethod
    def fluents(self) -> Set[Fluent]:
        """Current ground truth fluents."""
        ...

    @property
    @abstractmethod
    def objects_by_type(self) -> Dict[str, Set[str]]:
        """All known objects, organized by type."""
        ...

    @abstractmethod
    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Create an ActiveSkill for this action."""
        ...

    @abstractmethod
    def apply_effect(self, effect: GroundedEffect) -> None:
        """Apply an effect to the environment."""
        ...

    @abstractmethod
    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve which branch of a probabilistic effect occurs."""
        ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/railroad/tests/test_environment_base.py::test_environment_is_abstract -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/railroad/src/railroad/environment/environment.py packages/railroad/tests/test_environment_base.py
git commit -m "feat(environment): add abstract Environment base class structure"
```

---

## Task 2: Add Concrete Methods to Base Environment

**Files:**
- Modify: `packages/railroad/src/railroad/environment/environment.py`
- Test: `packages/railroad/tests/test_environment_base.py`

**Step 1: Write the failing test**

Add to `test_environment_base.py`:

```python
from railroad._bindings import Fluent as F, State
from railroad.core import Effect, Operator


class MinimalEnvironment(Environment):
    """Minimal concrete implementation for testing base class."""

    def __init__(self, state: State, operators: List[Operator], fluents: Set[F]):
        self._fluents_set = fluents
        self._objects = {"robot": {"robot1"}, "location": {"kitchen", "bedroom"}}
        super().__init__(state=state, operators=operators)

    @property
    def fluents(self) -> Set[F]:
        return self._fluents_set

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        return self._objects

    def create_skill(self, action, time):
        from railroad.environment.skill import SymbolicSkill
        return SymbolicSkill(action=action, start_time=time)

    def apply_effect(self, effect) -> None:
        for fluent in effect.resulting_fluents:
            if fluent.negated:
                self._fluents_set.discard(~fluent)
            else:
                self._fluents_set.add(fluent)

    def resolve_probabilistic_effect(self, effect, current_fluents):
        return [effect], current_fluents


# Add these imports at top
from typing import Dict, Set, List
from railroad.environment.environment import Environment


def test_environment_state_assembly():
    """Test that state property assembles fluents + upcoming effects."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    state = State(0.0, fluents, [])

    env = MinimalEnvironment(state=state, operators=[], fluents=fluents)

    assert env.time == 0.0
    assert F("at", "robot1", "kitchen") in env.state.fluents
    assert F("free", "robot1") in env.state.fluents


def test_environment_get_actions():
    """Test that get_actions instantiates from operators."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    state = State(0.0, fluents, [])

    move_op = Operator(
        name="move",
        parameters=["robot", "from", "to"],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[
            Effect(time=0.0, fluents=[~F("free ?robot")]),
            Effect(time=5.0, fluents=[~F("at ?robot ?from"), F("at ?robot ?to"), F("free ?robot")]),
        ]
    )

    env = MinimalEnvironment(state=state, operators=[move_op], fluents=fluents)
    actions = env.get_actions()

    action_names = [a.name for a in actions]
    assert "move robot1 kitchen bedroom" in action_names
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/railroad/tests/test_environment_base.py::test_environment_state_assembly packages/railroad/tests/test_environment_base.py::test_environment_get_actions -v`
Expected: FAIL with "AttributeError" (state, get_actions not implemented)

**Step 3: Write implementation**

Add to `Environment` class in `environment.py`:

```python
    @property
    def state(self) -> State:
        """Assemble state from fluents + upcoming effects from active skills."""
        self._update_skills()

        effects: List[Tuple[float, GroundedEffect]] = []
        for skill in self._active_skills:
            effects.extend(skill.upcoming_effects)

        return State(
            self._time,
            self.fluents,
            sorted(effects, key=lambda el: el[0]),
        )

    def _update_skills(self) -> None:
        """Advance skills to current time and remove completed ones."""
        for skill in self._active_skills:
            skill.advance(self._time, self)
        self._active_skills = [s for s in self._active_skills if not s.is_done]

    def get_actions(self) -> List[Action]:
        """Instantiate available actions from operators."""
        objects_by_type = self.objects_by_type

        # Add robot intermediate locations (robot_loc) to location set
        robot_locs = {
            f"{rob}_loc"
            for rob in objects_by_type.get("robot", set())
            if Fluent("at", rob, f"{rob}_loc") in self.fluents
        }
        objects_with_rloc: Dict[str, Collection[str]] = {
            k: set(v) for k, v in objects_by_type.items()
        }
        objects_with_rloc["location"] = (
            set(objects_with_rloc.get("location", set())) | robot_locs
        )

        all_actions: List[Action] = list(
            itertools.chain.from_iterable(
                op.instantiate(objects_with_rloc) for op in self._operators
            )
        )

        return [a for a in all_actions if self._is_valid_action(a)]

    def _is_valid_action(self, action: Action) -> bool:
        """Filter actions with infinite effects or invalid destinations."""
        if any(math.isinf(eff.time) for eff in action.effects):
            return False
        parts = action.name.split()
        if parts[0] == "move" and len(parts) > 3 and "_loc" in parts[3]:
            return False
        if parts[0] == "place" and len(parts) > 2 and "_loc" in parts[2]:
            return False
        if parts[0] == "search" and len(parts) > 2 and "_loc" in parts[2]:
            return False
        return True

    def is_goal_reached(self, goal_fluents: Collection[Fluent]) -> bool:
        """Check if all goal fluents are satisfied."""
        return all(f in self.state.fluents for f in goal_fluents)

    def _any_robot_free(self) -> bool:
        """Check if any robot is free."""
        return any(f.name == "free" for f in self.fluents)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/railroad/tests/test_environment_base.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/railroad/src/railroad/environment/environment.py packages/railroad/tests/test_environment_base.py
git commit -m "feat(environment): add state, get_actions, is_goal_reached to base class"
```

---

## Task 3: Add act() Method to Base Environment

**Files:**
- Modify: `packages/railroad/src/railroad/environment/environment.py`
- Test: `packages/railroad/tests/test_environment_base.py`

**Step 1: Write the failing test**

Add to `test_environment_base.py`:

```python
def test_environment_act_executes_action():
    """Test that act() executes an action and returns new state."""
    fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    state = State(0.0, fluents, [])

    move_op = Operator(
        name="move",
        parameters=["robot", "from", "to"],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[
            Effect(time=0.0, fluents=[~F("free ?robot")]),
            Effect(time=5.0, fluents=[~F("at ?robot ?from"), F("at ?robot ?to"), F("free ?robot")]),
        ]
    )

    env = MinimalEnvironment(state=state, operators=[move_op], fluents=fluents)
    actions = env.get_actions()
    move_action = next(a for a in actions if a.name == "move robot1 kitchen bedroom")

    result_state = env.act(move_action, do_interrupt=False)

    assert env.time == pytest.approx(5.0, abs=0.1)
    assert F("at", "robot1", "bedroom") in result_state.fluents
    assert F("free", "robot1") in result_state.fluents


def test_environment_act_rejects_invalid_preconditions():
    """Test that act() raises ValueError for invalid preconditions."""
    fluents = {F("at", "robot1", "kitchen")}  # Missing "free robot1"
    state = State(0.0, fluents, [])

    move_op = Operator(
        name="move",
        parameters=["robot", "from", "to"],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[Effect(time=5.0, fluents=[F("at ?robot ?to")])]
    )

    env = MinimalEnvironment(state=state, operators=[move_op], fluents=fluents)
    actions = env.get_actions()
    move_action = next(a for a in actions if a.name == "move robot1 kitchen bedroom")

    with pytest.raises(ValueError, match="preconditions not satisfied"):
        env.act(move_action)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/railroad/tests/test_environment_base.py::test_environment_act_executes_action packages/railroad/tests/test_environment_base.py::test_environment_act_rejects_invalid_preconditions -v`
Expected: FAIL with "AttributeError" (act not defined)

**Step 3: Write implementation**

Add to `Environment` class in `environment.py`:

```python
    def act(
        self,
        action: Action,
        do_interrupt: bool = True,
        loop_callback_fn: Optional[Callable[[], None]] = None,
    ) -> State:
        """Execute action, return state when a robot is free for new dispatch.

        Args:
            action: The action to execute.
            do_interrupt: Whether to interrupt interruptible skills when done.
            loop_callback_fn: Optional callback called each iteration.

        Returns:
            The new state after execution.

        Raises:
            ValueError: If action preconditions are not satisfied.
        """
        if not self.state.satisfies_precondition(action):
            raise ValueError(
                f"Action preconditions not satisfied: {action.name}"
            )

        skill = self.create_skill(action, self._time)
        self._active_skills.append(skill)

        # Apply immediate effects at current time
        for s in self._active_skills:
            s.advance(self._time, self)
        self._active_skills = [s for s in self._active_skills if not s.is_done]

        # Continue until any robot becomes free
        while not self._any_robot_free():
            if all(s.is_done for s in self._active_skills):
                break

            skill_times = [s.time_to_next_event for s in self._active_skills] or [float("inf")]
            next_time = min(skill_times)
            if next_time == float("inf"):
                break

            # Advance all skills to next event time
            for s in self._active_skills:
                s.advance(next_time, self)

            self._time = next_time
            self._active_skills = [s for s in self._active_skills if not s.is_done]

            if loop_callback_fn is not None:
                loop_callback_fn()

        # Interrupt interruptible skills if requested
        if do_interrupt:
            for skill in self._active_skills:
                if skill.is_interruptible and not skill.is_done:
                    skill.interrupt(self)

        self._active_skills = [s for s in self._active_skills if not s.is_done]
        return self.state
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/railroad/tests/test_environment_base.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/railroad/src/railroad/environment/environment.py packages/railroad/tests/test_environment_base.py
git commit -m "feat(environment): add act() method to base class"
```

---

## Task 4: Create SymbolicEnvironment Subclass

**Files:**
- Modify: `packages/railroad/src/railroad/environment/symbolic.py`
- Test: `packages/railroad/tests/test_symbolic_environment.py`

**Step 1: Write the failing test**

Create or update `test_symbolic_environment.py`:

```python
"""Tests for SymbolicEnvironment."""
import pytest
from railroad._bindings import Fluent as F, State
from railroad.core import Effect, Operator


def test_symbolic_environment_construction():
    """Test SymbolicEnvironment can be constructed with new API."""
    from railroad.environment.symbolic import SymbolicEnvironment

    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    initial_state = State(0.0, initial_fluents, [])

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[],
        true_object_locations={"kitchen": {"Knife"}},
    )

    assert F("at", "robot1", "kitchen") in env.fluents
    assert env.time == 0.0


def test_symbolic_environment_act():
    """Test SymbolicEnvironment executes actions correctly."""
    from railroad.environment.symbolic import SymbolicEnvironment

    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    initial_state = State(0.0, initial_fluents, [])

    move_op = Operator(
        name="move",
        parameters=["robot", "from", "to"],
        preconditions=[F("at ?robot ?from"), F("free ?robot")],
        effects=[
            Effect(time=0.0, fluents=[~F("free ?robot")]),
            Effect(time=5.0, fluents=[~F("at ?robot ?from"), F("at ?robot ?to"), F("free ?robot")]),
        ]
    )

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen", "bedroom"}},
        operators=[move_op],
    )

    actions = env.get_actions()
    move_action = next(a for a in actions if a.name == "move robot1 kitchen bedroom")

    env.act(move_action, do_interrupt=False)

    assert env.time == pytest.approx(5.0, abs=0.1)
    assert F("at", "robot1", "bedroom") in env.state.fluents
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/railroad/tests/test_symbolic_environment.py::test_symbolic_environment_construction packages/railroad/tests/test_symbolic_environment.py::test_symbolic_environment_act -v`
Expected: FAIL (SymbolicEnvironment doesn't exist with new signature or doesn't inherit from Environment)

**Step 3: Write implementation**

Replace contents of `packages/railroad/src/railroad/environment/symbolic.py`:

```python
"""Symbolic environment implementation."""

import random
from typing import Dict, List, Set, Tuple, Type

from railroad._bindings import Action, Fluent, GroundedEffect, State
from railroad.core import Operator

from .environment import Environment
from .skill import ActiveSkill, InterruptableMoveSymbolicSkill, SymbolicSkill


class SymbolicEnvironment(Environment):
    """Environment for symbolic (non-physical) execution.

    Suitable for planning simulations and unit tests where:
    - Fluents are tracked in-memory
    - Skills execute by stepping through effects
    - Probabilistic effects resolve via ground truth object locations
    """

    def __init__(
        self,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        true_object_locations: Dict[str, Set[str]] | None = None,
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
    ) -> None:
        """Initialize a symbolic environment.

        Args:
            state: Initial state (fluents, time, and optional upcoming effects).
            objects_by_type: Objects organized by type.
            operators: List of operators for action instantiation.
            true_object_locations: Ground truth object locations for search
                resolution. If None, search always fails.
            skill_overrides: Optional mapping from action type prefix to skill
                class. E.g., {"move": InterruptableMoveSymbolicSkill}
        """
        # Initialize subclass state before super().__init__
        # because _create_initial_effects_skill may need these
        self._fluents: Set[Fluent] = set(state.fluents)
        self._objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        self._objects_at_locations = (
            {k: set(v) for k, v in true_object_locations.items()}
            if true_object_locations else {}
        )
        self._skill_overrides = skill_overrides or {}

        super().__init__(state=state, operators=operators)

    @property
    def fluents(self) -> Set[Fluent]:
        return self._fluents

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        return self._objects_by_type

    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Create an ActiveSkill from the action.

        Routes to skill class based on:
        1. skill_overrides (if configured)
        2. Default mapping (move → InterruptableMoveSymbolicSkill, else SymbolicSkill)
        """
        parts = action.name.split()
        action_type = parts[0] if parts else ""

        # Check for override first
        if action_type in self._skill_overrides:
            skill_class = self._skill_overrides[action_type]
            return skill_class(action=action, start_time=time)

        # Default routing
        if action_type == "move":
            return InterruptableMoveSymbolicSkill(action=action, start_time=time)

        return SymbolicSkill(action=action, start_time=time)

    def apply_effect(self, effect: GroundedEffect) -> None:
        """Apply an effect, handling adds, removes, and probabilistic branches."""
        # Apply deterministic resulting_fluents
        for fluent in effect.resulting_fluents:
            if fluent.negated:
                self._fluents.discard(~fluent)
            else:
                self._fluents.add(fluent)

        # Handle probabilistic branches if present
        if effect.is_probabilistic:
            nested_effects, _ = self.resolve_probabilistic_effect(
                effect, self._fluents
            )
            for nested in nested_effects:
                self.apply_effect(nested)

        # Handle revelation (objects discovered when locations searched)
        self._handle_revelation()

    def _handle_revelation(self) -> None:
        """Reveal objects when locations are searched."""
        for fluent in list(self._fluents):
            if fluent.name == "searched":
                location = fluent.args[0]
                revealed_fluent = Fluent("revealed", location)

                if revealed_fluent not in self._fluents:
                    self._fluents.add(revealed_fluent)

                    # Reveal objects at this location
                    for obj in self._objects_at_locations.get(location, set()):
                        self._fluents.add(Fluent("found", obj))
                        self._fluents.add(Fluent("at", obj, location))
                        self._objects_by_type.setdefault("object", set()).add(obj)

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve probabilistic effect based on ground truth object locations.

        For search actions, checks if target object is actually at location.
        Otherwise, samples from the probability distribution.
        """
        if not effect.is_probabilistic:
            return [effect], current_fluents

        branches = effect.prob_effects
        if not branches:
            return [], current_fluents

        # Find success branch (contains positive "found" fluent)
        success_branch = None
        target_object = None
        location = None

        for branch in branches:
            _, branch_effects = branch
            for eff in branch_effects:
                for fluent in eff.resulting_fluents:
                    if fluent.name == "found" and not fluent.negated:
                        success_branch = branch
                        target_object = fluent.args[0]
                        location = self._find_search_location(eff, target_object)

        # If we can resolve from ground truth, do so deterministically
        if success_branch and target_object and location:
            if self._is_object_at_location(target_object, location):
                _, effects = success_branch
                return list(effects), current_fluents
            # Object not at location - sample from non-success branches
            other_branches = [b for b in branches if b is not success_branch]
            if other_branches:
                probs = [p for p, _ in other_branches]
                _, effects = random.choices(other_branches, weights=probs, k=1)[0]
                return list(effects), current_fluents

        # Can't determine from ground truth - sample from distribution
        probs = [p for p, _ in branches]
        _, effects = random.choices(branches, weights=probs, k=1)[0]
        return list(effects), current_fluents

    def _find_search_location(
        self, effect: GroundedEffect, target_object: str
    ) -> str | None:
        """Find the location from 'at object location' fluent in a branch."""
        for fluent in effect.resulting_fluents:
            if fluent.name == "at" and len(fluent.args) >= 2:
                if fluent.args[0] == target_object:
                    return fluent.args[1]
        return None

    def _is_object_at_location(self, obj: str, location: str) -> bool:
        """Check if object is at location using fluents + ground truth.

        Priority:
        1. If object is being held → not at any location
        2. If fluent says object is at this location → yes
        3. If fluent says object is at a different location → no
        4. Fall back to ground truth (for undiscovered objects)
        """
        # Check if held by any robot
        for f in self._fluents:
            if f.name == "holding" and len(f.args) >= 2 and f.args[1] == obj:
                return False

        # Check fluents for known location
        if Fluent("at", obj, location) in self._fluents:
            return True

        for f in self._fluents:
            if f.name == "at" and len(f.args) >= 2 and f.args[0] == obj:
                return False  # Object is at a different location

        # Fall back to ground truth
        return obj in self._objects_at_locations.get(location, set())
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/railroad/tests/test_symbolic_environment.py::test_symbolic_environment_construction packages/railroad/tests/test_symbolic_environment.py::test_symbolic_environment_act -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/railroad/src/railroad/environment/symbolic.py packages/railroad/tests/test_symbolic_environment.py
git commit -m "feat(environment): create SymbolicEnvironment inheriting from Environment base"
```

---

## Task 5: Add Backward Compatibility Alias

**Files:**
- Modify: `packages/railroad/src/railroad/environment/symbolic.py`
- Test: `packages/railroad/tests/test_symbolic_environment.py`

**Step 1: Write the failing test**

Add to `test_symbolic_environment.py`:

```python
def test_simple_symbolic_environment_alias():
    """Test SimpleSymbolicEnvironment still works for backward compatibility."""
    from railroad.environment.symbolic import SimpleSymbolicEnvironment

    initial_fluents = {F("at", "robot1", "kitchen"), F("free", "robot1")}
    initial_state = State(0.0, initial_fluents, [])

    # Old API - should still work
    env = SimpleSymbolicEnvironment(
        initial_state=initial_state,
        objects_by_type={"robot": {"robot1"}, "location": {"kitchen"}},
        objects_at_locations={},
    )

    assert F("at", "robot1", "kitchen") in env.fluents
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/railroad/tests/test_symbolic_environment.py::test_simple_symbolic_environment_alias -v`
Expected: FAIL (SimpleSymbolicEnvironment not found or signature doesn't match)

**Step 3: Write implementation**

Add to bottom of `symbolic.py`:

```python
class SimpleSymbolicEnvironment(SymbolicEnvironment):
    """Backward compatibility alias for SymbolicEnvironment.

    DEPRECATED: Use SymbolicEnvironment directly with new API.
    """

    def __init__(
        self,
        initial_state: State,
        objects_by_type: Dict[str, Set[str]],
        objects_at_locations: Dict[str, Set[str]],
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
    ) -> None:
        super().__init__(
            state=initial_state,
            objects_by_type=objects_by_type,
            operators=[],  # Old API didn't take operators
            true_object_locations=objects_at_locations,
            skill_overrides=skill_overrides,
        )

    @property
    def initial_state(self) -> State:
        """The initial state used to create this environment."""
        return State(0.0, self._fluents, [])
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/railroad/tests/test_symbolic_environment.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/railroad/src/railroad/environment/symbolic.py packages/railroad/tests/test_symbolic_environment.py
git commit -m "feat(environment): add SimpleSymbolicEnvironment backward compatibility alias"
```

---

## Task 6: Update Module Exports

**Files:**
- Modify: `packages/railroad/src/railroad/environment/__init__.py`
- Test: `packages/railroad/tests/test_symbolic_environment.py`

**Step 1: Write the failing test**

Add to `test_symbolic_environment.py`:

```python
def test_public_api_exports():
    """Test that new classes are exported from railroad.environment."""
    from railroad.environment import (
        Environment,
        SymbolicEnvironment,
        SimpleSymbolicEnvironment,
    )

    assert Environment is not None
    assert SymbolicEnvironment is not None
    assert SimpleSymbolicEnvironment is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/railroad/tests/test_symbolic_environment.py::test_public_api_exports -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Read current `__init__.py` first, then update to include new exports:

```python
"""Environment module for railroad."""

from .base import (
    AbstractEnvironment,
    SimpleEnvironment,
    SimpleOperatorEnvironment,
    SimulatedRobot,
    SkillStatus,
)
from .environment import Environment
from .interface import EnvironmentInterface
from .interface_v2 import EnvironmentInterfaceV2
from .skill import (
    ActiveSkill,
    InterruptableMoveSymbolicSkill,
    SymbolicSkill,
)
from .symbolic import SimpleSymbolicEnvironment, SymbolicEnvironment

__all__ = [
    # Legacy
    "AbstractEnvironment",
    "EnvironmentInterface",
    "EnvironmentInterfaceV2",
    "SimpleEnvironment",
    "SimpleOperatorEnvironment",
    "SimulatedRobot",
    "SkillStatus",
    # New architecture
    "ActiveSkill",
    "Environment",
    "InterruptableMoveSymbolicSkill",
    "SimpleSymbolicEnvironment",
    "SymbolicEnvironment",
    "SymbolicSkill",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/railroad/tests/test_symbolic_environment.py::test_public_api_exports -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/railroad/src/railroad/environment/__init__.py packages/railroad/tests/test_symbolic_environment.py
git commit -m "feat(environment): export Environment and SymbolicEnvironment from module"
```

---

## Task 7: Update Example to Use New API

**Files:**
- Modify: `packages/railroad/src/railroad/examples/multi_object_search.py`

**Step 1: Read current example**

Read the file to understand current structure.

**Step 2: Update imports and construction**

Change:
```python
from railroad.environment import EnvironmentInterfaceV2, SimpleSymbolicEnvironment
```

To:
```python
from railroad.environment import SymbolicEnvironment
```

Change construction from:
```python
initial_state = State(0.0, initial_fluents, [])
env = SimpleSymbolicEnvironment(initial_state, objects_by_type, OBJECTS_AT_LOCATIONS)
sim = EnvironmentInterfaceV2(env, [no_op, move_op, search_op, pick_op, place_op])
```

To:
```python
initial_state = State(0.0, initial_fluents, [])
env = SymbolicEnvironment(
    state=initial_state,
    objects_by_type=objects_by_type,
    operators=[no_op, move_op, search_op, pick_op, place_op],
    true_object_locations=OBJECTS_AT_LOCATIONS,
)
```

Change all `sim.` references to `env.`:
- `sim.state` → `env.state`
- `sim.get_actions()` → `env.get_actions()`
- `sim.advance(action, ...)` → `env.act(action, ...)`

**Step 3: Run example to verify it works**

Run: `uv run python -c "from railroad.examples.multi_object_search import main; print('Import OK')"`
Expected: "Import OK" (syntax check)

**Step 4: Commit**

```bash
git add packages/railroad/src/railroad/examples/multi_object_search.py
git commit -m "refactor(examples): update multi_object_search to use new SymbolicEnvironment API"
```

---

## Task 8: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all railroad tests**

Run: `uv run pytest packages/railroad/tests/ -v`
Expected: All tests PASS

**Step 2: Fix any failures**

If tests fail, identify whether:
1. Test needs updating to new API
2. Implementation has a bug

Fix accordingly and re-run.

**Step 3: Run type checker**

Run: `uv run ty check packages/railroad/src/railroad/environment/`
Expected: No errors

**Step 4: Commit verification**

```bash
git commit --allow-empty -m "chore: verify all tests pass after environment merge"
```

---

## Task 9: Remove Old EnvironmentInterfaceV2 (Optional Cleanup)

**Note:** Only do this if all tests pass and you want to remove legacy code.

**Files:**
- Delete: `packages/railroad/src/railroad/environment/interface_v2.py`
- Modify: `packages/railroad/src/railroad/environment/__init__.py`
- Update: Any remaining imports

**Step 1: Search for usages**

Run: `grep -r "EnvironmentInterfaceV2" packages/`

**Step 2: Update any remaining usages to use SymbolicEnvironment**

**Step 3: Remove from exports**

Update `__init__.py` to remove `EnvironmentInterfaceV2` from imports and `__all__`.

**Step 4: Delete file**

```bash
rm packages/railroad/src/railroad/environment/interface_v2.py
```

**Step 5: Run tests**

Run: `uv run pytest packages/railroad/tests/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor(environment): remove deprecated EnvironmentInterfaceV2"
```

---

## Verification

After completing all tasks:

1. **Unit tests**: `uv run pytest packages/railroad/tests/test_environment_base.py packages/railroad/tests/test_symbolic_environment.py -v`
2. **Full suite**: `uv run pytest packages/railroad/tests/ -v`
3. **Type check**: `uv run ty check packages/railroad/src/railroad/environment/`
4. **Example runs**: `uv run python packages/railroad/src/railroad/examples/multi_object_search.py` (optional, may take time)
