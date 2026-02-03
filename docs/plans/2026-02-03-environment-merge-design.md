# Environment Merge Design

**Goal:** Merge `EnvironmentInterfaceV2` and `Environment` protocol into a single unified class hierarchy for a better first-time user experience.

**Motivation:** The separation between "environment" and "interface" has become confusing. It's unclear what logic belongs where. Users currently need two objects (`SimpleSymbolicEnvironment` + `EnvironmentInterfaceV2`) when one would suffice.

---

## Desired API

```python
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State

initial_state = State(0.0, initial_fluents, [])
env = SymbolicEnvironment(
    state=initial_state,
    objects_by_type=objects_by_type,
    operators=[no_op, move_op, search_op, pick_op, place_op],
    true_object_locations=OBJECTS_AT_LOCATIONS,  # optional
)

# Single object for everything:
env.state          # current state (fluents + upcoming effects)
env.time           # current time
env.get_actions()  # available actions
env.act(action)    # execute action (renamed from advance)
env.is_goal_reached(goal_fluents)
```

---

## Class Hierarchy

```
Environment (abstract base class)
├── Concrete: _active_skills, _time, _operators
├── Concrete: state, time, get_actions(), act(), is_goal_reached()
├── Abstract: fluents, objects_by_type
├── Abstract: create_skill(), apply_effect(), resolve_probabilistic_effect()

SymbolicEnvironment(Environment)
├── Owns: _fluents, _objects_by_type, _objects_at_locations
├── Implements all abstract methods
└── Routes actions to appropriate skill classes
```

---

## Files Affected

| File | Change |
|------|--------|
| `skill.py` | Keep `ActiveSkill` protocol and skill classes. Remove `Environment` protocol. |
| `environment.py` (new) | Base `Environment` class with concrete + abstract methods |
| `symbolic.py` | Rename `SimpleSymbolicEnvironment` → `SymbolicEnvironment`, inherit from `Environment` |
| `interface_v2.py` | Delete (merged into base class) |
| `__init__.py` | Update exports |

---

## Base Environment Class

### Constructor

```python
class Environment(ABC):
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
```

### Concrete Methods

```python
    @property
    def time(self) -> float:
        return self._time

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

        all_actions = list(itertools.chain.from_iterable(
            op.instantiate(objects_with_rloc) for op in self._operators
        ))

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

### The act() Method

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

            next_time = min(
                (s.time_to_next_event for s in self._active_skills),
                default=float("inf")
            )
            if next_time == float("inf"):
                break

            for s in self._active_skills:
                s.advance(next_time, self)

            self._time = next_time
            self._active_skills = [s for s in self._active_skills if not s.is_done]

            if loop_callback_fn is not None:
                loop_callback_fn()

        if do_interrupt:
            for skill in self._active_skills:
                if skill.is_interruptible and not skill.is_done:
                    skill.interrupt(self)

        self._active_skills = [s for s in self._active_skills if not s.is_done]
        return self.state
```

### Abstract Methods

```python
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

---

## SymbolicEnvironment Subclass

```python
class SymbolicEnvironment(Environment):
    """Environment for symbolic (non-physical) execution."""

    def __init__(
        self,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        true_object_locations: Dict[str, Set[str]] | None = None,
        skill_overrides: Dict[str, Type[ActiveSkill]] | None = None,
    ) -> None:
        # Initialize subclass state before super().__init__
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
        parts = action.name.split()
        action_type = parts[0] if parts else ""

        if action_type in self._skill_overrides:
            skill_class = self._skill_overrides[action_type]
            return skill_class(action=action, start_time=time)

        if action_type == "move":
            return InterruptableMoveSymbolicSkill(action=action, start_time=time)

        return SymbolicSkill(action=action, start_time=time)

    def apply_effect(self, effect: GroundedEffect) -> None:
        for fluent in effect.resulting_fluents:
            if fluent.negated:
                self._fluents.discard(~fluent)
            else:
                self._fluents.add(fluent)

        if effect.is_probabilistic:
            nested_effects, _ = self.resolve_probabilistic_effect(
                effect, self._fluents
            )
            for nested in nested_effects:
                self.apply_effect(nested)

        self._handle_revelation()

    def _handle_revelation(self) -> None:
        for fluent in list(self._fluents):
            if fluent.name == "searched":
                location = fluent.args[0]
                revealed_fluent = Fluent("revealed", location)

                if revealed_fluent not in self._fluents:
                    self._fluents.add(revealed_fluent)
                    for obj in self._objects_at_locations.get(location, set()):
                        self._fluents.add(Fluent("found", obj))
                        self._fluents.add(Fluent("at", obj, location))
                        self._objects_by_type.setdefault("object", set()).add(obj)

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        if not effect.is_probabilistic:
            return [effect], current_fluents

        branches = effect.prob_effects
        if not branches:
            return [], current_fluents

        # Find success branch and resolve via ground truth
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

        if success_branch and target_object and location:
            if self._is_object_at_location(target_object, location):
                _, effects = success_branch
                return list(effects), current_fluents
            other_branches = [b for b in branches if b is not success_branch]
            if other_branches:
                probs = [p for p, _ in other_branches]
                _, effects = random.choices(other_branches, weights=probs, k=1)[0]
                return list(effects), current_fluents

        probs = [p for p, _ in branches]
        _, effects = random.choices(branches, weights=probs, k=1)[0]
        return list(effects), current_fluents

    def _find_search_location(
        self, effect: GroundedEffect, target_object: str
    ) -> str | None:
        for fluent in effect.resulting_fluents:
            if fluent.name == "at" and len(fluent.args) >= 2:
                if fluent.args[0] == target_object:
                    return fluent.args[1]
        return None

    def _is_object_at_location(self, obj: str, location: str) -> bool:
        for f in self._fluents:
            if f.name == "holding" and len(f.args) >= 2 and f.args[1] == obj:
                return False

        if Fluent("at", obj, location) in self._fluents:
            return True

        for f in self._fluents:
            if f.name == "at" and len(f.args) >= 2 and f.args[0] == obj:
                return False

        return obj in self._objects_at_locations.get(location, set())
```

---

## Migration Guide

### Before (current)

```python
from railroad.environment import SimpleSymbolicEnvironment, EnvironmentInterfaceV2
from railroad._bindings import State

initial_state = State(0.0, initial_fluents, [])
env = SimpleSymbolicEnvironment(initial_state, objects_by_type, OBJECTS_AT_LOCATIONS)
sim = EnvironmentInterfaceV2(env, [no_op, move_op, search_op, pick_op, place_op])

# Usage
sim.state
sim.time
sim.get_actions()
sim.advance(action, do_interrupt=False)
sim.is_goal_reached(goal_fluents)
```

### After (new)

```python
from railroad.environment import SymbolicEnvironment
from railroad._bindings import State

initial_state = State(0.0, initial_fluents, [])
env = SymbolicEnvironment(
    state=initial_state,
    objects_by_type=objects_by_type,
    operators=[no_op, move_op, search_op, pick_op, place_op],
    true_object_locations=OBJECTS_AT_LOCATIONS,
)

# Usage
env.state
env.time
env.get_actions()
env.act(action, do_interrupt=False)
env.is_goal_reached(goal_fluents)
```

### Changes Summary

| Old | New |
|-----|-----|
| `SimpleSymbolicEnvironment` + `EnvironmentInterfaceV2` | `SymbolicEnvironment` |
| `sim.advance(action)` | `env.act(action)` |
| `objects_at_locations` param | `true_object_locations` param |

---

## Future Work (not in scope)

1. **PhysicalEnvironment** - Subclass for real robot execution
2. **Deprecate legacy classes** - `EnvironmentInterface`, `SimpleSymbolicEnvironment`
3. **Remove old files** - Once migration complete
