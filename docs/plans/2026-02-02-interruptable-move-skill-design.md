# InterruptableMoveSymbolicSkill Design

## Overview

Add an `InterruptableMoveSymbolicSkill` that gets interrupted when another robot becomes free during multi-robot execution. Users can opt into this behavior via a skill override mapping on the environment.

## Requirements

1. When robot1 is executing an `InterruptableMoveSymbolicSkill` and robot2 becomes free, robot1's move should be interrupted
2. Interrupted moves rewrite destination fluents to an intermediate location (`robot_loc`)
3. Users specify a mapping from action type prefix to skill class
4. Default behavior (no override) should NOT auto-interrupt moves

## Design

### Skill Classes (`skill.py`)

**SymbolicSkill (modified)**
- Remove auto-detection of move actions as interruptible
- Set `_is_interruptible = False` by default
- `interrupt()` is a no-op

**InterruptableMoveSymbolicSkill (new)**
- Extends `SymbolicSkill`
- Sets `_is_interruptible = True`
- Extracts destination from action name ("move robot from to")
- `interrupt()` rewrites fluents to intermediate location

### Environment Skill Mapping (`symbolic.py`)

**SimpleSymbolicEnvironment (modified)**
- Add `skill_overrides: Dict[str, Type[ActiveSkill]] | None = None` parameter
- `create_skill()` checks action type prefix against overrides
- Falls back to default `SymbolicSkill` if no override

### Interface Behavior (`interface_v2.py`)

No changes needed. The existing logic already:
- Loops until `_any_robot_free()` returns True
- Calls `interrupt()` on all interruptible skills when exiting the loop

## Usage

```python
from railroad.environment.skill import InterruptableMoveSymbolicSkill
from railroad.environment.symbolic import SimpleSymbolicEnvironment

env = SimpleSymbolicEnvironment(
    initial_state=state,
    objects_by_type=objects_by_type,
    objects_at_locations=objects_at_locations,
    skill_overrides={"move": InterruptableMoveSymbolicSkill},
)
```

## Tests

1. **Unit test**: `InterruptableMoveSymbolicSkill.interrupt()` rewrites fluents correctly
2. **Integration test**: Multi-robot scenario where robot2 becoming free interrupts robot1's move
3. **Regression test**: Default `SymbolicSkill` moves are NOT interrupted
4. **Mapping test**: Skill overrides route to correct skill classes

## Files Changed

- `packages/railroad/src/railroad/environment/skill.py` - Add `InterruptableMoveSymbolicSkill`, modify `SymbolicSkill`
- `packages/railroad/src/railroad/environment/symbolic.py` - Add `skill_overrides` parameter
- `packages/railroad/tests/test_active_skill.py` - Add unit tests for new skill
- `packages/railroad/tests/test_interface_v2.py` or new file - Add integration tests
