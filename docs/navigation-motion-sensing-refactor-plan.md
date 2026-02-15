# Navigation Motion Sensing Refactor Plan

## Summary

Move periodic sensing cadence control out of `NavigationMoveSkill` and into `UnknownSpaceEnvironment`, while keeping motion-effect semantics in skills. This reduces `NavigationMoveSkill` complexity, keeps `Environment.act()` generic, and makes sensing policy environment-owned.

## Interface and API Updates

1. Add optional `MotionSkill` protocol in `packages/railroad/src/railroad/environment/skill.py`:
   - `controlled_robot: str`
   - `is_motion_active_at(time: float) -> bool`
2. Add scheduling hooks in `packages/railroad/src/railroad/environment/environment.py`:
   - `_cap_next_advance_time(proposed_next_time: float) -> float`
   - `_after_skills_advanced(advanced_to_time: float) -> None`
3. Keep `ActiveSkill` unchanged for compatibility.
4. Keep continuous robot state in `env.robot_poses`; do not continuously write `{robot}_loc` in `location_registry`.
5. Keep `{robot}_loc` registration for interrupt anchoring (and existing stale-location remap behavior).

## Implementation Checklist

1. Base scheduler hook integration:
   - Call `_cap_next_advance_time()` before advancing skills.
   - Call `_after_skills_advanced()` after advancing skills and cleaning completed skills.
2. Motion skill contract:
   - Introduce `MotionSkill` as optional runtime-checkable protocol.
3. Navigation move skill simplification:
   - Keep path generation + interpolation + move-duration retiming + effect scheduling + interrupt rewrite.
   - Remove internal sensor cadence state and sensing loop.
   - Continue updating `env.robot_poses` in `advance()`.
4. UnknownSpaceEnvironment sensing ownership:
   - Add motion-skill iteration helper.
   - Cap scheduler step to `sensor_dt` while motion active.
   - Sense moving robots in `_after_skills_advanced()` using current `robot_poses`.
   - Do not refresh frontiers/sync targets on each sensing micro-step.
5. Preserve symbolic policies:
   - `robot_poses` is continuous physical truth.
   - `{robot}_loc` remains transient symbolic anchor.
   - Keep existing `just-moved` cleanup for `{robot}_loc`.

## Validation and Tests

1. Keep existing navigation tests green.
2. Add cadence test: long move with small `sensor_dt` should produce repeated sensing updates.
3. Add no-motion test: non-motion action should not be forced into `sensor_dt` stepping.
4. Keep interrupt behavior test: aggressive thresholds still interrupt interruptible moves.
5. Add non-interruptible move test: aggressive thresholds should not interrupt when disabled.
6. Add pose-source policy test:
   - `robot_poses` updates during motion.
   - no continuous `robot1_loc` registry writes during normal movement.

## Acceptance Criteria

1. `NavigationMoveSkill` no longer owns periodic sensing cadence.
2. `UnknownSpaceEnvironment` enforces sensing cadence via `sensor_dt` when motion active.
3. Interrupt semantics and fluent rewrite behavior remain consistent.
4. Planner-visible fluent contracts remain intact.
5. Existing navigation suite plus new cadence/pose-policy tests pass.
