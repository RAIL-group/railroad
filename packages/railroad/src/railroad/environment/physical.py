"""Abstract base classes for physical environments.

This module provides a generic implementation of Environment and ActiveSkill
for physical robots or simulators that follow a similar pattern:
1. Actions are triggered asynchronously on the hardware/simulator.
2. Status is polled to detect completion.
3. Time in the planner tracks real (wall) time during execution.
"""

import time as time_module
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from railroad._bindings import Action, Fluent, GroundedEffect, State
from railroad.core import Operator
from .environment import Environment
from .skill import ActiveSkill, SkillStatus


class PhysicalScene(ABC):
    """Abstract base class for physical scene data providers.

    Scenes provide all information needed to initialize operators and
    environments, including location coordinates and ground truth object locations.
    """

    @property
    @abstractmethod
    def locations(self) -> Dict[str, Any]:
        """Location names mapped to their coordinates/poses."""
        ...

    @property
    @abstractmethod
    def objects(self) -> Set[str]:
        """All objects in the scene."""
        ...

    @property
    @abstractmethod
    def object_locations(self) -> Dict[str, Set[str]]:
        """Ground truth: location name -> set of object names at that location."""
        ...

    @abstractmethod
    def get_move_cost_fn(self) -> Callable[[str, str, str], float]:
        """Get move cost function for operator construction."""
        ...


class PhysicalSkill(ActiveSkill):
    """Generic skill for physical execution that polls for completion.

    Implements ActiveSkill protocol where:
    - Actions run asynchronously on physical hardware or a simulator.
    - Completion is detected by polling the environment's status methods.
    - Time in the planner tracks real time during execution.
    """

    def __init__(
        self,
        action: Action,
        start_time: float,
    ) -> None:
        self._action = action
        self._start_time = start_time

        parts = action.name.split()
        self._skill_name = parts[0] if parts else "unknown"
        self._skill_args = tuple(parts[1:])
        self._robot = parts[1] if len(parts) > 1 else "unknown"

        self._is_done = False
        self._is_interruptible = self._skill_name == "move"
        self._completion_time: Optional[float] = None

        self._wall_start_time = time_module.time()
        self._last_poll_done = False

        # Compute upcoming effects from action (using planned times initially)
        self._upcoming_effects: List[Tuple[float, GroundedEffect]] = sorted(
            [(start_time + eff.time, eff) for eff in action.effects],
            key=lambda el: el[0]
        )

        if self._upcoming_effects:
            self._end_time = max(t for t, _ in self._upcoming_effects)
        else:
            self._end_time = start_time

    @property
    def action(self) -> Action:
        return self._action

    @property
    def skill_name(self) -> str:
        return self._skill_name

    @property
    def skill_args(self) -> tuple:
        return self._skill_args

    @property
    def robot(self) -> str:
        return self._robot

    @property
    def is_done(self) -> bool:
        return self._is_done

    @property
    def is_interruptible(self) -> bool:
        return self._is_interruptible

    @property
    def upcoming_effects(self) -> List[Tuple[float, GroundedEffect]]:
        return self._upcoming_effects

    def _is_physical_action_done(self, env: Environment) -> bool:
        """Check if the physical robot action is complete."""
        # Grace period: don't check for completion for the first 0.2s to allow
        # the asynchronous action to actually start in the simulator/hardware.
        if time_module.time() - self._wall_start_time < 0.2:
            return False

        # We know env must be a PhysicalEnvironment to use this skill
        assert isinstance(env, PhysicalEnvironment)
        status = env.get_executed_skill_status(self._robot, self._skill_name)
        done = status == SkillStatus.DONE

        if done and self._completion_time is None:
            actual_duration = time_module.time() - self._wall_start_time
            self._completion_time = self._start_time + actual_duration

        return done

    @property
    def time_to_next_event(self) -> float:
        if not self._upcoming_effects:
            return float("inf")

        # Check for immediate effects (time=0 or time <= start_time)
        next_effect_time = self._upcoming_effects[0][0]
        if next_effect_time <= self._start_time + 1e-9:
            return next_effect_time

        # For completion effects, wait for physical action to be done
        if self._last_poll_done:
            # Return actual completion time instead of planned time
            return self._completion_time if self._completion_time is not None else next_effect_time

        # Physical action still running - poll based on real time
        elapsed = time_module.time() - self._wall_start_time
        return self._start_time + elapsed + 0.05

    def advance(self, time: float, env: Environment) -> None:
        """Advance skill, applying effects based on time and physical completion."""
        self._last_poll_done = self._is_physical_action_done(env)

        if not self._upcoming_effects:
            return

        due_effects = []

        # 1. Apply immediate effects (always applied when time reaches them)
        while self._upcoming_effects and self._upcoming_effects[0][0] <= self._start_time + 1e-9:
            if self._upcoming_effects[0][0] <= time + 1e-9:
                due_effects.append(self._upcoming_effects.pop(0))
            else:
                break

        # 2. Completion effects (only applied when physical action is done)
        if self._last_poll_done:
            while self._upcoming_effects:
                # Force apply remaining effects since physical action is complete
                due_effects.append(self._upcoming_effects.pop(0))

        for _, effect in due_effects:
            env.apply_effect(effect)

        # Mark done when no more effects and physical action complete
        if not self._upcoming_effects and self._last_poll_done:
            self._is_done = True

    def interrupt(self, env: Environment) -> None:
        """Interrupt this skill by stopping the physical robot."""
        if self._is_interruptible and not self._is_done:
            assert isinstance(env, PhysicalEnvironment)
            env.stop_robot(self._robot)

            # Record intermediate location for moves
            parts = self._action.name.split()
            if len(parts) >= 4 and parts[0] == "move":
                robot = parts[1]
                origin = parts[2]
                destination = parts[3]
                new_target = f"{robot}_loc"

                # Calculate progress based on real time vs planned duration
                registry = getattr(env, "location_registry", None)
                if registry is not None:
                    total_planned_time = self._end_time - self._start_time
                    if total_planned_time > 0:
                        elapsed = time_module.time() - self._wall_start_time
                        progress = max(0.0, min(1.0, elapsed / total_planned_time))

                        origin_coords = registry.get(origin)
                        dest_coords = registry.get(destination)
                        if origin_coords is not None and dest_coords is not None:
                            # We assume coordinates support basic math as per LocationRegistry doc
                            intermediate_coords = origin_coords + progress * (dest_coords - origin_coords)
                            registry.register(new_target, intermediate_coords)

                # Rewrite destination fluents
                new_fluents: Set[Fluent] = set()
                for _, eff in self._upcoming_effects:
                    for fluent in eff.resulting_fluents:
                        if (~fluent) in new_fluents:
                            new_fluents.remove(~fluent)
                        new_args = [arg if arg != destination else new_target for arg in fluent.args]
                        new_fluents.add(Fluent(" ".join([fluent.name] + new_args), negated=fluent.negated))

                for fluent in new_fluents:
                    if fluent.negated:
                        env.fluents.discard(~fluent)
                    else:
                        env.fluents.add(fluent)

            self._upcoming_effects = []
            self._is_done = True


class PhysicalEnvironment(Environment, ABC):
    """Abstract base class for physical environments.

    Provides generic implementations for state management and skill creation.
    Subclasses must implement methods for hardware/simulator interaction.
    """

    def __init__(
        self,
        scene: PhysicalScene,
        state: State,
        objects_by_type: Dict[str, Set[str]],
        operators: List[Operator],
        skill_overrides: Optional[Dict[str, Type[ActiveSkill]]] = None,
        location_registry: Any = None,
    ) -> None:
        self.scene = scene
        self._fluents = set(state.fluents)
        self._objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        self._skill_overrides = skill_overrides or {}
        self._location_registry = location_registry
        self._skill_history: List[Tuple[float, Action]] = []
        super().__init__(state, operators)

    @property
    def fluents(self) -> Set[Fluent]:
        return self._fluents

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        return self._objects_by_type

    @property
    def location_registry(self) -> Any:
        """Optional LocationRegistry for coordinate tracking."""
        return self._location_registry

    @property
    def skill_history(self) -> List[Tuple[float, Action]]:
        """History of executed actions with their execution times."""
        return self._skill_history

    def remove_object_from_location(self, obj: str, location: str) -> None:
        """Optional hook to remove an object from ground truth/perception."""
        pass

    def add_object_at_location(self, obj: str, location: str) -> None:
        """Optional hook to add an object to ground truth/perception."""
        pass

    @abstractmethod
    def execute_skill(self, robot_name: str, skill_name: str, *args: Any, **kwargs: Any) -> None:
        """Execute a skill on a robot."""
        ...

    @abstractmethod
    def get_executed_skill_status(self, robot_name: str, skill_name: str) -> SkillStatus:
        """Get the execution status of a skill."""
        ...

    @abstractmethod
    def stop_robot(self, robot_name: str) -> None:
        """Stop robot's current physical action."""
        ...

    def get_objects_at_location(self, location: str) -> Dict[str, Set[str]]:
        """Get objects at a location using scene ground truth."""
        objs = self.scene.object_locations.get(location, set())
        return {"object": objs}

    def _on_act_loop_iteration(self, dt: float) -> None:
        """Sleep briefly to avoid busy-waiting when polling physical status."""
        time_module.sleep(0.01)

    def _on_skill_completed(self, skill: ActiveSkill) -> None:
        """Handle physical completion side-effects (e.g., updating world state)."""
        if not isinstance(skill, PhysicalSkill):
            return

        # Stop hardware for completed actions
        self.stop_robot(skill.robot)

        # Update perception/world state for manipulation actions
        # skill_args is (robot, arg1, arg2, ...) from action name "skill robot arg1 arg2"
        name = skill.skill_name
        args = skill.skill_args
        if name == "pick" and len(args) >= 3:
            _robot, loc, obj = args[0], args[1], args[2]
            self.remove_object_from_location(obj, loc)
        elif name == "place" and len(args) >= 3:
            _robot, loc, obj = args[0], args[1], args[2]
            self.add_object_at_location(obj, loc)

    def _create_initial_effects_skill(
        self,
        start_time: float,
        upcoming_effects: List[Tuple[float, GroundedEffect]],
    ) -> ActiveSkill:
        from .symbolic import SymbolicSkill
        relative_effects = [
            GroundedEffect(abs_time - start_time, effect.resulting_fluents)
            for abs_time, effect in upcoming_effects
        ]
        action = Action(set(), relative_effects, name="_initial_effects")
        return SymbolicSkill(action=action, start_time=start_time)

    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        skill_name, robot_name, *skill_args = action.name.split()

        # Record execution
        self._skill_history.append((time, action))

        # Trigger physical execution immediately
        self.execute_skill(robot_name, skill_name, *skill_args)

        if skill_name in self._skill_overrides:
            skill_class = self._skill_overrides[skill_name]
            return skill_class(action=action, start_time=time)

        return PhysicalSkill(
            action=action,
            start_time=time,
        )
