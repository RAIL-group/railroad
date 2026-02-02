import environments
from railroad import operators
from railroad.planner import MCTSPlanner
from railroad.core import Fluent as F, get_action_by_name
from railroad.environment import EnvironmentInterfaceV2, SkillStatus
from railroad.environment.skill import ActiveSkill, SymbolicSkill
from railroad._bindings import Action, Fluent, GroundedEffect, ff_heuristic
from railroad.dashboard import PlannerDashboard
import argparse
import logging
from typing import Dict, List, Set, Tuple


# Fixed operator times for non-move skills
SEARCH_TIME = 1.0
PICK_TIME = 1.0
PLACE_TIME = 1.0


class PhysicalSkill:
    """Skill that executes on a physical robot and polls for completion."""

    def __init__(
        self,
        action: Action,
        start_time: float,
        robot: str,
        physical_env: "environments.pyrobosim.PyRoboSimEnv",
        skill_name: str,
        skill_args: tuple,
    ) -> None:
        self._action = action
        self._start_time = start_time
        self._robot = robot
        self._physical_env = physical_env
        self._skill_name = skill_name
        self._is_done = False
        self._is_interruptible = skill_name == "move"

        # Compute upcoming effects from action
        self._upcoming_effects: List[Tuple[float, GroundedEffect]] = sorted(
            [(start_time + eff.time, eff) for eff in action.effects],
            key=lambda el: el[0]
        )

        # Start physical execution
        physical_env.execute_skill(robot, skill_name, *skill_args)

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

    def _is_physical_action_done(self) -> bool:
        """Check if the physical robot action is complete."""
        # For no_op, check our own flag
        if self._skill_name == "no_op":
            return not self._physical_env.is_no_op_running.get(self._robot, False)
        # For other skills, check if robot is busy
        robot = self._physical_env.robots.get(self._robot)
        if robot is None:
            return True
        return not robot.is_busy()

    @property
    def time_to_next_event(self) -> float:
        if not self._upcoming_effects:
            return float("inf")

        # Check for immediate effects (time=0 or time <= start_time)
        next_effect_time = self._upcoming_effects[0][0]
        if next_effect_time <= self._start_time + 1e-9:
            return next_effect_time

        # For completion effects, wait for physical action to be done
        if self._is_physical_action_done():
            return next_effect_time

        # Physical action still running - poll again soon
        # Return a time slightly after start to keep polling
        return self._start_time + 0.01

    def advance(self, time: float, env: "PyRoboSimEnvironmentAdapter") -> None:
        """Advance skill, applying effects based on time and physical completion.

        - Immediate effects (time <= start_time): Apply based on time
        - Completion effects (time > start_time): Apply only when physical action done
        """
        if not self._upcoming_effects:
            return

        # Apply immediate effects (time=0) based on time alone
        immediate_effects = [
            (t, eff) for t, eff in self._upcoming_effects
            if t <= self._start_time + 1e-9 and t <= time + 1e-9
        ]
        for _, effect in immediate_effects:
            env.apply_effect(effect)

        # Remove applied immediate effects
        if immediate_effects:
            self._upcoming_effects = [
                (t, eff) for t, eff in self._upcoming_effects
                if t > self._start_time + 1e-9 or t > time + 1e-9
            ]

        # For completion effects, wait for physical action to be done
        if self._is_physical_action_done() and self._upcoming_effects:
            # Physical action complete - apply remaining effects
            for _, effect in self._upcoming_effects:
                env.apply_effect(effect)
            self._upcoming_effects = []

        # Mark done when no more effects and physical action complete
        if not self._upcoming_effects and self._is_physical_action_done():
            self._is_done = True

    def interrupt(self, env: "PyRoboSimEnvironmentAdapter") -> None:
        """Interrupt this skill by stopping the physical robot."""
        if self._is_interruptible and not self._is_done:
            self._physical_env.stop_robot(self._robot)
            # Mark as done without applying remaining effects
            self._upcoming_effects = []
            self._is_done = True


class PyRoboSimEnvironmentAdapter:
    """Adapter that wraps PyRoboSimEnv to implement the Environment protocol.

    Owns fluents and objects_by_type, delegates physical execution to the
    wrapped PyRoboSimEnv.
    """

    def __init__(
        self,
        physical_env: "environments.pyrobosim.PyRoboSimEnv",
        initial_fluents: Set[Fluent],
        objects_by_type: Dict[str, Set[str]],
    ) -> None:
        self._physical_env = physical_env
        self._fluents = set(initial_fluents)
        self._objects_by_type = {k: set(v) for k, v in objects_by_type.items()}
        # Ground truth for search resolution
        self._objects_at_locations: Dict[str, Set[str]] = {}

    @property
    def fluents(self) -> Set[Fluent]:
        return self._fluents

    @property
    def objects_by_type(self) -> Dict[str, Set[str]]:
        return self._objects_by_type

    def create_skill(self, action: Action, time: float) -> ActiveSkill:
        """Create a PhysicalSkill that executes on the robot."""
        parts = action.name.split()
        action_type = parts[0]
        robot = parts[1]

        if action_type == "move":
            # move robot from to
            loc_from, loc_to = parts[2], parts[3]
            return PhysicalSkill(
                action=action,
                start_time=time,
                robot=robot,
                physical_env=self._physical_env,
                skill_name="move",
                skill_args=(loc_from, loc_to),
            )
        elif action_type == "search":
            # search robot location object
            location = parts[2]
            return PhysicalSkill(
                action=action,
                start_time=time,
                robot=robot,
                physical_env=self._physical_env,
                skill_name="search",
                skill_args=(location,),
            )
        elif action_type == "pick":
            # pick robot location object
            location, obj = parts[2], parts[3]
            return PhysicalSkill(
                action=action,
                start_time=time,
                robot=robot,
                physical_env=self._physical_env,
                skill_name="pick",
                skill_args=(location, obj),
            )
        elif action_type == "place":
            # place robot location object
            location, obj = parts[2], parts[3]
            return PhysicalSkill(
                action=action,
                start_time=time,
                robot=robot,
                physical_env=self._physical_env,
                skill_name="place",
                skill_args=(location, obj),
            )
        elif action_type == "no_op":
            return PhysicalSkill(
                action=action,
                start_time=time,
                robot=robot,
                physical_env=self._physical_env,
                skill_name="no_op",
                skill_args=(),
            )
        else:
            # Default: use symbolic skill
            return SymbolicSkill(
                action=action,
                start_time=time,
                robot=robot,
                is_interruptible=False,
            )

    def apply_effect(self, effect: GroundedEffect) -> None:
        """Apply effect fluents to the state."""
        for fluent in effect.resulting_fluents:
            if fluent.negated:
                self._fluents.discard(~fluent)
            else:
                self._fluents.add(fluent)

        # Handle probabilistic effects
        if effect.is_probabilistic:
            nested_effects, _ = self.resolve_probabilistic_effect(effect, self._fluents)
            for nested in nested_effects:
                self.apply_effect(nested)

        # Handle revelation from search
        self._handle_revelation()

    def _handle_revelation(self) -> None:
        """Reveal objects when locations are searched."""
        for fluent in list(self._fluents):
            if fluent.name == "searched":
                location = fluent.args[0]
                revealed_fluent = Fluent("revealed", location)

                if revealed_fluent not in self._fluents:
                    self._fluents.add(revealed_fluent)

                    # Get objects from physical environment
                    objects_at_loc = self._physical_env.get_objects_at_location(location)
                    for obj in objects_at_loc.get("object", set()):
                        self._fluents.add(Fluent("found", obj))
                        self._fluents.add(Fluent("at", obj, location))
                        self._objects_by_type.setdefault("object", set()).add(obj)

    def resolve_probabilistic_effect(
        self,
        effect: GroundedEffect,
        current_fluents: Set[Fluent],
    ) -> Tuple[List[GroundedEffect], Set[Fluent]]:
        """Resolve probabilistic effects using physical environment ground truth."""
        if not effect.is_probabilistic:
            return [effect], current_fluents

        branches = effect.prob_effects
        if not branches:
            return [], current_fluents

        # For search, check physical environment for object presence
        for _, branch_effects in branches:
            for branch_eff in branch_effects:
                for fluent in branch_eff.resulting_fluents:
                    if fluent.name == "found" and not fluent.negated:
                        target_object = fluent.args[0]
                        # Find location from "at" fluent
                        location = self._find_search_location(branch_eff, target_object)
                        if location:
                            objects_at_loc = self._physical_env.get_objects_at_location(location)
                            if target_object in objects_at_loc.get("object", set()):
                                return list(branch_effects), current_fluents

        # Failure branch (last one)
        _, last_branch = branches[-1]
        return list(last_branch), current_fluents

    def _find_search_location(self, effect: GroundedEffect, target_object: str) -> str | None:
        """Find location from 'at object location' fluent."""
        for fluent in effect.resulting_fluents:
            if fluent.name == "at" and len(fluent.args) >= 2:
                if fluent.args[0] == target_object:
                    return fluent.args[1]
        return None

    def get_objects_at_location(self, location: str) -> Dict[str, Set[str]]:
        """Delegate to physical environment."""
        return self._physical_env.get_objects_at_location(location)

    def remove_object_from_location(self, obj: str, location: str) -> None:
        """No-op - physical env handles this during pick."""
        pass

    def add_object_at_location(self, obj: str, location: str) -> None:
        """No-op - physical env handles this during place."""
        pass


def main(args):
    physical_env = environments.pyrobosim.PyRoboSimEnv(
        args.world_file,
        show_plot=args.show_plot,
        record_plots=not args.no_video
    )
    objects = list(physical_env.objects.keys())[:2]  # Take first two objects for the demo

    objects_by_type = {
        "robot": set(physical_env.robots.keys()),
        "location": set(physical_env.locations.keys()),
        "object": set(objects),
    }

    initial_fluents = {
        F("revealed robot1_loc"),
        F("at robot1 robot1_loc"), F("free robot1"),
        F("revealed robot2_loc"),
        F("at robot2 robot2_loc"), F("free robot2"),
    }

    goal = F("at apple0 counter0") & F("at banana0 counter0")

    # Create operators - move uses distance-based time from simulator
    move_time_fn = physical_env._get_move_cost_fn()
    object_find_prob = lambda r, l, o: 0.8 if l == 'my_desk' and o == 'apple0' else 0.2
    move_op = operators.construct_move_operator_blocking(move_time_fn)
    search_op = operators.construct_search_operator(object_find_prob, SEARCH_TIME)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)

    # Create environment adapter
    env = PyRoboSimEnvironmentAdapter(physical_env, initial_fluents, objects_by_type)

    # Create interface
    env_interface = EnvironmentInterfaceV2(env, [no_op, pick_op, place_op, move_op, search_op])

    # Planning loop
    actions_taken = []
    max_iterations = 60  # Limit iterations to avoid infinite loops

    # Dashboard
    h_value = ff_heuristic(env_interface.state, goal, env_interface.get_actions())
    with PlannerDashboard(goal, initial_heuristic=h_value) as dashboard:
        # (Optional) initial dashboard update
        dashboard.update(sim_state=env_interface.state)

        for iteration in range(max_iterations):
            # Check if goal is reached
            if goal.evaluate(env_interface.state.fluents):
                break

            # Get available actions
            all_actions = env_interface.get_actions()
            # # Plan next action
            mcts = MCTSPlanner(all_actions)
            action_name = mcts(env_interface.state, goal, max_iterations=10000, c=300, max_depth=20)

            if action_name == 'NONE':
                dashboard.console.print("No more actions available. Goal may not be achievable.")
                break

            # Execute action
            action = get_action_by_name(all_actions, action_name)
            env_interface.advance(action, do_interrupt=False, loop_callback_fn=physical_env.canvas.update)
            actions_taken.append(action_name)

            tree_trace = mcts.get_trace_from_last_mcts_tree()
            h_value = ff_heuristic(env_interface.state, goal, env_interface.get_actions())
            relevant_fluents = {
                f for f in env_interface.state.fluents
                if any(keyword in f.name for keyword in ["at", "holding", "found", "searched"])
            }
            dashboard.update(
                sim_state=env_interface.state,
                relevant_fluents=relevant_fluents,
                tree_trace=tree_trace,
                step_index=iteration,
                last_action_name=action_name,
                heuristic_value=h_value,
            )

    # Print the full dashboard history to the console (optional)
    dashboard.print_history(env_interface.state, actions_taken)
    if not args.no_video:
        physical_env.canvas.save_animation(filepath='./data/pyrobosim_planning_demo.mp4')
    physical_env.canvas.wait_for_close()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--world-file", type=str, default='./resources/pyrobosim_worlds/test_world.yaml',
                      help="Path to the world YAML file.")
    args.add_argument("--show-plot", action='store_true', help="Whether to show the plot window.")
    args.add_argument("--no-video", action='store_true', help="Whether to disable generating video of the simulation.")
    args = args.parse_args()

    # Turn off all logging of level INFO and below (to suppress pyrobosim logs)
    logging.disable(logging.INFO)

    main(args)
