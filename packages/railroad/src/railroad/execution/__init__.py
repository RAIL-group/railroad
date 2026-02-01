"""Simplified execution module for PDDL plans.

This module provides a simplified alternative to railroad.environment for
executing PDDL plans. It prioritizes simplicity for testing and examples
while supporting all MCTS planner capabilities including probabilistic
effects with nested timing.

Key differences from railroad.environment:
- Single unified OngoingAction class (no subclasses)
- Handles nested effects inside prob_effects (fixes Issue #6)
- No skill execution status tracking (IDLE/RUNNING/DONE)
- No move interruption support
- No coordinate-based movement costs

Usage:
    from railroad.execution import Environment, EnvironmentInterface
    from railroad import operators
    from railroad.core import State, Fluent as F

    # Define operators
    move_op = operators.construct_move_operator_blocking(move_time=5.0)
    search_op = operators.construct_search_operator(0.5, search_time=3.0)

    # Set up ground truth
    objects_at_locations = {"kitchen": {"Knife"}, "bedroom": {"Pillow"}}

    # Create environment
    env = Environment(objects_at_locations)

    # Create interface
    initial_state = State(time=0, fluents={F("at robot1 kitchen"), F("free robot1")})
    objects_by_type = {"robot": ["robot1"], "location": ["kitchen", "bedroom"]}

    interface = EnvironmentInterface(
        initial_state, objects_by_type, [move_op, search_op], env
    )

    # Execute actions
    actions = interface.get_actions()
    interface.advance(actions[0])
"""

from .environment import Environment
from .interface import EnvironmentInterface, OngoingAction

__all__ = [
    "Environment",
    "EnvironmentInterface",
    "OngoingAction",
]
