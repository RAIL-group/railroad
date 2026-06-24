from railroad.core import Fluent, State, Heuristic, Goal
from railroad.operators import construct_move_visited_operator
from railroad.planner import MCTSPlanner
import pytest

F = Fluent

class MockCustomHeuristic(Heuristic):
    def __init__(self):
        self.called_with = []

    def __call__(self, state: State, goal: Goal, rpg: dict) -> float:
        self.called_with.append((state, goal, rpg))
        # Verify the structure of the RPG
        assert "current_time" in rpg
        assert "current_fluents" in rpg
        assert "goal" in rpg
        assert "cheapest_relaxed_plan" in rpg
        
        # Verify cheapest_relaxed_plan is a list of steps
        for step in rpg["cheapest_relaxed_plan"]:
            assert "achieves_fluent" in step
            assert "action" in step
            assert "exec_cost" in step
            assert "wait_cost" in step
            assert "requires_preconditions" in step

        # Let's return a dummy cost: say, the length of cheapest_relaxed_plan
        return float(len(rpg["cheapest_relaxed_plan"]) * 5.0)

def test_custom_heuristic_integration():
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "a", "b"],
    }
    move_op = construct_move_visited_operator(lambda *args: 5.0)
    all_actions = move_op.instantiate(objects_by_type)

    initial_state = State(
        time=0,
        fluents={F("at r1 start"), F("free r1"), F("visited start")}
    )
    goal = F("visited a")

    custom_h = MockCustomHeuristic()

    # Pass custom_heuristic to the planner constructor
    mcts = MCTSPlanner(all_actions, custom_heuristic=custom_h)

    # Run planner
    action_name = mcts(initial_state, goal, max_iterations=50, c=1.414)

    # Verify custom_heuristic was called at least once
    assert len(custom_h.called_with) > 0
    assert isinstance(action_name, str)
    assert action_name != "NONE"


def test_sandbox_evaluator_tournament():
    from railroad.experimental.llm_heuristic import LLMHeuristicGenerator, SandboxEvaluator

    # Generate a dummy prompt and check content
    gen = LLMHeuristicGenerator()
    prompt = gen.generate_prompt("domain desc", "problem desc", "{}")
    assert "class" in prompt
    assert "__call__" in prompt
    assert "Heuristic" in prompt

    # Define objects and actions for a simple problem
    objects_by_type = {
        "robot": ["r1"],
        "location": ["start", "a"],
    }
    move_op = construct_move_visited_operator(lambda *args: 5.0)
    all_actions = move_op.instantiate(objects_by_type)

    initial_state = State(
        time=0,
        fluents={F("at r1 start"), F("free r1"), F("visited start")}
    )
    goal = F("visited a")

    # Define candidate 1 (working candidate)
    candidate_1 = """
from railroad.core import Heuristic
class WorkingHeuristic(Heuristic):
    def __call__(self, state, goal, rpg):
        return 1.0
"""

    # Define candidate 2 (errored candidate)
    candidate_2 = """
from railroad.core import Heuristic
class ErrorHeuristic(Heuristic):
    def __call__(self, state, goal, rpg):
        raise ValueError("Simulated error")
"""

    # Define candidate 3 (non-working candidate)
    candidate_3 = "def dummy_func(): pass"

    evaluator = SandboxEvaluator(all_actions, initial_state, goal, max_iterations=50)

    # Test evaluating candidate 1
    res_1 = evaluator.evaluate_candidate(candidate_1)
    assert res_1["success"] is True
    assert res_1["error"] is None
    assert res_1["steps_taken"] > 0

    # Test evaluating candidate 2 (errored)
    res_2 = evaluator.evaluate_candidate(candidate_2)
    assert res_2["success"] is False
    assert "Simulated error" in res_2["error"]

    # Test evaluating candidate 3 (errored)
    res_3 = evaluator.evaluate_candidate(candidate_3)
    assert res_3["success"] is False
    assert "No class found" in res_3["error"]

    # Run tournament
    best_code, best_steps = evaluator.run_tournament([candidate_2, candidate_1])
    assert best_code == candidate_1
    assert best_steps == res_1["steps_taken"]

