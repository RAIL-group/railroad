import os
import sys
import time
import importlib
import numpy as np
from dotenv import load_dotenv

# Ensure virtualenv/packages are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../packages/railroad/src")))

load_dotenv()

from railroad import operators
from railroad.core import Fluent as F, State, Goal, get_action_by_name, transition
from railroad.environment import SymbolicEnvironment
from railroad.planner import MCTSPlanner
from railroad.experimental.llm_heuristic import HeuristicContextBuilder, LLMHeuristicGenerator, SandboxEvaluator

# ============================================================================
# Domain and Problem PDDL Descriptions for the LLM
# ============================================================================

PDDL_DOMAIN = """
(define (domain railroad-delivery)
  (:requirements :concurrent :probabilistic)
  (:predicates
     (at ?r - robot ?l - location)
     (free ?r - robot)
     (holding ?r - robot ?o - object)
     (at ?o - object ?l - location)
     (closed ?d - container-door)
     (blocking-access ?l - location)
     (has-door ?l - location ?d - container-door)
  )

  (:action move
     :parameters (?r - robot ?from - location ?to - location)
     :precondition (and (at ?r ?from) (free ?r))
     :effect (and (not (at ?r ?from)) (at ?r ?to))
  )

  (:action pick
     :parameters (?r - robot ?l - location ?o - object)
     :precondition (and (at ?r ?l) (at ?o ?l) (free ?r) (not (blocking-access ?l)))
     :effect (and (not (at ?o ?l)) (holding ?r ?o) (not (free ?r)))
  )

  (:action place
     :parameters (?r - robot ?l - location ?o - object)
     :precondition (and (at ?r ?l) (holding ?r ?o) (not (blocking-access ?l)))
     :effect (and (not (holding ?r ?o)) (at ?o ?l) (free ?r))
  )

  (:action open_container
     :parameters (?r - robot ?d - container-door ?l - location)
     :precondition (and (at ?r ?l) (closed ?d) (has-door ?l ?d) (free ?r))
     :effect (and (not (closed ?d)) (not (blocking-access ?l)))
  )
)
"""

PDDL_PROBLEM = """
(define (problem railroad-delivery-task)
  (:domain railroad-delivery)
  (:objects
     robot1 - robot
     start_loc table pantry bed cabinet fridge - location
     Notebook Clock Mug Pillow Waterbottle Plate Apple - object
     cabinet_door fridge_door - container-door
  )
  (:init
     (at robot1 start_loc)
     (free robot1)
     (at Notebook table)
     (at Clock table)
     (at Mug table)
     (at Pillow bed)
     (at Waterbottle bed)
     (at Plate cabinet)
     (has-door cabinet cabinet_door)
     (closed cabinet_door)
     (blocking-access cabinet)
     (has-door fridge fridge_door)
     (closed fridge_door)
     (blocking-access fridge)
  )
  (:goal (and (at Mug cabinet) (at Waterbottle fridge)))
)
"""

# ============================================================================
# Environment Setup (Matches pre_plan_context.py)
# ============================================================================

LOCATIONS = {
    "start_loc": np.array([-5, -5]),
    "table": np.array([0, 0]),
    "pantry": np.array([10, 0]),
    "bed": np.array([0, 12]),
    "cabinet": np.array([10, 12]),
    "fridge": np.array([15, 12]),
}

OBJECTS_AT_LOCATIONS = {
    "start_loc": set(),
    "table": {"Notebook", "Clock", "Mug"},
    "pantry": set(),
    "bed": {"Pillow", "Waterbottle"},
    "cabinet": {"Plate"},
    "fridge": {"Apple"},
}

ROBOT_VELOCITY = 1.0
PICK_TIME = 5.0
PLACE_TIME = 5.0
OPEN_TIME = 3.0

def setup_planning_problem():
    objects_of_interest = ["Notebook", "Clock", "Mug", "Pillow", "Waterbottle", "Plate", "Apple"]

    initial_fluents = {
        F("at", "robot1", "start_loc"),
        F("free", "robot1"),
        F("at Notebook table"),
        F("at Clock table"),
        F("at Cereal pantry"),
        F("at Pillow bed"),
        F("at Waterbottle bed"),
        F("at Mug table"),
        F("at Plate cabinet"),
        F("has-door cabinet cabinet_door"),
        F("closed cabinet_door"),
        F("blocking-access cabinet"),
        F("has-door fridge fridge_door"),
        F("closed fridge_door"),
        F("blocking-access fridge"),
    }

    goal = F("at Mug cabinet") & F("at Waterbottle fridge")

    objects_by_type = {
        "robot": {"robot1"},
        "location": set(LOCATIONS.keys()),
        "object": set(objects_of_interest),
        "container-door": {"cabinet_door", "fridge_door"}
    }

    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        distance = float(np.linalg.norm(LOCATIONS[loc_from] - LOCATIONS[loc_to]))
        return distance / ROBOT_VELOCITY

    move_op = operators.construct_move_operator_blocking(move_time)
    pick_op = operators.construct_pick_operator_blocking(PICK_TIME)
    place_op = operators.construct_place_operator_blocking(PLACE_TIME)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=100.0)
    open_op = operators.construct_open_container_door_operator_blocking(OPEN_TIME)

    initial_state = State(0.0, initial_fluents, [])

    env = SymbolicEnvironment(
        state=initial_state,
        objects_by_type=objects_by_type,
        operators=[no_op, move_op, pick_op, place_op, open_op],
        true_object_locations=OBJECTS_AT_LOCATIONS,
    )

    all_actions = env.get_actions()
    return env, goal, all_actions


def load_generated_heuristic():
    """
    Dynamically loads the compiled best candidate from the heuristics package.
    """
    module = importlib.import_module("railroad.heuristics.llm_generated")
    importlib.reload(module)
    
    from railroad.core import Heuristic
    for name, obj in vars(module).items():
        if isinstance(obj, type) and obj.__name__ != "Heuristic":
            return obj()
    raise ValueError("No custom heuristic class found in railroad/heuristics/llm_generated.py")


def main():
    print("--------------------------------------------------")
    print("MCTS concurrent planning with LLM-generated heuristic")
    print("--------------------------------------------------")

    env, goal, all_actions = setup_planning_problem()
    
    heuristics_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../packages/railroad/src/railroad/heuristics"))
    target_file = os.path.join(heuristics_dir, "llm_generated.py")

    # ============================================================================
    # Check Caching / LLM Generation Phase
    # ============================================================================
    if not os.path.exists(target_file):
        print("[Pipeline] No cached heuristic found at railroad/heuristics/llm_generated.py.")
        print("[Pipeline] Starting LLM prompt generation and tournament evaluator...")

        # Verify API Key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your environment or .env file "
                "to allow candidate generation."
            )

        # 1. Build initial context (RPG example)
        print("[Pipeline] Extracting Relaxed Planning Graph from initial state...")
        def fluent_filter(f):
            return any(kw in f.name for kw in ["at", "holding", "closed", "blocking"])

        context_builder = HeuristicContextBuilder(fluent_filter=fluent_filter)
        rpg_json = context_builder.to_json(env.state, goal, all_actions)

        # 2. Setup Generator and generate prompt
        generator = LLMHeuristicGenerator()
        prompt = generator.generate_prompt(PDDL_DOMAIN, PDDL_PROBLEM, rpg_json)

        # 3. Generate candidate variations (n=5)
        print("[Pipeline] Prompting LLM to generate 5 candidates...")
        candidates = generator.generate_heuristic_candidates(prompt, n=5)
        if not candidates:
            raise RuntimeError("LLM failed to generate any heuristic candidates.")

        # 4. Evaluate candidates in Sandbox Tournament
        print("[Pipeline] Running candidate tournament in Sandbox...")
        evaluator = SandboxEvaluator(all_actions, env.state, goal, max_iterations=500)
        best_code, best_steps = evaluator.run_tournament(candidates)

        if not best_code:
            raise RuntimeError("No candidate heuristic successfully solved the problem in the Sandbox!")

        print(f"[Pipeline] Best candidate selected! Succeeded in {best_steps} steps.")

        # 5. Write to heuristics package permanently
        print(f"[Pipeline] Writing best candidate permanently to {target_file}...")
        with open(target_file, "w") as f:
            f.write(best_code)
            
        print("[Pipeline] Heuristic written successfully.")
    else:
        print("[Pipeline] Found cached LLM-generated heuristic. Skipping LLM prompt generation!")

    # ============================================================================
    # Planning Execution Phase (Using the Custom Heuristic)
    # ============================================================================
    print("[Pipeline] Loading custom heuristic from heuristics package...")
    custom_h = load_generated_heuristic()

    print("[Pipeline] Initializing MCTS Planner with the custom heuristic...")
    mcts = MCTSPlanner(all_actions, custom_heuristic=custom_h)

    print("[Pipeline] Running the full MCTS planning loop to achieve the goal...")
    state = env.state
    step = 0
    max_steps = 30
    start_time = time.time()

    while step < max_steps:
        if goal.evaluate(state.fluents):
            print(f"🎉 GOAL REACHED in {step} steps!")
            break

        print(f"\n--- Step {step + 1} (time={state.time:.2f}) ---")
        
        # MCTS call uses the custom heuristic
        action_name = mcts(state, goal, max_iterations=2000, c=10.0)
        if action_name == "NONE":
            print("❌ MCTS failed to find an action!")
            break

        action = get_action_by_name(all_actions, action_name)
        state = transition(state, action)[0][0]
        print(f"Executed action: {action_name}")
        step += 1
    else:
        print("❌ Failed to reach the goal within maximum steps.")

    total_time = time.time() - start_time
    print(f"\n[Pipeline] Completed planning run in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
