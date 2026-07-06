import heapq
import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, Set, Optional, Dict, Tuple

# Ensure the scripts/ directory is on sys.path so sibling modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pyrefly: ignore [missing-import]
from planner_interface import call_planner, SimpleAction, MCTSPlanner, AStarPlanner  # pyrefly: ignore [unused-import]

# Switch between planners here — both have identical behaviour from the simulator's perspective
PLANNER = AStarPlanner  # change to MCTSPlanner to use MCTS instead
from railroad import operators
from railroad.core import Operator, Effect, Fluent

# Define Event types
@dataclass(order=True)
class Event:
    time: float
    robot_id: str
    event_type: str = field(compare=False)
    data: Any = field(compare=False)

def check_preconditions(action: SimpleAction, state: Set[str]) -> Optional[str]:
    """Check if all preconditions of the action hold in the state.
    Returns the first failing precondition (as a string) if any, or None if all hold.
    """
    for pre in action.preconditions:
        if pre.startswith("not "):
            pos_fluent = pre[4:]
            if pos_fluent in state:
                return pre
        else:
            if pre not in state:
                return pre
    return None

def apply_del_effects(action: SimpleAction, state: Set[str]) -> None:
    """Apply the delete effects of the action immediately at start.
    
    NOTE: If action abortion/preemption is ever added to this simulator,
    the delete effects applied here must be rolled back to avoid state corruption.
    """
    for d in action.del_effects:
        state.discard(d)

def apply_add_effects(action: SimpleAction, state: Set[str]) -> None:
    """Apply the add effects of the action at action end."""
    for a in action.add_effects:
        state.add(a)

# =============================================================================
# Part 2 — New Operators
# =============================================================================
F = Fluent

def construct_chop_operator() -> Operator:
    """Chop an item at the counter using the knife."""
    return Operator(
        name="chop",
        parameters=[("?r", "robot"), ("?item", "object")],
        preconditions=[
            F("at ?r counter"),
            F("free ?r"),
            F("holding ?r knife"),
            F("at ?item counter"),
            ~F("is_garnish ?item"),  # Garnish (herbs) must be chopped using chop_garnish operator instead
        ],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r")}),
            Effect(time=5.0, resulting_fluents={F("free ?r"), F("chopped ?item")}),
        ],
    )

def construct_stir_operator() -> Operator:
    """Stir the bowl at the counter. Requires empty hands."""
    return Operator(
        name="stir",
        parameters=[("?r", "robot")],
        preconditions=[
            F("at ?r counter"),
            F("free ?r"),
            ~F("hand-full ?r"),
            F("chopped carrot"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=12.0, resulting_fluents={F("free ?r"), F("stirred bowl")}),
        ],
    )

def construct_chop_garnish_operator() -> Operator:
    """Chop herbs (garnish) at the counter. Requires the stirred bowl."""
    return Operator(
        name="chop_garnish",
        parameters=[("?r", "robot")],
        preconditions=[
            F("at ?r counter"),
            F("free ?r"),
            F("holding ?r knife"),
            F("at herbs counter"),
            F("stirred bowl"),
        ],
        effects=[
            Effect(time=0, resulting_fluents={~F("free ?r")}),
            Effect(time=5.0, resulting_fluents={F("free ?r"), F("chopped herbs")}),
        ],
    )

# =============================================================================
# Unit Test for Part 1
# =============================================================================
def test_at_start_delete():
    print("Running unit test for at-start delete effects...")
    state = {
        "at robot1 counter",
        "free robot1",
        "at robot2 counter",
        "free robot2",
        "at knife counter",
    }
    
    action1 = SimpleAction(
        name="pick robot1 counter knife",
        duration=1.0,
        preconditions={"at robot1 counter", "free robot1", "at knife counter"},
        add_effects={"holding robot1 knife", "hand-full robot1", "free robot1"},
        del_effects={"at knife counter", "free robot1"}
    )
    
    action2 = SimpleAction(
        name="pick robot2 counter knife",
        duration=1.0,
        preconditions={"at robot2 counter", "free robot2", "at knife counter"},
        add_effects={"holding robot2 knife", "hand-full robot2", "free robot2"},
        del_effects={"at knife counter", "free robot2"}
    )
    
    # Initially both pass preconditions
    assert check_preconditions(action1, state) is None
    assert check_preconditions(action2, state) is None
    
    # Apply del effects of action1 immediately at start
    apply_del_effects(action1, state)
    
    # Now robot2's preconditions must fail because "at knife counter" is deleted
    failing_fluent = check_preconditions(action2, state)
    assert failing_fluent == "at knife counter", f"Expected failed fluent to be 'at knife counter', got: {failing_fluent}"
    print("Unit test passed successfully!")

# =============================================================================
# Main Simulation
# =============================================================================
def main():
    # Part 3 — Kitchen scenario objects and locations
    objects_by_type = {
        "robot": {"robot1", "robot2", "robot3"},
        "location": {"base1", "base2", "base3", "counter", "shelf", "table"},
        "object": {"knife", "carrot", "onion", "herbs", "plate"},
    }

    initial_fluents = {
        "at robot1 base1", "free robot1",
        "at robot2 base2", "free robot2",
        "at robot3 base3", "free robot3",
        "at knife counter",
        "at carrot counter", "at onion counter", "at herbs counter",
        "at plate shelf",
        "is_garnish herbs",
    }
    
    for loc in objects_by_type["location"]:
        initial_fluents.add(f"revealed {loc}")

    world_state = set(initial_fluents)

    def move_time_fn(robot: str, loc_from: str, loc_to: str) -> float:
        return 1.0
        
    move_op = operators.construct_move_operator_blocking(move_time=move_time_fn)
    pick_op = operators.construct_pick_operator_blocking(pick_time=1.0)
    place_op = operators.construct_place_operator_blocking(place_time=1.0)
    no_op = operators.construct_no_op_operator(no_op_time=5.0, extra_cost=50.0)
    
    chop_op = construct_chop_operator()
    stir_op = construct_stir_operator()
    chop_garnish_op = construct_chop_garnish_operator()

    my_operators = [move_op, pick_op, place_op, no_op, chop_op, stir_op, chop_garnish_op]

    event_queue: List[Event] = []
    
    # Push initial tasks for Part 3 (using conjunctive goals to force knife return)
    heapq.heappush(event_queue, Event(time=0.0, robot_id="robot1", event_type="TASK_ARRIVAL", data="chopped herbs & not hand-full robot1"))
    heapq.heappush(event_queue, Event(time=13.0, robot_id="robot2", event_type="TASK_ARRIVAL", data="chopped onion & not hand-full robot2"))
    # heapq.heappush(event_queue, Event(time=27.0, robot_id="robot3", event_type="TASK_ARRIVAL", data="at plate table"))

    robot_plans: Dict[str, List[SimpleAction]] = { "robot1": [], "robot2": [], "robot3": [] }
    robot_current_tasks: Dict[str, Optional[str]] = { "robot1": None, "robot2": None, "robot3": None }
    robot_replans: Dict[str, int] = { "robot1": 0, "robot2": 0, "robot3": 0 }        # mid-execution precondition failures
    robot_retries: Dict[str, int] = { "robot1": 0, "robot2": 0, "robot3": 0 }         # pre-execution planning failures
    
    # To assert a robot isn't given a task while busy
    robot_busy: Dict[str, bool] = { "robot1": False, "robot2": False, "robot3": False }

    RETRY_DELAY = 5.0  # seconds to wait before retrying a failed plan

    # Logging structures
    knife_possessions: List[Tuple[str, float, float]] = []
    knife_holder_start: Dict[str, float] = {}
    gantt_starts: Dict[str, float] = {}
    gantt_timeline: Dict[str, List[Tuple[str, float, float]]] = { "robot1": [], "robot2": [], "robot3": [] }

    def try_start_next_action(t: float, robot: str):
        if not robot_plans[robot]:
            robot_busy[robot] = False
            print(f"Time {t:05.1f} | {robot} finished all actions and is now idle.")
            return

        next_action = robot_plans[robot][0]
        
        failing_fluent = check_preconditions(next_action, world_state)
        if failing_fluent is None:
            # Success: Start action
            # Part 1: Apply delete effects immediately at start
            apply_del_effects(next_action, world_state)
            
            robot_busy[robot] = True
            print(f"Time {t:05.1f} | {robot} started: {next_action.name}")
            
            # Logging start
            gantt_starts[robot] = t
            if next_action.name.startswith("pick ") and "knife" in next_action.name:
                knife_holder_start[robot] = t
                
            end_time = t + next_action.duration
            heapq.heappush(event_queue, Event(time=end_time, robot_id=robot, event_type="ACTION_END", data=next_action))
        else:
            # Failure: Replanning
            robot_replans[robot] += 1
            print(f"Time {t:05.1f} | {robot} failed preconditions for {next_action.name} (failed fluent: '{failing_fluent}'). Replanning...")
            snapshot = set(world_state)
            current_task = robot_current_tasks[robot]
            assert current_task is not None
            new_plan = call_planner(snapshot, current_task, objects_by_type, my_operators, robot, planner_cls=PLANNER)

            if new_plan is None:
                # Goal already satisfied by another robot while we were waiting
                robot_busy[robot] = False
                print(f"Time {t:05.1f} | {robot} goal already satisfied — no longer needs to act.")
            elif new_plan:
                robot_plans[robot] = new_plan
                try_start_next_action(t, robot)
            else:
                # Still can't plan — retry later
                robot_busy[robot] = False
                robot_retries[robot] += 1
                retry_t = t + RETRY_DELAY
                print(f"Time {t:05.1f} | {robot} still can't replan. Retrying at t={retry_t:.1f}...")
                heapq.heappush(event_queue, Event(time=retry_t, robot_id=robot, event_type="TASK_ARRIVAL", data=robot_current_tasks[robot]))

    while event_queue:
        event = heapq.heappop(event_queue)
        t = event.time
        robot = event.robot_id
        
        if event.event_type == "TASK_ARRIVAL":
            task_goal = event.data
            if robot_busy[robot]:
                raise RuntimeError(f"Task arrival for {robot} at time {t} but it is already busy!")
                
            print(f"Time {t:05.1f} | [TASK_ARRIVAL] {robot} received task: {task_goal}")
            robot_current_tasks[robot] = task_goal
            snapshot = set(world_state)
            
            plan = call_planner(snapshot, task_goal, objects_by_type, my_operators, robot, planner_cls=PLANNER)
            if plan is None:
                print(f"Time {t:05.1f} | {robot} goal '{task_goal}' already satisfied in current state — no actions needed.")
            elif not plan:
                # Can't plan yet — schedule a retry
                robot_retries[robot] += 1
                retry_t = t + RETRY_DELAY
                print(f"Time {t:05.1f} | {robot} cannot plan for task '{task_goal}' yet. Retrying at t={retry_t:.1f}...")
                heapq.heappush(event_queue, Event(time=retry_t, robot_id=robot, event_type="TASK_ARRIVAL", data=task_goal))
            else:
                robot_plans[robot] = plan
                try_start_next_action(t, robot)
                
        elif event.event_type == "ACTION_END":
            action: SimpleAction = event.data
            
            # NOTE: There is an event tie at t=20 (robot1's stir ends and robot2's chop ends simultaneously).
            # The heap breaks ties by robot_id alphabetically, so robot1 is processed first and tries to
            # start its next action (pick knife). Since robot2 started picking the knife at t=14.0, it
            # applied the delete effects at start, so at t=20.0 the knife is not at the counter.
            # Thus, robot1 fails preconditions and is forced to replan. This tie-break is the intended conservative behavior.
            
            # Part 1: Apply add effects only at action end
            apply_add_effects(action, world_state)
            print(f"Time {t:05.1f} | {robot} ended: {action.name}")
            
            # Logging end
            if robot in gantt_starts:
                start_t = gantt_starts.pop(robot)
                gantt_timeline[robot].append((action.name, start_t, t))
            if action.name.startswith("place ") and "knife" in action.name:
                if robot in knife_holder_start:
                    start_t = knife_holder_start.pop(robot)
                    knife_possessions.append((robot, start_t, t))
            
            # Remove completed action
            robot_plans[robot].pop(0)
            
            # Start next
            try_start_next_action(t, robot)

    print("\n==========================================")
    print("Knife Possession Intervals:")
    print("==========================================")
    for holder, start_t, end_t in knife_possessions:
        print(f"  {holder} held knife: [{start_t:.1f} - {end_t:.1f}]")
        
    print("\n==========================================")
    print("Gantt Timeline:")
    print("==========================================")
    for r in sorted(objects_by_type["robot"]):
        line = f"  {r}:"
        for name, start_t, end_t in gantt_timeline[r]:
            # Simplify action name for display
            simple_name = name.split()[0]
            line += f"  {simple_name}({start_t:.1f}-{end_t:.1f})"
        print(line)

    print("\n==========================================")
    print("Simulation Counters:")
    print("==========================================")
    for r in sorted(objects_by_type["robot"]):
        print(f"  {r}: {robot_replans[r]} mid-execution replans, {robot_retries[r]} pre-execution retries")

if __name__ == "__main__":
    test_at_start_delete()
    print("\nStarting simulation...")
    main()
