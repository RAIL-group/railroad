# Re-import required dependencies due to kernel reset
from typing import Callable, List, Tuple, Dict, Set, Union, Sequence, Collection, Mapping, Optional, cast
import itertools

from railroad._bindings import GroundedEffect, Fluent, Action, State
from railroad._bindings import transition
from railroad._bindings import LiteralGoal, AndGoal, OrGoal, Goal

__all__ = ["transition"]  # re-exported from _bindings
from railroad._bindings import ff_heuristic as _ff_heuristic_cpp


def ff_heuristic(state: State, goal: Union[Goal, Fluent], all_actions: List[Action]) -> float:
    """Compute FF heuristic value for a state (probabilistic version).

    Args:
        state: The current state
        goal: Goal to achieve. Can be:
            - A Goal object: F("a") & F("b"), AndGoal([...]), etc.
            - A single Fluent: F("visited a") (auto-wrapped to LiteralGoal)
        all_actions: List of all available actions

    Returns:
        Heuristic value (estimated cost to reach goal)
    """
    # Normalize goal (wrap Fluent in LiteralGoal if needed)
    if isinstance(goal, Fluent):
        goal = LiteralGoal(goal)
    return _ff_heuristic_cpp(state, goal, all_actions)

Num = Union[float, int]


Binding = Dict[str, str]
Bindable = Callable[[Binding], Num]
OptExpr = Union[float, Tuple[Callable[..., float], List[str]]]


def _make_bindable(opt_expr: OptExpr) -> Bindable:
    if isinstance(opt_expr, Num):
        return lambda *args: opt_expr
    else:
        fn = opt_expr[0]
        args = opt_expr[1]
        return lambda b: fn(*[b.get(arg, arg) for arg in args])


class Effect:
    def __init__(
        self,
        time: OptExpr,
        prob_effects: List[Tuple[OptExpr, List["Effect"]]] = list(),
        resulting_fluents: Set[Fluent] = set(),
    ):
        self.time = _make_bindable(time)
        self.prob_effects = [
            (_make_bindable(prob), effects) for prob, effects in prob_effects
        ]
        self.resulting_fluents = resulting_fluents
        self.is_probabilistic = bool(self.prob_effects)

    def _ground(self, binding: Binding) -> "GroundedEffect":
        # def evaluate(expr): return expr(binding) if callable(expr) else expr
        if self.is_probabilistic:
            grounded_prob_effects = tuple(
                (prob(binding), tuple(e._ground(binding) for e in effect_list))
                for prob, effect_list in self.prob_effects
            )
        else:
            grounded_prob_effects = tuple()

        grounded_time: float = self.time(binding)
        grounded_resulting_fluents = set(
            Fluent(
                f.name, *[binding.get(arg, arg) for arg in f.args], negated=f.negated
            )
            for f in self.resulting_fluents
        )

        return GroundedEffect(
            grounded_time,
            prob_effects=grounded_prob_effects,
            resulting_fluents=grounded_resulting_fluents,
        )


class Operator:
    def __init__(
        self,
        name: str,
        parameters: List[Tuple[str, str]],
        preconditions: List[Fluent],
        effects: Sequence[Effect],
        extra_cost: float = 0.0,
    ):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects
        self.extra_cost = extra_cost

    def instantiate(self, objects_by_type: Mapping[str, Collection[str]]) -> List[Action]:
        grounded_actions = []
        domains = [objects_by_type[typ] for _, typ in self.parameters]
        for assignment in itertools.product(*domains):
            binding = {var: obj for (var, _), obj in zip(self.parameters, assignment)}
            if len(set(binding.values())) != len(binding):
                continue
            grounded_actions.append(self._ground(binding))
        return grounded_actions

    def _ground(self, binding: Dict[str, str]) -> Action:
        def evaluate(value):
            return value(binding) if callable(value) else value

        grounded_preconditions = frozenset(
            self._substitute_fluent(f, binding) for f in self.preconditions
        )
        grounded_effects = [eff._ground(binding) for eff in self.effects]

        name_str = f"{self.name} " + " ".join(
            binding[var] for var, _ in self.parameters
        )
        return Action(grounded_preconditions, grounded_effects, name=name_str, extra_cost=self.extra_cost)

    def _substitute_fluent(self, fluent: Fluent, binding: Dict[str, str]) -> Fluent:
        grounded_args = tuple(binding.get(arg, arg) for arg in fluent.args)
        return Fluent(fluent.name, *grounded_args, negated=fluent.negated)


def get_action_by_name(actions: List[Action], name: str) -> Action:
    for action in actions:
        if action.name == name:
            return action
    raise ValueError(f"No action found with name: {name}")


def get_next_actions(state: State, all_actions: List[Action]) -> List[Action]:
    # Step 1: Extract all `free(...)` fluents
    free_robot_fluents: List[Fluent] = cast(
        List[Fluent],
        sorted([f for f in state.fluents if f.name == "free"], key=lambda f: str(f)),
    )
    # neg_fluents = {~f for f in free_robot_fluents}
    neg_state = state.copy()
    neg_fluents: Set[Fluent] = {~f for f in free_robot_fluents}
    neg_state.update_fluents(neg_fluents)

    # Step 2: Check each robot individually
    for free_pred in free_robot_fluents:
        # Create a restricted version of the state
        combined_fluents: Set[Fluent] = cast(Set[Fluent], neg_state.fluents | {free_pred})
        temp_state = State(
            time=state.time,
            fluents=combined_fluents,
        )

        # Step 3: Check for applicable actions
        applicable = [a for a in all_actions if temp_state.satisfies_precondition(a)]
        if applicable:
            return applicable

    # Step 4: Otherwise, return any possible actions
    return [a for a in all_actions if state.satisfies_precondition(a)]


# ============================================================================
# Negative Precondition Preprocessing Functions
# ============================================================================


def extract_negative_preconditions(actions: List[Action]) -> Set[Fluent]:
    """Extract all negative preconditions from a list of actions.

    Args:
        actions: List of Action objects

    Returns:
        Set of Fluent objects that appear as negative preconditions.
        These fluents are the "flipped" versions (i.e., the positive form).
        For example, if action has precondition ~F("hand_full r1"),
        this returns {F("hand_full r1")}.
    """
    negative_fluents = set()
    for action in actions:
        # _neg_precond_flipped contains the positive version of negative preconditions
        negative_fluents.update(action._neg_precond_flipped)
    return negative_fluents


def extract_negative_goal_fluents(goal: Goal) -> Set[Fluent]:
    """Extract all negative fluents from a Goal object.

    This is needed to extend the negative-to-positive mapping to include
    goal fluents, not just action precondition fluents.

    Args:
        goal: A Goal object (LiteralGoal, AndGoal, OrGoal, etc.)

    Returns:
        Set of positive Fluent objects that appear negated in the goal.
        For example, if goal has ~F("at Book table"), returns {F("at Book table")}.
    """
    from railroad._bindings import GoalType

    negative_fluents = set()
    goal_type = goal.get_type()

    if isinstance(goal, LiteralGoal):
        fluent = goal.fluent()
        if fluent.negated:
            # Return the positive form
            negative_fluents.add(~fluent)
    elif goal_type in (GoalType.AND, GoalType.OR):
        for child in goal.children():
            negative_fluents.update(extract_negative_goal_fluents(child))

    return negative_fluents


def extract_all_negative_fluents(
    actions: Optional[List[Action]] = None,
    goal: Optional[Goal] = None
) -> Set[Fluent]:
    """Extract negative fluents from actions and/or goals.

    This unified function collects all fluents that appear in negated form,
    either as negative preconditions in actions or as negative literals in goals.
    The returned fluents are in positive form (ready for mapping creation).

    Args:
        actions: Optional list of Action objects
        goal: Optional Goal object

    Returns:
        Set of positive Fluent objects that appear negated in actions/goals.
    """
    negative_fluents = set()

    if actions:
        negative_fluents.update(extract_negative_preconditions(actions))

    if goal:
        negative_fluents.update(extract_negative_goal_fluents(goal))

    return negative_fluents


def create_positive_fluent_mapping(negative_fluents: Set[Fluent]) -> Dict[Fluent, Fluent]:
    """Create mapping from negative fluents to their positive "not-" versions.

    Args:
        negative_fluents: Set of fluents that appear in negative preconditions

    Returns:
        Dictionary mapping each fluent to its "not-" version.
        For example: F("hand_full r1") -> F("not-hand_full r1")
    """
    mapping = {}
    for fluent in negative_fluents:
        # Create positive version with "not-" prefix
        not_name = f"not-{fluent.name}"
        not_fluent = Fluent(not_name, *fluent.args)
        mapping[fluent] = not_fluent
    return mapping


def convert_state_to_positive_preconditions(
    state: State,
    neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> State:
    """Convert state to use positive versions of negative preconditions.

    For each negative precondition that could exist, adds the corresponding
    positive "not-" fluent if the original fluent is absent. Also converts
    upcoming effects to maintain consistency with the mapping.

    Args:
        state: Original state
        neg_to_pos_mapping: Mapping from fluents to their "not-" versions

    Returns:
        New State with additional positive fluents representing absence
        and converted upcoming effects.
        For example, if F("hand_full r1") is not in state.fluents,
        adds F("not-hand_full r1") to indicate hand is not full.
    """
    from railroad._bindings import GroundedEffect

    new_fluents = set(state.fluents)

    for original_fluent, not_fluent in neg_to_pos_mapping.items():
        # If the original fluent is NOT in the state, add the "not-" version
        if original_fluent not in state.fluents:
            new_fluents.add(not_fluent)

    # Convert upcoming effects using the same logic as convert_action_effects
    def augment_fluents(fluents: Set[Fluent]) -> Set[Fluent]:
        """Augment a set of fluents with consistency fluents."""
        augmented = set(fluents)
        for fluent in fluents:
            if fluent.negated:
                # Fluent is ~F("P") - removing P
                # Check if F("P") (the positive version) is in mapping
                positive_fluent = ~fluent  # Invert to get F("P")
                if positive_fluent in neg_to_pos_mapping:
                    # Add F("not-P") since P is being removed
                    augmented.add(neg_to_pos_mapping[positive_fluent])
            else:
                # Fluent is F("P") - adding P
                # Check if this P is in the mapping
                if fluent in neg_to_pos_mapping:
                    # Add ~F("not-P") since P is being added
                    augmented.add(~neg_to_pos_mapping[fluent])
        return augmented

    def convert_grounded_effect(effect: GroundedEffect) -> GroundedEffect:
        """Recursively convert a GroundedEffect and its probabilistic branches."""
        # Augment the immediate resulting fluents
        augmented_fluents = augment_fluents(effect.resulting_fluents)

        # Recursively convert probabilistic effects
        if effect.is_probabilistic:
            converted_prob_effects = []
            for prob_branch in effect.prob_effects:
                prob = prob_branch.prob
                converted_branch_effects = [
                    convert_grounded_effect(branch_eff)
                    for branch_eff in prob_branch.effects
                ]
                converted_prob_effects.append((prob, converted_branch_effects))

            return GroundedEffect(
                time=effect.time,
                resulting_fluents=augmented_fluents,
                prob_effects=converted_prob_effects
            )
        else:
            return GroundedEffect(
                time=effect.time,
                resulting_fluents=augmented_fluents
            )

    # Convert all upcoming effects (which are tuples of (time, effect))
    converted_effects = [(time, convert_grounded_effect(eff)) for time, eff in state.upcoming_effects]

    return State(time=state.time, fluents=new_fluents, upcoming_effects=converted_effects)


def convert_action_to_positive_preconditions(
    action: Action,
    neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> Action:
    """Convert action's negative preconditions to positive "not-" versions.

    Args:
        action: Original action with negative preconditions
        neg_to_pos_mapping: Mapping from fluents to their "not-" versions

    Returns:
        New Action with negative preconditions replaced by positive ones.
        For example, precondition ~F("hand_full r1") becomes F("not-hand_full r1").
    """
    new_preconditions = set()

    # Add all positive preconditions as-is
    new_preconditions.update(action._pos_precond)

    # Replace negative preconditions with positive "not-" versions
    for neg_fluent in action._neg_precond_flipped:
        if neg_fluent in neg_to_pos_mapping:
            new_preconditions.add(neg_to_pos_mapping[neg_fluent])
        else:
            # If not in mapping, keep as negative (shouldn't happen after preprocessing)
            new_preconditions.add(~neg_fluent)

    # Create new action with converted preconditions
    return Action(new_preconditions, action.effects, name=action.name, extra_cost=action.extra_cost)


def convert_action_effects(
    action: Action,
    neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> Action:
    """Convert action's effects to maintain consistency with positive preconditions.

    When an effect adds or removes a fluent that has a negative precondition mapping,
    this function adds the corresponding "not-" fluent to maintain consistency.

    For example:
    - If effect adds F("hand_full"), also add ~F("not-hand_full")
    - If effect removes F("hand_full") (i.e., ~F("hand_full")), also add F("not-hand_full")

    Args:
        action: Original action
        neg_to_pos_mapping: Mapping from fluents to their "not-" versions

    Returns:
        New Action with augmented effects
    """
    from railroad._bindings import GroundedEffect

    def augment_fluents(fluents: Set[Fluent]) -> Set[Fluent]:
        """Augment a set of fluents with consistency fluents."""
        augmented = set(fluents)
        for fluent in fluents:
            if fluent.negated:
                # Fluent is ~F("P") - removing P
                # Check if F("P") (the positive version) is in mapping
                positive_fluent = ~fluent  # Invert to get F("P")
                if positive_fluent in neg_to_pos_mapping:
                    # Add F("not-P") since P is being removed
                    augmented.add(neg_to_pos_mapping[positive_fluent])
            else:
                # Fluent is F("P") - adding P
                # Check if this P is in the mapping
                if fluent in neg_to_pos_mapping:
                    # Add ~F("not-P") since P is being added
                    augmented.add(~neg_to_pos_mapping[fluent])
        return augmented

    def convert_grounded_effect(effect: GroundedEffect) -> GroundedEffect:
        """Recursively convert a GroundedEffect and its probabilistic branches."""
        # Augment the immediate resulting fluents
        augmented_fluents = augment_fluents(effect.resulting_fluents)

        # Recursively convert probabilistic effects
        if effect.is_probabilistic:
            converted_prob_effects = []
            for prob_branch in effect.prob_effects:
                prob = prob_branch.prob
                converted_branch_effects = [
                    convert_grounded_effect(branch_eff)
                    for branch_eff in prob_branch.effects
                ]
                converted_prob_effects.append((prob, converted_branch_effects))

            return GroundedEffect(
                time=effect.time,
                resulting_fluents=augmented_fluents,
                prob_effects=converted_prob_effects
            )
        else:
            return GroundedEffect(
                time=effect.time,
                resulting_fluents=augmented_fluents
            )

    # Convert all effects
    converted_effects = [convert_grounded_effect(eff) for eff in action.effects]

    # Create new action with converted effects
    return Action(action.preconditions, converted_effects, name=action.name, extra_cost=action.extra_cost)


def preprocess_actions_for_relaxed_planning(
    actions: List[Action],
    initial_state: State
) -> Tuple[List[Action], State, Dict[Fluent, Fluent]]:
    """Preprocess actions and state to convert negative preconditions to positive.

    This is a one-time preprocessing step that should be done after actions
    are instantiated. The resulting actions and state can then be used for
    planning with algorithms like FF heuristic that work better with positive
    preconditions.

    Args:
        actions: List of instantiated actions
        initial_state: Initial state of the planning problem

    Returns:
        Tuple of (converted_actions, converted_state, mapping_dict):
        - converted_actions: Actions with negative preconditions replaced
        - converted_state: State with additional positive fluents
        - mapping_dict: Mapping used for conversion (for debugging/inspection)
    """
    # Step 1: Extract all negative preconditions
    negative_fluents = extract_negative_preconditions(actions)

    # Step 2: Create mapping to positive "not-" versions
    neg_to_pos_mapping = create_positive_fluent_mapping(negative_fluents)

    # Step 3: Convert all actions (preconditions and effects)
    converted_actions = []
    for action in actions:
        # First convert preconditions
        action_with_preconds = convert_action_to_positive_preconditions(action, neg_to_pos_mapping)
        # Then convert effects
        action_with_effects = convert_action_effects(action_with_preconds, neg_to_pos_mapping)
        converted_actions.append(action_with_effects)

    # Step 4: Convert initial state
    converted_state = convert_state_to_positive_preconditions(initial_state, neg_to_pos_mapping)

    return converted_actions, converted_state, neg_to_pos_mapping


def convert_goal_to_positive_preconditions(
    goal,  # Goal type from bindings
    neg_to_pos_mapping: Dict[Fluent, Fluent]
):
    """Convert a Goal's negative fluents to positive "not-" equivalents.

    This is necessary when using Goals with the MCTSPlanner, which converts
    negative preconditions to positive forms internally. Without this conversion,
    the heuristic function won't correctly evaluate goal literals.

    Args:
        goal: A Goal object (LiteralGoal, AndGoal, OrGoal, etc.)
        neg_to_pos_mapping: Mapping from fluents to their "not-" versions

    Returns:
        A new Goal with converted fluents
    """
    from railroad._bindings import (
        GoalType,
        LiteralGoal,
        TrueGoal,
        FalseGoal,
    )

    goal_type = goal.get_type()

    if goal_type == GoalType.TRUE_GOAL:
        return TrueGoal()
    elif goal_type == GoalType.FALSE_GOAL:
        return FalseGoal()
    elif goal_type == GoalType.LITERAL:
        fluent = goal.fluent()
        converted_fluent = _convert_fluent(fluent, neg_to_pos_mapping)
        return LiteralGoal(converted_fluent)
    elif goal_type == GoalType.AND:
        converted_children = [
            convert_goal_to_positive_preconditions(child, neg_to_pos_mapping)
            for child in goal.children()
        ]
        return AndGoal(converted_children)
    elif goal_type == GoalType.OR:
        converted_children = [
            convert_goal_to_positive_preconditions(child, neg_to_pos_mapping)
            for child in goal.children()
        ]
        return OrGoal(converted_children)
    else:
        # Unknown goal type, return as-is
        return goal


def _convert_fluent(
    fluent: Fluent,
    neg_to_pos_mapping: Dict[Fluent, Fluent]
) -> Fluent:
    """Convert a single fluent using the negative-to-positive mapping.

    Handles the conversion of negative fluents like ~F("P") to F("not-P").
    """
    if fluent.negated:
        # Fluent is ~F("P") - we want F("not-P")
        positive_fluent = ~fluent  # Get F("P")
        if positive_fluent in neg_to_pos_mapping:
            # Return F("not-P") instead of ~F("P")
            return neg_to_pos_mapping[positive_fluent]
    # No conversion needed
    return fluent
