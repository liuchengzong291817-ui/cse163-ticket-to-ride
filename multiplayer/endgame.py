"""Exact finite-belief endgames with one remaining focal action.

Research evaluator, not a playing policy. Hypothetical worlds are supplied
by a declared belief model; never supply the actual hidden world as a clue.
Only four-player states already in the final countdown are supported.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from .game import (Action, GameState, Observation, _connected, final_scores,
                   legal_actions, observe, step, winners)
from .policies import ReplanningGreedyPolicy


def action_key(action):
    return action.kind, action.route_id or ""


def action_name(action):
    return " ".join(part for part in action_key(action) if part)


@dataclass(frozen=True)
class BeliefWorld:
    name: str
    probability: Fraction
    state: GameState


@dataclass(frozen=True)
class Frontier:
    world: str
    probability: Fraction
    state: GameState
    events: tuple
    value: Fraction


@dataclass(frozen=True)
class Estimate:
    action: Action
    depth: Optional[int]
    value: Fraction
    float_value: float
    transitions: int
    frontier: tuple
    conditional_values: tuple


def validate_belief(worlds, focal, policies):
    worlds = tuple(worlds)
    if not worlds or type(focal) is not int or not 0 <= focal < 4:
        raise ValueError("supply worlds and a focal seat from 0 to 3")
    if (len(policies) != 4 or policies[focal] != "focal"
            or any(p not in {"replan", "uniform"}
                   for i, p in enumerate(policies) if i != focal)):
        raise ValueError("fix replan/uniform policies for three opponents")
    names = [w.name for w in worlds]
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("world names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("world names must be unique")
    if any(not isinstance(w.probability, Fraction) or w.probability <= 0
           for w in worlds):
        raise ValueError("positive exact Fraction probabilities required")
    if sum(w.probability for w in worlds) != 1:
        raise ValueError("world probabilities must sum exactly to one")
    for world in worlds:
        state = world.state
        if not isinstance(state, GameState):
            raise ValueError("world state must come from the game engine")
        if (state.rules.player_count != 4 or state.finished
                or state.current_player != focal
                or type(state.final_turns_remaining) is not int
                or not 1 <= state.final_turns_remaining <= 4):
            raise ValueError("need a live four-player final-turn focal state")
    view = observe(worlds[0].state, focal)
    if any(observe(w.state, focal) != view for w in worlds[1:]):
        raise ValueError("worlds must have identical focal observations")
    return tuple(sorted(worlds, key=lambda w: w.name))


def action_distribution(view, policy):
    """Opponent behavior uses its own observation, never a complete state."""
    if not isinstance(view, Observation):
        raise TypeError("opponent policy requires an Observation")
    actions = sorted(legal_actions(view), key=action_key)
    if not actions:
        raise ValueError("opponent needs a live observation of its turn")
    if policy == "replan":
        return ((ReplanningGreedyPolicy().choose(view), Fraction(1)),)
    if policy == "uniform":
        return tuple((a, Fraction(1, len(actions))) for a in actions)
    raise ValueError("unknown opponent policy")


def snapshot_scores(state):
    """Research-only hypothetical settlement, without ending the game."""
    scores = []
    for seat, player in enumerate(state.players):
        complete = [t for t in player.tickets
                    if _connected(state, seat, t.source, t.target)]
        total = (player.route_points + 2 * sum(t.points for t in complete)
                 - sum(t.points for t in player.tickets))
        scores.append((total, len(complete)))
    return tuple(scores)


def snapshot_value(state, focal):
    scores = snapshot_scores(state)
    best = max(scores)
    return (Fraction(1, scores.count(best))
            if scores[focal] == best else Fraction(0))


def terminal_value(state, focal):
    winning = winners(state)  # The rules reject nonterminal scoring.
    return (Fraction(1, len(winning)) if focal in winning
            else Fraction(0))


def _walk(state, depth, focal, policies, events):
    if state.finished:
        value = terminal_value(state, focal)
        return value, float(value), 0, [(Fraction(1), state, events, value)]
    if depth == 0:
        value = snapshot_value(state, focal)
        return value, float(value), 0, [(Fraction(1), state, events, value)]
    seat = state.current_player
    if seat == focal:
        raise ValueError("multiple focal decisions need information sets")
    value, floating, transitions, frontier = Fraction(0), 0.0, 0, []
    for action, probability in action_distribution(
            observe(state, seat), policies[seat]):
        after, event = step(state, seat, action)
        next_depth = None if depth is None else depth - 1
        child, child_float, count, leaves = _walk(
            after, next_depth, focal, policies, events + (event,))
        value += probability * child
        floating += float(probability) * child_float
        transitions += 1 + count
        frontier.extend((probability * p, s, trace, u)
                        for p, s, trace, u in leaves)
    return value, floating, transitions, frontier


def evaluate_action(worlds, action, depth=None, *, focal=0,
                    policies=("focal", "replan", "replan", "uniform")):
    """Enumerate to true termination, or h opponent actions after root.

    World averaging precedes root action selection. No max is taken inside
    a hidden world. Float recursion follows the same tree as Fraction.
    """
    if depth is not None and (type(depth) is not int or depth < 0):
        raise ValueError("depth must be None or a nonnegative integer")
    worlds = validate_belief(worlds, focal, policies)
    if action not in legal_actions(observe(worlds[0].state, focal)):
        raise ValueError("root action must be legal in the common view")
    value, floating, transitions = Fraction(0), 0.0, 0
    frontier, conditional = [], []
    for world in worlds:
        after, event = step(world.state, focal, action)
        child, child_float, count, leaves = _walk(
            after, depth, focal, policies, (event,))
        if sum(p for p, _, _, _ in leaves) != 1:
            raise ArithmeticError("conditional frontier mass is not one")
        value += world.probability * child
        floating += float(world.probability) * child_float
        transitions += 1 + count
        conditional.append((world.name, child))
        frontier.extend(Frontier(world.name, world.probability * p,
                                 state, events, utility)
                        for p, state, events, utility in leaves)
    if sum(leaf.probability for leaf in frontier) != 1:
        raise ArithmeticError("belief frontier mass is not one")
    return Estimate(action, depth, value, floating, transitions,
                    tuple(frontier), tuple(conditional))


def evaluate_actions(worlds, depth=None, *, focal=0,
                     policies=("focal", "replan", "replan", "uniform")):
    worlds = validate_belief(worlds, focal, policies)
    actions = sorted(legal_actions(observe(worlds[0].state, focal)),
                     key=action_key)
    return tuple(evaluate_action(worlds, action, depth,
                                 focal=focal, policies=policies)
                 for action in actions)


def diagnostics(reference, approximate):
    """Offline bounds using known reference error; not an online oracle."""
    if not reference or set(reference) != set(approximate):
        raise ValueError("match the same nonempty legal action set")
    if any(not isinstance(v, Fraction) or not 0 <= v <= 1
           for table in (reference, approximate) for v in table.values()):
        raise ValueError("values must be exact victory shares in [0, 1]")
    actions = sorted(reference, key=action_key)
    best, estimated_best = max(reference.values()), max(approximate.values())
    optimal = tuple(a for a in actions if reference[a] == best)
    maximizing = tuple(a for a in actions
                       if approximate[a] == estimated_best)
    selected = maximizing[0]
    errors = {a: approximate[a] - reference[a] for a in actions}
    epsilon = max(abs(e) for e in errors.values())
    regret = best - reference[selected]
    gap = (best - sorted(reference.values(), reverse=True)[1]
           if len(actions) > 1 else None)
    upper = all(e >= 0 for e in errors.values())
    generic_certificate = (len(actions) == 1 or
                           (len(optimal) == 1 and gap > 2 * epsilon))
    upper_certificate = upper and (len(actions) == 1 or
                                   (len(optimal) == 1 and gap > epsilon))
    return {
        "optimal_actions": optimal, "maximizing_actions": maximizing,
        "selected_action": selected,
        "selected_is_optimal": selected in optimal,
        "unique_best_gap": gap, "epsilon": epsilon, "regret": regret,
        "two_sided_regret_bound": 2 * epsilon,
        "two_sided_bound_holds": regret <= 2 * epsilon,
        "two_sided_unique_best_certificate": generic_certificate,
        "all_estimates_are_upper_bounds": upper,
        "one_sided_regret_bound": epsilon if upper else None,
        "one_sided_bound_holds": regret <= epsilon if upper else None,
        "one_sided_unique_best_certificate": upper_certificate,
    }


def score_record(state):
    if state.finished:
        return [(s.total, len(s.completed)) for s in final_scores(state)]
    return list(snapshot_scores(state))
