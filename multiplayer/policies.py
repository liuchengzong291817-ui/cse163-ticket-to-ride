"""Small observation-only baselines, not optimal or learned strategies.

The replanning baseline solves a shortest-path subproblem each turn. It does
not predict opponents, search a game tree, or estimate a game's win rate.
"""

import heapq
import random
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Optional, Tuple

from .game import Action, ROUTE_POINTS, _available, _connected, legal_actions


@dataclass(frozen=True)
class PathPlan:
    additional_trains: int
    route_ids: Tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    """Private explanation: do not put ticket information in public events."""

    action: Action
    reason: str
    ticket_id: Optional[str] = None
    plan: Optional[PathPlan] = None


def shortest_remaining_path(observation, source, target, *,
                            excluded_route_ids=()):
    """Dijkstra: owned routes cost zero; available routes cost their length.

    Opponent routes and closed parallel alternatives are excluded. Costs are
    additional trains, not remaining turns or a forecast of opponent actions.
    Sorted adjacency and strict distance improvements make ties deterministic
    and prevent parent cycles on zero-cost owned routes.
    Optional physical-edge deletion is a graph query, not an opponent move:
    it does not alter ownership or close other edges in a parallel group.
    """
    adjacency = {}
    excluded = frozenset(excluded_route_ids)
    for index, (route, owner) in enumerate(
            zip(observation.routes, observation.owners)):
        if route.route_id in excluded:
            continue
        if owner == observation.player_id:
            cost = 0
        elif _available(observation, index):
            cost = route.length
        else:
            continue
        adjacency.setdefault(route.source, []).append(
            (route.route_id, route.target, cost))
        adjacency.setdefault(route.target, []).append(
            (route.route_id, route.source, cost))
    distances, previous = {source: 0}, {}
    queue = [(0, source)]
    while queue:
        distance, city = heapq.heappop(queue)
        if distance != distances[city]:
            continue
        if city == target:
            route_ids = []
            while city != source:
                city, route_id = previous[city]
                route_ids.append(route_id)
            return PathPlan(distance, tuple(reversed(route_ids)))
        for route_id, neighbor, cost in sorted(adjacency.get(city, ())):
            candidate = distance + cost
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                previous[neighbor] = (city, route_id)
                heapq.heappush(queue, (candidate, neighbor))
    return None


def _current_actions(observation):
    actions = legal_actions(observation)
    if not actions:
        raise ValueError("policy needs a live observation of its own turn")
    return actions


class RandomLegalPolicy:
    """Uniform legal-action choice, using a private seeded RNG."""

    def __init__(self, seed):
        self._rng = random.Random(seed)

    def decide(self, observation):
        actions = sorted(_current_actions(observation),
                         key=lambda a: (a.kind, a.route_id or ""))
        action = self._rng.choice(actions)
        return Decision(action, "Uniform seeded choice among legal actions.")

    def choose(self, observation):
        return self.decide(observation).action


class ReplanningGreedyPolicy:
    """Recompute own goals; rank feasible unfinished tickets by 2*points/cost.

    Claim the shortest affordable new route on the chosen path, or gather.
    If no goal fits the train budget, take the best immediate route score.
    On a known final turn, maximize immediate own score including ticket
    completion. This heuristic has no claim of optimality or strategic skill.
    """

    _claim_reason = ("Replanned own ticket: shortest affordable "
                     "route on its minimum-additional-train path.")

    def decide(self, observation):
        actions = _current_actions(observation)
        claims = [a for a in actions if a.kind == "claim"]
        routes = {r.route_id: r for r in observation.routes}

        if observation.final_turns_remaining is not None:
            # After the final round starts, each player has at most one turn.
            if not claims:
                action = (Action("gather") if Action("gather") in actions
                          else Action("pass"))
                return Decision(action, "Final turn: no scoring claim is "
                                "available; take a required legal action.")
            gains = {a.route_id: self._immediate_gain(observation, a)
                     for a in claims}
            action = min(claims, key=lambda a: (
                -gains[a.route_id], routes[a.route_id].length, a.route_id))
            return Decision(action, "Final turn: largest immediate own score "
                            f"gain ({gains[action.route_id]} points).")

        goal = self._select_goal(observation)
        if goal is not None:
            ticket, plan = goal
            on_path = [a for a in claims if a.route_id in plan.route_ids]
            if on_path:
                action = self._select_claim(observation, on_path)
                return Decision(
                    action, self._claim_reason,
                    ticket.ticket_id, plan)
            if Action("gather") in actions:
                return Decision(
                    Action("gather"), "Goal fits remaining trains, but no "
                    "new route on its path is affordable yet.",
                    ticket.ticket_id, plan)

        if claims:
            action = min(claims, key=lambda a: (
                -ROUTE_POINTS[routes[a.route_id].length],
                routes[a.route_id].length, a.route_id))
            return Decision(action, "No feasible unfinished ticket: take "
                            "the largest immediate route score.")
        if Action("gather") in actions:
            return Decision(Action("gather"),
                            "No affordable claim; gather for a legal route.")
        return Decision(Action("pass"), "No claim or gather is legal.")

    def choose(self, observation):
        return self.decide(observation).action

    def _select_claim(self, observation, on_path):
        """Shared extension point: only rank affordable selected-path edges."""
        routes = {r.route_id: r for r in observation.routes}
        return min(on_path, key=lambda a: (
            routes[a.route_id].length, a.route_id))

    def _select_goal(self, observation):
        """Shared goal ranking; subclasses may retain a still-valid plan."""
        player = observation.players[observation.player_id]
        goals = []
        for ticket in observation.own_tickets:
            plan = shortest_remaining_path(
                observation, ticket.source, ticket.target)
            if (plan is not None
                    and 0 < plan.additional_trains <= player.trains):
                goals.append((ticket, plan))
        if not goals:
            return None
        return min(goals, key=lambda pair: (
            -Fraction(2 * pair[0].points, pair[1].additional_trains),
            pair[1].additional_trains, pair[0].ticket_id))

    @staticmethod
    def _immediate_gain(observation, action):
        owners = list(observation.owners)
        index = next(i for i, route in enumerate(observation.routes)
                     if route.route_id == action.route_id)
        owners[index] = observation.player_id
        after = replace(observation, owners=tuple(owners))
        gain = ROUTE_POINTS[observation.routes[index].length]
        for ticket in observation.own_tickets:
            before_path = shortest_remaining_path(
                observation, ticket.source, ticket.target)
            after_path = shortest_remaining_path(
                after, ticket.source, ticket.target)
            was_complete = (before_path is not None
                            and before_path.additional_trains == 0)
            is_complete = (after_path is not None
                           and after_path.additional_trains == 0)
            if is_complete and not was_complete:
                # A held ticket changes from -points to +points.
                gain += 2 * ticket.points
        return gain


class StaticPlanningPolicy(ReplanningGreedyPolicy):
    """Commit to one goal/path until completion, blocking, or budget failure.

    Repair uses exactly the dynamic policy's goal-ranking rule, including
    permission to choose a different goal. Resource shortage alone does not
    invalidate the plan. All action selection, fallback and final-turn rules
    are inherited unchanged. Create a fresh instance per player per game.
    """

    _claim_reason = ("Stored own ticket: shortest affordable "
                     "route on the retained or repaired path.")

    def __init__(self):
        self._ticket_id = None
        self._route_ids = ()
        self.last_plan_event = "not_started"

    def decide(self, observation):
        _current_actions(observation)
        if observation.final_turns_remaining is not None:
            self._ticket_id, self._route_ids = None, ()
            self.last_plan_event = "final_turn"
            return super().decide(observation)
        decision = super().decide(observation)
        return replace(decision, reason=(
            f"Static plan [{self.last_plan_event}]: {decision.reason}"))

    def _select_goal(self, observation):
        reason = "new_goal"
        ticket = next((t for t in observation.own_tickets
                       if t.ticket_id == self._ticket_id), None)
        if ticket is not None:
            # Only test owned-edge connectivity, without searching for a new
            # shortest route while the stored plan remains valid.
            if _connected(observation, observation.player_id,
                          ticket.source, ticket.target):
                reason = "completed"
            else:
                route_indices = {r.route_id: i
                                 for i, r in enumerate(observation.routes)}
                cost, blocked = 0, False
                for route_id in self._route_ids:
                    index = route_indices.get(route_id)
                    if index is None:
                        blocked = True
                        break
                    owner = observation.owners[index]
                    if owner == observation.player_id:
                        continue
                    if not _available(observation, index):
                        blocked = True
                        break
                    cost += observation.routes[index].length
                if blocked:
                    reason = "blocked"
                elif cost > observation.players[observation.player_id].trains:
                    reason = "over_budget"
                elif cost > 0:
                    self.last_plan_event = "retained"
                    return ticket, PathPlan(cost, self._route_ids)
                else:
                    reason = "completed"
        goal = super()._select_goal(observation)
        self.last_plan_event = reason
        if goal is None:
            self._ticket_id, self._route_ids = None, ()
        else:
            ticket, plan = goal
            self._ticket_id, self._route_ids = ticket.ticket_id, plan.route_ids
        return goal
