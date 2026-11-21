"""Rank affordable goal-path edges by own-ticket single-edge sensitivity.

Deleting a currently available physical edge measures consequence, not the
probability that an opponent will claim it. Only private own tickets and
public observations are used. The controlled comparator is dynamic greedy:
goal selection, candidate path, gather, fallback and final-turn rules match.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .game import _available
from .policies import ReplanningGreedyPolicy, shortest_remaining_path


@dataclass(frozen=True)
class TicketImpact:
    ticket_id: str
    ticket_points: int
    before_cost: int
    after_cost: Optional[int]
    status: str  # feasible, over_budget, or unreachable


@dataclass(frozen=True)
class EdgeImpact:
    route_id: str
    lost_ticket_swing: int
    weighted_detour: int
    tickets: Tuple[TicketImpact, ...]


def edge_impact(observation, route_id):
    """Delete one available edge, retaining all other costs and constraints.

    Include each own currently feasible unfinished ticket once. First sum
    2*points for tickets made unreachable or over the current train budget.
    Then sum 2*points*extra_trains for tickets that remain feasible. These
    are lexicographic heuristic components, not an expected score or a joint
    network cost. None records unreachable; infinity arithmetic is avoided.
    """
    index = next((i for i, route in enumerate(observation.routes)
                  if route.route_id == route_id), None)
    if index is None or not _available(observation, index):
        raise ValueError("impact requires a currently available physical edge")
    trains = observation.players[observation.player_id].trains
    lost, detour, impacts = 0, 0, []
    for ticket in sorted(observation.own_tickets, key=lambda t: t.ticket_id):
        before = shortest_remaining_path(
            observation, ticket.source, ticket.target)
        if before is None or not 0 < before.additional_trains <= trains:
            continue
        after = shortest_remaining_path(
            observation, ticket.source, ticket.target,
            excluded_route_ids=(route_id,))
        cost = None if after is None else after.additional_trains
        if cost is None:
            status = "unreachable"
            lost += 2 * ticket.points
        elif cost > trains:
            status = "over_budget"
            lost += 2 * ticket.points
        else:
            status = "feasible"
            detour += 2 * ticket.points * (cost - before.additional_trains)
        impacts.append(TicketImpact(ticket.ticket_id, ticket.points,
                                    before.additional_trains, cost, status))
    return EdgeImpact(route_id, lost, detour, tuple(impacts))


class BottleneckPriorityPolicy(ReplanningGreedyPolicy):
    """Dynamic greedy with only its within-path claim ordering replaced.

    Rank available affordable selected-path claims by lost ticket swing,
    then weighted detour, then the original shortest-edge/ID tiebreak.
    A longer unaffordable bottleneck does not change the gathering rule.
    """

    _claim_reason = ("Replanned own ticket: prioritize an affordable path "
                     "edge by own-ticket single-edge deletion impact.")

    def _select_claim(self, observation, on_path):
        routes = {r.route_id: r for r in observation.routes}
        impacts = {a.route_id: edge_impact(observation, a.route_id)
                   for a in on_path}
        return min(on_path, key=lambda a: (
            -impacts[a.route_id].lost_ticket_swing,
            -impacts[a.route_id].weighted_detour,
            routes[a.route_id].length, a.route_id))
