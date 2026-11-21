"""Immutable multiplayer rules; no dependency on the historical optimizer.

This is not a complete Ticket to Ride: Europe implementation. Resource
tokens replace train cards; tickets are fixed; stations, tunnel/ferry costs,
ticket draws and longest-route bonuses are omitted. See README.md.
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple


ROUTE_POINTS = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}


@dataclass(frozen=True)
class Route:
    route_id: str
    source: str
    target: str
    length: int
    parallel_group: Optional[str] = None


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    source: str
    target: str
    points: int


@dataclass(frozen=True)
class Rules:
    player_count: int = 4
    initial_trains: int = 45
    initial_resources: int = 4
    resource_cap: int = 8
    gather_amount: int = 2
    final_threshold: int = 2


@dataclass(frozen=True)
class Player:
    trains: int
    resources: int
    route_points: int
    tickets: Tuple[Ticket, ...]


@dataclass(frozen=True)
class PublicPlayer:
    trains: int
    resources: int
    route_points: int
    ticket_count: int


@dataclass(frozen=True)
class GameState:
    rules: Rules
    routes: Tuple[Route, ...]
    owners: Tuple[Optional[int], ...]
    players: Tuple[Player, ...]
    current_player: int
    turn: int = 0
    final_turns_remaining: Optional[int] = None
    consecutive_passes: int = 0
    finished: bool = False
    end_reason: Optional[str] = None


@dataclass(frozen=True)
class Observation:
    """Policy input: public state plus this player's tickets only."""

    player_id: int
    rules: Rules
    routes: Tuple[Route, ...]
    owners: Tuple[Optional[int], ...]
    players: Tuple[PublicPlayer, ...]
    own_tickets: Tuple[Ticket, ...]
    current_player: int
    turn: int
    final_turns_remaining: Optional[int]
    consecutive_passes: int
    finished: bool
    end_reason: Optional[str]


@dataclass(frozen=True)
class Action:
    kind: str
    route_id: Optional[str] = None


@dataclass(frozen=True)
class PublicEvent:
    turn: int
    player_id: int
    action: Action
    trains_after: int
    resources_after: int
    route_points_after: int
    final_turns_remaining: Optional[int]
    end_reason: Optional[str]


@dataclass(frozen=True)
class Score:
    player_id: int
    route_points: int
    completed: Tuple[Ticket, ...]
    failed: Tuple[Ticket, ...]
    ticket_points: int
    total: int


class IllegalAction(ValueError):
    """An action is rejected without changing the immutable game state."""


def new_game(routes, tickets_by_player, rules=Rules(), starting_player=0):
    """Validate and freeze supplied setup, without dealing cards."""
    values = (rules.player_count, rules.initial_trains,
              rules.initial_resources, rules.resource_cap,
              rules.gather_amount, rules.final_threshold)
    if any(type(value) is not int for value in values):
        raise ValueError("rule counts must be integers")
    if not 2 <= rules.player_count <= 5:
        raise ValueError("player_count must be between 2 and 5")
    if not (0 <= rules.final_threshold < rules.initial_trains
            and 0 <= rules.initial_resources <= rules.resource_cap
            and rules.gather_amount > 0 and rules.resource_cap > 0):
        raise ValueError("invalid resource, train, or final-round settings")
    if type(starting_player) is not int or not (
            0 <= starting_player < rules.player_count):
        raise ValueError("invalid starting player")
    routes = tuple(routes)
    tickets_by_player = tuple(tuple(tickets) for tickets in tickets_by_player)
    if not routes or len(tickets_by_player) != rules.player_count:
        raise ValueError("supply a map and tickets for every player")
    ids, cities, groups, pairs = set(), set(), {}, {}
    for route in routes:
        if (not route.route_id or route.route_id in ids
                or not route.source or not route.target
                or route.source == route.target
                or type(route.length) is not int
                or route.length not in ROUTE_POINTS
                or route.length > rules.resource_cap):
            raise ValueError("invalid or duplicate route")
        ids.add(route.route_id)
        cities.update((route.source, route.target))
        pair = tuple(sorted((route.source, route.target)))
        pairs.setdefault(pair, []).append(route.parallel_group)
        if route.parallel_group is not None:
            if not route.parallel_group:
                raise ValueError("parallel group must be nonempty")
            groups.setdefault(route.parallel_group, set()).add(pair)
    if any(len(group_pairs) != 1 for group_pairs in groups.values()):
        raise ValueError("parallel group must share endpoints")
    if any(len(group_ids) > 1
           and (None in group_ids or len(set(group_ids)) != 1)
           for group_ids in pairs.values()):
        raise ValueError("parallel routes require the same explicit group")
    ticket_ids = set()
    for tickets in tickets_by_player:
        for ticket in tickets:
            if (not ticket.ticket_id or ticket.ticket_id in ticket_ids
                    or ticket.source not in cities
                    or ticket.target not in cities
                    or ticket.source == ticket.target
                    or type(ticket.points) is not int or ticket.points <= 0):
                raise ValueError("invalid or duplicate ticket")
            ticket_ids.add(ticket.ticket_id)
    players = tuple(Player(rules.initial_trains, rules.initial_resources,
                           0, tickets) for tickets in tickets_by_player)
    return GameState(rules, routes, (None,) * len(routes), players,
                     starting_player)


def observe(state, player_id):
    """Construct an immutable public view without opponent tickets."""
    if type(player_id) is not int or not 0 <= player_id < len(state.players):
        raise ValueError("invalid player id")
    public = tuple(PublicPlayer(p.trains, p.resources, p.route_points,
                                len(p.tickets)) for p in state.players)
    return Observation(player_id, state.rules, state.routes, state.owners,
                       public, state.players[player_id].tickets,
                       state.current_player, state.turn,
                       state.final_turns_remaining, state.consecutive_passes,
                       state.finished, state.end_reason)


def _available(observation, route_index):
    if observation.owners[route_index] is not None:
        return False
    route = observation.routes[route_index]
    if route.parallel_group is None:
        return True
    for other, owner in zip(observation.routes, observation.owners):
        if other.parallel_group == route.parallel_group and owner is not None:
            if observation.rules.player_count <= 3:
                return False
            if owner == observation.player_id:
                return False
    return True


def legal_actions(observation):
    """List current-player actions using only their observation."""
    if observation.finished or (
            observation.player_id != observation.current_player):
        return ()
    player = observation.players[observation.player_id]
    usable = [route for index, route in enumerate(observation.routes)
              if _available(observation, index)
              and route.length <= player.trains]
    actions = [Action("claim", route.route_id) for route in usable
               if route.length <= player.resources]
    if usable and player.resources < observation.rules.resource_cap:
        actions.append(Action("gather"))
    return tuple(actions) if actions else (Action("pass"),)


def step(state, player_id, action):
    """Apply one legal action; return a new state and a ticket-free event."""
    if state.finished or player_id != state.current_player:
        raise IllegalAction("game finished or wrong player's turn")
    if action not in legal_actions(observe(state, player_id)):
        raise IllegalAction("action is not legal in this observation")
    player = state.players[player_id]
    owners = list(state.owners)
    if action.kind == "claim":
        index = next(i for i, r in enumerate(state.routes)
                     if r.route_id == action.route_id)
        route = state.routes[index]
        owners[index] = player_id
        player = replace(player, trains=player.trains - route.length,
                         resources=player.resources - route.length,
                         route_points=player.route_points
                         + ROUTE_POINTS[route.length])
    elif action.kind == "gather":
        player = replace(player, resources=min(
            state.rules.resource_cap,
            player.resources + state.rules.gather_amount))
    players = list(state.players)
    players[player_id] = player
    remaining = state.final_turns_remaining
    if remaining is not None:
        remaining -= 1
    elif player.trains <= state.rules.final_threshold:
        # Triggering turn is not one of the N final turns.
        remaining = state.rules.player_count
    passes = state.consecutive_passes + 1 if action.kind == "pass" else 0
    reason = None
    if remaining == 0:
        reason = "final_round"
    elif passes == state.rules.player_count:
        reason = "all_players_stuck"
    updated = replace(
        state, players=tuple(players), owners=tuple(owners),
        turn=state.turn + 1,
        current_player=(player_id + 1) % state.rules.player_count,
        final_turns_remaining=remaining, consecutive_passes=passes,
        finished=reason is not None, end_reason=reason)
    event = PublicEvent(updated.turn, player_id, action, player.trains,
                        player.resources, player.route_points,
                        remaining, reason)
    return updated, event


def _connected(state, player_id, source, target):
    adjacency = {}
    for route, owner in zip(state.routes, state.owners):
        if owner == player_id:
            adjacency.setdefault(route.source, []).append(route.target)
            adjacency.setdefault(route.target, []).append(route.source)
    seen, pending = {source}, [source]
    while pending:
        for neighbor in adjacency.get(pending.pop(), ()):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return target in seen


def final_scores(state):
    """Reveal and score every held ticket only after the game has ended."""
    if not state.finished:
        raise ValueError("tickets are not revealed before the game ends")
    scores = []
    for player_id, player in enumerate(state.players):
        completed, failed = [], []
        for ticket in player.tickets:
            target = (completed if _connected(state, player_id,
                                              ticket.source, ticket.target)
                      else failed)
            target.append(ticket)
        ticket_points = (sum(t.points for t in completed)
                         - sum(t.points for t in failed))
        scores.append(Score(player_id, player.route_points, tuple(completed),
                            tuple(failed), ticket_points,
                            player.route_points + ticket_points))
    return tuple(scores)


def winners(state):
    """Break score ties by completed tickets, then allow shared winners."""
    scores = final_scores(state)
    best = max((score.total, len(score.completed)) for score in scores)
    return tuple(s.player_id for s in scores
                 if (s.total, len(s.completed)) == best)
