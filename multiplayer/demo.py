"""A scripted four-player teaching game, not a strategy benchmark."""

from .game import (Action, Route, Rules, Ticket, final_scores, new_game,
                   observe, step, winners)


def demo_setup():
    """Use six cities, twelve physical routes, and eight trains per player."""
    routes = (
        Route("AB1", "A", "B", 2, "AB"),
        Route("AB2", "A", "B", 2, "AB"),
        Route("BC", "B", "C", 2), Route("CD", "C", "D", 2),
        Route("AE", "A", "E", 2), Route("EF", "E", "F", 2),
        Route("FD", "F", "D", 2), Route("BE", "B", "E", 1),
        Route("CE", "C", "E", 1), Route("BF", "B", "F", 2),
        Route("AC", "A", "C", 3), Route("DE", "D", "E", 2),
    )
    tickets = (
        (Ticket("P0-AF", "A", "F", 6), Ticket("P0-AD", "A", "D", 8)),
        (Ticket("P1-CF", "C", "F", 5),),
        (Ticket("P2-BE", "B", "E", 5),),
        (Ticket("P3-CF", "C", "F", 6),),
    )
    # Synthetic teaching tickets may share endpoints and have invented values.
    return new_game(routes, tickets, Rules(initial_trains=8))


class ScriptedPolicy:
    """Receive only an observation and return the next teaching action."""

    def __init__(self, actions):
        self.actions = iter(actions)

    def choose(self, observation):
        if observation.player_id != observation.current_player:
            raise ValueError("policy called out of turn")
        return next(self.actions)


def demo_policies():
    return (
        ScriptedPolicy((Action("claim", "AB1"), Action("gather"),
                        Action("claim", "AE"), Action("claim", "EF"),
                        Action("pass"))),
        ScriptedPolicy((Action("claim", "BC"), Action("claim", "BE"),
                        Action("gather"), Action("claim", "BF"))),
        ScriptedPolicy((Action("claim", "AB2"), Action("claim", "CE"),
                        Action("gather"), Action("claim", "AC"))),
        ScriptedPolicy((Action("claim", "CD"), Action("gather"),
                        Action("claim", "FD"), Action("claim", "DE"))),
    )


def play(initial_state, policies, max_turns=200):
    """Keep engine state inside the runner; policies get individual views."""
    if len(policies) != initial_state.rules.player_count:
        raise ValueError("one policy per player is required")
    state, events = initial_state, []
    while not state.finished:
        if len(events) >= max_turns:
            raise RuntimeError("turn guard reached; this is not a scored game")
        player_id = state.current_player
        action = policies[player_id].choose(observe(state, player_id))
        state, event = step(state, player_id, action)
        events.append(event)
    return state, tuple(events)


def render_replay(initial_state, events):
    """Reapply and validate events; reveal tickets only in final scoring."""
    state = initial_state
    cities = {city for r in state.routes for city in (r.source, r.target)}
    groups = {}
    for route in state.routes:
        if route.parallel_group is not None:
            groups.setdefault(route.parallel_group, []).append(route.route_id)
    parallel = "; ".join(
        "/".join(sorted(ids)) for _, ids in sorted(groups.items())
        if len(ids) > 1)
    lines = [
        f"SIMPLIFIED {state.rules.player_count}-PLAYER GAME "
        "(not full Europe rules)",
        f"{len(cities)} cities; {len(state.routes)} physical routes; "
        f"parallel groups: {parallel or 'none'}.",
        f"Replay begins at turn {state.turn}; "
        f"next player P{state.current_player}.",
        "Supplied-state budgets: " + "; ".join(
            f"P{i} trains={p.trains} resources={p.resources}"
            for i, p in enumerate(state.players)),
        "Private tickets are revealed only after the game ends.",
    ]
    for event in events:
        state, reproduced = step(state, event.player_id, event.action)
        if reproduced != event:
            raise ValueError("replay event does not match the rules")
        description = event.action.kind
        if event.action.route_id is not None:
            description += " " + event.action.route_id
        countdown = ("-" if event.final_turns_remaining is None
                     else str(event.final_turns_remaining))
        lines.append(
            f"Turn {event.turn:02d} P{event.player_id}: {description:<10} "
            f"trains={event.trains_after} resources={event.resources_after} "
            f"route_points={event.route_points_after} final_left={countdown}")
    if state.finished:
        lines.append(f"END: {state.end_reason}")
        for score in final_scores(state):
            completed = ", ".join(
                f"{t.source}-{t.target}(+{t.points})" for t in score.completed)
            failed = ", ".join(
                f"{t.source}-{t.target}(-{t.points})" for t in score.failed)
            lines.append(
                f"P{score.player_id}: routes={score.route_points}; "
                f"completed={completed or 'none'}; failed={failed or 'none'}; "
                f"TOTAL={score.total}")
        lines.append("Winner(s): " + ", ".join(
            f"P{i}" for i in winners(state)))
    return "\n".join(lines)


def main():
    initial = demo_setup()
    _, events = play(initial, demo_policies())
    print(render_replay(initial, events))


if __name__ == "__main__":
    main()
