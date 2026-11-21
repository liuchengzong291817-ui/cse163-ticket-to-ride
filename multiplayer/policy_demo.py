"""A blocking lesson and one four-player game; no performance study."""

from .demo import demo_setup, play, render_replay
from .game import Action, Route, Rules, Ticket, new_game, observe, step
from .policies import RandomLegalPolicy, ReplanningGreedyPolicy


def blocking_lesson():
    """Show P0's private decisions before and after P1 claims the needed BC."""
    routes = (Route("AB", "A", "B", 2), Route("BC", "B", "C", 2),
              Route("AD", "A", "D", 3), Route("DC", "D", "C", 2))
    state = new_game(
        routes, ((Ticket("P0-AC", "A", "C", 6),), (), (), ()),
        Rules(initial_trains=8))
    policy = ReplanningGreedyPolicy()
    before = policy.decide(observe(state, 0))
    state, _ = step(state, 0, before.action)
    for player_id, action in ((1, Action("claim", "BC")),
                              (2, Action("gather")), (3, Action("gather"))):
        state, _ = step(state, player_id, action)
    after = policy.decide(observe(state, 0))
    return before, after


class RecordedPolicy:
    """Store private explanations for display only after the game."""

    def __init__(self, policy, records):
        self.policy = policy
        self.records = records

    def choose(self, observation):
        decision = self.policy.decide(observation)
        self.records.append((observation.turn + 1, observation.player_id,
                             decision))
        return decision.action


def baseline_game():
    """Fixed teaching map and seats, with independent random-player seeds."""
    initial, records = demo_setup(), []
    policies = (ReplanningGreedyPolicy(), RandomLegalPolicy(164),
                ReplanningGreedyPolicy(), RandomLegalPolicy(166))
    wrapped = tuple(RecordedPolicy(policy, records) for policy in policies)
    final, events = play(initial, wrapped)
    return initial, final, events, tuple(records)


def main():
    before, after = blocking_lesson()
    print("PRIVATE P0 BLOCKING LESSON (four-player toy map)")
    snapshots = (("Before", before), ("After BC was claimed", after))
    for label, decision in snapshots:
        print(f"{label}: path={decision.plan.route_ids}; "
              f"new trains={decision.plan.additional_trains}; "
              f"action={decision.action.kind} {decision.action.route_id}")
    print("\nONE BASELINE GAME: P0/P2 replan; P1/P3 seeded random.")
    print("Asymmetric invented tickets, fixed seats: NOT win-rate evidence.")
    initial, final, events, records = baseline_game()
    print(render_replay(initial, events))
    if final.finished:
        print("\nPOST-GAME PRIVATE EXPLANATIONS (now safe to reveal):")
        for turn, player_id, decision in records:
            detail = ""
            if decision.plan is not None:
                detail = (f" ticket={decision.ticket_id}; "
                          f"path={decision.plan.route_ids}; "
                          f"new_trains={decision.plan.additional_trains};")
            print(f"Turn {turn:02d} P{player_id}:{detail} {decision.reason}")


if __name__ == "__main__":
    main()
