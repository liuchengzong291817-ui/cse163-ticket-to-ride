"""Constructed mechanism lessons; never included as performance samples."""

from .bottleneck import BottleneckPriorityPolicy, edge_impact
from .demo import play
from .game import (Action, Route, Rules, Ticket, final_scores, legal_actions,
                   new_game, observe, step)
from .planning_demo import lesson_setup, planning_lesson
from .policies import (ReplanningGreedyPolicy, StaticPlanningPolicy,
                       shortest_remaining_path)


class GatherOrFirstLegal:
    """Public-observation teaching actor; not a competitive benchmark."""

    def choose(self, observation):
        actions = legal_actions(observation)
        if Action("gather") in actions:
            return Action("gather")
        return actions[0]


class RaceOpponent(GatherOrFirstLegal):
    """First gather, then prefer BC over BD using only public availability."""

    def choose(self, observation):
        if observation.turn >= 4:
            for route_id in ("BC", "BD"):
                action = Action("claim", route_id)
                if action in legal_actions(observation):
                    return action
        return super().choose(observation)


def planning_race(policy_type):
    initial = lesson_setup()
    policies = (policy_type(), RaceOpponent(),
                GatherOrFirstLegal(), GatherOrFirstLegal())
    final, events = play(initial, policies)
    return initial, final, events


def bottleneck_setup(**settings):
    routes = (Route("AB", "A", "B", 1), Route("BC", "B", "C", 3),
              Route("AD", "A", "D", 1), Route("DB", "D", "B", 1))
    return new_game(routes, ((Ticket("AC", "A", "C", 6),), (), (), ()),
                    Rules(initial_trains=8, **settings))


def bottleneck_lesson():
    initial = bottleneck_setup()
    rows = []
    for name, policy in (("dynamic", ReplanningGreedyPolicy()),
                         ("bottleneck", BottleneckPriorityPolicy())):
        state = initial
        first = policy.decide(observe(state, 0))
        state, _ = step(state, 0, first.action)
        choices = legal_actions(observe(state, 1))
        reply = (Action("claim", "BC") if Action("claim", "BC") in choices
                 else Action("claim", "AB"))
        state, _ = step(state, 1, reply)
        for seat in (2, 3):
            state, _ = step(state, seat, Action("gather"))
        remaining = shortest_remaining_path(observe(state, 0), "A", "C")
        rows.append((name, first, reply, remaining))
    impacts = tuple(edge_impact(observe(initial, 0), route_id)
                    for route_id in ("AB", "BC"))
    return initial, impacts, tuple(rows)


def main():
    print("CONSTRUCTED PRIVATE LESSONS: not evaluation scenarios.")
    before, after, _ = planning_lesson()
    print("\nStatic vs dynamic: same first AB, different next goals.")
    for name, decision in zip(("static", "dynamic"), after):
        print(f"{name}: ticket={decision.ticket_id}, "
              f"next_claim={decision.action.route_id}")
    for name, policy in (("static", StaticPlanningPolicy),
                         ("dynamic", ReplanningGreedyPolicy)):
        _, final, _ = planning_race(policy)
        print(f"{name} versus scripted public opponent: "
              f"scores={[s.total for s in final_scores(final)]}; "
              f"end={final.end_reason}; turns={final.turn}")
    print("\nBottleneck lesson: AC=6, costs AB=1, BC=3, AD=1, DB=1.")
    _, impacts, rows = bottleneck_lesson()
    for impact in impacts:
        detail = impact.tickets[0]
        print(f"Delete {impact.route_id}: {detail.before_cost} -> "
              f"{detail.after_cost}; status={detail.status}; "
              f"lost_swing={impact.lost_ticket_swing}; "
              f"weighted_detour={impact.weighted_detour}")
    for name, first, reply, remaining in rows:
        cost = None if remaining is None else remaining.additional_trains
        print(f"{name}: first={first.action.route_id}, "
              f"opponent={reply.route_id}, own AC remaining_cost={cost}")


if __name__ == "__main__":
    main()
