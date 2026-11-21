"""Hand-checkable private planning lessons, separate from evaluation maps."""

from .game import Action, Route, Rules, Ticket, new_game, observe, step
from .policies import ReplanningGreedyPolicy, StaticPlanningPolicy


def lesson_setup(blocking=False):
    if blocking:
        routes = (Route("AB", "A", "B", 2), Route("BC", "B", "C", 2),
                  Route("AD", "A", "D", 3), Route("DC", "D", "C", 2))
        tickets = (Ticket("AC", "A", "C", 6),)
    else:
        routes = (Route("AB", "A", "B", 1), Route("BD", "B", "D", 3),
                  Route("BC", "B", "C", 1))
        tickets = (Ticket("AD", "A", "D", 8), Ticket("AC", "A", "C", 3))
    return new_game(routes, (tickets, (), (), ()), Rules(initial_trains=8))


def planning_lesson(blocking=False):
    """Both policies see the same legal history through their second turn."""
    state = lesson_setup(blocking)
    static, dynamic = StaticPlanningPolicy(), ReplanningGreedyPolicy()
    before = (static.decide(observe(state, 0)),
              dynamic.decide(observe(state, 0)))
    if before[0].action != before[1].action:
        raise RuntimeError("lesson requires the same first action")
    state, _ = step(state, 0, before[0].action)
    for seat in (1, 2, 3):
        action = (Action("claim", "BC") if blocking and seat == 1
                  else Action("gather"))
        state, _ = step(state, seat, action)
    after = (static.decide(observe(state, 0)),
             dynamic.decide(observe(state, 0)))
    return before, after, static.last_plan_event


def main():
    print("PRIVATE P0 LESSONS; four-player synthetic maps, not performance.")
    for blocking, label in ((False, "No blocking: priorities change"),
                            (True, "Opponent blocks the planned BC")):
        before, after, event = planning_lesson(blocking)
        print(f"\n{label}; static event={event}")
        for stage, decisions in (("First turn", before),
                                 ("Next turn", after)):
            for policy, decision in zip(("static", "dynamic"), decisions):
                print(f"{stage} {policy}: ticket={decision.ticket_id}; "
                      f"path={decision.plan.route_ids}; "
                      f"new_trains={decision.plan.additional_trains}; "
                      f"action={decision.action.kind} "
                      f"{decision.action.route_id}")


if __name__ == "__main__":
    main()
