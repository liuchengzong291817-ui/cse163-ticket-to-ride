"""Exercise plan retention and repair on hand-checkable legal histories."""

import unittest
from dataclasses import replace

from .demo import demo_setup, play
from .game import (Action, Route, Rules, Ticket, legal_actions, new_game,
                   observe, step)
from .planning_demo import lesson_setup, planning_lesson
from .policies import (PathPlan, ReplanningGreedyPolicy, StaticPlanningPolicy)


def finish_other_turns(state):
    for seat in (1, 2, 3):
        state, _ = step(state, seat, Action("gather"))
    return state


class StaticPlanningTests(unittest.TestCase):
    def test_first_plan_matches_then_retains_despite_priority_change(self):
        before, after, event = planning_lesson()
        for decision in before:
            self.assertEqual(decision.ticket_id, "AD")
            self.assertEqual(decision.plan, PathPlan(4, ("AB", "BD")))
            self.assertEqual(decision.action, Action("claim", "AB"))
        self.assertEqual(event, "retained")
        self.assertEqual(after[0].ticket_id, "AD")
        self.assertEqual(after[0].plan, PathPlan(3, ("AB", "BD")))
        self.assertEqual(after[0].action, Action("claim", "BD"))
        self.assertEqual(after[1].ticket_id, "AC")
        self.assertEqual(after[1].action, Action("claim", "BC"))

    def test_blocked_plan_repairs_with_the_same_ranking_and_action(self):
        before, after, event = planning_lesson(blocking=True)
        self.assertEqual(before[0].plan, PathPlan(4, ("AB", "BC")))
        self.assertEqual(event, "blocked")
        for decision in after:
            self.assertEqual(decision.plan, PathPlan(5, ("AD", "DC")))
            self.assertEqual(decision.action, Action("claim", "DC"))
            self.assertNotIn("BC", decision.plan.route_ids)

    def test_resource_shortage_gathers_without_replacing_a_valid_goal(self):
        state = lesson_setup()
        state = new_game(state.routes, (state.players[0].tickets, (), (), ()),
                         Rules(initial_trains=8, initial_resources=1))
        policy = StaticPlanningPolicy()
        first = policy.decide(observe(state, 0))
        state, _ = step(state, 0, first.action)
        state = finish_other_turns(state)
        before = repr(state)
        second = policy.decide(observe(state, 0))
        self.assertEqual(first.action, Action("claim", "AB"))
        self.assertEqual(second.action, Action("gather"))
        self.assertEqual(second.ticket_id, "AD")
        self.assertEqual(second.plan.route_ids, first.plan.route_ids)
        self.assertEqual(policy.last_plan_event, "retained")
        self.assertEqual(repr(state), before)

    def test_completed_goal_is_replaced_on_next_turn(self):
        state, policy = lesson_setup(), StaticPlanningPolicy()
        for expected in ("AB", "BD"):
            decision = policy.decide(observe(state, 0))
            self.assertEqual(decision.action, Action("claim", expected))
            state, _ = step(state, 0, decision.action)
            state = finish_other_turns(state)
        decision = policy.decide(observe(state, 0))
        self.assertEqual(policy.last_plan_event, "completed")
        self.assertEqual(decision.ticket_id, "AC")
        self.assertEqual(decision.plan.additional_trains, 1)
        self.assertEqual(decision.action, Action("gather"))

    def test_blocking_repair_may_choose_a_different_ticket(self):
        original = lesson_setup(blocking=True)
        tickets = original.players[0].tickets + (Ticket("AD", "A", "D", 4),)
        state = new_game(original.routes, (tickets, (), (), ()),
                         original.rules)
        static = StaticPlanningPolicy()
        first = static.decide(observe(state, 0))
        self.assertEqual(first.ticket_id, "AC")
        state, _ = step(state, 0, first.action)
        state, _ = step(state, 1, Action("claim", "BC"))
        for seat in (2, 3):
            state, _ = step(state, seat, Action("gather"))
        view = observe(state, 0)
        repaired = static.decide(view)
        dynamic = ReplanningGreedyPolicy().decide(view)
        self.assertEqual(static.last_plan_event, "blocked")
        self.assertEqual(repaired.ticket_id, "AD")
        self.assertEqual((repaired.plan, repaired.action),
                         (dynamic.plan, dynamic.action))

    def test_train_budget_failure_uses_shared_legal_fallback(self):
        original = lesson_setup(blocking=True)
        state = new_game(original.routes + (Route("EF", "E", "F", 6),),
                         (original.players[0].tickets, (), (), ()),
                         Rules(initial_trains=9, initial_resources=6))
        policy = StaticPlanningPolicy()
        first = policy.decide(observe(state, 0))
        self.assertEqual(first.plan.additional_trains, 4)
        # Deliberately deviate from its recommendation by a legal action.
        # This defensive repair is not normally needed when following a plan.
        state, _ = step(state, 0, Action("claim", "EF"))
        state = finish_other_turns(state)
        view = observe(state, 0)
        decision = policy.decide(view)
        self.assertEqual(policy.last_plan_event, "over_budget")
        self.assertIsNone(decision.ticket_id)
        self.assertEqual(decision.action,
                         ReplanningGreedyPolicy().choose(view))
        self.assertIn(decision.action, legal_actions(view))

    def test_parallel_closure_can_force_repair(self):
        routes = (Route("AB1", "A", "B", 3, "AB"),
                  Route("AB2", "A", "B", 3, "AB"),
                  Route("BC", "B", "C", 2),
                  Route("AC", "A", "C", 6))
        for count in (2, 4):
            state = new_game(
                routes, ((Ticket("goal", "A", "C", 9),),)
                + ((),) * (count - 1),
                Rules(player_count=count, initial_resources=1))
            static = StaticPlanningPolicy()
            first = static.decide(observe(state, 0))
            self.assertEqual(first.plan.route_ids, ("AB1", "BC"))
            state, _ = step(state, 0, first.action)
            for seat in range(1, count):
                state, _ = step(state, seat, Action("gather"))
            # Still following the original plan, P0 claims BC first.
            second = static.decide(observe(state, 0))
            state, _ = step(state, 0, second.action)
            # P1 claims the other parallel edge, closing AB1 only in 2P.
            state, _ = step(state, 1, Action("claim", "AB2"))
            for seat in range(2, count):
                state, _ = step(state, seat, Action("gather"))
            decision = static.decide(observe(state, 0))
            if count == 2:
                self.assertEqual(static.last_plan_event, "blocked")
                self.assertEqual(decision.plan.route_ids, ("AC",))
            else:
                self.assertEqual(static.last_plan_event, "retained")
                self.assertEqual(decision.plan.route_ids, ("AB1", "BC"))

    def test_final_turn_override_is_identical_and_ignores_cached_goal(self):
        initial = lesson_setup()
        state = new_game(initial.routes + (Route("EF", "E", "F", 6),),
                         (initial.players[0].tickets, (), (), ()),
                         Rules(initial_trains=8, initial_resources=6))
        static = StaticPlanningPolicy()
        first = static.decide(observe(state, 0))
        state, _ = step(state, 0, first.action)
        state, _ = step(state, 1, Action("claim", "EF"))
        for seat in (2, 3):
            state, _ = step(state, seat, Action("gather"))
        view = observe(state, 0)
        self.assertIsNotNone(view.final_turns_remaining)
        self.assertEqual(static.decide(view),
                         ReplanningGreedyPolicy().decide(view))
        self.assertEqual(static.last_plan_event, "final_turn")
        self.assertEqual(static.choose(view), Action("claim", "BD"))

    def test_hidden_tickets_do_not_affect_stateful_decisions(self):
        state = lesson_setup()
        players = list(state.players)
        players[1] = replace(
            players[1], tickets=(Ticket("secret", "A", "B", 2),))
        first = replace(state, players=tuple(players))
        players[1] = replace(
            players[1], tickets=(Ticket("other", "C", "D", 99),))
        second = replace(state, players=tuple(players))
        a, b = StaticPlanningPolicy(), StaticPlanningPolicy()
        self.assertEqual(a.decide(observe(first, 0)),
                         b.decide(observe(second, 0)))
        for _ in range(2):
            self.assertEqual(a.decide(observe(first, 0)),
                             b.decide(observe(second, 0)))

    def test_full_static_game_is_repeatable_legal_with_separate_memory(self):
        initial = demo_setup()
        first = play(initial, tuple(StaticPlanningPolicy() for _ in range(4)))
        second = play(initial, tuple(StaticPlanningPolicy() for _ in range(4)))
        self.assertEqual(first, second)
        final, events = first
        state = initial
        for event in events:
            self.assertIn(event.action, legal_actions(observe(
                state, event.player_id)))
            state, checked = step(state, event.player_id, event.action)
            self.assertEqual(checked, event)
        self.assertEqual(state, final)
        self.assertTrue(final.finished)

    def test_invalid_turns_are_rejected_before_changing_memory(self):
        state, policy = lesson_setup(), StaticPlanningPolicy()
        for view in (observe(state, 1),
                     observe(replace(state, finished=True), 0)):
            with self.assertRaises(ValueError):
                policy.choose(view)
            self.assertEqual(policy.last_plan_event, "not_started")


if __name__ == "__main__":
    unittest.main()
