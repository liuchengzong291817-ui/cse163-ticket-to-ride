"""Hand-checkable strategy tests; these do not measure playing strength."""

import unittest
from dataclasses import replace

from .game import (Action, Route, Rules, Ticket, legal_actions, new_game,
                   observe, step)
from .policy_demo import baseline_game, blocking_lesson
from .policies import (PathPlan, RandomLegalPolicy, ReplanningGreedyPolicy,
                       shortest_remaining_path)


class PolicyTests(unittest.TestCase):
    def diamond(self, tickets=None, **settings):
        routes = (Route("AB", "A", "B", 2), Route("BC", "B", "C", 2),
                  Route("AD", "A", "D", 3), Route("DC", "D", "C", 2))
        if tickets is None:
            tickets = (Ticket("goal", "A", "C", 6),)
        return new_game(routes, (tickets, (), (), ()),
                        Rules(initial_trains=8, **settings))

    def test_blocking_changes_path_and_action(self):
        before, after = blocking_lesson()
        self.assertEqual(before.plan, PathPlan(4, ("AB", "BC")))
        self.assertEqual(before.action, Action("claim", "AB"))
        self.assertEqual(after.plan, PathPlan(5, ("AD", "DC")))
        self.assertEqual(after.action, Action("claim", "DC"))
        self.assertNotIn("BC", after.plan.route_ids)

    def test_owned_routes_have_zero_cost_and_cycles_terminate(self):
        routes = (Route("AB", "A", "B", 1), Route("BC", "B", "C", 1),
                  Route("CA", "C", "A", 1), Route("CD", "C", "D", 2))
        state = new_game(routes, ((), (), (), ()))
        # A valid public board fixture containing an owned zero-cost cycle.
        state = replace(state, owners=(0, 0, 0, None))
        plan = shortest_remaining_path(observe(state, 0), "A", "D")
        self.assertEqual(plan.additional_trains, 2)
        self.assertEqual(plan.route_ids[-1], "CD")
        self.assertEqual(len(plan.route_ids), len(set(plan.route_ids)))
        self.assertEqual(shortest_remaining_path(
            observe(state, 0), "A", "C").additional_trains, 0)

    def test_gather_when_goal_path_is_unaffordable(self):
        state = self.diamond(initial_resources=1)
        decision = ReplanningGreedyPolicy().decide(observe(state, 0))
        self.assertEqual(decision.action, Action("gather"))
        self.assertEqual(decision.plan.additional_trains, 4)
        self.assertEqual(decision.ticket_id, "goal")

    def test_ticket_priority_uses_points_over_additional_cost(self):
        tickets = (Ticket("AC", "A", "C", 6), Ticket("AD", "A", "D", 4))
        state = self.diamond(tickets=tickets)
        decision = ReplanningGreedyPolicy().decide(observe(state, 0))
        # 2*6/4 = 3 is greater than 2*4/3.
        self.assertEqual(decision.ticket_id, "AC")
        self.assertEqual(decision.action, Action("claim", "AB"))

    def test_unreachable_and_over_budget_goals_use_legal_fallback(self):
        state = self.diamond()
        # Both routes out of A are occupied by opponents.
        blocked = replace(state, owners=(1, None, 2, None))
        self.assertIsNone(shortest_remaining_path(
            observe(blocked, 0), "A", "C"))
        decision = ReplanningGreedyPolicy().decide(observe(blocked, 0))
        self.assertIsNone(decision.ticket_id)
        self.assertIn(decision.action, legal_actions(observe(blocked, 0)))
        # The shortest path needs 4 trains, but this player has only 3 left.
        players = list(state.players)
        players[0] = replace(players[0], trains=3)
        short = replace(state, players=tuple(players))
        decision = ReplanningGreedyPolicy().decide(observe(short, 0))
        self.assertIsNone(decision.ticket_id)
        self.assertEqual(decision.action, Action("claim", "AD"))

    def test_completed_ticket_is_not_planned_again(self):
        state = self.diamond(tickets=(Ticket("AB", "A", "B", 5),))
        state, _ = step(state, 0, Action("claim", "AB"))
        for player_id in (1, 2, 3):
            state, _ = step(state, player_id, Action("gather"))
        decision = ReplanningGreedyPolicy().decide(observe(state, 0))
        self.assertIsNone(decision.ticket_id)
        self.assertNotEqual(decision.action, Action("claim", "AB"))

    def test_final_turn_prefers_ticket_completion_to_route_score(self):
        routes = (Route("AB", "A", "B", 1), Route("AC", "A", "C", 3),
                  Route("DE", "D", "E", 1))
        state = new_game(routes, ((Ticket("win", "A", "B", 5),), (), (), ()),
                         Rules(initial_trains=3), starting_player=3)
        state, _ = step(state, 3, Action("claim", "DE"))
        policy = ReplanningGreedyPolicy()
        view = observe(state, 0)
        self.assertEqual(view.final_turns_remaining, 4)
        self.assertEqual(policy.choose(view), Action("claim", "AB"))
        self.assertEqual(policy._immediate_gain(view, Action("claim", "AB")),
                         11)  # Route +1; ticket changes from -5 to +5.
        players = list(state.players)
        players[0] = replace(players[0], resources=0)
        empty = observe(replace(state, players=tuple(players)), 0)
        decision = policy.decide(empty)
        self.assertEqual(decision.action, Action("gather"))
        self.assertIsNone(decision.plan)

    def test_parallel_availability_is_shared_with_engine(self):
        routes = (Route("AB1", "A", "B", 2, "AB"),
                  Route("AB2", "A", "B", 2, "AB"))
        for count in (2, 4):
            tickets = ((), (Ticket("goal", "A", "B", 5),))
            tickets += ((),) * (count - 2)
            state = new_game(routes, tickets, Rules(player_count=count))
            state, _ = step(state, 0, Action("claim", "AB1"))
            view = observe(state, 1)
            plan = shortest_remaining_path(view, "A", "B")
            decision = ReplanningGreedyPolicy().decide(view)
            if count == 4:
                self.assertEqual(plan, PathPlan(2, ("AB2",)))
                self.assertEqual(decision.action, Action("claim", "AB2"))
            else:
                self.assertIsNone(plan)
                self.assertEqual(decision.action, Action("pass"))

    def test_hidden_opponent_tickets_cannot_change_decisions(self):
        state = self.diamond()
        players = list(state.players)
        # Keep public ticket counts identical while altering hidden content.
        players[1] = replace(
            players[1], tickets=(Ticket("secret1", "A", "B", 8),))
        first = replace(state, players=tuple(players))
        players[1] = replace(
            players[1], tickets=(Ticket("secret2", "C", "D", 99),))
        second = replace(state, players=tuple(players))
        a, b = observe(first, 0), observe(second, 0)
        self.assertEqual(a, b)
        self.assertEqual(ReplanningGreedyPolicy().decide(a),
                         ReplanningGreedyPolicy().decide(b))
        self.assertEqual(RandomLegalPolicy(163).choose(a),
                         RandomLegalPolicy(163).choose(b))

    def test_deterministic_ties_and_no_observation_mutation(self):
        state = self.diamond()
        view, before = observe(state, 0), repr(state)
        reversed_map = replace(state, routes=tuple(reversed(state.routes)),
                               owners=tuple(reversed(state.owners)))
        policy = ReplanningGreedyPolicy()
        self.assertEqual(policy.decide(view), policy.decide(view))
        self.assertEqual(policy.decide(view),
                         policy.decide(observe(reversed_map, 0)))
        self.assertEqual(repr(state), before)

    def test_seeded_random_is_repeatable_and_does_not_share_rng(self):
        view = observe(self.diamond(), 0)
        first, second = RandomLegalPolicy(163), RandomLegalPolicy(163)
        noise = RandomLegalPolicy(999)
        for _ in range(20):
            noise.choose(view)
            a, b = first.choose(view), second.choose(view)
            self.assertEqual(a, b)
            self.assertIn(a, legal_actions(view))

    def test_full_four_player_game_is_legal_and_repeatable(self):
        initial, final, events, records = baseline_game()
        self.assertEqual((initial, final, events, records), baseline_game())
        self.assertTrue(final.finished)
        self.assertEqual(len(events), len(records))
        self.assertEqual({event.player_id for event in events}, {0, 1, 2, 3})
        state = initial
        for event in events:
            view = observe(state, event.player_id)
            self.assertIn(event.action, legal_actions(view))
            state, replayed = step(state, event.player_id, event.action)
            self.assertEqual(replayed, event)
            for player_id, player in enumerate(state.players):
                used = sum(r.length for r, o in zip(state.routes, state.owners)
                           if o == player_id)
                self.assertEqual(player.trains + used, 8)
                self.assertGreaterEqual(player.resources, 0)
                self.assertLessEqual(player.resources, 8)
        self.assertEqual(state, final)

    def test_policies_reject_out_of_turn_and_finished_views(self):
        state = self.diamond()
        views = (observe(state, 1), observe(replace(state, finished=True), 0))
        for view in views:
            for policy in (RandomLegalPolicy(163), ReplanningGreedyPolicy()):
                with self.assertRaises(ValueError):
                    policy.choose(view)


if __name__ == "__main__":
    unittest.main()
