"""Hand calculations and invariants for the edge-deletion heuristic."""

import unittest
from dataclasses import replace

from .bottleneck import BottleneckPriorityPolicy, edge_impact
from .bottleneck_demo import (bottleneck_lesson, bottleneck_setup,
                              planning_race)
from .game import (Action, Route, Rules, Ticket, final_scores, legal_actions,
                   new_game, observe, step)
from .policies import (ReplanningGreedyPolicy, StaticPlanningPolicy,
                       shortest_remaining_path)


class BottleneckTests(unittest.TestCase):
    def test_constructed_static_dynamic_race_has_different_final_scores(self):
        for policy, scores in ((StaticPlanningPolicy, [10, 1, 0, 0]),
                               (ReplanningGreedyPolicy, [-3, 4, 0, 0])):
            initial, final, events = planning_race(policy)
            state = initial
            for event in events:
                state, checked = step(state, event.player_id, event.action)
                self.assertEqual(checked, event)
            self.assertEqual(state, final)
            self.assertEqual(final.end_reason, "all_players_stuck")
            self.assertEqual([s.total for s in final_scores(final)], scores)

    def test_bridge_and_finite_detour_match_hand_calculation(self):
        _, impacts, rows = bottleneck_lesson()
        ab, bc = impacts
        self.assertEqual((ab.lost_ticket_swing, ab.weighted_detour), (0, 12))
        self.assertEqual((ab.tickets[0].before_cost,
                          ab.tickets[0].after_cost), (4, 5))
        self.assertEqual((bc.lost_ticket_swing, bc.weighted_detour), (12, 0))
        self.assertIsNone(bc.tickets[0].after_cost)
        self.assertEqual(bc.tickets[0].status, "unreachable")
        self.assertEqual(rows[0][1].action, Action("claim", "AB"))
        self.assertEqual(rows[1][1].action, Action("claim", "BC"))
        self.assertEqual(rows[0][1].ticket_id, rows[1][1].ticket_id)
        self.assertEqual(rows[0][1].plan, rows[1][1].plan)
        self.assertIsNone(rows[0][3])
        self.assertEqual(rows[1][3].additional_trains, 2)

    def test_finite_detour_breaks_tie_without_disconnection(self):
        routes = (Route("AB", "A", "B", 1), Route("BC", "B", "C", 2),
                  Route("AD", "A", "D", 1), Route("DB", "D", "B", 1),
                  Route("BE", "B", "E", 2), Route("EC", "E", "C", 2))
        state = new_game(routes, ((Ticket("AC", "A", "C", 5),), (), (), ()),
                         Rules(initial_trains=8))
        view = observe(state, 0)
        ab, bc = edge_impact(view, "AB"), edge_impact(view, "BC")
        self.assertEqual((ab.lost_ticket_swing, bc.lost_ticket_swing), (0, 0))
        self.assertEqual((ab.weighted_detour, bc.weighted_detour), (10, 20))
        self.assertEqual(BottleneckPriorityPolicy().choose(view),
                         Action("claim", "BC"))

    def test_budget_failure_is_distinct_from_unreachable(self):
        original = bottleneck_setup()
        state = new_game(original.routes,
                         ((Ticket("AC", "A", "C", 6),), (), (), ()),
                         Rules(initial_trains=4))
        view = observe(state, 0)
        ab, bc = edge_impact(view, "AB"), edge_impact(view, "BC")
        self.assertEqual(ab.tickets[0].status, "over_budget")
        self.assertEqual(ab.tickets[0].after_cost, 5)
        self.assertEqual(bc.tickets[0].status, "unreachable")
        self.assertEqual((ab.lost_ticket_swing, bc.lost_ticket_swing),
                         (12, 12))
        self.assertEqual((ab.weighted_detour, bc.weighted_detour), (0, 0))
        # Equal impact returns to the original shortest-edge tiebreak.
        self.assertEqual(BottleneckPriorityPolicy().choose(view),
                         ReplanningGreedyPolicy().choose(view))

    def test_multiple_own_tickets_aggregate_once_each(self):
        original = bottleneck_setup()
        tickets = (Ticket("AC", "A", "C", 6), Ticket("BC", "B", "C", 4))
        state = new_game(original.routes, (tickets, (), (), ()),
                         original.rules)
        impact = edge_impact(observe(state, 0), "BC")
        self.assertEqual(impact.lost_ticket_swing, 20)
        self.assertEqual([t.ticket_id for t in impact.tickets], ["AC", "BC"])
        self.assertEqual(impact.weighted_detour, 0)
        self.assertEqual(edge_impact(observe(state, 0), "AB").weighted_detour,
                         12)

    def test_completed_and_initially_infeasible_tickets_are_excluded(self):
        original = bottleneck_setup()
        tickets = (Ticket("AC", "A", "C", 6), Ticket("AD", "A", "D", 5))
        state = new_game(original.routes, (tickets, (), (), ()),
                         original.rules)
        state, _ = step(state, 0, Action("claim", "AD"))
        impact = edge_impact(observe(state, 0), "BC")
        self.assertEqual([t.ticket_id for t in impact.tickets], ["AC"])
        self.assertEqual(edge_impact(observe(state, 0), "AB").weighted_detour,
                         0)  # Owned AD has zero additional train cost.
        # A fresh three-train player cannot finish AC (cost four) at all.
        short = new_game(original.routes, (tickets, (), (), ()),
                         Rules(initial_trains=3))
        impact = edge_impact(observe(short, 0), "AB")
        self.assertEqual([t.ticket_id for t in impact.tickets], ["AD"])
        self.assertEqual(impact.lost_ticket_swing, 0)
        blocked, _ = step(short, 0, Action("gather"))
        blocked, _ = step(blocked, 1, Action("claim", "BC"))
        impact = edge_impact(observe(blocked, 0), "AB")
        self.assertNotIn("AC", [t.ticket_id for t in impact.tickets])

    def test_parallel_physical_edges_are_not_merged_or_all_deleted(self):
        routes = (Route("AB1", "A", "B", 1, "AB"),
                  Route("AB2", "A", "B", 1, "AB"),
                  Route("BC", "B", "C", 2))
        state = new_game(routes, ((Ticket("AC", "A", "C", 6),), (), (), ()))
        view = observe(state, 0)
        before = repr(view)
        impact = edge_impact(view, "AB1")
        self.assertEqual(impact.tickets[0].after_cost, 3)
        self.assertEqual((impact.lost_ticket_swing, impact.weighted_detour),
                         (0, 0))
        self.assertEqual(shortest_remaining_path(
            view, "A", "C", excluded_route_ids=("AB1",)).route_ids,
            ("AB2", "BC"))
        self.assertEqual(repr(view), before)

    def test_unavailable_edges_rejected_and_view_is_never_mutated(self):
        state = bottleneck_setup()
        view, before = observe(state, 0), repr(state)
        with self.assertRaises(ValueError):
            edge_impact(view, "unknown")
        edge_impact(view, "BC")
        self.assertEqual(repr(state), before)
        state, _ = step(state, 0, Action("claim", "AB"))
        for seat in (0, 1):
            with self.assertRaises(ValueError):
                edge_impact(observe(state, seat), "AB")

    def test_affordability_and_gather_match_dynamic_rules(self):
        for resources, expected in ((0, Action("gather")),
                                    (1, Action("claim", "AB"))):
            view = observe(bottleneck_setup(initial_resources=resources), 0)
            a, b = (p.decide(view) for p in
                    (ReplanningGreedyPolicy(), BottleneckPriorityPolicy()))
            self.assertEqual((a.ticket_id, a.plan), (b.ticket_id, b.plan))
            self.assertEqual(a.action, b.action)
            self.assertEqual(b.action, expected)
            self.assertIn(b.action, legal_actions(view))

    def test_final_turn_and_no_goal_fallback_are_inherited_unchanged(self):
        original = bottleneck_setup()
        no_goal = new_game(original.routes, ((), (), (), ()), original.rules)
        final = replace(original, final_turns_remaining=4)
        for state in (no_goal, final):
            view = observe(state, 0)
            self.assertEqual(BottleneckPriorityPolicy().decide(view),
                             ReplanningGreedyPolicy().decide(view))

    def test_higher_impact_off_goal_edge_is_not_added_to_candidates(self):
        routes = (Route("AB", "A", "B", 1), Route("BC", "B", "C", 2),
                  Route("XY", "X", "Y", 8))
        tickets = (Ticket("AC", "A", "C", 3), Ticket("XY", "X", "Y", 6))
        state = new_game(routes, (tickets, (), (), ()),
                         Rules(initial_trains=10, initial_resources=8))
        view = observe(state, 0)
        self.assertGreater(edge_impact(view, "XY").lost_ticket_swing,
                           edge_impact(view, "AB").lost_ticket_swing)
        a, b = (p.decide(view) for p in
                (ReplanningGreedyPolicy(), BottleneckPriorityPolicy()))
        self.assertEqual((a.ticket_id, a.plan), (b.ticket_id, b.plan))
        self.assertEqual(b.ticket_id, "AC")
        self.assertEqual(b.action, Action("claim", "AB"))

    def test_opponent_secrets_and_map_order_cannot_change_results(self):
        state = bottleneck_setup()
        players = list(state.players)
        players[1] = replace(players[1], tickets=(Ticket("one", "A", "B", 2),))
        first = replace(state, players=tuple(players))
        players[1] = replace(
            players[1], tickets=(Ticket("two", "B", "C", 99),))
        second = replace(state, players=tuple(players))
        a, b = observe(first, 0), observe(second, 0)
        self.assertEqual(a, b)
        self.assertEqual(edge_impact(a, "BC"), edge_impact(b, "BC"))
        reversed_map = replace(a, routes=tuple(reversed(a.routes)),
                               owners=tuple(reversed(a.owners)))
        policy = BottleneckPriorityPolicy()
        self.assertEqual(policy.decide(a), policy.decide(b))
        self.assertEqual(policy.decide(a), policy.decide(reversed_map))

    def test_out_of_turn_and_finished_decisions_are_rejected(self):
        state = bottleneck_setup()
        for view in (observe(state, 1),
                     observe(replace(state, finished=True), 0)):
            with self.assertRaises(ValueError):
                BottleneckPriorityPolicy().decide(view)


if __name__ == "__main__":
    unittest.main()
