"""Regression checks from the research-design and correctness review."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from .bottleneck import BottleneckPriorityPolicy
from .demo import demo_policies, demo_setup, play, render_replay
from .evaluate import _run_game, make_scenarios, paired_record
from .game import Action, Route, Rules, Ticket, new_game, observe, step
from .policies import (RandomLegalPolicy, ReplanningGreedyPolicy,
                       StaticPlanningPolicy)


class ReviewTests(unittest.TestCase):
    def test_invalid_focal_seats_fail_before_a_mislabeled_game_can_run(self):
        scenario = make_scenarios((101,))[0]
        with patch("multiplayer.evaluate.play") as runner:
            for seat in (-1, 4, True, False, 0.0, "0"):
                with self.subTest(seat=seat):
                    with self.assertRaises(ValueError):
                        _run_game(scenario, 0, 163, seat, "bottleneck")
            runner.assert_not_called()

    def test_paired_rows_reject_wrong_scenario_replicate_or_player(self):
        scenario = make_scenarios((101,))[0]
        _, baseline, _ = _run_game(scenario, 0, 163, 0, "replan")
        _, candidate, _ = _run_game(scenario, 0, 163, 0, "bottleneck")
        pair = paired_record(scenario, 0, 0, baseline[0], candidate[0])
        self.assertEqual(pair["delta_score"],
                         candidate[0]["score"] - baseline[0]["score"])
        changes = ({"seat": 1}, {"scenario_id": "another"}, {"replicate": 1})
        for change in changes:
            for side in (0, 1):
                rows = [baseline[0], candidate[0]]
                rows[side] = {**rows[side], **change}
                with self.assertRaises(ValueError):
                    paired_record(scenario, 0, 0, *rows)
        for seat in (-1, 4, True, 0.0):
            with self.assertRaises(ValueError):
                paired_record(scenario, 0, seat, baseline[0], candidate[0])

    def test_replay_metadata_describes_the_actual_supplied_setup(self):
        initial = new_game(
            (Route("XY", "X", "Y", 1),), ((), ()),
            Rules(player_count=2, initial_trains=3, initial_resources=0))
        text = render_replay(initial, ())
        self.assertIn("SIMPLIFIED 2-PLAYER GAME", text)
        self.assertIn("2 cities; 1 physical routes; parallel groups: none",
                      text)
        self.assertIn("P0 trains=3 resources=0", text)
        self.assertNotIn("12 routes", text)
        self.assertNotIn("AB1/AB2", text)

    def test_midgame_replay_has_real_budgets_and_no_hidden_tickets(self):
        state, _ = step(demo_setup(), 0, Action("claim", "AB1"))
        text = render_replay(state, ())
        self.assertIn("6 cities; 12 physical routes", text)
        self.assertIn("AB1/AB2", text)
        self.assertIn("Replay begins at turn 1; next player P1", text)
        self.assertIn("P0 trains=6 resources=2", text)
        self.assertIn("P1 trains=8 resources=4", text)
        self.assertNotIn("P0-AF", text)
        self.assertNotIn("A-D(-8)", text)

    def test_same_public_history_keeps_decisions_secret_independent(self):
        initial = demo_setup()
        players = list(initial.players)
        players[1] = replace(players[1],
                             tickets=(Ticket("HIDDEN", "A", "D", 99),))
        hidden_changed = replace(initial, players=tuple(players))
        _, events = play(initial, demo_policies())
        factories = (lambda: RandomLegalPolicy(163), ReplanningGreedyPolicy,
                     StaticPlanningPolicy, BottleneckPriorityPolicy)
        for factory in factories:
            a, b = factory(), factory()
            first, second = initial, hidden_changed
            for event in events:
                if event.player_id == 0:
                    x, y = observe(first, 0), observe(second, 0)
                    self.assertEqual(x, y)
                    self.assertNotIn("HIDDEN", repr(x))
                    self.assertEqual(a.choose(x), b.choose(y))
                first, public_a = step(first, event.player_id, event.action)
                second, public_b = step(second, event.player_id, event.action)
                self.assertEqual(public_a, public_b)

    def test_later_players_crossing_threshold_do_not_restart_final_round(self):
        routes = (Route("AB", "A", "B", 1), Route("BC", "B", "C", 1),
                  Route("CD", "C", "D", 1), Route("DE", "D", "E", 1),
                  Route("EF", "E", "F", 1))
        state = new_game(routes, ((), (), (), ()), Rules(initial_trains=3))
        for seat, route_id, countdown in ((0, "AB", 4), (1, "BC", 3),
                                          (2, "CD", 2), (3, "DE", 1),
                                          (0, "EF", 0)):
            state, _ = step(state, seat, Action("claim", route_id))
            self.assertEqual(state.final_turns_remaining, countdown)
            self.assertEqual(state.finished, countdown == 0)
        self.assertEqual(state.end_reason, "final_round")
        self.assertEqual(state.turn, 5)


if __name__ == "__main__":
    unittest.main()
