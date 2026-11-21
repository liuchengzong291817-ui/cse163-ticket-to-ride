"""Standard-library tests of the four-player slice and supported variants."""

import unittest
from dataclasses import FrozenInstanceError, replace

from .demo import demo_policies, demo_setup, play, render_replay
from .game import (Action, IllegalAction, Observation, Route, Rules, Ticket,
                   final_scores, legal_actions, new_game, observe, step,
                   winners)


class GameTests(unittest.TestCase):
    def basic(self, count=4, **settings):
        routes = (Route("AB", "A", "B", 1), Route("BC", "B", "C", 2),
                  Route("AC", "A", "C", 3))
        tickets = tuple((Ticket(f"T{i}", "A", "C", 5),)
                        for i in range(count))
        return new_game(routes, tickets, Rules(player_count=count, **settings))

    def test_four_player_rotation_and_immutable_input(self):
        initial = self.basic()
        state = initial
        for player_id in (0, 1, 2, 3):
            state, _ = step(state, player_id, Action("gather"))
        self.assertEqual(state.current_player, 0)
        self.assertEqual(state.turn, 4)
        self.assertEqual(initial.turn, 0)
        self.assertTrue(all(p.resources == 4 for p in initial.players))

    def test_resource_cap_and_claim_accounting(self):
        state = self.basic(initial_resources=7)
        state, _ = step(state, 0, Action("gather"))
        self.assertEqual(state.players[0].resources, 8)
        state, _ = step(state, 1, Action("claim", "AC"))
        self.assertEqual((state.players[1].resources, state.players[1].trains,
                          state.players[1].route_points), (4, 42, 4))
        self.assertEqual(state.owners[2], 1)

    def test_illegal_actions_do_not_mutate(self):
        state = self.basic(initial_resources=0)
        snapshot = repr(state)
        actions = [Action("claim", "AB"), Action("claim", "missing"),
                   Action("pass"), Action("unknown"), Action("gather", "AB")]
        for action in actions:
            with self.assertRaises(IllegalAction):
                step(state, 0, action)
        with self.assertRaises(IllegalAction):
            step(state, 1, Action("gather"))
        self.assertEqual(repr(state), snapshot)

    def test_cannot_claim_twice_or_exceed_train_budget(self):
        state = self.basic(initial_trains=3)
        state, _ = step(state, 0, Action("claim", "AB"))
        with self.assertRaises(IllegalAction):
            step(state, 1, Action("claim", "AB"))
        for player_id in (1, 2, 3):
            state, _ = step(state, player_id, Action("gather"))
        self.assertNotIn(Action("claim", "AC"),
                         legal_actions(observe(state, 0)))
        with self.assertRaises(IllegalAction):
            step(state, 0, Action("claim", "AC"))

    def test_opponent_private_tickets_do_not_affect_observation(self):
        state = self.basic()
        players = list(state.players)
        players[1] = replace(
            players[1], tickets=(Ticket("SECRET", "A", "B", 99),))
        changed = replace(state, players=tuple(players))
        for player_id in (0, 2, 3):
            self.assertEqual(observe(state, player_id),
                             observe(changed, player_id))
        self.assertNotEqual(observe(state, 1), observe(changed, 1))
        view = observe(changed, 0)
        self.assertNotIn("SECRET", repr(view))
        self.assertFalse(hasattr(view.players[1], "tickets"))
        with self.assertRaises(FrozenInstanceError):
            view.players[1].resources = 999
        with self.assertRaises(ValueError):
            final_scores(state)
        _, event = step(changed, 0, Action("claim", "AB"))
        self.assertNotIn("SECRET", repr(event))

    def test_policy_receives_observation_only(self):
        seen = []

        class Policy:
            def choose(self, observation):
                seen.append(observation)
                assert isinstance(observation, Observation)
                return legal_actions(observation)[0]

        final, _ = play(self.basic(), tuple(Policy() for _ in range(4)))
        self.assertTrue(final.finished)
        self.assertTrue(seen)
        for view in seen:
            self.assertEqual(view.own_tickets[0].ticket_id,
                             f"T{view.player_id}")

    def test_four_player_parallel_routes_have_distinct_owners(self):
        state = demo_setup()
        state, _ = step(state, 0, Action("claim", "AB1"))
        view = observe(replace(state, current_player=0), 0)
        self.assertNotIn(Action("claim", "AB2"), legal_actions(view))
        state, _ = step(state, 1, Action("claim", "AB2"))
        self.assertEqual(state.owners[:2], (0, 1))

    def test_two_and_three_player_parallel_group_closes(self):
        original = demo_setup()
        for count in (2, 3):
            tickets = tuple(p.tickets for p in original.players[:count])
            state = new_game(original.routes, tickets,
                             Rules(player_count=count, initial_trains=8))
            state, _ = step(state, 0, Action("claim", "AB1"))
            self.assertNotIn(Action("claim", "AB2"),
                             legal_actions(observe(state, 1)))

    def test_final_round_exactly_n_more_actions_including_trigger(self):
        for count in (2, 4, 5):
            state = self.basic(count, initial_trains=3)
            state, _ = step(state, 0, Action("claim", "AB"))
            self.assertEqual(state.final_turns_remaining, count)
            for offset, player_id in enumerate([*range(1, count), 0], start=1):
                state, _ = step(state, player_id, Action("gather"))
                self.assertEqual(state.final_turns_remaining, count - offset)
                self.assertEqual(state.finished, offset == count)
            self.assertEqual(state.turn, count + 1)
            self.assertEqual(state.end_reason, "final_round")
            self.assertEqual(
                legal_actions(observe(state, state.current_player)), ())
            with self.assertRaises(IllegalAction):
                step(state, state.current_player, Action("pass"))

    def test_four_forced_passes_end_a_stuck_game(self):
        state = new_game((Route("AB", "A", "B", 4),),
                         ((), (), (), ()), Rules(initial_trains=3))
        for player_id in range(4):
            self.assertEqual(legal_actions(observe(state, player_id)),
                             (Action("pass"),))
            state, _ = step(state, player_id, Action("pass"))
            self.assertEqual(state.finished, player_id == 3)
        self.assertEqual(state.end_reason, "all_players_stuck")

    def test_final_round_wraps_from_another_starting_player(self):
        base = self.basic(initial_trains=3)
        state = new_game(base.routes, tuple(p.tickets for p in base.players),
                         base.rules, starting_player=2)
        state, _ = step(state, 2, Action("claim", "AB"))
        for player_id in (3, 0, 1, 2):
            self.assertEqual(state.current_player, player_id)
            state, _ = step(state, player_id, Action("gather"))
        self.assertTrue(state.finished)
        self.assertEqual(state.turn, 5)

    def test_score_tie_uses_completed_ticket_count(self):
        state = new_game(
            (Route("AB", "A", "B", 1), Route("AC", "A", "C", 2)),
            ((Ticket("x", "A", "B", 5),),
             (Ticket("y", "A", "C", 2), Ticket("z", "A", "C", 2)),
             (), ()))
        state, _ = step(state, 0, Action("claim", "AB"))
        state, _ = step(state, 1, Action("claim", "AC"))
        for player_id in (2, 3, 0, 1):
            state, _ = step(state, player_id, Action("pass"))
        self.assertEqual([s.total for s in final_scores(state)], [6, 6, 0, 0])
        self.assertEqual(winners(state), (1,))

    def test_negative_scores_and_opponent_edges_do_not_connect(self):
        state = new_game((Route("AB", "A", "B", 1),),
                         ((Ticket("a", "A", "B", 5),),
                          (Ticket("b", "A", "B", 7),), (), ()))
        state, _ = step(state, 0, Action("claim", "AB"))
        for player_id in (1, 2, 3, 0):
            state, _ = step(state, player_id, Action("pass"))
        scores = final_scores(state)
        self.assertEqual([s.total for s in scores], [6, -7, 0, 0])
        self.assertEqual(winners(state), (0,))

    def test_demo_scoring_replay_determinism_and_countdown(self):
        initial = demo_setup()
        final, events = play(initial, demo_policies())
        again, repeated = play(demo_setup(), demo_policies())
        self.assertEqual((final, events), (again, repeated))
        self.assertEqual([s.total for s in final_scores(final)],
                         [4, 10, 12, 12])
        self.assertEqual(winners(final), (2, 3))
        self.assertEqual([e.final_turns_remaining for e in events[12:]],
                         [4, 3, 2, 1, 0])
        self.assertEqual([e.player_id for e in events[13:]], [1, 2, 3, 0])
        text = render_replay(initial, events)
        self.assertIn("A-D(-8)", text)
        self.assertNotIn("A-D(-8)", render_replay(initial, events[:10]))
        self.assertEqual(text, render_replay(demo_setup(), repeated))
        corrupted = (replace(events[0], resources_after=99), *events[1:])
        with self.assertRaises(ValueError):
            render_replay(initial, corrupted)
        for player_id, player in enumerate(final.players):
            used = sum(r.length for r, o in zip(final.routes, final.owners)
                       if o == player_id)
            self.assertEqual(player.trains + used, 8)
            self.assertGreaterEqual(player.resources, 0)

    def test_invalid_setup_is_rejected(self):
        route = Route("x", "A", "B", 1)
        for routes in [(route, route),
                       (route, Route("y", "B", "A", 1)),
                       (Route("x", "A", "B", 2, "g"),
                        Route("y", "B", "C", 2, "g"))]:
            with self.assertRaises(ValueError):
                new_game(routes, ((), (), (), ()))
        with self.assertRaises(ValueError):
            new_game((route,), ((), ()))
        with self.assertRaises(ValueError):
            new_game((route,), ((), (), (), ()), Rules(initial_resources=9))
        with self.assertRaises(ValueError):
            new_game((route,), ((Ticket("t", "A", "Z", 3),), (), (), ()))

    def test_turn_guard_is_not_a_scored_termination(self):
        with self.assertRaises(RuntimeError):
            play(demo_setup(), demo_policies(), max_turns=1)


if __name__ == "__main__":
    unittest.main()
