"""Exact known examples and boundaries; no held-out performance cases."""

from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from .endgame import (BeliefWorld, action_distribution, diagnostics,
                      evaluate_action, evaluate_actions, snapshot_scores,
                      snapshot_value, terminal_value, validate_belief)
from .endgame_experiment import (CASE_PATH, POLICIES, canonical, load_case,
                                 run_experiment, sha, write_report)
from .game import (Action, Observation, ROUTE_POINTS, Route, Rules, Ticket,
                   final_scores, legal_actions, new_game, observe, step)
from .policies import ReplanningGreedyPolicy


class EndgameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case, _, cls.worlds, cls.prefix = load_case()
        cls.reference = evaluate_actions(cls.worlds)
        cls.report = run_experiment()

    def scratch(self):
        root = Path(__file__).resolve().parent.parent
        temporary = (root / "tmp").resolve()
        if not temporary.is_relative_to(root):
            raise ValueError("scratch must stay inside the project")
        temporary.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="endgame-", dir=temporary)

    def test_hand_values_all_actions_and_four_depths(self):
        expected = {
            "claim AB": ["1", "1/4", "1/4", "1/4"],
            "claim CD": ["1", "1/2", "1/2", "3/8"],
            "claim EF": ["1/2", "1/4", "1/6", "1/12"],
            "claim GH": ["1/2", "0", "0", "0"],
            "gather": ["0", "0", "0", "0"],
        }
        for row in self.report["values"]:
            depth = row["h"] if row["phase"] == "truncated" else 3
            self.assertEqual(row["value_exact"],
                             expected[row["action"]][depth])
        old = json.loads(CASE_PATH.with_name("handcheck.json").read_text())
        self.assertEqual(old["case_sha256"], self.report["case_sha256"])
        reference = {
            r["action"]: r["value_exact"] for r in self.report["values"]
            if r["phase"] == "reference"}
        self.assertEqual(reference,
                         {r["action"]: r["Q_ref"] for r in old["rows"]})

    def test_actual_depth_work_and_full_reference_agree(self):
        self.assertEqual([r["transitions"] for r in self.report["decisions"]],
                         [10, 20, 30, 52])
        self.assertEqual([r["epsilon"] for r in self.report["decisions"]],
                         ["3/4", "1/6", "1/8", "0"])
        self.assertEqual([r["regret"] for r in self.report["decisions"]],
                         ["1/8", "0", "0", "0"])
        self.assertEqual(sum(len(e.frontier) for e in self.reference), 22)
        for a, b in zip(self.reference, evaluate_actions(self.worlds, 9)):
            self.assertEqual((a.value, a.transitions, a.frontier),
                             (b.value, b.transitions, b.frontier))

    def test_upper_bounds_and_monotone_values_for_this_model(self):
        reference = {e.action: e.value for e in self.reference}
        previous = None
        for depth in range(4):
            estimates = {e.action: e.value
                         for e in evaluate_actions(self.worlds, depth)}
            for action in reference:
                self.assertGreaterEqual(estimates[action], reference[action])
                if previous is not None:
                    self.assertLessEqual(estimates[action], previous[action])
            diagnosis = diagnostics(reference, estimates)
            self.assertTrue(diagnosis["one_sided_bound_holds"])
            self.assertTrue(diagnosis["two_sided_bound_holds"])
            previous = estimates

    def test_reference_replays_probabilities_budgets_and_scores(self):
        worlds = {w.name: w.state for w in self.worlds}
        for estimate in self.reference:
            self.assertEqual(sum(f.probability for f in estimate.frontier), 1)
            self.assertEqual(sum(f.probability * f.value
                                 for f in estimate.frontier), estimate.value)
            root_scores = set()
            for leaf in estimate.frontier:
                state = worlds[leaf.world]
                for saved in leaf.events:
                    self.assertIn(saved.action, legal_actions(
                        observe(state, saved.player_id)))
                    state, actual = step(state, saved.player_id, saved.action)
                    self.assertEqual(saved, actual)
                    for seat, player in enumerate(state.players):
                        routes = [r for r, owner in
                                  zip(state.routes, state.owners)
                                  if owner == seat]
                        self.assertEqual(
                            player.trains + sum(r.length for r in routes), 4)
                        self.assertEqual(player.route_points,
                                         sum(ROUTE_POINTS[r.length]
                                             for r in routes))
                        self.assertTrue(0 <= player.resources <= 8)
                self.assertEqual(state, leaf.state)
                self.assertTrue(state.finished)
                self.assertEqual(state.turn, 8)
                self.assertEqual(state.final_turns_remaining, 0)
                self.assertEqual(terminal_value(state, 0), leaf.value)
                self.assertEqual(snapshot_value(state, 0), leaf.value)
                root_scores.add(final_scores(state)[0].total)
            self.assertEqual(len(root_scores), 1)

    def test_hypothetical_settlement_does_not_end_live_games(self):
        initial = self.worlds[0].state
        before = repr(initial)
        with self.assertRaises(ValueError):
            terminal_value(initial, 0)
        snapshot_scores(initial)
        estimates = evaluate_actions(self.worlds, 0)
        self.assertTrue(all(not leaf.state.finished
                            for e in estimates for leaf in e.frontier))
        self.assertEqual(repr(initial), before)
        self.assertFalse(initial.finished)

    def test_information_boundary_and_average_before_max(self):
        self.assertEqual(observe(self.worlds[0].state, 0),
                         observe(self.worlds[1].state, 0))
        calls = []
        original = ReplanningGreedyPolicy.choose

        def inspect(policy, view):
            self.assertIsInstance(view, Observation)
            self.assertTrue(all(not hasattr(p, "tickets")
                                for p in view.players))
            calls.append(view.player_id)
            return original(policy, view)

        with patch.object(ReplanningGreedyPolicy, "choose", inspect):
            result = evaluate_actions(self.worlds)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {1, 2})
        self.assertEqual(max(e.value for e in result), F(3, 8))
        clairvoyant = sum(w.probability * max(
            dict(e.conditional_values)[w.name] for e in result)
                          for w in self.worlds)
        self.assertEqual(clairvoyant, F(5, 8))
        self.assertNotEqual(clairvoyant, max(e.value for e in result))
        with self.assertRaises(TypeError):
            action_distribution(self.worlds[0].state, "uniform")

    def test_bad_beliefs_and_unsupported_horizons_are_rejected(self):
        h, other = self.worlds
        bad_worlds = [(), (h, h), (replace(h, probability=0.5), other),
                      (replace(h, probability=F(-1, 2)), other),
                      (replace(h, probability=F(1, 3)), other),
                      (replace(h, state=replace(h.state, finished=True)),
                       other)]
        for countdown in (None, 5):
            bad_worlds.append((replace(h, state=replace(
                h.state, final_turns_remaining=countdown)), other))
        players = list(other.state.players)
        players[1] = replace(players[1], resources=players[1].resources + 1)
        bad_worlds.append((h, replace(other, state=replace(
            other.state, players=tuple(players)))))
        for worlds in bad_worlds:
            with self.assertRaises(ValueError):
                validate_belief(worlds, 0, POLICIES)
        for depth in (-1, True, 1.0):
            with self.assertRaises(ValueError):
                evaluate_actions(self.worlds, depth)
        with self.assertRaises(ValueError):
            evaluate_action(self.worlds, Action("claim", "BE"))
        with self.assertRaises(ValueError):
            validate_belief(self.worlds, 0,
                            ("focal", "static", "replan", "uniform"))

    def test_uniform_distribution_includes_gather_and_forced_pass(self):
        view = observe(self.worlds[0].state, 0)
        distribution = action_distribution(view, "uniform")
        self.assertEqual(len(distribution), 5)
        self.assertIn((Action("gather"), F(1, 5)), distribution)
        state = new_game((Route("AB", "A", "B", 1),), ((),) * 4,
                         Rules(initial_trains=2, final_threshold=1), 3)
        state, _ = step(state, 3, Action("claim", "AB"))
        self.assertEqual(action_distribution(observe(state, 0), "uniform"),
                         ((Action("pass"), F(1)),))

    def test_terminal_before_requested_depth_uses_real_utility(self):
        state = new_game((Route("AB", "A", "B", 1),), ((),) * 4,
                         Rules(initial_trains=2, final_threshold=1), 3)
        state, _ = step(state, 3, Action("claim", "AB"))
        for seat in range(3):
            state, _ = step(state, seat, Action("pass"))
        self.assertEqual(state.final_turns_remaining, 1)
        worlds = (BeliefWorld("only", F(1), state),)
        policies = ("replan", "replan", "uniform", "focal")
        with patch("multiplayer.endgame.snapshot_value",
                   side_effect=AssertionError("terminal must not use cutoff")):
            for depth in (None, 0, 9):
                estimate = evaluate_action(worlds, Action("pass"), depth,
                                           focal=3, policies=policies)
                self.assertEqual(estimate.value, 1)
                self.assertEqual(estimate.transitions, 1)
                self.assertEqual(len(estimate.frontier[0].events), 1)

    def test_joint_winner_fractions_and_completed_ticket_tiebreak(self):
        for count in (2, 3, 4):
            routes = tuple(Route(f"R{i}", f"A{i}", f"B{i}", 1)
                           for i in range(count))
            state = new_game(routes, ((),) * 4,
                             Rules(initial_trains=2, final_threshold=1))
            for seat in range(4):
                action = (Action("claim", f"R{seat}") if seat < count
                          else Action("pass"))
                state, _ = step(state, seat, action)
            state, _ = step(state, 0, Action("pass"))
            self.assertEqual(terminal_value(state, 0), F(1, count))
            self.assertEqual(snapshot_value(state, 0), F(1, count))
        state = new_game((Route("AB", "A", "B", 1), Route("CD", "C", "D", 2)),
                         ((Ticket("AB-ticket", "A", "B", 1),), (), (), ()),
                         Rules(initial_trains=3, final_threshold=1))
        for seat, action in ((0, Action("claim", "AB")),
                             (1, Action("claim", "CD")),
                             (2, Action("pass")), (3, Action("pass")),
                             (0, Action("pass")), (1, Action("pass"))):
            state, _ = step(state, seat, action)
        self.assertEqual(snapshot_scores(state)[:2], ((2, 1), (2, 0)))
        self.assertEqual(terminal_value(state, 0), 1)

    def test_strict_gap_bounds_and_true_optimal_sets(self):
        a, b = Action("claim", "AB"), Action("claim", "CD")
        reference = {a: F(2, 5), b: F(3, 5)}
        two_sided = diagnostics(reference, {a: F(1, 2), b: F(1, 2)})
        self.assertEqual(two_sided["epsilon"], F(1, 10))
        self.assertEqual(two_sided["regret"], F(1, 5))
        self.assertTrue(two_sided["two_sided_bound_holds"])
        self.assertFalse(two_sided["two_sided_unique_best_certificate"])
        one_sided = diagnostics(reference, {a: F(3, 5), b: F(3, 5)})
        self.assertEqual(one_sided["regret"], one_sided["epsilon"])
        self.assertFalse(one_sided["one_sided_unique_best_certificate"])
        certified = diagnostics(reference, {a: F(9, 20), b: F(3, 5)})
        self.assertTrue(certified["two_sided_unique_best_certificate"])
        self.assertTrue(certified["one_sided_unique_best_certificate"])
        tie = diagnostics({a: F(1, 2), b: F(1, 2)},
                          {a: F(1, 2), b: F(1, 2)})
        self.assertEqual(tie["optimal_actions"], (a, b))
        self.assertEqual(tie["unique_best_gap"], 0)
        self.assertEqual(tie["regret"], 0)
        self.assertFalse(tie["two_sided_unique_best_certificate"])
        for wrong in ({a: F(1)}, {a: 0.4, b: 0.6}):
            with self.assertRaises(ValueError):
                diagnostics(reference, wrong)

    def test_world_order_invariance_and_belief_threshold(self):
        self.assertEqual(evaluate_actions(tuple(reversed(self.worlds))),
                         self.reference)
        worlds = tuple(replace(w, probability=F(2, 5) if w.name == "H"
                               else F(3, 5)) for w in self.worlds)
        values = {e.action: e.value for e in evaluate_actions(worlds)}
        self.assertEqual(values[Action("claim", "AB")], F(3, 10))
        self.assertEqual(values[Action("claim", "CD")], F(3, 10))
        diagnosis = diagnostics(values, values)
        self.assertEqual(diagnosis["optimal_actions"],
                         (Action("claim", "AB"), Action("claim", "CD")))

    def test_reproducible_export_snapshot_and_no_overwrite(self):
        self.assertEqual(run_experiment(), self.report)
        with self.scratch() as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            write_report(self.report, first)
            run = subprocess.run([
                sys.executable, "-B", "-m", "multiplayer.endgame_experiment",
                "--output", str(second)], cwd=first / "source",
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            content = {p.relative_to(first).as_posix(): p.read_bytes()
                       for p in first.rglob("*") if p.is_file()}
            self.assertEqual(len(content), 12)
            self.assertEqual(content,
                             {p.relative_to(second).as_posix(): p.read_bytes()
                              for p in second.rglob("*") if p.is_file()})
            with self.assertRaises(FileExistsError):
                write_report(self.report, first)
            self.assertEqual(content,
                             {p.relative_to(first).as_posix(): p.read_bytes()
                              for p in first.rglob("*") if p.is_file()})
            manifest = json.loads(content["manifest.json"])
            for name, digest in manifest["result_sha256"].items():
                self.assertEqual(sha(content[name]), digest)

    def test_declared_policy_mismatch_and_changed_sources_rejected(self):
        with self.scratch() as temp:
            root = Path(temp)
            case = json.loads(CASE_PATH.read_text())
            case["opponent_policies_after_prefix"]["3"] = "replan"
            path = root / "bad-case.json"
            path.write_text(canonical(case), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_case(path)
            report = {**self.report, "source_sha256": {}}
            with self.assertRaises(ValueError):
                write_report(report, root / "bad-result")
            self.assertFalse((root / "bad-result").exists())


if __name__ == "__main__":
    unittest.main()
