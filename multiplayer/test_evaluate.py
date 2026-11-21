"""Check controls, paired units, replay legality, and output safety."""

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from .demo import demo_policies, demo_setup, play
from .evaluate import (METRICS, make_scenarios, run_pilot, score_records,
                       summarize_pairs, write_report)
from .game import Action, legal_actions, observe, step


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = make_scenarios((101,))
        cls.report = run_pilot(cls.scenarios)

    def test_deals_are_fixed_with_two_unique_tickets_per_seat(self):
        self.assertEqual(make_scenarios(), make_scenarios())
        for scenario in make_scenarios():
            state = scenario.initial_state
            self.assertEqual(state.rules.player_count, 4)
            self.assertEqual(state.current_player, 0)
            self.assertEqual([len(p.tickets) for p in state.players], [2] * 4)
            ids = [t.ticket_id for p in state.players for t in p.tickets]
            self.assertEqual(len(set(ids)), 8)

    def test_pairing_changes_one_policy_and_no_setup_or_seed(self):
        report = self.report
        games = {row["game_id"]: row for row in report["games"]}
        self.assertEqual(len(games), 10)  # 2 RNG replicates * (1 + 4) games.
        self.assertEqual(len(report["pairs"]), 8)
        for pair in report["pairs"]:
            before = games[pair["baseline_game_id"]]
            after = games[pair["candidate_game_id"]]
            self.assertEqual(before["initial_state_sha256"],
                             after["initial_state_sha256"])
            self.assertEqual(before["rng_seeds_by_seat"],
                             after["rng_seeds_by_seat"])
            a = json.loads(before["policies_by_seat"])
            b = json.loads(after["policies_by_seat"])
            self.assertEqual(a, ["random"] * 4)
            changed = [seat for seat in range(4) if a[seat] != b[seat]]
            self.assertEqual(changed, [pair["focal_seat"]])
            self.assertEqual(b[pair["focal_seat"]], "replan")
            self.assertEqual(pair["focal_seat"], pair["turn_position"])
        for replicate in (0, 1):
            pairs = [p for p in report["pairs"] if p["replicate"] == replicate]
            self.assertEqual({p["focal_seat"] for p in pairs}, set(range(4)))
            self.assertEqual(len({p["baseline_game_id"] for p in pairs}), 1)
        first, second = report["games"][0], report["games"][5]
        self.assertNotEqual(first["rng_seeds_by_seat"],
                            second["rng_seeds_by_seat"])

    def test_every_recorded_action_replays_legally(self):
        initials = {s.scenario_id: s.initial_state for s in self.scenarios}
        games = {r["game_id"]: r for r in self.report["games"]}
        for replay in self.report["replays"]:
            state = initials[replay["scenario_id"]]
            for saved in replay["events"]:
                action = Action(**saved["action"])
                self.assertIn(action, legal_actions(
                    observe(state, saved["player_id"])))
                state, event = step(state, saved["player_id"], action)
                self.assertEqual(asdict(event), saved)
                for seat, player in enumerate(state.players):
                    used = sum(
                        route.length for route, owner
                        in zip(state.routes, state.owners) if owner == seat)
                    self.assertEqual(player.trains + used, 8)
                    self.assertTrue(0 <= player.resources <= 8)
            self.assertTrue(state.finished)
            self.assertEqual(state.turn, games[replay["game_id"]]["turns"])
            actual = [r for r in self.report["players"]
                      if r["game_id"] == replay["game_id"]]
            for expected, row in zip(score_records(state), actual):
                self.assertEqual(expected, {key: row[key] for key in expected})

    def test_pair_metrics_and_split_victory_accounting(self):
        players = {(r["game_id"], r["seat"]): r
                   for r in self.report["players"]}
        for pair in self.report["pairs"]:
            a = players[(pair["baseline_game_id"], pair["focal_seat"])]
            b = players[(pair["candidate_game_id"], pair["focal_seat"])]
            for metric in METRICS:
                self.assertEqual(pair[f"delta_{metric}"],
                                 b[metric] - a[metric])
        for game in self.report["games"]:
            credits = [players[(game["game_id"], seat)]["win_credit"]
                       for seat in range(4)]
            self.assertAlmostEqual(sum(credits), 1)
        final, _ = play(demo_setup(), demo_policies())
        # P2 and P3 are joint winners of the known scripted game.
        rows = score_records(final)
        self.assertEqual([r["win_credit"] for r in rows], [0, 0, 0.5, 0.5])
        self.assertEqual([r["sole_win"] for r in rows], [0, 0, 0, 0])
        self.assertEqual([r["shared_win"] for r in rows], [0, 0, 1, 1])

    def test_identical_inputs_reproduce_all_tables_and_events(self):
        before = repr(self.scenarios)
        self.assertEqual(run_pilot(self.scenarios), self.report)
        self.assertEqual(repr(self.scenarios), before)

    def test_scenarios_get_equal_weight_not_game_row_weight(self):
        # Synthetic numbers isolate the aggregation rule, not game performance.
        pairs = []
        examples = (("A", (0, 1), 0.1), ("B", (0,), 0.9))
        for scenario, replicates, effect in examples:
            for replicate in replicates:
                for seat in range(4):
                    row = {"scenario_id": scenario, "replicate": replicate,
                           "focal_seat": seat}
                    for metric in METRICS:
                        row[f"baseline_{metric}"] = 0
                        row[f"candidate_{metric}"] = effect
                        row[f"delta_{metric}"] = effect
                    pairs.append(row)
        summaries = summarize_pairs(pairs)
        self.assertEqual(summaries, summarize_pairs(list(reversed(pairs))))
        for row in summaries[2]:
            self.assertEqual(row["scenario_count"], 2)
            self.assertEqual(row["paired_seat_cells"], 12)
            self.assertAlmostEqual(row["mean_paired_difference"], 0.5)
            self.assertEqual(row["min_scenario_difference"], 0.1)
            self.assertEqual(row["max_scenario_difference"], 0.9)
        with self.assertRaises(ValueError):
            summarize_pairs(pairs[1:])

    def test_paired_metric_ties_are_not_winner_ties(self):
        pairs = []
        for seat, change in enumerate((-0.25, 0, 0.25, 0)):
            row = {"scenario_id": "A", "replicate": 0, "focal_seat": seat}
            for metric in METRICS:
                row[f"baseline_{metric}"] = 0.5
                row[f"candidate_{metric}"] = 0.5 + change
                row[f"delta_{metric}"] = change
            pairs.append(row)
        for row in summarize_pairs(pairs)[2]:
            self.assertEqual(row["paired_increase_rate"], 0.25)
            self.assertEqual(row["paired_tie_rate"], 0.5)
            self.assertEqual(row["paired_decrease_rate"], 0.25)
            self.assertEqual(row["mean_paired_difference"], 0)
            self.assertIsNone(row["scenario_difference_sd"])

    def test_duplicate_or_played_scenarios_are_rejected(self):
        with self.assertRaises(ValueError):
            make_scenarios((101, 101))
        with self.assertRaises(ValueError):
            run_pilot(self.scenarios * 2)
        with self.assertRaises(ValueError):
            run_pilot(self.scenarios, replicates=(0, 0))
        scenario = self.scenarios[0]
        played, _ = step(scenario.initial_state, 0, Action("gather"))
        with self.assertRaises(ValueError):
            run_pilot((replace(scenario, initial_state=played),))

    def test_export_is_repeatable_and_refuses_overwrite(self):
        root = Path(__file__).resolve().parent.parent
        scratch = (root / "tmp").resolve()
        if not scratch.is_relative_to(root):
            raise ValueError("test scratch directory must stay in the project")
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix="evaluation-", dir=scratch) as temporary:
            self.assertEqual(Path(temporary).resolve().parent, scratch)
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            write_report(self.report, first)
            write_report(self.report, second)
            originals = {p.name: p.read_bytes() for p in first.iterdir()}
            copies = {p.name: p.read_bytes() for p in second.iterdir()}
            self.assertEqual(originals, copies)
            with self.assertRaises(FileExistsError):
                write_report(self.report, first)
            unchanged = {p.name: p.read_bytes() for p in first.iterdir()}
            self.assertEqual(originals, unchanged)
            manifest = json.loads(
                (first / "manifest.json").read_text(encoding="utf-8"))
            for name, expected in manifest["result_sha256"].items():
                content = (first / name).read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                self.assertEqual(digest, expected)
            with (first / "pairs.csv").open(newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 8)


if __name__ == "__main__":
    unittest.main()
