"""Check controls, legality, repeatability and historical compatibility."""

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from .compare_planning import run_comparison
from .evaluate import METRICS, make_scenarios, score_records, write_report
from .game import Action, legal_actions, observe, step


class PlanningComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = make_scenarios((101,))
        cls.report = run_comparison(cls.scenarios)

    def test_only_focal_strategy_changes_with_all_four_seats_paired(self):
        games = {r["game_id"]: r for r in self.report["games"]}
        self.assertEqual(len(games), 16)
        self.assertEqual(len(self.report["pairs"]), 8)
        for pair in self.report["pairs"]:
            a, b = (games[pair[f"{arm}_game_id"]]
                    for arm in ("baseline", "candidate"))
            self.assertEqual(a["initial_state_sha256"],
                             b["initial_state_sha256"])
            self.assertEqual(a["rng_seeds_by_seat"], b["rng_seeds_by_seat"])
            first = json.loads(a["policies_by_seat"])
            second = json.loads(b["policies_by_seat"])
            seat = pair["focal_seat"]
            self.assertEqual((first[seat], second[seat]), ("static", "replan"))
            self.assertEqual(first[:seat] + first[seat+1:], ["random"] * 3)
            self.assertEqual(second[:seat] + second[seat+1:], ["random"] * 3)
        for replicate in (0, 1):
            cells = [r for r in self.report["pairs"]
                     if r["replicate"] == replicate]
            self.assertEqual({r["focal_seat"] for r in cells}, set(range(4)))
            self.assertEqual({r["turn_position"] for r in cells},
                             set(range(4)))
            self.assertEqual(len({r["baseline_game_id"] for r in cells}), 4)

    def test_all_actions_replay_and_scores_and_budgets_match(self):
        initials = {s.scenario_id: s.initial_state for s in self.scenarios}
        for replay in self.report["replays"]:
            state = initials[replay["scenario_id"]]
            initial_trains = state.rules.initial_trains
            for saved in replay["events"]:
                action = Action(**saved["action"])
                self.assertIn(action, legal_actions(observe(
                    state, saved["player_id"])))
                state, event = step(state, saved["player_id"], action)
                self.assertEqual(asdict(event), saved)
                for seat, player in enumerate(state.players):
                    used = sum(r.length for r, owner
                               in zip(state.routes, state.owners)
                               if owner == seat)
                    self.assertEqual(player.trains + used, initial_trains)
                    self.assertTrue(0 <= player.resources
                                    <= state.rules.resource_cap)
            self.assertTrue(state.finished)
            actual = [r for r in self.report["players"]
                      if r["game_id"] == replay["game_id"]]
            expected = score_records(state)
            self.assertAlmostEqual(sum(r["win_credit"] for r in expected), 1)
            for row, saved in zip(expected, actual):
                self.assertEqual(row, {key: saved[key] for key in row})

    def test_differences_have_dynamic_minus_static_sign(self):
        rows = {(r["game_id"], r["seat"]): r for r in self.report["players"]}
        for pair in self.report["pairs"]:
            a, b = (rows[(pair[f"{arm}_game_id"], pair["focal_seat"])]
                    for arm in ("baseline", "candidate"))
            for metric in METRICS:
                self.assertEqual(pair[f"delta_{metric}"],
                                 b[metric] - a[metric])
        for row in self.report["summary"]:
            self.assertEqual(row["scenario_count"], 1)
            self.assertEqual(row["paired_seat_cells"], 8)
            self.assertIsNone(row["scenario_difference_sd"])

    def test_fresh_instances_make_repeated_and_reordered_runs_identical(self):
        self.assertEqual(self.report, run_comparison(self.scenarios))
        reversed_order = run_comparison(self.scenarios, replicates=(1, 0))
        for table, key in (("games", "game_id"), ("replays", "game_id")):
            self.assertEqual(sorted(self.report[table], key=lambda r: r[key]),
                             sorted(reversed_order[table],
                                    key=lambda r: r[key]))
        self.assertEqual(self.report["summary"], reversed_order["summary"])

    def test_dynamic_trajectories_match_historical_pilot(self):
        archive = (Path(__file__).resolve().parent.parent
                   / "outputs/multiplayer_pilot_v1")
        with (archive / "games.csv").open(newline="", encoding="utf-8") as f:
            old_games = {r["game_id"]: r for r in csv.DictReader(f)}
        lines = (archive / "replays.jsonl").read_text(
            encoding="utf-8").splitlines()
        old_replays = {r["game_id"]: r
                       for r in (json.loads(line) for line in lines)}
        new_replays = {r["game_id"]: r for r in self.report["replays"]}
        for game in self.report["games"]:
            if game["arm"].startswith("replan"):
                self.assertEqual({k: str(v) for k, v in game.items()},
                                 old_games[game["game_id"]])
                self.assertEqual(new_replays[game["game_id"]],
                                 old_replays[game["game_id"]])

    def test_rejects_invalid_design_and_export_overwrite(self):
        for scenarios, replicates in (((), (0,)), (self.scenarios, (0, 0)),
                                      (self.scenarios * 2, (0,))):
            with self.assertRaises(ValueError):
                run_comparison(scenarios, replicates)
        root = Path(__file__).resolve().parent.parent
        scratch = (root / "tmp").resolve()
        if not scratch.is_relative_to(root):
            raise ValueError("scratch must remain in project")
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix="planning-", dir=scratch) as temp:
            self.assertEqual(Path(temp).resolve().parent, scratch)
            first, second = Path(temp) / "first", Path(temp) / "second"
            write_report(self.report, first)
            write_report(run_comparison(self.scenarios), second)
            content = {p.name: p.read_bytes() for p in first.iterdir()}
            self.assertEqual(content,
                             {p.name: p.read_bytes()
                              for p in second.iterdir()})
            with self.assertRaises(FileExistsError):
                write_report(self.report, first)
            self.assertEqual(content,
                             {p.name: p.read_bytes() for p in first.iterdir()})
            manifest = json.loads(content["manifest.json"])
            self.assertIn("compare_planning.py", manifest["source_sha256"])
            for name, digest in manifest["result_sha256"].items():
                self.assertEqual(hashlib.sha256(content[name]).hexdigest(),
                                 digest)


if __name__ == "__main__":
    unittest.main()
