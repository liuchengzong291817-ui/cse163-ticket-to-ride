"""Same setups and goals; only within-path claim order varies."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from .bottleneck import BottleneckPriorityPolicy
from .compare_bottleneck import run_comparison
from .evaluate import METRICS, make_scenarios, score_records, write_report
from .game import Action, legal_actions, observe, step
from .policies import ReplanningGreedyPolicy


class BottleneckComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = make_scenarios((101,))
        cls.report = run_comparison(cls.scenarios)

    def test_pairing_holds_setup_opponents_seeds_and_all_seats_fixed(self):
        games = {g["game_id"]: g for g in self.report["games"]}
        self.assertEqual(len(games), 16)
        self.assertEqual(len(self.report["pairs"]), 8)
        players = {(r["game_id"], r["seat"]): r
                   for r in self.report["players"]}
        for pair in self.report["pairs"]:
            a, b = (games[pair[f"{arm}_game_id"]]
                    for arm in ("baseline", "candidate"))
            self.assertEqual(a["initial_state_sha256"],
                             b["initial_state_sha256"])
            self.assertEqual(a["rng_seeds_by_seat"], b["rng_seeds_by_seat"])
            x, y = (json.loads(g["policies_by_seat"]) for g in (a, b))
            seat = pair["focal_seat"]
            self.assertEqual((x[seat], y[seat]), ("replan", "bottleneck"))
            self.assertEqual(x[:seat] + x[seat+1:], ["random"] * 3)
            self.assertEqual(y[:seat] + y[seat+1:], ["random"] * 3)
            for metric in METRICS:
                before = players[(a["game_id"], seat)][metric]
                after = players[(b["game_id"], seat)][metric]
                self.assertEqual(pair[f"delta_{metric}"], after - before)
        for replicate in (0, 1):
            pairs = [p for p in self.report["pairs"]
                     if p["replicate"] == replicate]
            self.assertEqual({p["focal_seat"] for p in pairs}, set(range(4)))
            self.assertEqual({p["turn_position"] for p in pairs},
                             set(range(4)))

    def test_identical_observations_keep_goal_path_and_nonclaim_rules(self):
        initials = {s.scenario_id: s.initial_state for s in self.scenarios}
        games = {r["game_id"]: r for r in self.report["games"]}
        baseline = ReplanningGreedyPolicy()
        candidate = BottleneckPriorityPolicy()
        for replay in self.report["replays"]:
            game = games[replay["game_id"]]
            if not game["arm"].startswith("bottleneck"):
                continue
            state = initials[replay["scenario_id"]]
            for saved in replay["events"]:
                action = Action(**saved["action"])
                if state.current_player == game["focal_seat"]:
                    view = observe(state, state.current_player)
                    a, b = baseline.decide(view), candidate.decide(view)
                    self.assertEqual((a.ticket_id, a.plan),
                                     (b.ticket_id, b.plan))
                    self.assertEqual(b.action, action)
                    if a.action != b.action:
                        self.assertEqual((a.action.kind, b.action.kind),
                                         ("claim", "claim"))
                        self.assertIn(b.action, legal_actions(view))
                        self.assertIn(a.action.route_id, a.plan.route_ids)
                        self.assertIn(b.action.route_id, a.plan.route_ids)
                state, _ = step(state, state.current_player, action)

    def test_every_event_is_legal_with_conserved_budget_and_exact_score(self):
        initials = {s.scenario_id: s.initial_state for s in self.scenarios}
        for replay in self.report["replays"]:
            state = initials[replay["scenario_id"]]
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
                    self.assertEqual(player.trains + used,
                                     state.rules.initial_trains)
                    self.assertTrue(0 <= player.resources
                                    <= state.rules.resource_cap)
            self.assertTrue(state.finished)
            expected = score_records(state)
            actual = [r for r in self.report["players"]
                      if r["game_id"] == replay["game_id"]]
            for row, saved in zip(expected, actual):
                self.assertEqual(row, {key: saved[key] for key in row})
            self.assertAlmostEqual(sum(r["win_credit"] for r in expected), 1)

    def test_repeated_and_reordered_runs_are_identical_without_mutation(self):
        original = repr(self.scenarios)
        self.assertEqual(self.report, run_comparison(self.scenarios))
        reordered = run_comparison(self.scenarios, replicates=(1, 0))
        for table in ("games", "replays"):
            self.assertEqual(sorted(self.report[table],
                                    key=lambda row: row["game_id"]),
                             sorted(reordered[table],
                                    key=lambda row: row["game_id"]))
        self.assertEqual(self.report["summary"], reordered["summary"])
        self.assertEqual(repr(self.scenarios), original)

    def test_invalid_design_and_output_overwrite_are_rejected(self):
        for scenarios, replicates in (((), (0,)), (self.scenarios, (0, 0)),
                                      (self.scenarios * 2, (0,))):
            with self.assertRaises(ValueError):
                run_comparison(scenarios, replicates)
        root = Path(__file__).resolve().parent.parent
        scratch = (root / "tmp").resolve()
        if not scratch.is_relative_to(root):
            raise ValueError("scratch must stay inside the project")
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
                prefix="bottleneck-", dir=scratch) as temp:
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
            self.assertIn("bottleneck.py", manifest["source_sha256"])
            self.assertIn("compare_bottleneck.py", manifest["source_sha256"])
            for name, digest in manifest["result_sha256"].items():
                self.assertEqual(hashlib.sha256(content[name]).hexdigest(),
                                 digest)


if __name__ == "__main__":
    unittest.main()
