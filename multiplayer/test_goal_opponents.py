"""Use development deals only; never evaluate the reserved test cohort here."""

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from .evaluate import METRICS, make_scenarios, score_records
from .game import Action, legal_actions, new_game, observe, step
from .goal_opponents import (PROTOCOL_PATH, assert_split_disjoint,
                             create_freeze, hand_partition_key, load_protocol,
                             rotated_scenario, run_pilot, summarize_rotations,
                             verify_freeze, write_report)


class GoalOpponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deliberately use the already-observed 101/202/303 cases only.
        cls.report = run_pilot("development")
        cls.protocol = load_protocol()
        cls.cases = {f"development-{s.deal_seed}": s
                     for s in make_scenarios()}

    def temporary_directory(self):
        root = Path(__file__).resolve().parent.parent
        scratch = (root / "tmp").resolve()
        if not scratch.is_relative_to(root):
            raise ValueError("test scratch must remain in the project")
        scratch.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(
            prefix="goal-opponents-", dir=scratch)

    def test_protocol_seeds_are_disjoint_and_old_sources_are_frozen(self):
        p = self.protocol
        self.assertEqual(p["development_deal_seeds"], [101, 202, 303])
        names = ("development_deal_seeds", "test_deal_seeds",
                 "reserved_not_evaluated_deal_seeds")
        pools = [set(p[k]) for k in names]
        self.assertFalse(pools[0] & pools[1] or pools[1] & pools[2])
        self.assertEqual(len(p["frozen_preexisting_sources_sha256"]), 7)
        self.assertEqual(self.report["split"], "development")
        self.assertFalse(
            self.report["split_audit"]["reserved_deals_generated"])

    def test_test_run_without_freeze_fails_before_deals_are_generated(self):
        with patch("multiplayer.goal_opponents._split_scenarios") as generator:
            with self.assertRaises(ValueError):
                run_pilot("test")
            generator.assert_not_called()

    def test_split_check_detects_same_hands_despite_new_ids_and_seats(self):
        a, b = make_scenarios((101, 202))
        hands = tuple(tuple(reversed(p.tickets))
                      for p in reversed(a.initial_state.players))
        permuted = replace(
            a, scenario_id="different-name",
            initial_state=new_game(a.initial_state.routes, hands,
                                   a.initial_state.rules))
        self.assertEqual(hand_partition_key(a), hand_partition_key(permuted))
        with self.assertRaises(ValueError):
            assert_split_disjoint((a,), (permuted,))
        with self.assertRaises(ValueError):
            assert_split_disjoint((a,), (b, replace(b, scenario_id="another")))
        assert_split_disjoint((a,), (b,))

    def test_balanced_starts_and_seats_with_shared_baseline(self):
        report = self.report
        games = {g["game_id"]: g for g in report["games"]}
        self.assertEqual(len(games), 60)
        self.assertEqual(len(report["pairs"]), 48)
        self.assertEqual(sum(g["arm"] == "all_replan" for g in games.values()),
                         12)
        for pair in report["pairs"]:
            a, b = (games[pair[f"{arm}_game_id"]]
                    for arm in ("baseline", "candidate"))
            self.assertEqual(a["initial_state_sha256"],
                             b["initial_state_sha256"])
            keys = ("scenario_id", "starting_player", "split", "deal_seed")
            for key in keys:
                self.assertEqual(a[key], b[key])
            x, y = (json.loads(g["policies_by_seat"]) for g in (a, b))
            self.assertEqual(x, ["replan"] * 4)
            seat = pair["focal_seat"]
            self.assertEqual(y[seat], "bottleneck")
            self.assertEqual(y[:seat] + y[seat+1:], ["replan"] * 3)
            self.assertEqual(pair["turn_position"],
                             (seat - pair["starting_player"]) % 4)
        for case in self.cases:
            for start in range(4):
                cells = [r for r in report["pairs"]
                         if r["scenario_id"] == case
                         and r["starting_player"] == start]
                self.assertEqual({r["focal_seat"] for r in cells},
                                 set(range(4)))
                self.assertEqual(len({r["baseline_game_id"] for r in cells}),
                                 1)
            for seat in range(4):
                cells = [r for r in report["pairs"]
                         if r["scenario_id"] == case
                         and r["focal_seat"] == seat]
                self.assertEqual({r["turn_position"] for r in cells},
                                 set(range(4)))

    def test_rotation_changes_only_the_starting_player_in_a_fresh_state(self):
        scenario = make_scenarios((101,))[0]
        for start in range(4):
            rotated = rotated_scenario(scenario, start)
            self.assertEqual(rotated.initial_state,
                             replace(scenario.initial_state,
                                     current_player=start))
        self.assertEqual(scenario.initial_state.current_player, 0)
        for invalid in (-1, 4, True, "0"):
            with self.assertRaises(ValueError):
                rotated_scenario(scenario, invalid)

    def test_all_events_replay_legally_with_budgets_and_scores_preserved(self):
        for replay in self.report["replays"]:
            state = rotated_scenario(self.cases[replay["scenario_id"]],
                                     replay["starting_player"]).initial_state
            for saved in replay["events"]:
                action = Action(**saved["action"])
                self.assertIn(action, legal_actions(observe(
                    state, saved["player_id"])))
                state, event = step(state, saved["player_id"], action)
                self.assertEqual(asdict(event), saved)
                for seat, player in enumerate(state.players):
                    used = sum(
                        r.length for r, owner in
                        zip(state.routes, state.owners)
                        if owner == seat)
                    self.assertEqual(player.trains + used, 8)
                    self.assertTrue(0 <= player.resources <= 8)
            actual = [r for r in self.report["players"]
                      if r["game_id"] == replay["game_id"]]
            expected = score_records(state)
            for row, saved in zip(expected, actual):
                self.assertEqual(row, {key: saved[key] for key in row})
            self.assertAlmostEqual(sum(r["win_credit"] for r in expected), 1)
        for row in self.report["summary"]:
            if row["metric"] == "win_credit":
                self.assertAlmostEqual(row["baseline_mean"], 0.25)

    def test_deal_summary_and_incomplete_rotation_checks(self):
        pairs = []
        for case, effect in (("a", 0.1), ("b", 0.9)):
            for start in range(4):
                for seat in range(4):
                    row = {"scenario_id": case, "split": "development",
                           "starting_player": start, "focal_seat": seat,
                           "turn_position": (seat - start) % 4}
                    for metric in METRICS:
                        row[f"baseline_{metric}"] = 0
                        row[f"candidate_{metric}"] = effect
                        row[f"delta_{metric}"] = effect
                    pairs.append(row)
        summaries = summarize_rotations(pairs)
        self.assertEqual(summaries, summarize_rotations(list(reversed(pairs))))
        for row in summaries[2]:
            self.assertEqual(row["scenario_count"], 2)
            self.assertEqual(row["paired_seat_cells"], 32)
            self.assertEqual(row["mean_paired_difference"], 0.5)
        for invalid in (pairs[1:], [r for r in pairs if r["starting_player"]],
                        [{**r, "turn_position": 9} for r in pairs]):
            with self.assertRaises(ValueError):
                summarize_rotations(invalid)
        mixed = copy.deepcopy(pairs)
        mixed[0]["split"] = "test"
        with self.assertRaises(ValueError):
            summarize_rotations(mixed)

    def test_rerunning_deterministic_games_adds_no_new_randomness(self):
        self.assertEqual(self.report, run_pilot("development"))
        self.assertNotIn("replicate", self.report["pairs"][0])
        self.assertNotIn("rng_replicates", self.report["scenario_summary"][0])

    def test_freeze_and_exports_are_reproducible_and_refuse_overwrite(self):
        with self.temporary_directory() as temp:
            root = Path(temp)
            frozen = root / "freeze.json"
            create_freeze(PROTOCOL_PATH, frozen)
            verify_freeze(PROTOCOL_PATH, frozen)
            with self.assertRaises(FileExistsError):
                create_freeze(PROTOCOL_PATH, frozen)
            a, b = root / "a", root / "b"
            write_report(self.report, a, freeze_path=frozen)
            write_report(run_pilot("development"), b, freeze_path=frozen)
            original = {p.name: p.read_bytes() for p in a.iterdir()}
            self.assertEqual(len(original), 12)
            self.assertEqual(original,
                             {p.name: p.read_bytes() for p in b.iterdir()})
            with self.assertRaises(FileExistsError):
                write_report(self.report, a, freeze_path=frozen)
            self.assertEqual(original,
                             {p.name: p.read_bytes() for p in a.iterdir()})
            manifest = json.loads(original["manifest.json"])
            for name, digest in manifest["result_sha256"].items():
                self.assertEqual(hashlib.sha256(original[name]).hexdigest(),
                                 digest)

    def test_changed_freeze_or_overlapping_seeds_fail_before_test(self):
        with self.temporary_directory() as temp:
            root = Path(temp)
            frozen = root / "freeze.json"
            record = create_freeze(PROTOCOL_PATH, frozen)
            record["source_sha256"]["goal_opponents.py"] = "0" * 64
            bad_freeze = root / "bad-freeze.json"
            bad_freeze.write_text(json.dumps(record), encoding="utf-8")
            with patch("multiplayer.goal_opponents._split_scenarios") as gen:
                with self.assertRaises(ValueError):
                    run_pilot("test", freeze_path=bad_freeze)
                gen.assert_not_called()
            protocol = copy.deepcopy(self.protocol)
            protocol["test_deal_seeds"] = [101]
            bad_protocol = root / "bad-protocol.json"
            bad_protocol.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_protocol(bad_protocol)


if __name__ == "__main__":
    unittest.main()
