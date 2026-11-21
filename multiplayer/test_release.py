"""Release regression tests use development seeds and the observed endgame only."""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as F
import json
import unittest
from unittest.mock import patch

from .competition_experiment import (aggregate, fingerprint, hand_key,
                                     overlap_audit, run as competition_run,
                                     run_game, seat_seeds)
from .endgame import evaluate_actions
from .endgame_experiment import load_case
from .evaluate import METRICS, _scenario_record, make_scenarios
from .game import Action, Observation, legal_actions, observe, step
from .mc_experiment import make_plan, run as mc_run, sample_once
from .policies import RandomLegalPolicy, ReplanningGreedyPolicy
from .release_common import bootstrap_mean, percentile, stable_seed, wilson
from .release_verify import replay


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.worlds, _ = load_case()
        cls.exact = evaluate_actions(cls.worlds)
        cls.small = mc_run(repeats=4, budgets=(1, 4), depths=(0, 3), bootstrap=20)

    def test_exact_stratified_outcomes_agree_with_tree(self):
        class IndexRng:
            def __init__(self, index):
                self.index = index
                self.sizes = []

            def randrange(self, n):
                self.sizes.append(n)
                return self.index

        for estimate in self.exact:
            result = F(0)
            for world in self.worlds:
                rng = IndexRng(0)
                sample_once(world, estimate.action, 3, 0, rng=rng)
                self.assertEqual(len(rng.sizes), 1)
                count = rng.sizes[0]
                conditional = sum(sample_once(
                    world, estimate.action, 3, 0, rng=IndexRng(i))[0]
                    for i in range(count)) / count
                result += world.probability * conditional
            self.assertEqual(result, estimate.value)

    def test_sample_plan_is_nested_reproducible_and_world_order_invariant(self):
        short = make_plan(self.worlds, 3, 23, 0)
        self.assertEqual(short, make_plan(self.worlds, 6, 23, 0)[:3])
        self.assertEqual(short, make_plan(tuple(reversed(self.worlds)), 3, 23, 0))
        self.assertNotEqual(short, make_plan(self.worlds, 3, 23, 1))
        self.assertEqual({p["world"] for p in make_plan(self.worlds, 128, 23, 0)},
                         {w.name for w in self.worlds})

    def test_actual_transitions_and_terminal_only_at_full_depth(self):
        for world in self.worlds:
            for estimate in self.exact:
                for h in range(4):
                    _, count, final, events = sample_once(world, estimate.action, h, 17)
                    self.assertEqual(count, h + 1)
                    self.assertEqual(len(events), count)
                    self.assertEqual(final.finished, h == 3)

    def test_error_decomposition_and_bound(self):
        tables, _, _ = self.small
        for row in tables["values"]:
            self.assertEqual(F(row["total_error"]),
                             F(row["sampling_error"]) + F(row["truncation_error"]))
            self.assertGreaterEqual(F(row["float_rounding_error"]), 0)
        for row in tables["decisions"]:
            self.assertTrue(row["generic_bound_holds"])
            self.assertEqual(row["transitions"], 5 * row["samples"] * (row["depth"] + 1))

    def test_sampling_cannot_fix_depth_zero_tie(self):
        for row in self.small[0]["decisions"]:
            if row["depth"] == 0:
                self.assertEqual(row["selected_action"], "claim AB")
                self.assertEqual(row["regret"], "1/8")
                self.assertEqual(row["depth_selection_error"], 0)
                self.assertEqual(row["terminal_selection_error"], 1)

    def test_sampling_repeat_is_bit_reproducible(self):
        again = mc_run(repeats=4, budgets=(1, 4), depths=(0, 3), bootstrap=20)
        self.assertEqual(self.small, again)

    def test_uniform_utility_support_matches_exact_leaves(self):
        for estimate in self.exact:
            for world in self.worlds:
                support = {leaf.value for leaf in estimate.frontier
                           if leaf.world == world.name}
                for seed in range(10):
                    self.assertIn(sample_once(world, estimate.action, 3, seed)[0],
                                  support)

    def test_invalid_sampling_design(self):
        for kwargs in ({"repeats": 0}, {"budgets": (4, 1)},
                       {"budgets": (1, 1)}, {"depths": (4,)}):
            with self.assertRaises(ValueError):
                mc_run(**kwargs)


class CompetitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = make_scenarios((101,))[0]
        cls.small = competition_run(deal_seeds=(101,), starts=(0, 1),
                                    replicates=(0, 1), bootstrap=20)

    def test_unique_game_count_and_reused_cells(self):
        tables, documents, _ = self.small
        self.assertEqual(len(tables["games"]), 62)
        self.assertEqual(len(tables["players"]), 248)
        self.assertEqual(len(tables["pairs"]), 64)
        self.assertEqual(len({g["game_id"] for g in tables["games"]}), 62)
        self.assertEqual(documents["parameters"]["deal_blocks"], 1)
        self.assertEqual(len(tables["deal_effects"]), 4 * len(METRICS))

    def test_exact_replay_scores_and_state_hashes(self):
        tables, _, streams = self.small
        games = {g["game_id"]: g for g in tables["games"]}
        for row in streams["replays"]:
            initial, final = replay(_scenario_record(self.scenario),
                                    row["starting_player"], row["events"])
            game = games[row["game_id"]]
            self.assertEqual(fingerprint(asdict(initial)), game["initial_state_sha256"])
            self.assertEqual(fingerprint(asdict(final)), game["final_state_sha256"])
            self.assertEqual(final.turn, game["turns"])

    def test_replay_detects_tampering(self):
        row = deepcopy(self.small[2]["replays"][0])
        row["events"][0]["resources_after"] += 1
        with self.assertRaises(ValueError):
            replay(_scenario_record(self.scenario), row["starting_player"], row["events"])

    def test_seed_pairing_by_seat_and_start(self):
        a = seat_seeds(5, 101, 0, 0)
        self.assertEqual(a, seat_seeds(5, 101, 0, 0))
        self.assertEqual(len(set(a)), 4)
        self.assertNotEqual(a, seat_seeds(5, 101, 1, 0))
        games = {g["game_id"]: g for g in self.small[0]["games"]}
        for pair in self.small[0]["pairs"]:
            for arm in ("baseline", "candidate"):
                game = games[pair[arm + "_game_id"]]
                actual = json.loads(game["rng_seeds_by_seat"])
                expected = json.loads(pair["paired_rng_seeds"])
                self.assertEqual(game["starting_player"], pair["starting_player"])
                for seat, seed in enumerate(actual):
                    if seed is not None:
                        self.assertEqual(seed, expected[seat])

    def test_whole_deal_aggregation_and_single_deal_degenerate_interval(self):
        for row in self.small[0]["summary"]:
            self.assertEqual(row["deal_blocks"], 1)
            self.assertEqual(row["paired_cells"], 16)
            self.assertEqual(row["increases"] + row["ties"] + row["decreases"], 16)
            self.assertEqual(row["bootstrap_low"], row["mean_difference"])
            self.assertEqual(row["bootstrap_high"], row["mean_difference"])

    def test_incomplete_block_rejected(self):
        pairs = self.small[0]["pairs"]
        with self.assertRaises(ValueError):
            aggregate(pairs[1:], bootstrap=2, master=1)
        with self.assertRaises(ValueError):
            aggregate(pairs + pairs[:1], bootstrap=2, master=1)

    def test_overlap_audit_ignores_seat_permutation(self):
        scenario = self.scenario
        state = scenario.initial_state
        permuted = replace(scenario, scenario_id="permuted", initial_state=replace(
            state, players=tuple(reversed(state.players))))
        self.assertEqual(hand_key(permuted), hand_key(scenario))
        audit = overlap_audit((scenario, permuted), (scenario,))
        self.assertEqual(len(audit["observed_overlap"]), 2)
        self.assertEqual(len(audit["within_evaluation_duplicates"]), 1)

    def test_policies_only_receive_observations(self):
        seen = []
        original = ReplanningGreedyPolicy.choose

        def spy(policy, view):
            self.assertIsInstance(view, Observation)
            self.assertFalse(any(hasattr(p, "tickets") for p in view.players))
            seen.append(view)
            return original(policy, view)

        with patch.object(ReplanningGreedyPolicy, "choose", spy):
            run_game(self.scenario, 0, ("replan",) * 4, (0,) * 4)
        self.assertTrue(seen)

    def test_deterministic_policies_ignore_rng_seed_and_run_repeats(self):
        a = run_game(self.scenario, 0, ("replan",) * 4, (0,) * 4)
        b = run_game(self.scenario, 0, ("replan",) * 4, (99,) * 4)
        self.assertEqual(a, b)
        again = competition_run(deal_seeds=(101,), starts=(0, 1),
                                replicates=(0, 1), bootstrap=20)
        self.assertEqual(self.small, again)

    def test_random_policy_remains_legal_after_opponent_blocks(self):
        state, _ = step(self.scenario.initial_state, 0, Action("claim", "AB1"))
        view = observe(state, 1)
        self.assertNotIn(Action("claim", "AB1"), legal_actions(view))
        self.assertIn(Action("claim", "AB2"), legal_actions(view))
        for seed in range(25):
            self.assertIn(RandomLegalPolicy(seed).choose(view), legal_actions(view))


class UncertaintyTests(unittest.TestCase):
    def test_wilson_known_boundaries_and_half(self):
        self.assertAlmostEqual(wilson(0, 10)[1], 0.2775327998628892)
        self.assertAlmostEqual(wilson(10, 10)[0], 0.7224672001371107)
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(lo + hi, 1)
        with self.assertRaises(ValueError):
            wilson(0, 0)

    def test_percentile_interpolation_and_bootstrap_reproducibility(self):
        self.assertEqual(percentile([0, 10], .25), 2.5)
        self.assertEqual(bootstrap_mean([2] * 5, 20, 1), (2, 2))
        a = bootstrap_mean([-1, 0, 1], 100, 3)
        self.assertEqual(a, bootstrap_mean([-1, 0, 1], 100, 3))
        self.assertLessEqual(a[0], 0)
        self.assertGreaterEqual(a[1], 0)

    def test_seed_components_are_unambiguous(self):
        self.assertNotEqual(stable_seed("1", "23"), stable_seed("12", "3"))
        self.assertNotEqual(stable_seed(1), stable_seed("1"))


if __name__ == "__main__":
    unittest.main()
