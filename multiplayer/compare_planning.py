"""Paired static-with-repair versus dynamic planning on the four-player toy.

Only the focal policy changes. Goal ranking, resources, final-turn rules,
initial hands, seats and opponent RNG streams are shared. This is a small
development comparison, not evidence of statistical or real-game strength.
"""

import argparse
from pathlib import Path

from .evaluate import (MASTER_SEED, REPLICATES, _run_game, _scenario_record,
                       make_scenarios, paired_record, summarize_pairs,
                       validate_design, write_report)


def run_comparison(scenarios=None, replicates=REPLICATES,
                   master_seed=MASTER_SEED):
    """Run two games per focal seat, with a fresh policy in every game."""
    scenarios = make_scenarios() if scenarios is None else tuple(scenarios)
    replicates = tuple(replicates)
    validate_design(scenarios, replicates, master_seed)
    games, players, replays, pairs = [], [], [], []
    for scenario in scenarios:
        for replicate in replicates:
            for seat in range(4):
                focal_results = {}
                for policy in ("static", "replan"):
                    game, rows, replay = _run_game(
                        scenario, replicate, master_seed, seat, policy)
                    games.append(game)
                    players.extend(rows)
                    replays.append(replay)
                    focal_results[policy] = rows[seat]
                pairs.append(paired_record(
                    scenario, replicate, seat,
                    focal_results["static"], focal_results["replan"]))
    seed_rows, scenario_rows, summary = summarize_pairs(pairs)
    return {
        "games": games, "players": players, "pairs": pairs, "replays": replays,
        "seed_summary": seed_rows, "scenario_summary": scenario_rows,
        "summary": summary,
        "scenarios": [_scenario_record(s) for s in scenarios],
        "extra_source_files": ("compare_planning.py",),
        "protocol": {
            "master_seed": master_seed, "replicates": list(replicates),
            "outer_unit": "initial-deal scenario on the fixed synthetic map",
            "paired_unit": "scenario x RNG replicate x focal seat",
            "baseline": "StaticPlanningPolicy, retaining one goal/path",
            "candidate": "ReplanningGreedyPolicy, recomputing each turn",
            "repair": "on completion, blocking, or remaining-train failure; "
                      "rerank all own unfinished feasible goals",
            "shared_rules": "same goal ranking, route choice, gather, "
                            "fallback and final-turn override",
            "baseline_reuse": "separate static game per paired focal seat",
            "opponents": "three separately seeded RandomLegalPolicy instances",
            "rng_pairing": "same opponent streams, not forced same actions",
            "policy_lifetime": "fresh instance per seat per game",
            "aggregation": "equal seats, replicates, then scenarios",
            "tie_rule": "split victory 1/k after score and ticket tiebreak",
            "uncertainty": "scenario spread only; no significance test",
            "scope": "same development scenarios as earlier pilot; "
                     "no held-out or real-game claim",
        },
    }


def main():
    default = (Path(__file__).resolve().parent.parent
               / "outputs/multiplayer_static_vs_dynamic_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default,
                        help="New directory; existing paths are refused")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new --output directory")
    report = run_comparison()
    write_report(report, args.output)
    print(f"Static vs dynamic: {len(report['games'])} unique games, "
          f"{len(report['pairs'])} paired seat comparisons; "
          f"{len(report['scenarios'])} initial-deal scenarios.")
    print("Delta = dynamic - static. Descriptive toy-model results only.")
    for row in report["summary"][:3]:
        print(f"{row['metric']}: static={row['baseline_mean']:.6f}; "
              f"dynamic={row['candidate_mean']:.6f}; "
              f"delta={row['mean_paired_difference']:.6f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
