"""Controlled small pilot: dynamic shortest-edge versus bottleneck priority.

Use the unchanged development map, deals, seeds, seats and opponent policy.
Constructed teaching counterexamples are deliberately outside this sample.
Only selected-path claim order changes; this is not a significance study.
"""

import argparse
from pathlib import Path

from .evaluate import (MASTER_SEED, REPLICATES, _run_game, _scenario_record,
                       make_scenarios, paired_record, summarize_pairs,
                       validate_design, write_report)


def run_comparison(scenarios=None, replicates=REPLICATES,
                   master_seed=MASTER_SEED):
    scenarios = make_scenarios() if scenarios is None else tuple(scenarios)
    replicates = tuple(replicates)
    validate_design(scenarios, replicates, master_seed)
    games, players, replays, pairs = [], [], [], []
    for scenario in scenarios:
        for replicate in replicates:
            for seat in range(4):
                focal_results = {}
                for policy in ("replan", "bottleneck"):
                    game, rows, replay = _run_game(
                        scenario, replicate, master_seed, seat, policy)
                    games.append(game)
                    players.extend(rows)
                    replays.append(replay)
                    focal_results[policy] = rows[seat]
                pairs.append(paired_record(
                    scenario, replicate, seat,
                    focal_results["replan"], focal_results["bottleneck"]))
    seed_rows, scenario_rows, summary = summarize_pairs(pairs)
    return {
        "games": games, "players": players, "pairs": pairs, "replays": replays,
        "seed_summary": seed_rows, "scenario_summary": scenario_rows,
        "summary": summary,
        "scenarios": [_scenario_record(s) for s in scenarios],
        "extra_source_files": ("compare_bottleneck.py",),
        "protocol": {
            "master_seed": master_seed, "replicates": list(replicates),
            "outer_unit": "initial-deal scenario on the fixed synthetic map",
            "paired_unit": "scenario x RNG replicate x focal seat",
            "baseline": "ReplanningGreedyPolicy, shortest affordable edge",
            "candidate": "BottleneckPriorityPolicy, own-ticket "
                         "deletion impact",
            "changed_factor": "claim ordering among the same affordable "
                              "edges on the same selected goal path",
            "shared_rules": "dynamic goal ranking, path search, gather, "
                            "fallback, final turn and deterministic tiebreak",
            "impact": "delete one available physical edge; first 2*points "
                      "made unreachable/over-budget, then 2*points*detour "
                      "for still-feasible own unfinished tickets",
            "impact_limits": "single-edge consequence, not opponent "
                             "probability, expected loss, "
                             "or joint network cost",
            "baseline_reuse": "separate dynamic game per paired focal seat",
            "opponents": "three separately seeded RandomLegalPolicy instances",
            "rng_pairing": "same opponent streams, not forced same actions",
            "policy_lifetime": "fresh instance per seat per game",
            "aggregation": "equal seats, replicates, then scenarios",
            "tie_rule": "split victory 1/k after score and ticket tiebreak",
            "uncertainty": "scenario spread only; no significance test",
            "scope": "unchanged development scenarios; excludes constructed "
                     "lessons; no held-out, real-game or runtime claim",
        },
    }


def main():
    default = (Path(__file__).resolve().parent.parent
               / "outputs/multiplayer_bottleneck_vs_dynamic_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default,
                        help="New directory; existing paths are refused")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new --output directory")
    report = run_comparison()
    write_report(report, args.output)
    print(f"Bottleneck vs dynamic: {len(report['games'])} games, "
          f"{len(report['pairs'])} pairs, "
          f"{len(report['scenarios'])} scenarios.")
    print("Delta = bottleneck - dynamic; descriptive development results.")
    for row in report["summary"][:3]:
        print(f"{row['metric']}: dynamic={row['baseline_mean']:.6f}; "
              f"bottleneck={row['candidate_mean']:.6f}; "
              f"delta={row['mean_paired_difference']:.6f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
