"""Fresh-deal exploratory pilot against three deterministic dynamic players.

Use development deals for tests, freeze the evaluator before the test run,
and keep starting-player rotations separate from stochastic repetitions.
No strategy implementation or game rule is changed by this module.
"""

import argparse
import csv
import hashlib
import json
import platform
import statistics
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from .bottleneck import BottleneckPriorityPolicy
from .demo import play
from .evaluate import (METRICS, _check_final_state, _fingerprint, _json,
                       _scenario_record, make_scenarios, score_records)
from .game import new_game, winners
from .policies import ReplanningGreedyPolicy


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "multiplayer/protocols/goal_opponents_pilot_v1.json"
FREEZE_PATH = ROOT / "validation/goal_opponents_pilot_v1_freeze.json"
FIELDS = ("baseline_mean", "candidate_mean", "mean_paired_difference",
          "paired_increase_rate", "paired_tie_rate", "paired_decrease_rate")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path=PROTOCOL_PATH):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    pools = []
    for name in ("development_deal_seeds", "test_deal_seeds",
                 "reserved_not_evaluated_deal_seeds"):
        seeds = protocol[name]
        if (not isinstance(seeds, list) or not seeds
                or any(type(s) is not int or s < 0 for s in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError(
                f"{name} must contain unique nonnegative integers")
        pools.append(set(seeds))
    if any(pools[i] & pools[j] for i in range(3) for j in range(i)):
        raise ValueError(
            "development, test and reserved seeds must be disjoint")
    if (protocol["starting_players"] != list(range(4))
            or protocol["focal_seats"] != list(range(4))):
        raise ValueError("this pilot requires all four starts and focal seats")
    expected = ("replan", "bottleneck", "replan")
    actual = tuple(protocol[k] for k in
                   ("baseline_policy", "candidate_policy", "opponent_policy"))
    if actual != expected:
        raise ValueError("this controlled pilot fixes all policy roles")
    for name, digest in protocol["frozen_preexisting_sources_sha256"].items():
        if _sha(ROOT / "multiplayer" / name) != digest:
            raise ValueError(f"pre-existing source changed: {name}")
    return protocol


def source_hashes(protocol):
    names = (*protocol["frozen_preexisting_sources_sha256"],
             "goal_opponents.py")
    return {name: _sha(ROOT / "multiplayer" / name) for name in names}


def create_freeze(protocol_path=PROTOCOL_PATH, freeze_path=FREEZE_PATH):
    """Record source/runtime identity without generating any test deals."""
    protocol = load_protocol(protocol_path)
    record = {
        "protocol_id": protocol["protocol_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "protocol_sha256": _sha(protocol_path),
        "source_sha256": source_hashes(protocol),
        "purpose": "freeze before fresh-deal exploratory test; no inference "
                   "of confirmatory validity from this timestamp",
    }
    with Path(freeze_path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def verify_freeze(protocol_path=PROTOCOL_PATH, freeze_path=FREEZE_PATH):
    protocol = load_protocol(protocol_path)
    record = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if (record["protocol_id"] != protocol["protocol_id"]
            or record["protocol_sha256"] != _sha(protocol_path)
            or record["source_sha256"] != source_hashes(protocol)
            or record["python_version"] != platform.python_version()):
        raise ValueError(
            "protocol, source or Python version differs from freeze")
    return record


def hand_partition_key(scenario):
    """Detect reused hands after changing seats or within-hand card order."""
    hands = sorted(tuple(sorted(t.ticket_id for t in p.tickets))
                   for p in scenario.initial_state.players)
    return _fingerprint(hands)


def assert_split_disjoint(development, test):
    """Predeclared novelty check; abort rather than select by game outcomes."""
    dev_keys = {hand_partition_key(s) for s in development}
    test_keys = [hand_partition_key(s) for s in test]
    if dev_keys.intersection(test_keys):
        raise ValueError("test hand partition repeats a development deal")
    if len(set(test_keys)) != len(test_keys):
        raise ValueError(
            "test hand partitions repeat; review the frozen design")


def _split_scenarios(protocol, split):
    development = make_scenarios(tuple(protocol["development_deal_seeds"]))
    if split == "development":
        cases = development
    elif split == "test":
        cases = make_scenarios(tuple(protocol["test_deal_seeds"]))
        assert_split_disjoint(development, cases)
    else:
        raise ValueError("split must be development or test")
    cases = tuple(replace(s, scenario_id=f"{split}-{s.deal_seed}")
                  for s in cases)
    audit = {
        "split": split, "development_deal_seeds":
        protocol["development_deal_seeds"],
        "evaluated_deal_seeds": [s.deal_seed for s in cases],
        "development_hand_partition_sha256":
        [hand_partition_key(s) for s in development],
        "evaluated_hand_partition_sha256":
        [hand_partition_key(s) for s in cases],
        "comparison_ignores": "seat permutation and within-hand card order",
        "disjoint_check_performed": split == "test",
        "reserved_deals_generated": False,
    }
    return cases, audit


def rotated_scenario(scenario, starting_player):
    if type(starting_player) is not int or not 0 <= starting_player < 4:
        raise ValueError("starting player must be an integer from 0 to 3")
    initial = scenario.initial_state
    rotated = new_game(initial.routes,
                       tuple(p.tickets for p in initial.players),
                       initial.rules, starting_player=starting_player)
    return replace(scenario, initial_state=rotated)


def _play_case(scenario, starting_player, split, focal_seat=None):
    if focal_seat is not None and (
            type(focal_seat) is not int or not 0 <= focal_seat < 4):
        raise ValueError("invalid focal seat")
    initial = rotated_scenario(scenario, starting_player).initial_state
    policies = tuple(BottleneckPriorityPolicy() if seat == focal_seat
                     else ReplanningGreedyPolicy() for seat in range(4))
    names = ["bottleneck" if seat == focal_seat else "replan"
             for seat in range(4)]
    final, events = play(initial, policies)
    _check_final_state(initial, final)
    arm = ("all_replan" if focal_seat is None
           else f"bottleneck_seat_{focal_seat}")
    game_id = f"{scenario.scenario_id}/start{starting_player}/{arm}"
    context = {"game_id": game_id, "scenario_id": scenario.scenario_id,
               "split": split, "starting_player": starting_player}
    game = {
        **context, "deal_seed": scenario.deal_seed, "arm": arm,
        "focal_seat": -1 if focal_seat is None else focal_seat,
        "policies_by_seat": _json(names),
        "initial_state_sha256": _fingerprint(asdict(initial)),
        "turns": final.turn, "end_reason": final.end_reason,
        "winner_seats": _json(winners(final)),
        "final_state_sha256": _fingerprint(asdict(final)),
    }
    players = [{**context, "policy": names[r["seat"]], **r}
               for r in score_records(final)]
    replay = {**context, "events": [asdict(event) for event in events]}
    return game, players, replay


def _pair(scenario_id, start, seat, split, before, after):
    expected = (scenario_id, start, seat, split)
    for row in (before, after):
        actual = tuple(row[k] for k in
                       ("scenario_id", "starting_player", "seat", "split"))
        if actual != expected:
            raise ValueError(
                "pairing must preserve deal, start, seat and split")
    pair = {"scenario_id": scenario_id, "split": split,
            "starting_player": start, "focal_seat": seat,
            "turn_position": (seat - start) % 4,
            "baseline_game_id": before["game_id"],
            "candidate_game_id": after["game_id"]}
    for metric in METRICS:
        pair[f"baseline_{metric}"] = before[metric]
        pair[f"candidate_{metric}"] = after[metric]
        pair[f"delta_{metric}"] = after[metric] - before[metric]
    return pair


def _cell_means(cells, metric):
    differences = [r[f"delta_{metric}"] for r in cells]
    return {
        "baseline_mean": statistics.mean(
            r[f"baseline_{metric}"] for r in cells),
        "candidate_mean": statistics.mean(
            r[f"candidate_{metric}"] for r in cells),
        "mean_paired_difference": statistics.mean(differences),
        "paired_increase_rate": sum(
            d > 1e-12 for d in differences) / len(cells),
        "paired_tie_rate": sum(
            abs(d) <= 1e-12 for d in differences) / len(cells),
        "paired_decrease_rate": sum(
            d < -1e-12 for d in differences) / len(cells),
    }


def summarize_rotations(pairs):
    """A rotation is a fixed design factor, never an RNG replicate."""
    if not pairs or len({r["split"] for r in pairs}) != 1:
        raise ValueError("summarize one nonempty split at a time")
    rotations, scenarios, overall = [], [], []
    split = pairs[0]["split"]
    for scenario_id in sorted({r["scenario_id"] for r in pairs}):
        subset = [r for r in pairs if r["scenario_id"] == scenario_id]
        if {r["starting_player"] for r in subset} != set(range(4)):
            raise ValueError("each deal requires all four starting players")
        for start in range(4):
            cells = [r for r in subset if r["starting_player"] == start]
            if sorted(r["focal_seat"] for r in cells) != list(range(4)):
                raise ValueError(
                    "each starting player requires four focal seats")
            if any(r["turn_position"] != (r["focal_seat"] - start) % 4
                   for r in cells):
                raise ValueError("incorrect relative turn position")
            for metric in METRICS:
                rotations.append({
                    "scenario_id": scenario_id, "split": split,
                    "starting_player": start, "metric": metric,
                    "paired_seat_cells": 4, **_cell_means(cells, metric)})
        for metric in METRICS:
            blocks = [r for r in rotations if r["scenario_id"] == scenario_id
                      and r["metric"] == metric]
            scenarios.append({
                "scenario_id": scenario_id, "split": split, "metric": metric,
                "starting_player_count": 4, "paired_seat_cells": 16,
                **{f: statistics.mean(r[f] for r in blocks) for f in FIELDS}})
    for metric in METRICS:
        blocks = [r for r in scenarios if r["metric"] == metric]
        effects = [r["mean_paired_difference"] for r in blocks]
        overall.append({
            "split": split, "metric": metric, "scenario_count": len(blocks),
            "paired_seat_cells": len(pairs),
            **{f: statistics.mean(r[f] for r in blocks) for f in FIELDS},
            "min_scenario_difference": min(effects),
            "max_scenario_difference": max(effects),
            "scenario_difference_sd": statistics.stdev(effects)
            if len(effects) > 1 else None})
    return rotations, scenarios, overall


def run_pilot(split="development", protocol_path=PROTOCOL_PATH,
              freeze_path=None):
    protocol = load_protocol(protocol_path)
    if split == "test":
        if freeze_path is None:
            raise ValueError("test run requires a verified pre-run freeze")
        verify_freeze(protocol_path, freeze_path)
    elif split != "development":
        raise ValueError("unknown split")
    scenarios, split_audit = _split_scenarios(protocol, split)
    games, players, replays, pairs = [], [], [], []
    for scenario in scenarios:
        for start in protocol["starting_players"]:
            game, baseline, replay = _play_case(scenario, start, split)
            games.append(game)
            players.extend(baseline)
            replays.append(replay)
            for seat in protocol["focal_seats"]:
                game, candidate, replay = _play_case(
                    scenario, start, split, seat)
                games.append(game)
                players.extend(candidate)
                replays.append(replay)
                pairs.append(_pair(scenario.scenario_id, start, seat, split,
                                   baseline[seat], candidate[seat]))
    rotations, scenario_rows, summary = summarize_rotations(pairs)
    return {
        "split": split, "games": games, "players": players, "pairs": pairs,
        "replays": replays, "rotation_summary": rotations,
        "scenario_summary": scenario_rows, "summary": summary,
        "scenarios": [{**_scenario_record(s), "split": split,
                       "evaluated_starting_players":
                       protocol["starting_players"]}
                      for s in scenarios],
        "split_audit": split_audit, "protocol_sha256": _sha(protocol_path),
        "source_sha256": source_hashes(protocol),
    }


def write_report(report, directory, protocol_path=PROTOCOL_PATH,
                 freeze_path=FREEZE_PATH):
    freeze = verify_freeze(protocol_path, freeze_path)
    if (report["source_sha256"] != freeze["source_sha256"]
            or report["protocol_sha256"] != freeze["protocol_sha256"]):
        raise ValueError("report does not match the frozen implementation")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for table in ("games", "players", "pairs", "rotation_summary",
                  "scenario_summary", "summary"):
        rows = report[table]
        with (directory / f"{table}.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name in ("scenarios", "split_audit"):
        (directory / f"{name}.json").write_text(
            json.dumps(report[name], indent=2) + "\n", encoding="utf-8")
    with (directory / "replays.jsonl").open("w", encoding="utf-8") as stream:
        for replay in report["replays"]:
            stream.write(_json(replay) + "\n")
    (directory / "protocol.json").write_bytes(
        Path(protocol_path).read_bytes())
    (directory / "freeze.json").write_bytes(Path(freeze_path).read_bytes())
    manifest = {
        "protocol_id": freeze["protocol_id"], "split": report["split"],
        "python_version": freeze["python_version"],
        "unique_games": len(report["games"]),
        "paired_cells": len(report["pairs"]),
        "outer_deal_count": len(report["scenarios"]),
        "policy_randomness": "none; deterministic policies, no RNG replicates",
        "aggregation": "equal focal seats, starting players, then deals",
        "interpretation": "exploratory fresh-deal pilot, not confirmation",
        "test_cohort_status_after_run": "observed; not fresh for future tuning"
        if report["split"] == "test" else "test not evaluated by this run",
        "source_sha256": freeze["source_sha256"],
        "protocol_sha256": freeze["protocol_sha256"],
        "result_sha256": {p.name: _sha(p)
                          for p in sorted(directory.iterdir())},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "test"),
                        default="test")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--freeze-file", type=Path, default=FREEZE_PATH)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze_only:
        create_freeze(args.protocol, args.freeze_file)
        print("Freeze created; no test deals generated or games run.")
        return
    output = args.output or ROOT / (
        f"outputs/multiplayer_goal_opponents_{args.split}_v1")
    if output.exists():
        parser.error("output already exists; choose a new directory")
    verify_freeze(args.protocol, args.freeze_file)
    report = run_pilot(args.split, args.protocol, args.freeze_file)
    write_report(report, output, args.protocol, args.freeze_file)
    print(f"{args.split}: {len(report['scenarios'])} deals, "
          f"{len(report['games'])} games, "
          f"{len(report['pairs'])} paired cells.")
    print("Three dynamic opponents; delta = bottleneck - dynamic.")
    for row in report["summary"][:3]:
        print(f"{row['metric']}: baseline={row['baseline_mean']:.6f}; "
              f"candidate={row['candidate_mean']:.6f}; "
              f"delta={row['mean_paired_difference']:.6f}")
    print("Exploratory pilot only; rotations are not independent deals.")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
