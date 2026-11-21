"""Small paired four-player pilot, using only the Python standard library.

One all-random game is reused as the baseline for four single-seat policy
replacements per scenario and RNG replicate. Scenarios, not player rows or
paired seat comparisons, are the outer aggregation units. No inference of
statistical significance or real-game playing strength is made.
"""

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .bottleneck import BottleneckPriorityPolicy
from .demo import demo_setup, play
from .game import ROUTE_POINTS, Rules, Ticket, final_scores, new_game, winners
from .policies import (RandomLegalPolicy, ReplanningGreedyPolicy,
                       StaticPlanningPolicy)


DEAL_SEEDS = (101, 202, 303)
REPLICATES = (0, 1)
MASTER_SEED = 163
METRICS = ("win_credit", "margin_to_best_opponent", "score", "completed_count",
           "failed_ticket_points", "sole_win", "shared_win")
TOY_TICKETS = (
    Ticket("AF", "A", "F", 6), Ticket("AD", "A", "D", 8),
    Ticket("AC", "A", "C", 6), Ticket("BD", "B", "D", 5),
    Ticket("BF", "B", "F", 4), Ticket("BE", "B", "E", 3),
    Ticket("CF", "C", "F", 5), Ticket("CE", "C", "E", 3),
    Ticket("DF", "D", "F", 4), Ticket("DE", "D", "E", 4),
    Ticket("AE", "A", "E", 4), Ticket("AB", "A", "B", 4),
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    deal_seed: int
    initial_state: object


def make_scenarios(deal_seeds=DEAL_SEEDS):
    """Fix two distinct synthetic tickets per seat on the existing toy map."""
    if not deal_seeds or len(set(deal_seeds)) != len(deal_seeds):
        raise ValueError("deal seeds must be nonempty and unique")
    scenarios = []
    for seed in deal_seeds:
        if type(seed) is not int or seed < 0:
            raise ValueError("deal seeds must be nonnegative integers")
        deal = random.Random(seed).sample(TOY_TICKETS, 8)
        hands = tuple(tuple(deal[2 * i:2 * i + 2]) for i in range(4))
        initial = new_game(demo_setup().routes, hands,
                           Rules(initial_trains=8), starting_player=0)
        scenarios.append(Scenario(f"toy-{seed}", seed, initial))
    return tuple(scenarios)


def player_seed(master_seed, scenario_id, replicate, seat):
    """Stable separate streams; avoid Python's process-randomized hash()."""
    key = f"{master_seed}|{scenario_id}|{replicate}|{seat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _scenario_record(scenario):
    state = scenario.initial_state
    return {
        "scenario_id": scenario.scenario_id, "deal_seed": scenario.deal_seed,
        "rules": asdict(state.rules), "starting_player": state.current_player,
        "routes": [asdict(route) for route in state.routes],
        "tickets_by_seat": [[asdict(ticket) for ticket in player.tickets]
                            for player in state.players],
    }


def score_records(final):
    """Split a shared victory into 1/k credits instead of full wins."""
    scores, winning_seats = final_scores(final), winners(final)
    records = []
    for score in scores:
        won = score.player_id in winning_seats
        records.append({
            "seat": score.player_id, "score": score.total,
            "route_points": score.route_points,
            "completed_count": len(score.completed),
            "failed_ticket_points": sum(t.points for t in score.failed),
            "margin_to_best_opponent": score.total - max(
                s.total for s in scores if s.player_id != score.player_id),
            "win_credit": 1 / len(winning_seats) if won else 0.0,
            "sole_win": int(won and len(winning_seats) == 1),
            "shared_win": int(won and len(winning_seats) > 1),
            "completed_ticket_ids": _json(
                [t.ticket_id for t in score.completed]),
            "failed_ticket_ids": _json([t.ticket_id for t in score.failed]),
        })
    return records


def _check_final_state(initial, final):
    """Independently check conservation and claimed-route score accounting."""
    if not final.finished:
        raise RuntimeError("an unfinished game must not be scored")
    for seat, player in enumerate(final.players):
        claimed = [r for r, owner in zip(final.routes, final.owners)
                   if owner == seat]
        if player.trains + sum(r.length for r in claimed) != (
                initial.rules.initial_trains):
            raise RuntimeError("train conservation failed")
        if not 0 <= player.resources <= initial.rules.resource_cap:
            raise RuntimeError("resource bounds failed")
        if player.route_points != sum(
                ROUTE_POINTS[r.length] for r in claimed):
            raise RuntimeError("route score accounting failed")


def _run_game(scenario, replicate, master_seed, focal_seat=None,
              focal_policy="replan"):
    if focal_seat is not None and (
            type(focal_seat) is not int or not 0 <= focal_seat < 4):
        raise ValueError("focal_seat must be None or an integer from 0 to 3")
    factories = {"replan": ReplanningGreedyPolicy,
                 "static": StaticPlanningPolicy,
                 "bottleneck": BottleneckPriorityPolicy}
    if focal_policy not in factories:
        raise ValueError("unknown focal policy")
    seeds = tuple(player_seed(
        master_seed, scenario.scenario_id, replicate, seat)
                  for seat in range(4))
    policies = tuple(factories[focal_policy]() if seat == focal_seat
                     else RandomLegalPolicy(seeds[seat]) for seat in range(4))
    initial = scenario.initial_state
    final, events = play(initial, policies)
    _check_final_state(initial, final)
    arm = ("all_random" if focal_seat is None
           else f"{focal_policy}_seat_{focal_seat}")
    game_id = f"{scenario.scenario_id}/r{replicate}/{arm}"
    names = [focal_policy if seat == focal_seat else "random"
             for seat in range(4)]
    game = {
        "game_id": game_id, "scenario_id": scenario.scenario_id,
        "replicate": replicate, "arm": arm,
        "focal_seat": -1 if focal_seat is None else focal_seat,
        "initial_state_sha256": _fingerprint(asdict(initial)),
        "policies_by_seat": _json(names),
        "rng_seeds_by_seat": _json(seeds),
        "turns": final.turn, "end_reason": final.end_reason,
        "winner_seats": _json(winners(final)),
        "final_state_sha256": _fingerprint(asdict(final)),
    }
    players = [{"game_id": game_id, "scenario_id": scenario.scenario_id,
                "replicate": replicate, "policy": names[row["seat"]], **row}
               for row in score_records(final)]
    replay = {"game_id": game_id, "scenario_id": scenario.scenario_id,
              "events": [asdict(event) for event in events]}
    return game, players, replay


def summarize_pairs(pairs):
    """Average seats then RNG replicates within each initial-deal scenario.

    Scenario effect spread is descriptive. No confidence intervals, p-values,
    or standard errors based on correlated seat/game/player rows are emitted.
    """
    if not pairs:
        raise ValueError("at least one complete paired scenario is required")
    seed_rows, scenario_rows, overall = [], [], []
    scenarios = sorted({row["scenario_id"] for row in pairs})
    for scenario_id in scenarios:
        subset = [row for row in pairs if row["scenario_id"] == scenario_id]
        reps = sorted({row["replicate"] for row in subset})
        for replicate in reps:
            cells = [row for row in subset if row["replicate"] == replicate]
            if sorted(row["focal_seat"] for row in cells) != list(range(4)):
                raise ValueError(
                    "each scenario/replicate needs four focal seats")
            for metric in METRICS:
                differences = [row[f"delta_{metric}"] for row in cells]
                seed_rows.append({
                    "scenario_id": scenario_id, "replicate": replicate,
                    "metric": metric, "paired_seats": 4,
                    "baseline_mean": statistics.mean(
                        row[f"baseline_{metric}"] for row in cells),
                    "candidate_mean": statistics.mean(
                        row[f"candidate_{metric}"] for row in cells),
                    "mean_paired_difference": statistics.mean(differences),
                    "paired_increase_rate": sum(
                        d > 1e-12 for d in differences) / 4,
                    "paired_tie_rate": sum(
                        abs(d) <= 1e-12 for d in differences) / 4,
                    "paired_decrease_rate": sum(
                        d < -1e-12 for d in differences) / 4,
                })
        for metric in METRICS:
            blocks = [r for r in seed_rows if r["scenario_id"] == scenario_id
                      and r["metric"] == metric]
            fields = ("baseline_mean", "candidate_mean",
                      "mean_paired_difference",
                      "paired_increase_rate", "paired_tie_rate",
                      "paired_decrease_rate")
            scenario_rows.append({
                "scenario_id": scenario_id, "metric": metric,
                "rng_replicates": len(reps), "paired_seat_cells": len(subset),
                **{f: statistics.mean(r[f] for r in blocks) for f in fields},
            })
    for metric in METRICS:
        blocks = [r for r in scenario_rows if r["metric"] == metric]
        effects = [r["mean_paired_difference"] for r in blocks]
        overall.append({
            "metric": metric, "scenario_count": len(scenarios),
            "paired_seat_cells": len(pairs),
            "baseline_mean": statistics.mean(
                r["baseline_mean"] for r in blocks),
            "candidate_mean": statistics.mean(
                r["candidate_mean"] for r in blocks),
            "mean_paired_difference": statistics.mean(effects),
            "min_scenario_difference": min(effects),
            "max_scenario_difference": max(effects),
            "scenario_difference_sd": statistics.stdev(effects)
            if len(effects) > 1 else None,
            "paired_increase_rate": statistics.mean(
                r["paired_increase_rate"] for r in blocks),
            "paired_tie_rate": statistics.mean(
                r["paired_tie_rate"] for r in blocks),
            "paired_decrease_rate": statistics.mean(
                r["paired_decrease_rate"] for r in blocks),
        })
    return seed_rows, scenario_rows, overall


def validate_design(scenarios, replicates, master_seed):
    """Shared setup checks for the historical and planning comparisons."""
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    ids = {s.scenario_id for s in scenarios}
    if not scenarios or len(ids) != len(scenarios):
        raise ValueError("scenarios must be nonempty with unique identifiers")
    if (not replicates or len(set(replicates)) != len(replicates)
            or any(type(r) is not int or r < 0 for r in replicates)):
        raise ValueError("replicates must be unique nonnegative integers")
    for scenario in scenarios:
        state = scenario.initial_state
        wrong_hand = any(len(p.tickets) != 2 for p in state.players)
        if state.rules.player_count != 4 or wrong_hand:
            raise ValueError(
                "pilot requires four players with two tickets each")
        fresh = new_game(state.routes, tuple(p.tickets for p in state.players),
                         state.rules, starting_player=state.current_player)
        if fresh != state:
            raise ValueError("scenarios must be fresh initial states")


def paired_record(scenario, replicate, seat, before, after):
    """Compare the same focal seat; positive delta means a numeric increase."""
    if type(seat) is not int or not 0 <= seat < 4:
        raise ValueError("paired seat must be an integer from 0 to 3")
    expected = (scenario.scenario_id, replicate, seat)
    for row in (before, after):
        actual = (row.get("scenario_id"), row.get("replicate"),
                  row.get("seat"))
        if actual != expected:
            raise ValueError(
                "paired rows must match scenario, replicate, seat")
    pair = {
        "scenario_id": scenario.scenario_id,
        "replicate": replicate,
        "focal_seat": seat,
        "turn_position": (seat - scenario.initial_state.current_player) % 4,
        "baseline_game_id": before["game_id"],
        "candidate_game_id": after["game_id"],
    }
    for metric in METRICS:
        pair[f"baseline_{metric}"] = before[metric]
        pair[f"candidate_{metric}"] = after[metric]
        pair[f"delta_{metric}"] = after[metric] - before[metric]
    return pair


def run_pilot(scenarios=None, replicates=REPLICATES, master_seed=MASTER_SEED):
    """Run 5 games per scenario/replicate, preserving the immutable setup."""
    scenarios = make_scenarios() if scenarios is None else tuple(scenarios)
    replicates = tuple(replicates)
    validate_design(scenarios, replicates, master_seed)
    games, players, replays, pairs = [], [], [], []
    for scenario in scenarios:
        for replicate in replicates:
            game, baseline, replay = _run_game(
                scenario, replicate, master_seed)
            games.append(game)
            players.extend(baseline)
            replays.append(replay)
            for seat in range(4):
                game, candidate, replay = _run_game(
                    scenario, replicate, master_seed, focal_seat=seat)
                games.append(game)
                players.extend(candidate)
                replays.append(replay)
                pairs.append(paired_record(
                    scenario, replicate, seat,
                    baseline[seat], candidate[seat]))
    seed_rows, scenario_rows, summary = summarize_pairs(pairs)
    return {
        "games": games, "players": players, "pairs": pairs, "replays": replays,
        "seed_summary": seed_rows, "scenario_summary": scenario_rows,
        "summary": summary,
        "scenarios": [_scenario_record(s) for s in scenarios],
        "protocol": {
            "master_seed": master_seed, "replicates": list(replicates),
            "outer_unit": "initial-deal scenario on the fixed synthetic map",
            "paired_unit": "scenario x RNG replicate x focal seat",
            "baseline_reuse": "one all-random game supplies all four seats",
            "opponents": "three separately seeded RandomLegalPolicy instances",
            "candidate": "ReplanningGreedyPolicy, replacing one seat only",
            "aggregation": "equal seats, replicates, then scenarios",
            "tie_rule": "split victory 1/k after score and ticket tiebreak",
            "uncertainty": "scenario spread only; no significance test",
            "scope": "development pilot; no real-game or held-out claim",
        },
    }


def write_report(report, output_directory):
    """Create a new result directory; never replace an existing directory."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    for table in ("games", "players", "pairs", "seed_summary",
                  "scenario_summary", "summary"):
        rows = report[table]
        destination = directory / f"{table}.csv"
        with destination.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (directory / "scenarios.json").write_text(
        json.dumps(report["scenarios"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with (directory / "replays.jsonl").open("w", encoding="utf-8") as f:
        for replay in report["replays"]:
            f.write(_json(replay) + "\n")
    source = Path(__file__).resolve().parent
    source_names = (("game.py", "demo.py", "policies.py", "evaluate.py",
                     "bottleneck.py")
                    + tuple(report.get("extra_source_files", ())))
    manifest = {
        **report["protocol"], "python_version": platform.python_version(),
        "unique_games": len(report["games"]),
        "paired_cells": len(report["pairs"]),
        "scenario_count": len(report["scenarios"]),
        "scenario_sha256": _fingerprint(report["scenarios"]),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in source_names},
    }
    manifest["result_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir()) if p.is_file()}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def main():
    default = (Path(__file__).resolve().parent.parent
               / "outputs/multiplayer_pilot_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default,
                        help="New directory; existing paths are refused")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new --output directory")
    report = run_pilot()
    write_report(report, args.output)
    print(f"Pilot complete: {len(report['games'])} unique games, "
          f"{len(report['pairs'])} paired seat comparisons; "
          f"{len(report['scenarios'])} initial-deal scenarios.")
    print("Descriptive toy-model results only; no significance claim.")
    for row in report["summary"][:3]:
        print(f"{row['metric']}: paired mean difference "
              f"{row['mean_paired_difference']:.6f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
