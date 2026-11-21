"""Frozen, paired four-player evaluation with full replay and deal blocks."""

import argparse
from dataclasses import asdict
from pathlib import Path
import statistics

from .bottleneck import BottleneckPriorityPolicy
from .demo import play
from .evaluate import (METRICS, _check_final_state, _scenario_record,
                       make_scenarios, score_records)
from .game import new_game, winners
from .policies import RandomLegalPolicy, ReplanningGreedyPolicy
from .release_common import (bootstrap_mean, canonical, export, protocol,
                             sha_bytes, stable_seed, verify_freeze)


def fingerprint(value):
    return sha_bytes(canonical(value).encode("utf-8"))


def hand_key(scenario):
    return tuple(sorted(tuple(sorted(t.ticket_id for t in p.tickets))
                        for p in scenario.initial_state.players))


def overlap_audit(scenarios, observed):
    known = {hand_key(s): s.scenario_id for s in observed}
    seen, duplicates, overlaps = {}, [], []
    for scenario in scenarios:
        key = hand_key(scenario)
        if key in known:
            overlaps.append([scenario.scenario_id, known[key]])
        if key in seen:
            duplicates.append([scenario.scenario_id, seen[key]])
        seen[key] = scenario.scenario_id
    return {"comparison": "Full ticket-hand partitions, ignoring seat permutation",
            "observed_overlap": overlaps, "within_evaluation_duplicates": duplicates,
            "retention_rule": "Retain all predeclared draws, regardless of results"}


def seat_seeds(master, deal, start, replicate):
    return tuple(stable_seed(master, "competition", deal, start, replicate, seat)
                 for seat in range(4))


def run_game(scenario, start, names, seeds):
    factories = {"replan": ReplanningGreedyPolicy,
                 "bottleneck": BottleneckPriorityPolicy}
    if (len(names) != 4 or len(seeds) != 4
            or any(n not in {*factories, "random"} for n in names)):
        raise ValueError("four known policy names and four seeds required")
    original = scenario.initial_state
    initial = new_game(original.routes, tuple(p.tickets for p in original.players),
                       original.rules, starting_player=start)
    policies = tuple(RandomLegalPolicy(seed) if name == "random"
                     else factories[name]() for name, seed in zip(names, seeds))
    active_seeds = tuple(seed if name == "random" else None
                         for name, seed in zip(names, seeds))
    key = (scenario.scenario_id, start, names, active_seeds)
    game_id = "g-" + fingerprint(key)[:20]
    final, events = play(initial, policies)
    _check_final_state(initial, final)
    scores = score_records(final)
    game = {"game_id": game_id, "scenario_id": scenario.scenario_id,
            "deal_seed": scenario.deal_seed, "starting_player": start,
            "policies_by_seat": canonical(names),
            "rng_seeds_by_seat": canonical(active_seeds),
            "initial_state_sha256": fingerprint(asdict(initial)),
            "final_state_sha256": fingerprint(asdict(final)),
            "turns": final.turn, "end_reason": final.end_reason,
            "winner_seats": canonical(winners(final))}
    replay = {"game_id": game_id, "scenario_id": scenario.scenario_id,
              "starting_player": start, "events": [asdict(e) for e in events]}
    players = [{"game_id": game_id, "scenario_id": scenario.scenario_id,
                "policy": names[row["seat"]], **row} for row in scores]
    return game, players, replay


def aggregate(pairs, *, bootstrap, master):
    """All cells stay inside their original deal; do not bootstrap games/turns."""
    blocks, summary = [], []
    for profile in sorted({p["opponent_profile"] for p in pairs}):
        for comparison in ("bottleneck-minus-replan", "random-minus-replan"):
            group = [p for p in pairs if p["opponent_profile"] == profile
                     and p["comparison"] == comparison]
            if not group:
                continue
            deal_ids = sorted({p["scenario_id"] for p in group})
            for metric in METRICS:
                selected_blocks = []
                for deal in deal_ids:
                    cells = [p for p in group if p["scenario_id"] == deal]
                    indices = [(p["starting_player"], p["focal_seat"],
                                p["replicate"]) for p in cells]
                    if len(set(indices)) != len(indices):
                        raise ValueError("duplicate paired cell")
                    expected = {(s, f, r)
                                for s in {p["starting_player"] for p in group}
                                for f in {p["focal_seat"] for p in group}
                                for r in {p["replicate"] for p in group}}
                    if set(indices) != expected:
                        raise ValueError("incomplete balanced deal block")
                    delta = [p["delta_" + metric] for p in cells]
                    block = {
                        "opponent_profile": profile, "comparison": comparison,
                        "scenario_id": deal, "metric": metric,
                        "paired_cells": len(cells),
                        "baseline_mean": statistics.fmean(
                            p["baseline_" + metric] for p in cells),
                        "candidate_mean": statistics.fmean(
                            p["candidate_" + metric] for p in cells),
                        "mean_difference": statistics.fmean(delta),
                        "increases": sum(d > 1e-12 for d in delta),
                        "ties": sum(abs(d) <= 1e-12 for d in delta),
                        "decreases": sum(d < -1e-12 for d in delta)}
                    blocks.append(block)
                    selected_blocks.append(block)
                effects = [b["mean_difference"] for b in selected_blocks]
                low, high = bootstrap_mean(
                    effects, bootstrap, stable_seed(master, "deal-bootstrap"))
                summary.append({
                    "opponent_profile": profile, "comparison": comparison,
                    "metric": metric, "deal_blocks": len(deal_ids),
                    "paired_cells": len(group),
                    "baseline_mean": statistics.fmean(
                        b["baseline_mean"] for b in selected_blocks),
                    "candidate_mean": statistics.fmean(
                        b["candidate_mean"] for b in selected_blocks),
                    "mean_difference": statistics.fmean(effects),
                    "bootstrap_low": low, "bootstrap_high": high,
                    "min_deal_effect": min(effects), "max_deal_effect": max(effects),
                    "increases": sum(b["increases"] for b in selected_blocks),
                    "ties": sum(b["ties"] for b in selected_blocks),
                    "decreases": sum(b["decreases"] for b in selected_blocks)})
    return blocks, summary


def run(*, deal_seeds=None, starts=None, seats=None, replicates=None,
        profiles=None, bootstrap=None, master=None, progress=False):
    config = protocol()
    spec = config["competition"]
    deal_seeds = spec["evaluation_deal_seeds"] if deal_seeds is None else deal_seeds
    starts = spec["starting_players"] if starts is None else starts
    seats = spec["focal_seats"] if seats is None else seats
    replicates = spec["random_replicates"] if replicates is None else replicates
    profiles = spec["opponent_profiles"] if profiles is None else profiles
    bootstrap = spec["bootstrap_resamples"] if bootstrap is None else bootstrap
    master = config["master_seed"] if master is None else master
    for values, allowed in ((starts, range(4)), (seats, range(4)),
                            (profiles, ("replan", "random"))):
        if (not values or len(set(values)) != len(values)
                or any(v not in allowed for v in values)):
            raise ValueError("unique valid starts, seats and profiles required")
    if (not replicates or len(set(replicates)) != len(replicates)
            or any(type(r) is not int or r < 0 for r in replicates)):
        raise ValueError("unique nonnegative replicates required")
    scenarios = make_scenarios(deal_seeds)
    audit = overlap_audit(
        scenarios, make_scenarios(spec["previously_observed_deal_seeds"]))
    games, players, replays, pairs = [], [], [], []
    used_ids = set()
    for i, scenario in enumerate(scenarios):
        cache = {}

        def get_game(start, names, seeds):
            active_seeds = tuple(seed if name == "random" else None
                                 for name, seed in zip(names, seeds))
            key = (start, tuple(names), active_seeds)
            if key not in cache:
                game, score_rows, replay = run_game(scenario, start, names, seeds)
                if game["game_id"] in used_ids:
                    raise RuntimeError("game ID collision")
                used_ids.add(game["game_id"])
                games.append(game)
                players.extend(score_rows)
                replays.append(replay)
                cache[key] = (game["game_id"], score_rows)
            return cache[key]

        for profile in profiles:
            for start in starts:
                for replicate in replicates:
                    seeds_for_game = seat_seeds(master, scenario.deal_seed,
                                                start, replicate)
                    for focal in seats:
                        results = {}
                        for candidate in ("random", "replan", "bottleneck"):
                            names = [candidate if s == focal else profile
                                     for s in range(4)]
                            results[candidate] = get_game(start, names, seeds_for_game)
                        baseline_id, baseline = results["replan"]
                        for candidate in ("bottleneck", "random"):
                            candidate_id, candidate_rows = results[candidate]
                            pair = {
                                "opponent_profile": profile,
                                "comparison": candidate + "-minus-replan",
                                "scenario_id": scenario.scenario_id,
                                "deal_seed": scenario.deal_seed,
                                "starting_player": start, "focal_seat": focal,
                                "turn_position": (focal - start) % 4,
                                "replicate": replicate,
                                "baseline_game_id": baseline_id,
                                "candidate_game_id": candidate_id,
                                "paired_rng_seeds": canonical(seeds_for_game)}
                            for metric in METRICS:
                                before, after = (baseline[focal][metric],
                                                 candidate_rows[focal][metric])
                                pair["baseline_" + metric] = before
                                pair["candidate_" + metric] = after
                                pair["delta_" + metric] = after - before
                            pairs.append(pair)
        if progress:
            print(f"Competition deals: {i + 1}/{len(scenarios)}; "
                  f"unique games: {len(games)}", flush=True)
    blocks, summary = aggregate(pairs, bootstrap=bootstrap, master=master)
    parameters = {
        "deal_seeds": list(deal_seeds), "starting_players": list(starts),
        "focal_seats": list(seats), "random_replicates": list(replicates),
        "opponent_profiles": list(profiles), "master_seed": master,
        "bootstrap_resamples": bootstrap, "unique_games": len(games),
        "paired_cells": len(pairs), "deal_blocks": len(scenarios),
        "transitions": sum(g["turns"] for g in games),
        "inference_unit": spec["bootstrap_unit"]}
    return ({"games": games, "players": players, "pairs": pairs,
             "deal_effects": blocks, "summary": summary},
            {"parameters": parameters, "overlap_audit": audit,
             "scenarios": [_scenario_record(s) for s in scenarios]},
            {"replays": replays})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_freeze()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    tables, documents, streams = run(progress=True)
    export(args.output, "four_player_competition", tables, documents, streams,
           documents["parameters"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
