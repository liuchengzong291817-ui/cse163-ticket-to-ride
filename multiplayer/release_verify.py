"""Verify saved result bytes and replay all competition games independently."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from .competition_experiment import fingerprint
from .evaluate import _check_final_state, score_records
from .game import Action, Route, Rules, Ticket, legal_actions, new_game, observe, step
from .release_common import file_sha, verify_freeze


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["result_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha(path) != expected:
            raise ValueError(f"result hash differs: {name}")
    return manifest


def replay(scenario, starting_player, events):
    initial = new_game(tuple(Route(**r) for r in scenario["routes"]),
                       tuple(tuple(Ticket(**t) for t in hand)
                             for hand in scenario["tickets_by_seat"]),
                       Rules(**scenario["rules"]), starting_player=starting_player)
    state = initial
    for saved in events:
        action = Action(**saved["action"])
        if action not in legal_actions(observe(state, state.current_player)):
            raise ValueError("replay contains illegal action")
        state, event = step(state, saved["player_id"], action)
        if asdict(event) != saved:
            raise ValueError("replay event differs from recomputed transition")
    _check_final_state(initial, state)
    return initial, state


def verify_competition(directory):
    directory = Path(directory)
    manifest = verify_manifest(directory)
    scenarios = {s["scenario_id"]: s for s in json.loads(
        (directory / "scenarios.json").read_text(encoding="utf-8"))}
    games = {g["game_id"]: g for g in read_csv(directory / "games.csv")}
    players = {(r["game_id"], int(r["seat"])): r
               for r in read_csv(directory / "players.csv")}
    seen, transitions = set(), 0
    with (directory / "replays.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            game = games[row["game_id"]]
            if row["game_id"] in seen:
                raise ValueError("duplicate replay")
            seen.add(row["game_id"])
            if (row["scenario_id"] != game["scenario_id"]
                    or row["starting_player"] != int(game["starting_player"])):
                raise ValueError("replay belongs to different scenario or start")
            initial, final = replay(scenarios[row["scenario_id"]],
                                    row["starting_player"], row["events"])
            if (fingerprint(asdict(initial)) != game["initial_state_sha256"]
                    or fingerprint(asdict(final)) != game["final_state_sha256"]):
                raise ValueError("state fingerprint differs")
            if final.turn != int(game["turns"]) or final.end_reason != game["end_reason"]:
                raise ValueError("termination metadata differs")
            for score in score_records(final):
                saved = players[(row["game_id"], score["seat"])]
                if any(str(value) != saved[name] for name, value in score.items()):
                    raise ValueError("recomputed scoring differs")
            transitions += len(row["events"])
    if seen != set(games) or len(players) != 4 * len(games):
        raise ValueError("missing game replay/player records")
    pairs = read_csv(directory / "pairs.csv")
    for pair in pairs:
        focal = int(pair["focal_seat"])
        expected_seeds = json.loads(pair["paired_rng_seeds"])
        candidate = pair["comparison"].split("-minus-")[0]
        for arm, policy in (("baseline", "replan"), ("candidate", candidate)):
            game = games[pair[arm + "_game_id"]]
            if (game["scenario_id"] != pair["scenario_id"]
                    or game["starting_player"] != pair["starting_player"]):
                raise ValueError("unmatched paired starting state")
            names = json.loads(game["policies_by_seat"])
            seeds = json.loads(game["rng_seeds_by_seat"])
            expected = [policy if s == focal else pair["opponent_profile"]
                        for s in range(4)]
            if names != expected:
                raise ValueError("paired policy assignment differs")
            for seat, name in enumerate(names):
                if name == "random" and seeds[seat] != expected_seeds[seat]:
                    raise ValueError("paired random streams differ")
            score = players[(game["game_id"], focal)]
            for field in [key for key in pair if key.startswith(arm + "_")
                          and key != arm + "_game_id"]:
                metric = field[len(arm) + 1:]
                if float(pair[field]) != float(score[metric]):
                    raise ValueError("paired score differs")
        for field in [key for key in pair if key.startswith("delta_")]:
            metric = field[6:]
            expected = float(pair["candidate_" + metric]) - float(
                pair["baseline_" + metric])
            if float(pair[field]) != expected:
                raise ValueError("paired difference differs")
    expected = manifest["parameters"]
    if (len(seen) != expected["unique_games"]
            or len(pairs) != expected["paired_cells"]
            or transitions != expected["transitions"]):
        raise ValueError("counts differ from manifest")
    return {"games_replayed": len(seen), "transitions_replayed": transitions,
            "player_scores_checked": len(players), "paired_cells_checked": len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/release_v1"))
    args = parser.parse_args()
    verify_freeze()
    verify_manifest(args.results / "mc")
    print(json.dumps(verify_competition(args.results / "competition"), indent=2))


if __name__ == "__main__":
    main()
