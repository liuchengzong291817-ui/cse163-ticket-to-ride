"""Check one declared teaching example, not a general search agent or study.

From the project root: python -B -m research_design.endgame_depth_v1.verify_hand_example
An optional --output JSON path is created exclusively, never overwritten.
"""
import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform

from multiplayer.game import (Action, Route, Rules, Ticket, final_scores,
                              legal_actions, new_game, observe, step, winners)
from multiplayer.policies import ReplanningGreedyPolicy


def label(action):
    return action.kind + (" " + action.route_id if action.route_id else "")


def hypothetical_scores(state):
    """Counterfactual settlement; do not call a live state terminal."""
    result = []
    for seat, player in enumerate(state.players):
        adjacency = {}
        for route, owner in zip(state.routes, state.owners):
            if owner == seat:
                adjacency.setdefault(route.source, set()).add(route.target)
                adjacency.setdefault(route.target, set()).add(route.source)
        completed = []
        for ticket in player.tickets:
            seen, pending = {ticket.source}, [ticket.source]
            while pending:
                for city in adjacency.get(pending.pop(), set()) - seen:
                    seen.add(city)
                    pending.append(city)
            if ticket.target in seen:
                completed.append(ticket)
        total = (player.route_points + 2 * sum(t.points for t in completed)
                 - sum(t.points for t in player.tickets))
        result.append((total, len(completed)))
    return result


def share(state):
    scores = hypothetical_scores(state)
    best = max(scores)
    return F(1, scores.count(best)) if scores[0] == best else F(0)


def calculate():
    directory = Path(__file__).resolve().parent
    case_path = directory / "case.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    worlds, prefixes = [], []
    for world in case["hidden_worlds"]:
        hands = [tuple(Ticket(**t) for t in case["fixed_tickets"].get(str(i), []))
                 for i in range(4)]
        hands[1] = tuple(Ticket(**t) for t in world["player_1_tickets"])
        state = new_game(tuple(Route(**r) for r in case["routes"]), hands,
                         Rules(**case["rules"]))
        events = []
        for row in case["public_prefix"]:
            state, event = step(state, row["player_id"], Action(**row["action"]))
            events.append(asdict(event))
        assert state.turn == 4 and state.current_player == 0
        assert state.final_turns_remaining == 4 and not state.finished
        worlds.append((world["id"], F(world["probability"]), state))
        prefixes.append(events)
    assert prefixes[0] == prefixes[1]
    assert observe(worlds[0][2], 0) == observe(worlds[1][2], 0)
    assert sum(p for _, p, _ in worlds) == 1
    actions = sorted(legal_actions(observe(worlds[0][2], 0)),
                     key=lambda a: (a.kind, a.route_id or ""))
    rows, leaves, transition_count = [], [], 0
    for action in actions:
        exact, floats, conditional = [F(0) for _ in range(4)], [0.0] * 4, {}
        for world_id, prior, initial in worlds:
            state, event = step(initial, 0, action)
            history, mass, value = [asdict(event)], F(0), F(0)
            transition_count += 1
            for depth in range(3):
                utility = share(state)
                exact[depth] += prior * utility
                floats[depth] += float(prior) * float(utility)
                if depth < 2:
                    seat = state.current_player
                    reply = ReplanningGreedyPolicy().choose(observe(state, seat))
                    state, event = step(state, seat, reply)
                    history.append(asdict(event))
                    transition_count += 1
            replies = sorted(legal_actions(observe(state, 3)),
                             key=lambda a: (a.kind, a.route_id or ""))
            for reply in replies:
                final, event = step(state, 3, reply)
                transition_count += 1
                assert final.finished and final.turn == 8
                assert final.final_turns_remaining == 0
                utility = F(1, len(winners(final))) if 0 in winners(final) else F(0)
                assert utility == share(final)
                assert hypothetical_scores(final) == [
                    (s.total, len(s.completed)) for s in final_scores(final)]
                probability = F(1, len(replies))
                mass += probability
                value += probability * utility
                floats[3] += float(prior) * float(probability) * float(utility)
                leaves.append({"action": label(action), "world": world_id,
                               "joint_probability": str(prior * probability),
                               "events": history + [asdict(event)],
                               "scores_and_completed": hypothetical_scores(final),
                               "utility": str(utility)})
            assert mass == 1
            conditional[world_id] = str(value)
            exact[3] += prior * value
        assert exact[3] == F(case["hand_predictions_before_engine_check"][label(action)])
        rows.append({"action": label(action), "Q_h": [str(x) for x in exact],
                     "Q_ref": str(exact[3]), "conditional_Q_ref": conditional,
                     "float_error": [str(F.from_float(x) - y) for x, y in zip(floats, exact)]})
    depths = []
    optimum = max(F(row["Q_ref"]) for row in rows)
    for depth in range(4):
        best = max(F(row["Q_h"][depth]) for row in rows)
        maximizing = [row for row in rows if F(row["Q_h"][depth]) == best]
        epsilon = max(abs(F(row["Q_h"][depth]) - F(row["Q_ref"])) for row in rows)
        depths.append({"h": depth, "max_abs_error": str(epsilon),
                       "maximizing_actions": [row["action"] for row in maximizing],
                       "selected_action": maximizing[0]["action"],
                       "regret": str(optimum - F(maximizing[0]["Q_ref"]))})
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    root = directory.parent.parent
    return {"case_id": case["case_id"], "purpose": case["purpose"],
            "python_version": platform.python_version(), "case_sha256": sha(case_path),
            "checker_sha256": sha(Path(__file__)),
            "engine_sha256": {n: sha(root / "multiplayer" / n) for n in ("game.py", "policies.py")},
            "public_root_observation": asdict(observe(worlds[0][2], 0)),
            "identical_public_prefix_in_both_worlds": prefixes[0],
            "rows": rows, "depths": depths, "terminal_leaves": leaves,
            "terminal_leaf_count": len(leaves), "transitions_excluding_setup": transition_count,
            "root_observation_equal": True, "all_hand_predictions_match": True,
            "sampling_used": False, "independent_test_cases_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output exists; choose a new path")
    result = calculate()
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    for row in result["rows"]:
        print(row["action"], "Q_0..Q_3=", row["Q_h"], "exact=", row["Q_ref"])
    print("Depth decisions:", result["depths"])
    print("Exact terminal leaves:", result["terminal_leaf_count"])


if __name__ == "__main__":
    main()
