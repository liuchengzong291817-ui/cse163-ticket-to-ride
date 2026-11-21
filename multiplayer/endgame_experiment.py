"""Repeatable experiment on the already-observed constructed endgame.

No new performance sample, sampling, tuning, or playing strategy is added.
Outputs include exact frontier probabilities and a runnable source snapshot.
"""

import argparse
import csv
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform

from .endgame import (BeliefWorld, action_name, diagnostics, evaluate_actions,
                      score_record, validate_belief)
from .game import Action, Route, Rules, Ticket, new_game, observe, step


ROOT = Path(__file__).resolve().parent.parent
CASE_PATH = ROOT / "research_design/endgame_depth_v1/case.json"
POLICIES = ("focal", "replan", "replan", "uniform")
SOURCE_FILES = ("__init__.py", "game.py", "policies.py", "endgame.py",
                "endgame_experiment.py")


def sha(content):
    return hashlib.sha256(content).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def source_hashes():
    return {name: sha((ROOT / "multiplayer" / name).read_bytes())
            for name in SOURCE_FILES}


def load_case(case_path=CASE_PATH):
    raw = Path(case_path).read_bytes()
    case = json.loads(raw)
    if (type(case["focal_player"]) is not int
            or case["focal_player"] != 0):
        raise ValueError("this input adapter fixes focal player P0")
    expected = {"1": "ReplanningGreedyPolicy", "2": "ReplanningGreedyPolicy",
                "3": "ideal uniform distribution over current legal actions"}
    if case["opponent_policies_after_prefix"] != expected:
        raise ValueError("input policy description differs from execution")
    rules = Rules(**case["rules"])
    routes = tuple(Route(**r) for r in case["routes"])
    worlds, traces = [], []
    for hidden in case["hidden_worlds"]:
        if not isinstance(hidden["probability"], str):
            raise ValueError("probabilities must be exact fraction strings")
        hands = [tuple(Ticket(**t) for t in
                       case["fixed_tickets"].get(str(i), []))
                 for i in range(4)]
        hands[1] = tuple(Ticket(**t) for t in hidden["player_1_tickets"])
        state = new_game(routes, hands, rules)
        trace = []
        for row in case["public_prefix"]:
            state, event = step(
                state, row["player_id"], Action(**row["action"]))
            trace.append(event)
        worlds.append(BeliefWorld(hidden["id"],
                                  Fraction(hidden["probability"]), state))
        traces.append(tuple(trace))
    worlds = validate_belief(worlds, 0, POLICIES)
    if any(trace != traces[0] for trace in traces[1:]):
        raise ValueError("hypothetical worlds must share the public prefix")
    return case, raw, worlds, traces[0]


def _cell(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Action):
        return action_name(value)
    if isinstance(value, tuple):
        return canonical([_cell(item) for item in value])
    return value


def _record_estimates(estimates, phase, depth, rows, frontiers):
    for estimate in estimates:
        error = Fraction.from_float(estimate.float_value) - estimate.value
        rows.append({
            "phase": phase, "h": depth, "action": action_name(estimate.action),
            "value_exact": str(estimate.value),
            "value_decimal": float(estimate.value),
            "float_recursion_value": estimate.float_value,
            "rounding_error_exact": str(error),
            "transitions": estimate.transitions,
            "frontier_count": len(estimate.frontier),
            "terminal_count": sum(f.state.finished for f in estimate.frontier),
            "probability_mass": str(sum(f.probability
                                        for f in estimate.frontier)),
            "conditional_values": canonical({
                name: str(value)
                for name, value in estimate.conditional_values})})
        for leaf in estimate.frontier:
            frontiers.append({
                "phase": phase, "h": depth,
                "action": action_name(estimate.action), "world": leaf.world,
                "joint_probability": str(leaf.probability),
                "utility": str(leaf.value), "terminal": leaf.state.finished,
                "events": [asdict(e) for e in leaf.events],
                "state_sha256": sha(canonical(asdict(leaf.state)).encode()),
                "scores_and_completed": score_record(leaf.state)})


def run_experiment(case_path=CASE_PATH):
    sources = source_hashes()
    case, raw, worlds, prefix = load_case(case_path)
    reference = evaluate_actions(worlds, policies=POLICIES)
    ref = {row.action: row.value for row in reference}
    values, frontiers, decisions = [], [], []
    _record_estimates(reference, "reference", -1, values, frontiers)
    previous = None
    depths = range(worlds[0].state.final_turns_remaining)
    for depth in depths:
        estimates = evaluate_actions(worlds, depth, policies=POLICIES)
        approximate = {row.action: row.value for row in estimates}
        diagnosis = diagnostics(ref, approximate)
        if not diagnosis["all_estimates_are_upper_bounds"]:
            raise ArithmeticError("snapshot upper-bound property failed")
        if previous and any(approximate[a] > previous[a] for a in ref):
            raise ArithmeticError("snapshot values must decrease with depth")
        if not diagnosis["one_sided_bound_holds"]:
            raise ArithmeticError("one-sided regret bound failed")
        previous = approximate
        decisions.append({
            "h": depth, **{k: _cell(v) for k, v in diagnosis.items()},
            "transitions": sum(row.transitions for row in estimates),
            "frontier_count": sum(len(row.frontier) for row in estimates)})
        _record_estimates(estimates, "truncated", depth, values, frontiers)
    if previous != ref:
        raise ArithmeticError("full-depth values differ from exact reference")
    if source_hashes() != sources:
        raise ValueError("execution source changed during the experiment")
    return {
        "case_id": case["case_id"], "case_bytes": raw,
        "case_sha256": sha(raw), "source_sha256": sources,
        "values": values, "decisions": decisions, "frontiers": frontiers,
        "worlds": {
            "public_prefix": [asdict(e) for e in prefix],
            "focal_observation": asdict(observe(worlds[0].state, 0)),
            "focal_observations_identical": True,
            "hypothetical_worlds": [
                {"id": w.name, "probability": str(w.probability),
                 "state": asdict(w.state)} for w in worlds]},
    }


def write_report(report, directory):
    if report["source_sha256"] != source_hashes():
        raise ValueError("report no longer matches current execution source")
    if sha(report["case_bytes"]) != report["case_sha256"]:
        raise ValueError("input case changed inside report")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for table in ("values", "decisions"):
        rows = report[table]
        with (directory / f"{table}.csv").open(
                "x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (directory / "input_case.json").write_bytes(report["case_bytes"])
    (directory / "worlds.json").write_text(
        json.dumps(report["worlds"], indent=2) + "\n", encoding="utf-8")
    with (directory / "frontiers.jsonl").open("x", encoding="utf-8") as stream:
        for row in report["frontiers"]:
            stream.write(canonical(row) + "\n")
    source_dir = directory / "source/multiplayer"
    source_dir.mkdir(parents=True)
    for name in SOURCE_FILES:
        content = (ROOT / "multiplayer" / name).read_bytes()
        if sha(content) != report["source_sha256"][name]:
            raise ValueError("source changed while making snapshot")
        (source_dir / name).write_bytes(content)
    source_case = directory / (
        "source/research_design/endgame_depth_v1/case.json")
    source_case.parent.mkdir(parents=True)
    source_case.write_bytes(report["case_bytes"])
    manifest = {
        "experiment_id": "constructed_endgame_depth_experiment_v1",
        "case_id": report["case_id"],
        "case_role": "one previously observed development/teaching case",
        "independent_performance_samples": 0,
        "policy_by_seat": list(POLICIES), "focal_player": 0,
        "information": "declared finite belief; no actual-world input",
        "belief_world_count": len(report["worlds"]["hypothetical_worlds"]),
        "utility": "score, completed tickets, then equal winner shares",
        "cutoff": "counterfactual settlement after h opponent actions",
        "scope": "final countdown only; focal player never acts again",
        "reference": "all true terminal branches with exact Fraction weights",
        "comparison": "independent truncated traversals for each depth",
        "sampling_used": False, "random_seed": None,
        "transition_counts_exclude": "setup, validation, scoring, no caching",
        "reference_terminal_leaves": sum(
            r["phase"] == "reference" for r in report["frontiers"]),
        "stored_frontiers_are_not_independent_games": True,
        "certificates": "offline, use exact reference error; "
                        "not online bounds",
        "research_status": "exploratory method validation, not a complete "
                           "numerical analysis study or new playing strategy",
        "python_version": platform.python_version(),
        "source_sha256": report["source_sha256"],
        "case_sha256": report["case_sha256"],
        "result_sha256": {
            p.relative_to(directory).as_posix(): sha(p.read_bytes())
            for p in sorted(directory.rglob("*")) if p.is_file()},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE_PATH)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/multiplayer_endgame_depth_v1")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new directory")
    report = run_experiment(args.case)
    write_report(report, args.output)
    for row in report["decisions"]:
        print(f"h={row['h']}: epsilon={row['epsilon']}; "
              f"choose={row['selected_action']}; regret={row['regret']}; "
              f"transitions={row['transitions']}")
    print("One observed teaching case; no independent performance evidence.")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
