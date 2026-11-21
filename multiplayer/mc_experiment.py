"""Paired sampling/truncation experiment on the observed teaching endgame.

This is an offline model experiment, not a hidden-information playing policy.
Sampling plans depend on the declared belief, never on a realized secret world.
"""

import argparse
from dataclasses import asdict
from fractions import Fraction
import math
from pathlib import Path
import random
import statistics

from .endgame import (action_key, action_name, diagnostics, evaluate_actions,
                      snapshot_value, terminal_value, validate_belief)
from .endgame_experiment import load_case
from .game import legal_actions, observe, step
from .policies import ReplanningGreedyPolicy
from .release_common import (ROOT, bootstrap_mean, export, protocol, stable_seed,
                             verify_freeze, wilson)

POLICIES = ("focal", "replan", "replan", "uniform")


def make_plan(worlds, count, master, repeat):
    """Draw rational-probability worlds exactly; keep opponent RNG separate."""
    worlds = validate_belief(worlds, 0, POLICIES)
    if type(count) is not int or count <= 0:
        raise ValueError("positive sample count required")
    denominator = math.lcm(*(w.probability.denominator for w in worlds))
    weights = [int(w.probability * denominator) for w in worlds]
    world_rng = random.Random(stable_seed(master, "mc-world", repeat))
    opponent_rng = random.Random(stable_seed(master, "mc-opponent", repeat))
    plan = []
    for trial in range(count):
        draw = world_rng.randrange(denominator)
        index = 0
        while draw >= weights[index]:
            draw -= weights[index]
            index += 1
        plan.append({"repeat": repeat, "trial": trial,
                     "world": worlds[index].name,
                     "opponent_seed": opponent_rng.getrandbits(64)})
    return plan


def sample_once(world, action, depth, opponent_seed, *, rng=None):
    """Actually traverse root plus <=depth opponent actions, using own views."""
    if type(depth) is not int or depth < 0:
        raise ValueError("nonnegative depth required")
    rng = random.Random(opponent_seed) if rng is None else rng
    state, event = step(world.state, 0, action)
    events = [event]
    remaining = depth
    while not state.finished and remaining:
        seat = state.current_player
        view = observe(state, seat)
        if POLICIES[seat] == "replan":
            chosen = ReplanningGreedyPolicy().choose(view)
        elif POLICIES[seat] == "uniform":
            actions = sorted(legal_actions(view), key=action_key)
            chosen = actions[rng.randrange(len(actions))]
        else:
            raise ValueError("only one focal action is supported")
        state, event = step(state, seat, chosen)
        events.append(event)
        remaining -= 1
    value = terminal_value(state, 0) if state.finished else snapshot_value(
        state, 0)
    return value, len(events), state, tuple(events)


def run(*, repeats=None, budgets=None, depths=None, bootstrap=None,
        master=None, progress=False):
    config = protocol()
    spec = config["monte_carlo"]
    repeats = spec["independent_algorithm_repeats"] if repeats is None else repeats
    budgets = spec["samples_per_action"] if budgets is None else list(budgets)
    depths = spec["depths"] if depths is None else list(depths)
    bootstrap = spec["bootstrap_resamples"] if bootstrap is None else bootstrap
    master = config["master_seed"] if master is None else master
    if (type(repeats) is not int or repeats <= 0 or not budgets or not depths
            or budgets != sorted(set(budgets))
            or any(type(n) is not int or n <= 0 for n in budgets)
            or len(depths) != len(set(depths))
            or any(type(h) is not int or not 0 <= h <= 3 for h in depths)):
        raise ValueError("positive repeats/budgets and unique depths 0..3 needed")
    _, _, worlds, _ = load_case(ROOT / spec["case"])
    worlds = validate_belief(worlds, 0, POLICIES)
    by_name = {w.name: w for w in worlds}
    exact = evaluate_actions(worlds)
    reference = {e.action: e.value for e in exact}
    truncated = {h: {e.action: e.value for e in evaluate_actions(worlds, h)}
                 for h in depths}
    actions = sorted(reference, key=action_key)
    values, decisions, plans, traces = [], [], [], []
    for repeat in range(repeats):
        plan = make_plan(worlds, max(budgets), master, repeat)
        plans.extend(plan)
        for depth in depths:
            sums = {a: Fraction(0) for a in actions}
            floats = {a: 0.0 for a in actions}
            counts = {a: 0 for a in actions}
            for n, trial in enumerate(plan, 1):
                for action in actions:
                    utility, count, state, events = sample_once(
                        by_name[trial["world"]], action, depth,
                        trial["opponent_seed"])
                    sums[action] += utility
                    floats[action] += float(utility)
                    counts[action] += count
                    if repeat == 0 and n <= 2 and depth == 3:
                        traces.append({**trial, "action": asdict(action),
                                       "depth": depth, "utility": str(utility),
                                       "events": [asdict(e) for e in events],
                                       "finished": state.finished})
                if n not in budgets:
                    continue
                estimated = {a: sums[a] / n for a in actions}
                actual = diagnostics(reference, estimated)
                sampling = diagnostics(truncated[depth], estimated)
                for action in actions:
                    estimate, target = estimated[action], truncated[depth][action]
                    float_mean = floats[action] / n
                    rounding = abs(Fraction.from_float(float_mean) - estimate)
                    values.append({
                        "repeat": repeat, "depth": depth, "samples": n,
                        "action": action_name(action), "estimate": str(estimate),
                        "exact_depth_value": str(target),
                        "exact_terminal_value": str(reference[action]),
                        "sampling_error": str(estimate - target),
                        "truncation_error": str(target - reference[action]),
                        "total_error": str(estimate - reference[action]),
                        "float_mean": float_mean,
                        "float_rounding_error": str(rounding),
                        "transitions": counts[action]})
                decisions.append({
                    "repeat": repeat, "depth": depth, "samples": n,
                    "selected_action": action_name(actual["selected_action"]),
                    "terminal_selection_error": int(
                        not actual["selected_is_optimal"]),
                    "depth_selection_error": int(
                        not sampling["selected_is_optimal"]),
                    "regret": str(actual["regret"]),
                    "sampling_sup_error": str(sampling["epsilon"]),
                    "total_sup_error": str(actual["epsilon"]),
                    "generic_bound_holds": actual["two_sided_bound_holds"],
                    "generic_certificate": actual[
                        "two_sided_unique_best_certificate"],
                    "transitions": sum(counts.values())})
        if progress and (repeat + 1) % 16 == 0:
            print(f"MC repeats: {repeat + 1}/{repeats}", flush=True)
    summaries = []
    for depth in depths:
        for n in budgets:
            cells = [r for r in decisions
                     if r["depth"] == depth and r["samples"] == n]
            errors = sum(r["terminal_selection_error"] for r in cells)
            low, high = wilson(errors, repeats)
            regrets = [float(Fraction(r["regret"])) for r in cells]
            rlow, rhigh = bootstrap_mean(
                regrets, bootstrap, stable_seed(master, "mc-bootstrap"))
            summaries.append({
                "depth": depth, "samples_per_action": n, "repeats": repeats,
                "selection_errors": errors, "selection_error_rate": errors / repeats,
                "error_wilson_low": low, "error_wilson_high": high,
                "mean_regret": statistics.fmean(regrets),
                "regret_bootstrap_low": rlow, "regret_bootstrap_high": rhigh,
                "mean_sampling_sup_error": statistics.fmean(
                    float(Fraction(r["sampling_sup_error"])) for r in cells),
                "mean_total_sup_error": statistics.fmean(
                    float(Fraction(r["total_sup_error"])) for r in cells),
                "mean_transitions": statistics.fmean(
                    r["transitions"] for r in cells),
                "exact_full_tree_transitions": sum(e.transitions for e in exact)})
    parameters = {"repeats": repeats, "budgets": budgets, "depths": depths,
                  "bootstrap_resamples": bootstrap, "master_seed": master,
                  "case": spec["case"], "policies": POLICIES,
                  "unique_samples": repeats * max(budgets),
                  "evaluated_transitions": sum(
                      r["transitions"] for r in decisions
                      if r["samples"] == max(budgets)),
                  "exact_reference_transitions": sum(e.transitions for e in exact),
                  "case_role": spec["case_role"]}
    return ({"values": values, "decisions": decisions, "summary": summaries},
            {"parameters": parameters}, {"sample_plan": plans, "traces": traces})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_freeze()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    tables, documents, streams = run(progress=True)
    export(args.output, "mc_teaching_endgame", tables, documents, streams,
           documents["parameters"])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
