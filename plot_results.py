"""Optional Matplotlib publication artifacts, regenerated only from saved CSV."""

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402

from multiplayer.demo import demo_setup  # noqa: E402
from multiplayer.release_common import file_sha  # noqa: E402
from multiplayer.release_verify import read_csv  # noqa: E402

COLORS = ["#2364aa", "#e07a20", "#2a8c6c", "#8e5ea2"]


def save(fig, directory, name):
    fig.savefig(directory / (name + ".png"), dpi=220,
                metadata={"Software": "TicketToRide research v1.0"})
    fig.savefig(directory / (name + ".svg"), metadata={"Date": None})
    fig.savefig(directory / (name + ".pdf"),
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def maps(directory):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    coords = {"A": (0, 1), "B": (1, 1), "C": (2, 1),
              "D": (2, 0), "E": (0, 0), "F": (1, 0)}
    ax = axes[0]
    for route in demo_setup().routes:
        x, y = coords[route.source], coords[route.target]
        bend = .22 if route.route_id == "AB1" else (
            -.22 if route.route_id == "AB2" else 0)
        if route.route_id == "AC":
            bend = -.25
        ax.add_patch(FancyArrowPatch(
            x, y, arrowstyle="-", linewidth=1.8, color="#888888",
            connectionstyle=f"arc3,rad={bend}", shrinkA=16, shrinkB=16))
        mx, my = ((x[0] + y[0]) / 2, (x[1] + y[1]) / 2)
        if route.route_id.startswith("AB"):
            my += -.1 if route.route_id == "AB1" else .1
        if route.route_id == "AC":
            my += .43
        if route.route_id == "BE":
            mx -= .06
        if route.route_id == "CE":
            mx, my = 1.4, .7
        if route.route_id == "BF":
            my = .32
        ax.text(mx, my, f"{route.route_id}:{route.length}", ha="center", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    for node, (x, y) in coords.items():
        ax.scatter(x, y, s=650, color="#e8f1fa", edgecolor=COLORS[0], zorder=3)
        ax.text(x, y, node, ha="center", va="center", weight="bold", zorder=4)
    ax.set(xlim=(-.3, 2.35), ylim=(-.3, 1.65), title="Competition map: 6 cities, 12 routes")
    ax.axis("off")
    ax = axes[1]
    coords = {"O": (0, 1), "A": (1, 1), "B": (2, 1), "E": (3, 1),
              "D": (3, 2), "C": (4, 2), "F": (4, 1), "G": (5, 1),
              "H": (6, 1), "N": (0, 0)}
    case = json.loads((ROOT / "research_design/endgame_depth_v1/case.json").read_text())
    owners = {r["action"]["route_id"]: r["player_id"] for r in case["public_prefix"]}
    for route in case["routes"]:
        x, y = coords[route["source"]], coords[route["target"]]
        color = COLORS[owners[route["route_id"]]] if route["route_id"] in owners else "#777777"
        ax.plot([x[0], y[0]], [x[1], y[1]], color=color, lw=3)
        ax.text((x[0] + y[0]) / 2, (x[1] + y[1]) / 2 + .12,
                str(route["length"]), ha="center", fontsize=10)
    for node, (x, y) in coords.items():
        ax.scatter(x, y, s=280, color="white", edgecolor="#444444", zorder=3)
        ax.text(x, y, node, ha="center", va="center", zorder=4)
    ax.set(xlim=(-.4, 6.4), ylim=(-.35, 2.5),
           title="Observed endgame: colored routes already owned")
    ax.axis("off")
    fig.suptitle("Two synthetic maps, two distinct research questions", fontsize=15)
    save(fig, directory, "01_maps")


def depth_plot(results, directory):
    values = read_csv(results / "depth/values.csv")
    decisions = read_csv(results / "depth/decisions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for action in ("claim AB", "claim CD", "claim EF", "claim GH", "gather"):
        rows = [r for r in values if r["phase"] == "truncated" and r["action"] == action]
        axes[0].plot([int(r["h"]) for r in rows],
                     [float(Fraction(r["value_exact"])) for r in rows], "o-", label=action)
    axes[0].set(xlabel="Opponent moves searched (h)", ylabel="Exact truncated victory share",
                xticks=range(4), title="Depth changes action values")
    axes[0].legend(fontsize=9)
    for field, label in (("epsilon", "Maximum value error"), ("regret", "Decision regret")):
        axes[1].plot([int(r["h"]) for r in decisions],
                     [float(Fraction(r[field])) for r in decisions], "o-", label=label)
    axes[1].axhline(1 / 8, color="#888888", ls="--", label="True action gap = 1/8")
    axes[1].set(xlabel="Opponent moves searched (h)", ylabel="Victory-share units",
                xticks=range(4), title="Correct action can precede a sufficient certificate")
    axes[1].legend(fontsize=9)
    fig.suptitle("One constructed endgame; 22 exact terminal leaves", fontsize=14)
    save(fig, directory, "02_depth")


def mc_plot(results, directory):
    rows = read_csv(results / "mc/summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for h, color in enumerate(COLORS):
        cells = [r for r in rows if int(r["depth"]) == h]
        n = [int(r["samples_per_action"]) for r in cells]
        error = np.array([float(r["selection_error_rate"]) for r in cells])
        low = np.array([float(r["error_wilson_low"]) for r in cells])
        high = np.array([float(r["error_wilson_high"]) for r in cells])
        axes[0].errorbar(n, error, yerr=[np.maximum(0, error - low), np.maximum(0, high - error)],
                         fmt="o-", color=color, label=f"h={h}", capsize=3)
        costs = [float(r["mean_transitions"]) for r in cells]
        regret = [float(r["mean_regret"]) for r in cells]
        axes[1].plot(costs, regret, "o-", color=color, label=f"h={h}")
        axes[1].fill_between(costs, [float(r["regret_bootstrap_low"]) for r in cells],
                             [float(r["regret_bootstrap_high"]) for r in cells],
                             color=color, alpha=.13)
    axes[0].set(xlabel="Samples per root action (nested N)", ylabel="Wrong-action frequency",
                title="128 paired algorithm repeats; 95% Wilson intervals", ylim=(-.03, 1.06))
    axes[1].axvline(52, ls="--", color="black", label="Exact tree: 52 transitions")
    axes[1].set(xlabel="Actual transitions per decision", ylabel="Mean exact decision regret",
                title="95% repeat-bootstrap bands; no runtime-speedup claim")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    save(fig, directory, "03_sampling_budget")


def competition_plot(results, directory):
    summary = read_csv(results / "competition/summary.csv")
    blocks = read_csv(results / "competition/deal_effects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
    keys = [("replan", "bottleneck-minus-replan"), ("random", "bottleneck-minus-replan"),
            ("replan", "random-minus-replan"), ("random", "random-minus-replan")]
    labels = ["B - G | three G opponents", "B - G | three R opponents",
              "R - G | three G opponents", "R - G | three R opponents"]
    for ax, metric in zip(axes, ("win_credit", "score")):
        for y, (profile, comparison) in enumerate(keys):
            row = next(r for r in summary if r["opponent_profile"] == profile
                       and r["comparison"] == comparison and r["metric"] == metric)
            cells = [r for r in blocks if r["opponent_profile"] == profile
                     and r["comparison"] == comparison and r["metric"] == metric]
            effects = [float(r["mean_difference"]) for r in cells]
            ax.scatter(effects, y + np.linspace(-.13, .13, len(cells)), s=14,
                       color=COLORS[y], alpha=.35)
            value, lo, hi = (float(row[k]) for k in
                             ("mean_difference", "bootstrap_low", "bootstrap_high"))
            ax.errorbar(value, y, xerr=[[max(0, value - lo)], [max(0, hi - value)]],
                        fmt="o", color="black", capsize=5, ms=5)
        ax.axvline(0, color="#888888", ls="--")
        ax.set(yticks=range(4), yticklabels=labels,
               xlabel="Paired effect (candidate minus G)", title=metric)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("36 deal blocks; dots = deal effects; bars = 95% block-bootstrap intervals\n"
                 "Fixed synthetic map; exploratory, pointwise intervals", fontsize=13)
    save(fig, directory, "04_competition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/release_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "ticket-to-ride-research-v1"})
    maps(args.output)
    depth_plot(args.results, args.output)
    mc_plot(args.results, args.output)
    competition_plot(args.results, args.output)
    manifest = {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                "script_sha256": file_sha(__file__),
                "source_result_manifest_sha256": {
                    name: file_sha(args.results / name / "manifest.json")
                    for name in ("depth", "mc", "competition")},
                "figure_sha256": {p.name: file_sha(p) for p in sorted(args.output.iterdir())}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created 4 figures in PNG, SVG and PDF: {args.output}")


if __name__ == "__main__":
    main()
