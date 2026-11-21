"""Post-result descriptive tables and explicitly selected illustrative replays."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from multiplayer.demo import render_replay
from multiplayer.game import Action, PublicEvent
from multiplayer.release_common import ROOT, file_sha
from multiplayer.release_verify import read_csv, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/release_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    competition = args.results / "competition"
    rows = read_csv(competition / "summary.csv")
    lines = ["# 自动导出的结果表", "", "数值来自已保存 CSV；不增加新实验或独立样本。", "",
             "## 四人配对效果", "",
             "每行 36 个发票情景、2,304 个配对格。区间按整个情景重抽样 4,000 次。", "",
             "| 对手 | 比较 | 指标 | G 均值 | 候选均值 | 差值 | 95% 区间 | 增/平/减格数 |",
             "|---|---|---|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join([
            row["opponent_profile"], row["comparison"], row["metric"],
            f'{float(row["baseline_mean"]):.6f}', f'{float(row["candidate_mean"]):.6f}',
            f'{float(row["mean_difference"]):+.6f}',
            f'[{float(row["bootstrap_low"]):.6f}, {float(row["bootstrap_high"]):.6f}]',
            "/".join(row[k] for k in ("increases", "ties", "decreases"))]) + " |")
    lines += ["", "## Monte Carlo 预算", "",
              "同一个已观察的教学残局。每行 128 个算法随机流；不是 128 个独立局面。", "",
              "| h | 每动作 N | 选错次数 | 选错率及 95% Wilson 区间 | 平均 regret | 均值区间 | 转移数 |",
              "|---:|---:|---:|---|---:|---|---:|"]
    for row in read_csv(args.results / "mc/summary.csv"):
        lines.append("| " + " | ".join([
            row["depth"], row["samples_per_action"], row["selection_errors"] + "/128",
            f'{float(row["selection_error_rate"]):.6f} '
            f'[{float(row["error_wilson_low"]):.6f}, {float(row["error_wilson_high"]):.6f}]',
            f'{float(row["mean_regret"]):.6f}',
            f'[{float(row["regret_bootstrap_low"]):.6f}, '
            f'{float(row["regret_bootstrap_high"]):.6f}]',
            row["mean_transitions"]]) + " |")
    lines += ["", "没有观测到错误时，regret 的百分位 bootstrap 区间可能退化为 [0,0]，",
              "不能据此声称真实错误概率为零；同时查看 Wilson 上界。", ""]
    (args.output / "RESULT_TABLES_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    pairs = [p for p in read_csv(competition / "pairs.csv")
             if p["opponent_profile"] == "replan" and p["replicate"] == "0"
             and p["comparison"] == "bottleneck-minus-replan"]
    ordering = lambda p: (float(p["delta_score"]), p["scenario_id"],  # noqa: E731
                          int(p["starting_player"]), int(p["focal_seat"]))
    selected = {"largest_score_loss": min(pairs, key=ordering),
                "largest_score_gain": max(pairs, key=ordering)}
    ids = {p[arm + "_game_id"] for p in selected.values() for arm in ("baseline", "candidate")}
    traces = {}
    with (competition / "replays.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["game_id"] in ids:
                traces[row["game_id"]] = row
    scenarios = {r["scenario_id"]: r for r in json.loads(
        (competition / "scenarios.json").read_text(encoding="utf-8"))}
    examples = {}
    for name, pair in selected.items():
        for arm in ("baseline", "candidate"):
            trace = traces[pair[arm + "_game_id"]]
            initial, final = replay(scenarios[pair["scenario_id"]],
                                    int(pair["starting_player"]), trace["events"])
            events = [PublicEvent(**{**event, "action": Action(**event["action"])})
                      for event in trace["events"]]
            (args.output / f"{name}_{arm}.txt").write_text(
                render_replay(initial, events) + "\n", encoding="utf-8")
        before = traces[pair["baseline_game_id"]]["events"]
        after = traces[pair["candidate_game_id"]]["events"]
        first = next(i for i, (a, b) in enumerate(zip(before, after))
                     if a["action"] != b["action"])
        examples[name] = {
            "selection": "Post-hoc extreme score difference among "
                         "B-G versus GGG, replicate 0; teaching illustration only",
            "pair": pair, "scenario": scenarios[pair["scenario_id"]],
            "first_difference": {"baseline": before[first], "candidate": after[first]}}
    (args.output / "posthoc_examples.json").write_text(
        json.dumps(examples, indent=2) + "\n", encoding="utf-8")
    mc_values = read_csv(args.results / "mc/values.csv")
    worst = max(mc_values, key=lambda r: Fraction(r["float_rounding_error"]))
    diagnostics = {"maximum_sample_mean_rounding_error": worst,
                   "ggg_bottleneck_unique_cells": len(pairs),
                   "ggg_bottleneck_unique_win_counts": {
                       "increase": sum(float(p["delta_win_credit"]) > 1e-12 for p in pairs),
                       "tie": sum(abs(float(p["delta_win_credit"])) <= 1e-12 for p in pairs),
                       "decrease": sum(float(p["delta_win_credit"]) < -1e-12 for p in pairs)},
                   "posthoc_analysis_does_not_change_policies": True}
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    manifest = {"script_sha256": file_sha(__file__),
                "result_manifest_sha256": {name: file_sha(args.results / name / "manifest.json")
                                           for name in ("depth", "mc", "competition")},
                "files_sha256": {p.name: file_sha(p) for p in sorted(args.output.iterdir())}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "diagnostics": diagnostics,
        "examples": {k: {"pair": v["pair"], "first_difference": v["first_difference"]}
                     for k, v in examples.items()}}, indent=2))


if __name__ == "__main__":
    main()
