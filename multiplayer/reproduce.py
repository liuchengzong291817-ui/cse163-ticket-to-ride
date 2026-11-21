"""Reproduce all release studies into a NEW directory and compare result bytes."""

import argparse
import json
from pathlib import Path

from . import competition_experiment, mc_experiment
from .endgame_experiment import run_experiment, write_report
from .release_common import ROOT, export, file_sha, verify_freeze
from .release_verify import verify_competition, verify_manifest


def compare_directories(expected, actual):
    expected, actual = Path(expected), Path(actual)
    left = {p.relative_to(expected).as_posix(): file_sha(p)
            for p in expected.rglob("*") if p.is_file()}
    right = {p.relative_to(actual).as_posix(): file_sha(p)
             for p in actual.rglob("*") if p.is_file()}
    if left != right:
        differences = sorted(k for k in left.keys() | right.keys()
                             if left.get(k) != right.get(k))
        raise ValueError(f"Result bytes differ: {differences}")
    return {"identical_files": len(left)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path,
                        help="Existing study root to compare byte for byte")
    args = parser.parse_args()
    verify_freeze()
    args.output.mkdir(parents=True, exist_ok=False)
    print("Reproducing exact depth experiment", flush=True)
    write_report(run_experiment(), args.output / "depth")
    for name, module in (("mc", mc_experiment),
                         ("competition", competition_experiment)):
        tables, documents, streams = module.run(progress=True)
        export(args.output / name, name, tables, documents, streams,
               documents["parameters"])
        verify_manifest(args.output / name)
    verification = verify_competition(args.output / "competition")
    verification["historical_depth"] = compare_directories(
        ROOT / "outputs/multiplayer_endgame_depth_v1", args.output / "depth")
    if args.compare:
        verification["reproduced_studies"] = compare_directories(args.compare, args.output)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
