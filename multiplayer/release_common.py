"""Shared reproducible output and uncertainty calculations for this release."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "config/release_v1.json"
FREEZE = ROOT / "validation/release_v1_freeze.json"
SOURCES = ("__init__.py", "game.py", "policies.py", "bottleneck.py",
           "demo.py", "evaluate.py", "endgame.py", "endgame_experiment.py",
           "release_common.py", "mc_experiment.py", "competition_experiment.py",
           "release_verify.py", "reproduce.py")


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def file_sha(path):
    return sha_bytes(Path(path).read_bytes())


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest()[:8],
                          "big")


def protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def source_hashes():
    return {name: file_sha(ROOT / "multiplayer" / name) for name in SOURCES}


def freeze():
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": file_sha(PROTOCOL),
              "case_sha256": file_sha(ROOT / protocol()["monte_carlo"]["case"]),
              "sources_sha256": source_hashes(),
              "python_version": platform.python_version(),
              "purpose": "Local design freeze before evaluation, not formal "
                         "preregistration or proof of confirmatory validity."}
    with FREEZE.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return record


def verify_freeze():
    record = json.loads(FREEZE.read_text(encoding="utf-8"))
    if (record["protocol_sha256"] != file_sha(PROTOCOL)
            or record["case_sha256"] != file_sha(
                ROOT / protocol()["monte_carlo"]["case"])
            or record["sources_sha256"] != source_hashes()
            or record["python_version"] != platform.python_version()):
        raise ValueError("Code, protocol or Python differs from release freeze")
    return record


def percentile(values, q):
    ordered = sorted(values)
    if not ordered or not 0 <= q <= 1:
        raise ValueError("nonempty values and quantile in [0,1] required")
    position = (len(ordered) - 1) * q
    left = math.floor(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def bootstrap_mean(values, repetitions, seed):
    """Percentile interval resampling whole caller-defined independent units."""
    if not values or type(repetitions) is not int or repetitions <= 0:
        raise ValueError("nonempty unit values and positive repetitions required")
    rng = random.Random(seed)
    n = len(values)
    means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n))
             for _ in range(repetitions)]
    return percentile(means, 0.025), percentile(means, 0.975)


def wilson(successes, n):
    """95% Wilson interval for a repeat-level Bernoulli event."""
    if type(n) is not int or n <= 0 or not 0 <= successes <= n:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p, z2 = successes / n, z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def export(directory, study, tables, documents, streams, parameters):
    """Create new output only, preserving deterministic bytes and provenance."""
    verify_freeze()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        if not rows:
            raise ValueError("empty result table")
        with (directory / (name + ".csv")).open(
                "x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name, value in documents.items():
        (directory / (name + ".json")).write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8")
    for name, rows in streams.items():
        with (directory / (name + ".jsonl")).open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(canonical(row) + "\n")
    (directory / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    (directory / "freeze.json").write_bytes(FREEZE.read_bytes())
    manifest = {"study": study, "parameters": parameters,
                "python_version": platform.python_version(),
                "source_sha256": source_hashes(),
                "protocol_sha256": file_sha(PROTOCOL),
                "result_sha256": {p.name: file_sha(p)
                                  for p in sorted(directory.iterdir())}}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
