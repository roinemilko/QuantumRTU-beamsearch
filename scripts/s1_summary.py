"""S1: paired LER of frontier freezing (results/ler_s1) against the unpruned gpu-fp32 tallies (results/ler).

Both decode the same shots (ler.py --match). Per (p, preset, freezing config), X+Z summed:
failures of each, shots failing only unpruned / only frozen / both, and an exact two-sided
sign-test p-value on the discordant shots (McNemar). Speed (tally shots/s) is indicative only:
the unpruned tallies were timed in an earlier session. Writes results/s1_ler.json and .md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "gpu-fp32"


def main() -> None:
    rows: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for path in sorted((ROOT / "results" / "ler_s1").glob("*/*.json")):
        _, basis, p = path.parent.name.split("_")
        preset, name = path.stem.split("__")
        frozen = json.loads(path.read_text())
        base = json.loads((ROOT / "results" / "ler" / path.parent.name / f"{preset}__{BASE}.json").read_text())
        shots = min(frozen["shots"], base["shots"])
        a, b = ({i for i in t["failed"] if i < shots} for t in (base, frozen))
        row = rows[(p[1:], preset, name.removeprefix(BASE + "-"))]
        row["shots"] += shots
        row["fail_unpruned"] += len(a)
        row["fail_frozen"] += len(b)
        row["only_unpruned"] += len(a - b)
        row["only_frozen"] += len(b - a)
        row["both"] += len(a & b)
        row["seconds_unpruned"] += base["seconds"] * shots / base["shots"]
        row["seconds_frozen"] += frozen["seconds"] * shots / frozen["shots"]
        row["bases"] += 1
    report = {}
    for (p, preset, config), r in sorted(rows.items()):
        n, k = int(r["only_unpruned"] + r["only_frozen"]), int(min(r["only_unpruned"], r["only_frozen"]))
        report[f"{p}/{preset}/{config}"] = {**r, "ler_ratio": r["fail_frozen"] / max(r["fail_unpruned"], 1),
                                            "sign_test_p": sign_test(n, k),
                                            "speedup_indicative": r["seconds_unpruned"] / max(r["seconds_frozen"], 1e-9)}
    (ROOT / "results" / "s1_ler.json").write_text(json.dumps(report, indent=2))
    lines = ["| p | preset | freezing | shots (X+Z) | failures unpruned → frozen | ratio | only unpruned / only frozen / both "
             "| sign-test p | speed (indicative) |", "|---|---|---|---|---|---|---|---|---|"]
    for key, r in report.items():
        p, preset, config = key.split("/")
        lines.append(f"| {p} | {preset} | {config} | {int(r['shots']):,} | {int(r['fail_unpruned'])} → {int(r['fail_frozen'])} "
                     f"| {r['ler_ratio']:.2f} | {int(r['only_unpruned'])} / {int(r['only_frozen'])} / {int(r['both'])} "
                     f"| {r['sign_test_p']:.2g} | {r['speedup_indicative']:.1f}× |")
    (ROOT / "results" / "s1_ler.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def sign_test(n: int, k: int) -> float:
    """Two-sided exact binomial(n, 1/2) probability of a split at least as uneven as k vs n - k."""
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


if __name__ == "__main__":
    main()
