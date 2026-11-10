"""LER vs physical error rate for the CUDA baseline, with IonQ's published curves overlaid.

Reads results/ler/*/*.json (scripts/ler.py); writes results/ler.png, results/ler.md, results/ler.json.
Per-round LER = Σ_{X,Z} failures / shots / 12 (IonQ's definition); error bars sum the per-basis
binomial likelihood-ratio intervals (sinter.fit_binomial, factor 1000).
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sinter
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

Run = dict[str, Any]  # one tally JSON from scripts/ler.py

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ROUNDS = 12
PRESETS = ("beam8_230", "beam32_340", "beam64_640")
LABELS = {"beam8_230": "beam8", "beam32_340": "beam32", "beam64_640": "beam64"}
COLORS = {"beam8_230": "#2a78d6", "beam32_340": "#eb6834", "beam64_640": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Read off external/BeamSearchDecoder/simulation_results/beam_144_12.pdf (±~10%, log axis).
PAPER = {
    "beam8_230": {0.001: 2.8e-7, 0.002: 1.9e-5, 0.003: 3.3e-4, 0.004: 3.1e-3, 0.005: 1.8e-2, 0.006: 6.0e-2},
    "beam32_340": {0.001: 6.8e-8, 0.002: 5.6e-6, 0.003: 1.1e-4, 0.004: 1.4e-3, 0.005: 1.1e-2, 0.006: 4.7e-2},
    "beam64_640": {0.001: 5.4e-8, 0.002: 4.0e-6, 0.003: 8.0e-5, 0.004: 8.5e-4, 0.005: 5.2e-3, 0.006: 2.5e-2},
    "bp30+osd": {0.001: 3.7e-7, 0.002: 2.7e-5, 0.003: 5.1e-4, 0.004: 4.8e-3, 0.005: 2.5e-2, 0.006: 7.5e-2},
}


def load() -> dict[tuple[str, str, str], dict[str, Run]]:
    """{(p, preset, backend): {basis: tally}}."""
    runs: dict[tuple[str, str, str], dict[str, Run]] = defaultdict(dict)
    for path in sorted((RESULTS / "ler").glob("*/*.json")):
        _, basis, p = path.parent.name.split("_")
        preset, backend = path.stem.split("__")
        runs[(p[1:], preset, backend)][basis] = json.loads(path.read_text())
    return runs


def per_round(tallies: dict[str, Run]) -> dict[str, float]:
    fits = [sinter.fit_binomial(num_shots=t["shots"], num_hits=t["errors"], max_likelihood_factor=1e3)
            for t in tallies.values()]
    return {"ler": sum(f.best for f in fits) / ROUNDS, "low": sum(f.low for f in fits) / ROUNDS,
            "high": sum(f.high for f in fits) / ROUNDS,
            "errors": sum(t["errors"] for t in tallies.values()), "shots": sum(t["shots"] for t in tallies.values()),
            "unsatisfied": sum(t["unsatisfied"] for t in tallies.values()),
            "shots_per_s": sum(t["shots"] for t in tallies.values()) / sum(t["seconds"] for t in tallies.values())}


def paired(a: Run, b: Run) -> tuple[int, int, int, int]:
    """(common shots, fail only in a, only in b, both) on the shots both runs decoded."""
    shots = min(a["shots"], b["shots"])
    fa, fb = ({i for i in t["failed"] if i < shots} for t in (a, b))
    return shots, len(fa - fb), len(fb - fa), len(fa & fb)


def main() -> None:
    runs = load()
    summary = {f"{p}/{preset}/{backend}": per_round(t) for (p, preset, backend), t in runs.items() if len(t) == 2}
    (RESULTS / "ler.json").write_text(json.dumps(summary, indent=2))
    draw(summary).savefig(RESULTS / "ler.png", dpi=160, facecolor=SURFACE)
    (RESULTS / "ler.md").write_text(table(summary, runs))


def draw(summary: dict[str, dict[str, float]]) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 6.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for preset in PRESETS:
        pts = sorted((float(k.split("/")[0]), v) for k, v in summary.items()
                     if k.split("/")[1:] == [preset, "gpu-fp32"])
        if pts:
            xs, ys = [x for x, _ in pts], [v["ler"] for _, v in pts]
            err = [[v["ler"] - v["low"] for _, v in pts], [v["high"] - v["ler"] for _, v in pts]]
            ax.errorbar(xs, ys, yerr=err, color=COLORS[preset], lw=2, marker="o", ms=8, capsize=3,
                        mec=SURFACE, mew=1.5, zorder=3)
        paper = sorted(PAPER[preset].items())
        ax.plot([x for x, _ in paper], [y for _, y in paper], color=COLORS[preset], lw=1.2, ls="--",
                marker="o", ms=6, mfc="none", zorder=2)
    cpu = sorted((float(k.split("/")[0]), v["ler"]) for k, v in summary.items() if k.endswith("beam8_230/ionq"))
    ax.scatter([x for x, _ in cpu], [y for _, y in cpu], marker="x", s=70, color=INK, lw=1.5, zorder=4)
    paper = sorted(PAPER["bp30+osd"].items())
    ax.plot([x for x, _ in paper], [y for _, y in paper], color=MUTED, lw=1.2, ls=":", zorder=2)
    ax.text(paper[-1][0] * 1.03, paper[-1][1], "BP+OSD\n(paper)", fontsize=8.5, color=MUTED, va="center")
    ticks = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.00085, 0.0072)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("physical error rate p (circuit-level, uniform)", color=MUTED)
    ax.set_ylabel("logical error rate per syndrome round  (X + Z)", color=MUTED)
    ax.legend(handles=[*(Line2D([], [], color=COLORS[k], lw=6, label=LABELS[k]) for k in PRESETS),
                       Line2D([], [], color=MUTED, lw=2, marker="o", label="this work: CUDA float32, RTX 5070"),
                       Line2D([], [], color=MUTED, lw=1.2, ls="--", marker="o", mfc="none",
                              label="IonQ paper (C++), read off their figure"),
                       Line2D([], [], color=INK, marker="x", ls="", label="IonQ C++ re-run here, beam8 (same shots)")],
              loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle("Task A: logical error rate of the CUDA beam-search baseline", x=0.01, ha="left",
                 fontsize=14, color=INK)
    fig.text(0.01, 0.905, "BB[[144,12,12]] · memory-X and memory-Z · 12 rounds · up to 100 failures per basis "
             "(p = 0.001 capped at 2·10⁷ shots/basis)\nerror bars: binomial likelihood-ratio intervals (1000×)",
             ha="left", fontsize=9, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def table(summary: dict[str, dict[str, float]], runs: dict[tuple[str, str, str], dict[str, Run]]) -> str:
    rows = ["| p | preset | backend | failures / shots (X+Z) | LER per round [interval] | paper | "
            "non-converged failures | shots/s |", "|---|---|---|---|---|---|---|---|"]
    for key in sorted(summary, key=lambda k: (float(k.split("/")[0]), k)):
        p, preset, backend = key.split("/")
        v = summary[key]
        rows.append(f"| {p} | {LABELS.get(preset, preset)} | {backend} | {v['errors']} / {v['shots']:,} | "
                    f"{v['ler']:.2e} [{v['low']:.1e}, {v['high']:.1e}] | {PAPER.get(preset, {}).get(float(p), 0):.1e} | "
                    f"{v['unsatisfied']} | {v['shots_per_s']:,.0f} |")
    pairs = ["", "## Paired comparison (same shots)", "",
             "| p | basis | preset | A | B | common shots | fail only A | fail only B | both |", "|---|---|---|---|---|---|---|---|---|"]
    by_job: dict[tuple[str, str, str], dict[str, Run]] = defaultdict(dict)
    for (p, preset, backend), tallies in runs.items():
        for basis, t in tallies.items():
            by_job[(p, basis, preset)][backend] = t
    for (p, basis, preset), backends in sorted(by_job.items()):
        for a, b in combinations(sorted(backends), 2):
            if backends[a]["batch"] == backends[b]["batch"]:
                pairs.append(f"| {p} | {basis} | {LABELS.get(preset, preset)} | {a} | {b} | "
                             + " | ".join(f"{n:,}" for n in paired(backends[a], backends[b])) + " |")
    return "# Task A: logical error rate\n\n" + "\n".join(rows + pairs) + "\n"


if __name__ == "__main__":
    main()
