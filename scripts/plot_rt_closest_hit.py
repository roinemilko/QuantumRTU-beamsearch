"""R0/R1 figure: closest-hit min-sum ("messages as geometry") vs CUDA and the RT baseline.

(a) one BP iteration end to end vs batch size; (b) where a closest-hit iteration's GPU time goes
(RT trace of the check minimums vs BVH refit), next to a whole CUDA iteration for scale.
Reads results/ch_iteration.json (quiet-GPU re-run); writes results/rt_closest_hit.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SERIES = {"cuda_us_per_iter": ("CUDA kernel", "#eb6834"), "rt_us_per_iter": ("RT baseline (neighbour lookup)", "#1baf7a"),
          "ch_refit_us_per_iter": ("RT closest-hit min-sum", "#8b5fbf")}
PARTS = {"ch_min_trace_gpu_us": ("RT trace: check minimums", "#1baf7a"), "ch_refit_gpu_us": ("BVH refit (geometry update)", "#8b5fbf")}


def main() -> None:
    data = json.loads((RESULTS / "ch_iteration.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    draw_iteration(a, data)
    draw_split(b, data)
    fig.tight_layout()
    out = RESULTS / "rt_closest_hit.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


def draw_iteration(ax: Axes, data: dict[str, dict[str, float]]) -> None:
    runs = sorted(int(r) for r in data)
    for key, (label, color) in SERIES.items():
        ax.plot(runs, [data[str(r)][key] for r in runs], "o-", color=color, lw=2, ms=7, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(runs, [str(r) for r in runs])
    ax.set_xlabel("BP runs processed together")
    ax.set_ylabel("µs per BP iteration (log)")
    ax.set_title("(a) One BP iteration, end to end", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(color="#eee")
    ax.spines[["top", "right"]].set_visible(False)


def draw_split(ax: Axes, data: dict[str, dict[str, float]]) -> None:
    runs = sorted(int(r) for r in data)
    bars = {**PARTS, "cuda_us_per_iter": ("whole CUDA iteration (for scale)", "#eb6834")}
    for i, r in enumerate(runs):
        for j, (key, (_, color)) in enumerate(bars.items()):
            ax.bar(i + (j - 1) * 0.27, data[str(r)][key], color=color, width=0.25)
    ax.set_yscale("log")
    ax.set_xticks(range(len(runs)), [str(r) for r in runs])
    ax.set_xlabel("BP runs processed together")
    ax.set_ylabel("GPU µs per iteration (log)")
    ax.set_title("(b) Closest-hit: the minimums are cheap, the refit is not", loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=c, label=label) for label, c in bars.values()], frameon=False, loc="upper left")
    ax.grid(axis="y", color="#eee")
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    main()
