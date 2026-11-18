"""Task B figure: baseline RT-core (OptiX) mapping vs the CUDA kernel and the CPU.

Reads results/{bp_iteration,rt_baseline,gpu_baseline,cpu_baseline}.json;
writes results/rt_baseline.png and results/rt_baseline.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CASE = "bb144_z_p0.001"
PRESETS = ("beam8_230", "beam32_340", "beam64_640")
LABELS = {"beam8_230": "beam8", "beam32_340": "beam32", "beam64_640": "beam64"}
CPU, CUDA, RT = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
Json = dict[str, Any]


def main() -> None:
    data = {k: json.loads((RESULTS / f"{k}.json").read_text())
            for k in ("bp_iteration", "rt_baseline", "gpu_baseline", "cpu_baseline")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), facecolor=SURFACE)
    iteration_panel(axes[0], data["bp_iteration"])
    latency_panel(axes[1], data)
    throughput_panel(axes[2], data)
    fig.suptitle("Task B: baseline RT-core mapping (OptiX, edges = triangles, nodes = rays) vs CUDA kernel",
                 x=0.01, y=0.985, ha="left", fontsize=15, color=INK)
    fig.text(0.01, 0.895, "BB[[144,12,12]] memory-Z, 936 checks × 8,784 bits, 30,672 edges (= triangles) · RTX 5070 · "
             "float32 · RT results bit-identical to CUDA float32 (3,550/3,550 golden shots), RT float64 bit-identical "
             "to IonQ", ha="left", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(RESULTS / "rt_baseline.png", dpi=150, facecolor=SURFACE)
    (RESULTS / "rt_baseline.md").write_text(table(data))


def iteration_panel(ax: Axes, it: Json) -> None:
    style(ax)
    ax.set_xscale("log")
    runs = [int(r) for r in it]
    series = ((CUDA, "-", "cuda_us_per_iter", "CUDA kernel"), (RT, "-", "rt_us_per_iter", "RT, end-to-end"),
              (RT, "--", "rt_check_pass_us", "RT check pass (row rays)"), (RT, ":", "rt_bit_pass_us", "RT bit pass (column rays)"))
    for color, ls, key, label in series:
        ax.plot(runs, [it[str(r)][key] for r in runs], color=color, ls=ls, lw=2, marker="o" if ls == "-" else None,
                ms=6, label=label)
    ax.set_xlabel("BP runs launched together (shots or beam children, log)", color=MUTED)
    ax.set_ylabel("µs per BP iteration (log)", color=MUTED)
    ax.set_title("1. One BP iteration, GPU time", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")


def latency_panel(ax: Axes, data: Json) -> None:
    style(ax)
    for i, preset in enumerate(PRESETS):
        values = ((CPU, data["cpu_baseline"][CASE][preset]),
                  (CUDA, data["gpu_baseline"][f"{CASE}/{preset}/float32"]["latency"]),
                  (RT, data["rt_baseline"][f"{CASE}/{preset}/float32"]["latency"]))
        for j, (color, v) in enumerate(values):
            x = i + (j - 1) * 0.27
            ax.bar(x, v["mean_ms"], width=0.25, color=color, zorder=2)
            ax.scatter(x, v["p999_ms"], marker="D", s=40, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
            ax.text(x, v["mean_ms"] * 1.12, f"{v['mean_ms']:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(PRESETS)), [LABELS[p] for p in PRESETS], color=INK)
    ax.set_ylim(0.3, 200)
    ax.set_ylabel("ms per decode, batch = 1 (log)", color=MUTED)
    ax.set_title("2. Decoder latency, p = 0.001: bar = mean, ◆ = p99.9", loc="left", fontsize=11, color=INK)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c) for c in (CPU, CUDA, RT)],
              labels=["CPU (IonQ C++, 1 core)", "CUDA", "RT"], frameon=False, fontsize=8.5, loc="upper left")


def throughput_panel(ax: Axes, data: Json) -> None:
    style(ax)
    ax.set_xscale("log")
    for color, key in ((CUDA, "gpu_baseline"), (RT, "rt_baseline")):
        for preset, ls in (("beam8_230", "-"), ("beam64_640", "--")):
            points = data[key][f"{CASE}/{preset}/float32"]["throughput_shots_per_s"]
            batches = [int(b) for b in points]
            ax.plot(batches, [points[str(b)] for b in batches], color=color, lw=2, ls=ls, marker="o", ms=5)
    ax.set_xlabel("shots per decode_batch call (log)", color=MUTED)
    ax.set_ylabel("shots / s (log)", color=MUTED)
    ax.set_title("3. Decoder throughput, p = 0.001", loc="left", fontsize=11, color=INK)
    ax.legend(handles=[Line2D([], [], color=CUDA, lw=2), Line2D([], [], color=RT, lw=2),
                       Line2D([], [], color=MUTED, lw=2), Line2D([], [], color=MUTED, lw=2, ls="--")],
              labels=["CUDA", "RT", "beam8", "beam64"], frameon=False, fontsize=8.5, loc="upper left")


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_yscale("log")
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def table(data: Json) -> str:
    it = data["bp_iteration"]
    lines = ["# Task B: baseline RT mapping vs CUDA (RTX 5070, float32)", "", "## One BP iteration (GPU time, µs)", "",
             "| runs | CUDA | RT total | RT check pass | RT bit pass | RT / CUDA |", "|---|---|---|---|---|---|"]
    for r, v in it.items():
        lines.append(f"| {r} | {v['cuda_us_per_iter']:.0f} | {v['rt_us_per_iter']:.0f} | {v['rt_check_pass_us']:.0f} | "
                     f"{v['rt_bit_pass_us']:.0f} | {v['rt_us_per_iter'] / v['cuda_us_per_iter']:.2f}× |")
    lines += ["", "## Decoder (beam search), batch = 1 latency and best throughput", "",
              "| p | preset | CPU mean / p99.9 ms | CUDA mean / p50 / p99.9 ms | RT mean / p50 / p99.9 ms | "
              "CUDA best shots/s | RT best shots/s |", "|---|---|---|---|---|---|---|"]
    for key, rt in data["rt_baseline"].items():
        if key == "host" or not key.endswith("float32"):
            continue
        case, preset, _ = key.split("/")
        cpu, cuda = data["cpu_baseline"][case][preset], data["gpu_baseline"][key]
        c, r = cuda["latency"], rt["latency"]
        lines.append(f"| {case.split('_p')[1]} | {LABELS[preset]} | {cpu['mean_ms']:.2f} / {cpu['p999_ms']:.1f} | "
                     f"{c['mean_ms']:.2f} / {c['p50_ms']:.2f} / {c['p999_ms']:.1f} | "
                     f"{r['mean_ms']:.2f} / {r['p50_ms']:.2f} / {r['p999_ms']:.1f} | "
                     f"{max(cuda['throughput_shots_per_s'].values()):,.0f} | {max(rt['throughput_shots_per_s'].values()):,.0f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
