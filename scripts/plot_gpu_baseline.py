"""Task A figure: CUDA beam search (RTX 5070) vs the single-core CPU baseline.

Reads results/gpu_baseline.json + results/cpu_baseline.json; writes results/gpu_baseline.png/.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CASE = "bb144_z_p0.001"
PRESETS = ("beam8_230", "beam32_340", "beam64_640")
LABELS = {"beam8_230": "beam8", "beam32_340": "beam32", "beam64_640": "beam64"}
CPU_COLOR, GPU_COLOR = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ROUNDS = 12
SOTA_US_PER_SHOT = 42.5 * ROUNDS


def main() -> None:
    gpu = json.loads((RESULTS / "gpu_baseline.json").read_text())
    cpu = json.loads((RESULTS / "cpu_baseline.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6.4), facecolor=SURFACE)
    latency_panel(left, gpu, cpu)
    throughput_panel(right, gpu, cpu)
    fig.suptitle("Task A: CUDA beam search on RTX 5070 vs single-core CPU (IonQ C++)", x=0.01, y=0.985,
                 ha="left", fontsize=15, color=INK)
    fig.text(0.01, 0.875, ("BB[[144,12,12]] · memory-Z · circuit-level noise p = 0.001 · 12 rounds · "
                           "936 × 8,784 check matrix, 30,672 edges · same 10,000 seeded shots as CPU baseline\n"
                           "GPU: float32 kernel, one thread block per shot/child, all BP iterations in-kernel; "
                           "float64 mode reproduces IonQ bit-for-bit (3,550/3,550 shots)."),
             ha="left", fontsize=9.5, color=MUTED, linespacing=1.6)
    fig.text(0.01, 0.012, f"Host: {gpu['host']['gpu']}; host CPU shared (load avg {gpu['host']['loadavg']}).  "
             "CPU: Intel i7-1355U, 1 core.  Data: results/gpu_baseline.json, results/cpu_baseline.json",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.85))
    fig.savefig(RESULTS / "gpu_baseline.png", dpi=160, facecolor=SURFACE)
    (RESULTS / "gpu_baseline.md").write_text(table(gpu, cpu))


def latency_panel(ax: Axes, gpu: dict, cpu: dict) -> None:
    style(ax, log=True)
    for i, preset in enumerate(PRESETS):
        c, g = cpu[CASE][preset], gpu[f"{CASE}/{preset}/float32"]["latency"]
        for dx, color, mean, p999 in ((-0.17, CPU_COLOR, c["mean_ms"], c["p999_ms"]),
                                      (0.17, GPU_COLOR, g["mean_ms"], g["p999_ms"])):
            ax.bar(i + dx, mean, width=0.32, color=color, zorder=2)
            ax.scatter(i + dx, p999, marker="D", s=50, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(i + dx, mean * 1.12, f"{mean:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
            ax.text(i + dx + 0.07, p999, f"{p999:.1f}", ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(PRESETS)), [LABELS[p] for p in PRESETS], color=INK)
    ax.set_ylim(0.3, 200)
    ax.set_ylabel("ms per decode, batch = 1 (log)", color=MUTED)
    ax.set_title("Latency: bar = mean, ◆ = p99.9", loc="left", fontsize=11, color=INK)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CPU_COLOR), plt.Rectangle((0, 0), 1, 1, color=GPU_COLOR)],
              labels=["CPU, 1 core", "GPU, float32"], loc="upper left", frameon=False, fontsize=9)


def throughput_panel(ax: Axes, gpu: dict, cpu: dict) -> None:
    style(ax, log=True)
    ax.set_xscale("log")
    points = gpu[f"{CASE}/beam8_230/float32"]["throughput_shots_per_s"]
    batches = [int(b) for b in points]
    us = [1e6 / points[str(b)] for b in batches]
    ax.plot(batches, us, color=GPU_COLOR, lw=2, marker="o", ms=7, zorder=3)
    for b, u in zip(batches, us):
        ax.text(b, u * 1.18, f"{u:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(cpu[CASE]["beam8_230"]["mean_ms"] * 1e3, color=CPU_COLOR, lw=1.5, ls="--")
    ax.text(1.1, cpu[CASE]["beam8_230"]["mean_ms"] * 1e3 * 1.12, "CPU beam8, 1 core: 3,990 µs", fontsize=9, color=INK)
    ax.axhline(SOTA_US_PER_SHOT, color=INK, lw=1.2, ls=":")
    ax.text(30, SOTA_US_PER_SHOT * 1.12, "SOTA ref: cudaq-qec BP+OSD, A100, batch 256: 510 µs", fontsize=9, color=INK)
    ax.set_ylim(10, 20000)
    ax.set_xlabel("shots per decode_batch call (log)", color=MUTED)
    ax.set_ylabel("amortized µs per shot (log)", color=MUTED)
    ax.set_title("Throughput, beam8 float32: time per shot vs batch size", loc="left", fontsize=11, color=INK)


def style(ax: Axes, log: bool) -> None:
    ax.set_facecolor(SURFACE)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def table(gpu: dict, cpu: dict) -> str:
    rows = ["| p | preset | precision | GPU mean ms | GPU p50 | GPU p99.9 | CPU mean | CPU p99.9 | speedup mean / p99.9 "
            "| GPU best shots/s | µs/shot | µs/round |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, v in gpu.items():
        if key == "host":
            continue
        case, preset, precision = key.split("/")
        g, c, best = v["latency"], cpu[case][preset], max(v["throughput_shots_per_s"].values())
        rows.append(f"| {case.split('_p')[1]} | {LABELS[preset]} | {precision} | {g['mean_ms']:.2f} | {g['p50_ms']:.2f} | "
                    f"{g['p999_ms']:.1f} | {c['mean_ms']:.2f} | {c['p999_ms']:.1f} | "
                    f"{c['mean_ms'] / g['mean_ms']:.1f}× / {c['p999_ms'] / g['p999_ms']:.1f}× | {best:,.0f} | "
                    f"{1e6 / best:.1f} | {1e6 / best / ROUNDS:.2f} |")
    return "# Task A: GPU baseline (RTX 5070) vs CPU (1 core)\n\n" + "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
