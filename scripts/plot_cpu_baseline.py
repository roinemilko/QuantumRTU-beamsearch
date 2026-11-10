"""Plot the single-core CPU baseline (results/cpu_baseline.json) as the target GPU decoders must beat.

Writes results/cpu_baseline.png and results/cpu_baseline.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEM_STEM = "BB[[144,12,12]],memory_Z,error_rate=0.001,syndrome_rounds=12"
ROUNDS = 12
CPU = "Intel Core i7-1355U (laptop, WSL2), 1 core"

DECODERS = ("beam8_230", "beam32_340", "beam64_640", "bp30+osd")
LABELS = {"beam8_230": "beam8", "beam32_340": "beam32", "beam64_640": "beam64", "bp30+osd": "BP+OSD"}
COLORS = {"beam8_230": "#2a78d6", "beam32_340": "#eb6834", "beam64_640": "#1baf7a", "bp30+osd": "#eda100"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# IonQ paper (arXiv 2512.07057) Tables 2-3: single Apple M3 Pro core, ms per 12-round decode.
PAPER = {
    "0.0005": {"beam8_230": (1.627, 8.704), "beam32_340": (3.202, 11.26), "beam64_640": (5.881, 14.53),
               "bp30+osd": (3.55, 272.5)},
    "0.001": {"beam8_230": (2.318, 11.01), "beam32_340": (3.837, 14.18), "beam64_640": (6.698, 17.70),
              "bp30+osd": (10.59, 289.0)},
}
# arXiv 2605.30765: cudaq-qec BP30+OSD-0 on one A100, batch 256, mean 42.5 us/round at p=1e-3 (amortized).
GPU_SOTA_MS = 0.0425 * ROUNDS


@dataclass(frozen=True)
class Point:
    """One decoder at one error rate: measured mean / p99.9 per full decode, in ms."""

    decoder: str
    mean: float
    p999: float
    shots: int


def main() -> None:
    measured = json.loads((RESULTS / "cpu_baseline.json").read_text())
    stats = json.loads((ROOT / "data" / "dem" / "stats.json").read_text())[DEM_STEM]
    panels = {p: points(measured[f"bb144_z_p{p}"]) for p in ("0.0005", "0.001")}
    draw(panels, stats).savefig(RESULTS / "cpu_baseline.png", dpi=160, facecolor=SURFACE)
    (RESULTS / "cpu_baseline.md").write_text(table(panels, stats))


def points(run: dict[str, dict[str, float]]) -> list[Point]:
    return [Point(d, run[d]["mean_ms"], run[d]["p999_ms"], int(run[d]["shots"])) for d in DECODERS if d in run]


def draw(panels: dict[str, list[Point]], stats: dict[str, float]) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), sharey=True, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [len(pts) for pts in panels.values()]})
    for ax, (p, pts) in zip(axes, panels.items()):
        draw_panel(ax, p, pts)
    axes[0].set_ylabel("ms per decode (12 syndrome rounds), log scale", color=MUTED)
    legend(axes[0])
    fig.suptitle("CPU baseline to beat: IonQ beam search and BP+OSD, single core", x=0.01, y=0.985,
                 ha="left", fontsize=15, color=INK)
    fig.text(0.01, 0.855, subtitle(stats), ha="left", fontsize=9.5, color=MUTED, linespacing=1.6)
    fig.text(0.01, 0.012, footer(), ha="left", fontsize=8.5, color=MUTED, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.075, 1, 0.83))
    return fig


def draw_panel(ax: Axes, p: str, pts: list[Point]) -> None:
    style(ax)
    for i, pt in enumerate(pts):
        ax.bar(i, pt.mean, width=0.55, color=COLORS[pt.decoder], zorder=2)
        ax.scatter(i, pt.p999, marker="D", s=56, color=COLORS[pt.decoder], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(i, pt.mean / 1.2, f"{pt.mean:.2f}", ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold")
        ax.text(i + 0.12, pt.p999, f"{pt.p999:.0f}", ha="left", va="center", fontsize=9.5, color=INK)
        ax.scatter([i - 0.38] * 2, PAPER[p][pt.decoder], marker="_", s=140, linewidth=2, color=MUTED, zorder=3)
    ax.axhline(ROUNDS, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_xlim(-0.65, len(pts) - 0.35)
    ax.set_xticks(range(len(pts)), [LABELS[pt.decoder] for pt in pts], color=INK, fontsize=10.5)
    ax.set_title(f"physical error rate p = {p}   ({pts[0].shots:,} shots)", loc="left", fontsize=11, color=INK)
    if p == "0.001":
        ax.axhline(GPU_SOTA_MS, color=INK, lw=1.2, ls=":", zorder=4)
        ax.annotate("Task A target: beat beam8\n3.99 ms mean · 21.9 ms p99.9", xy=(0, 4.9), xytext=(0.3, 170),
                    fontsize=10.5, color=INK, fontweight="bold",
                    arrowprops={"arrowstyle": "->", "color": INK, "lw": 1})


def legend(ax: Axes) -> None:
    handles = [
        Patch(color=MUTED, label="mean (bar, value inside)"),
        Line2D([], [], marker="D", ls="", color=MUTED, label="99.9th percentile (value beside)"),
        Line2D([], [], marker="_", ms=12, mew=2, ls="", color=MUTED, label="IonQ paper, M3 Pro core: mean & p99.9"),
        Line2D([], [], ls="--", color=MUTED, label="real-time budget: 1 ms/round = 12 ms"),
        Line2D([], [], ls=":", lw=1.2, color=INK, label="GPU SOTA ref (p=1e-3): cudaq-qec BP+OSD,\nA100, batch 256 → 0.51 ms"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3000)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def subtitle(stats: dict[str, float]) -> str:
    return (f"Code: BB[[144,12,12]] gross code · memory-Z experiment · circuit-level depolarizing noise · "
            f"{ROUNDS} syndrome rounds\n"
            f"Decoding graph: {int(stats['checks'])} detectors × {int(stats['bits']):,} fault mechanisms = "
            f"{int(stats['edges']):,} Tanner edges (check degree {int(stats['check_degree_min'])}-"
            f"{int(stats['check_degree_max'])}, fault degree {int(stats['bit_degree_min'])}-{int(stats['bit_degree_max'])})\n"
            "Every decode reproduced its syndrome; 0 logical failures in every set (too few shots to measure LER).")


def footer() -> str:
    return (f"Hardware: {CPU}; glibc mmap threshold raised to remove an allocator page-fault artifact.  "
            "Data: results/cpu_baseline.json\n"
            "beam8 = 30 + 10×20 BP iterations, width 8 · beam32 = 40 + 10×30, width 32 · "
            "beam64 = 40 + 20×30, width 64 · BP+OSD = min-sum 30 iterations + OSD-CS order 10")


def table(panels: dict[str, list[Point]], stats: dict[str, float]) -> str:
    rows = ["| p | decoder | shots | mean ms | p99.9 ms | mean ms/round | p99.9 ms/round | paper mean / p99.9 ms |",
            "|---|---|---|---|---|---|---|---|"]
    for p, pts in panels.items():
        rows += [f"| {p} | {LABELS[pt.decoder]} | {pt.shots:,} | {pt.mean:.2f} | {pt.p999:.1f} | "
                 f"{pt.mean / ROUNDS:.3f} | {pt.p999 / ROUNDS:.2f} | {PAPER[p][pt.decoder][0]} / {PAPER[p][pt.decoder][1]} |"
                 for pt in pts]
    head = (f"# CPU baseline\n\nBB[[144,12,12]], memory-Z, circuit-level noise, {ROUNDS} rounds; "
            f"{int(stats['checks'])} × {int(stats['bits'])} check matrix, {int(stats['edges'])} edges. {CPU}.\n\n")
    target = ("\n**Target for Task A (p = 1e-3, beam8):** mean 3.99 ms, p99.9 21.9 ms per decode. "
              f"GPU SOTA reference: cudaq-qec BP+OSD on A100, {GPU_SOTA_MS:.2f} ms per 12-round decode "
              "(amortized, batch 256).\n")
    return head + "\n".join(rows) + "\n" + target


if __name__ == "__main__":
    main()
