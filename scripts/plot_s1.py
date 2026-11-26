"""S1 figure: (a) paired LER ratio of frontier freezing vs unpruned CUDA, (b) batch-one latency split.

Reads results/s1_ler.json (scripts/s1_summary.py) and results/latency_breakdown.json; writes
results/s1_freezing.png.
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
P_COLORS = {"0.003": "#eb6834", "0.002": "#1baf7a"}
PARTS = {"initial_gpu_mean_ms": ("initial BP (GPU)", "#eb6834"), "rounds_gpu_mean_ms": ("beam rounds (GPU)", "#f4a582"),
         "other_mean_ms": ("host / copies", "#b9b8b3")}
CONFIGS = {"fr8w0": "τ 8", "fr8w3": "τ 8, warm 3", "fr16w0": "τ 16", "fr32w0": "τ 32", "fr32w3": "τ 32, warm 3",
           "fr64w0": "τ 64", "fr64w3": "τ 64, warm 3", "fr32w0r2": "τ 32, refresh 2", "fr32w3r5": "τ 32, warm 3, refresh 5",
           "fr32w0r5": "τ 32, refresh 5", "fr32w0i5r5": "τ 32, refresh 5,\n+ initial-BP freeze"}
LATENCY = {"unpruned": "unpruned", "tau32": "τ 32", "tau32_w3_init5": "τ 32, warm 3\n+ initial-BP freeze"}


def main() -> None:
    fig, (ler, latency) = plt.subplots(1, 2, figsize=(14, 6), facecolor="white", gridspec_kw={"width_ratios": (1.1, 1)})
    draw_ler(ler, json.loads((RESULTS / "s1_ler.json").read_text()))
    draw_latency(latency, json.loads((RESULTS / "latency_breakdown.json").read_text()))
    fig.tight_layout()
    out = RESULTS / "s1_freezing.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


def draw_ler(ax: Axes, report: dict[str, dict[str, float]]) -> None:
    names = list(CONFIGS)
    for offset, (p, color) in zip((-0.15, 0.15), P_COLORS.items()):
        for row, name in enumerate(names):
            r = report.get(f"{p}/beam8_230/{name}")
            if r is None:
                continue
            significant = r["sign_test_p"] < 0.05
            ax.plot(r["ler_ratio"], row + offset, "o", ms=8, color=color, mfc=color if significant else "white", mew=2)
    ax.axvline(1.0, color="#555", lw=1, zorder=0)
    ax.set_yticks(range(len(names)), [CONFIGS[n] for n in names])
    ax.set_xlabel("failures with freezing / failures unpruned (same shots, X+Z)")
    ax.set_title("(a) Paired LER, beam8", loc="left", fontweight="bold")
    for p, color in P_COLORS.items():
        ax.plot([], [], "o", color=color, ms=8, label=f"p = {p}")
    ax.plot([], [], "o", color="#555", mfc="white", mew=2, ms=8, label="not significant (sign test p ≥ 0.05)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="x", color="#eee")
    ax.spines[["top", "right"]].set_visible(False)


def draw_latency(ax: Axes, report: dict[str, dict[str, float]]) -> None:
    rows = [(f"z/{p}/{key}", f"p = {p}: {label}") for p in ("0.001", "0.003") for key, label in LATENCY.items()]
    rows = [(key, label) for key, label in rows if key in report]
    for i, (key, _) in enumerate(rows):
        left = 0.0
        for part, (_, color) in PARTS.items():
            value = report[key][part]
            ax.barh(i, value, left=left, color=color, edgecolor="white", linewidth=2, height=0.7)
            left += value
        ax.text(left + 0.05, i, f"{report[key]['wall_mean_ms']:.2f} ms", va="center", fontsize=9, color="#333")
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.invert_yaxis()
    ax.set_xlabel("mean batch-one latency (ms), 300 non-trivial shots")
    ax.set_title("(b) Where batch-one latency goes", loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=color, label=label) for label, color in PARTS.values()], loc="upper right",
              frameon=False)
    ax.set_xlim(0, 6.5)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    main()
