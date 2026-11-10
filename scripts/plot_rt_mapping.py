"""Explainer figure for docs/rt_mapping.md: Tanner graph edges as triangles, checks/bits as rays.

Toy example: the [7,4] Hamming code (3 checks × 7 bits). Writes docs/rt_mapping.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])
ROW_RAY, COL_RAY = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE, TRI = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#8a8984"


def main() -> None:
    fig, (graph, scene) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE,
                                       gridspec_kw={"width_ratios": [1, 1.45]})
    tanner(graph)
    geometry(scene)
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "rt_mapping.png", dpi=150, facecolor=SURFACE)


def tanner(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    checks, bits = H.shape
    cx = np.linspace(1.5, 4.5, checks)
    for c, v in zip(*np.nonzero(H)):
        color = ROW_RAY if c == 1 else (COL_RAY if v == 4 else GRID)
        ax.plot([cx[c], v], [1, 0], color=color, lw=2.2 if color != GRID else 1.2, zorder=1)
    ax.scatter(range(bits), [0] * bits, s=260, color=SURFACE, edgecolor=INK, lw=1.5, zorder=2)
    ax.scatter(cx, [1] * checks, s=260, marker="s", color=SURFACE, edgecolor=INK, lw=1.5, zorder=2)
    for v in range(bits):
        ax.text(v, -0.2, f"bit {v}", ha="center", va="top", fontsize=9, color=INK)
    for c in range(checks):
        ax.text(cx[c], 1.17, f"check {c}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_title("1. Tanner graph (toy: Hamming [7,4])", loc="left", fontsize=11, color=INK)
    ax.set_xlim(-0.7, bits - 0.3)
    ax.set_ylim(-0.55, 1.5)
    ax.axis("off")


def geometry(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    checks, bits = H.shape
    for c, v in zip(*np.nonzero(H)):
        hit = c == 1 or v == 4
        ax.add_patch(Polygon([(v - 0.22, c), (v, c + 0.22), (v + 0.22, c), (v, c - 0.22)],
                             closed=True, color=INK if hit else TRI, zorder=2))
    ax.add_patch(FancyArrowPatch((-0.9, 1), (bits - 0.1, 1), arrowstyle="-|>", mutation_scale=16,
                                 color=ROW_RAY, lw=2.2, zorder=3))
    ax.add_patch(FancyArrowPatch((4, -0.9), (4, checks - 0.25), arrowstyle="-|>", mutation_scale=16,
                                 color=COL_RAY, lw=2.2, zorder=3))
    ax.text(-0.9, 1.3, "row ray = check 1\n(check-node update)", fontsize=9, color=INK, va="bottom")
    ax.text(4.15, -0.85, "column ray = bit 4\n(bit-node update)", fontsize=9, color=INK, va="bottom")
    ax.set_xticks(range(bits), [str(v) for v in range(bits)])
    ax.set_yticks(range(checks), [str(c) for c in range(checks)])
    ax.set_xlabel("x = bit index", color=MUTED)
    ax.set_ylabel("y = check index", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(-1.1, bits)
    ax.set_ylim(-1.1, checks + 0.2)
    ax.set_aspect("equal")
    ax.set_title("2. Scene: one triangle per edge (H[c,v] = 1); black = hits of the two rays",
                 loc="left", fontsize=11, color=INK)


if __name__ == "__main__":
    main()
