"""Toy figure: Tanner graph (left) and the same edges as triangles hit by a row and a column ray (right).

Check c1 = row ray (check-node update), bit v3 = column ray (bit-node update). Writes docs/rt_rays.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
EDGES = {0: (0, 1, 3), 1: (1, 2, 4), 2: (3, 4, 5)}  # check -> bits
ROW, COL = 1, 3  # highlighted check (row ray) and bit (column ray)
CHECK, BIT = "#eb6834", "#a8431b"  # orange family: check side, bit side (darker)
CHECK_FILL, BIT_FILL = "#fde3d6", "#f6d5c4"
INK, MUTED, IDLE, SURFACE = "#0b0b0b", "#52514e", "#b9b8b3", "#fcfcfb"


def main() -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": (1, 1.15)})
    tanner(left)
    scene(right)
    fig.tight_layout(w_pad=4)
    out = ROOT / "docs" / "rt_rays.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(out)


def tanner(ax: Axes) -> None:
    blank(ax, (-0.6, 5.6), (-0.6, 3.6))
    cx = {c: 1 + 1.5 * c for c in EDGES}  # checks spread over the bits
    for c, bits in EDGES.items():
        for v in bits:
            color, lw = (CHECK, 2.4) if c == ROW else (BIT, 2.4) if v == COL else (IDLE, 1.0)
            ax.plot([cx[c], v], [3, 0], color=color, lw=lw, zorder=1, solid_capstyle="round")
    for c, x in cx.items():
        on = c == ROW
        ax.add_patch(FancyBboxPatch((x - 0.27, 3 - 0.27), 0.54, 0.54, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=CHECK_FILL, ec=CHECK, lw=2.4 if on else 1.0, zorder=2))
        ax.text(x, 3, f"c{c}", ha="center", va="center", fontsize=13, color=INK,
                fontweight="bold" if on else "normal", zorder=3)
    for v in range(6):
        on = v == COL
        ax.add_patch(Circle((v, 0), 0.3, fc=BIT_FILL, ec=BIT, lw=2.4 if on else 1.0, zorder=2))
        ax.text(v, 0, f"v{v}", ha="center", va="center", fontsize=13, color=INK,
                fontweight="bold" if on else "normal", zorder=3)


def scene(ax: Axes) -> None:
    blank(ax, (-1.3, 6.4), (-0.75, 3.0))
    y = {c: 2 - c for c in EDGES}  # c0 on top
    ax.annotate("", xy=(6.2, y[ROW]), xytext=(-0.6, y[ROW]),
                arrowprops={"arrowstyle": "-|>", "color": CHECK, "lw": 2.2, "mutation_scale": 18}, zorder=1)
    ax.annotate("", xy=(COL, -0.55), xytext=(COL, 2.75),
                arrowprops={"arrowstyle": "-|>", "color": BIT, "lw": 2.2, "mutation_scale": 18}, zorder=1)
    for c, bits in EDGES.items():
        for v in bits:
            color = CHECK if c == ROW else BIT if v == COL else IDLE
            ax.add_patch(Polygon([(v - 0.2, y[c] - 0.16), (v + 0.2, y[c] - 0.16), (v, y[c] + 0.2)],
                                 fc=color, ec=SURFACE, lw=1.5, zorder=2))
    for c in EDGES:
        ax.text(-1.05, y[c], f"c{c}", ha="center", va="center", fontsize=13, color=MUTED)
    for v in range(6):
        ax.text(v, -0.75, f"v{v}", ha="center", va="center", fontsize=13, color=MUTED)
    ax.text(6.2, y[ROW] + 0.28, "row ray = c1", ha="right", fontsize=11, color=INK)
    ax.text(COL + 0.12, 2.65, "column ray = v3", ha="left", va="top", fontsize=11, color=INK)


def blank(ax: Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    main()
