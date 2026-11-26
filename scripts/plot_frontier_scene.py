"""S3 figure: the same frontier pruning done with CUDA index lists vs a per-round RT scene.

(a) cost of one 20-iteration beam round vs batch size (warm-up k = 3): unpruned CUDA, pruned
CUDA lists, pruned RT scene; (b) what a pruned round is made of, CUDA vs RT, at each batch size.
Reads results/frontier_scene.json; writes results/frontier_scene.png.
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
K = 3
ITERS = 20
SERIES = {"round_unpruned_cuda_us": ("unpruned CUDA", "#b9b8b3"), "round_cuda_us": ("pruned, CUDA index lists", "#eb6834"),
          "round_rt_us": ("pruned, RT scene", "#1baf7a")}
PARTS = (("setup (frontier choice, lists)", "#b9b8b3"), ("RT scene build (BVH)", "#8b5fbf"),
         ("20 × check pass", "#eb6834"), ("20 × bit pass", "#f4a582"))


def main() -> None:
    data = json.loads((RESULTS / "frontier_scene.json").read_text())
    rows = {int(key.split("/")[0]): row for key, row in data.items() if key.endswith(f"/{K}")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    draw_rounds(a, rows)
    draw_parts(b, rows)
    fig.tight_layout()
    out = RESULTS / "frontier_scene.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


def draw_rounds(ax: Axes, rows: dict[int, dict[str, float]]) -> None:
    runs = sorted(rows)
    for key, (label, color) in SERIES.items():
        ax.plot(runs, [rows[r][key] / 1e3 for r in runs], "o-", color=color, lw=2, ms=7, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(runs, [str(r) for r in runs])
    ax.set_xlabel("BP runs processed together")
    ax.set_ylabel("ms per 20-iteration beam round (log)")
    ax.set_title("(a) Pruning pays; RT pruning pays less", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(color="#eee")
    ax.spines[["top", "right"]].set_visible(False)


def draw_parts(ax: Axes, rows: dict[int, dict[str, float]]) -> None:
    runs = sorted(rows)
    labels = []
    for i, r in enumerate(runs):
        row = rows[r]
        for j, (variant, check) in enumerate((("CUDA", "check_cuda_us"), ("RT", "check_rt_us"))):
            values = (row["setup_shared_us"], row["setup_rt_scene_us"] if variant == "RT" else 0.0,
                      ITERS * row[check], ITERS * row["bit_pass_us"])
            bottom, x = 0.0, i * 2.6 + j
            for value, (_, color) in zip(values, PARTS):
                ax.bar(x, value / 1e3, bottom=bottom, color=color, edgecolor="white", linewidth=1.5, width=0.85)
                bottom += value / 1e3
            labels.append((x, f"{variant}\n{r} run{'s' if r > 1 else ''}"))
    ax.set_xticks([x for x, _ in labels], [label for _, label in labels], fontsize=9)
    ax.set_ylabel("ms per pruned round")
    ax.set_title("(b) Inside a pruned round", loc="left", fontweight="bold")
    ax.legend(handles=[Patch(color=c, label=label) for label, c in PARTS], frameon=False, loc="upper left")
    ax.grid(axis="y", color="#eee")
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    main()
