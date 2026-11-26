"""Plot matched RTU dispatch latency and derived BP-run throughput."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COLORS = {"RTU": "#eb6834", "Cached edge IDs": "#1baf7a", "Cached + compacted": "#8b5fbf"}
SERIES = tuple(COLORS)


@dataclass(frozen=True)
class Measurement:
    runs: int
    milliseconds: float

    @property
    def bp_runs_per_second(self) -> float:
        return self.runs / (self.milliseconds / 1_000)


@dataclass(frozen=True)
class Panel:
    metric: str
    title: str


def main() -> None:
    data = read_results()
    fig, (latency, throughput) = plt.subplots(1, 2, figsize=(13, 6), facecolor="white",
                                               gridspec_kw={"width_ratios": (1, 1.25)})
    draw_panel(latency, data, Panel("milliseconds", "BP latency: four iterations, 64 fixed bits per run"))
    draw_panel(throughput, data, Panel("bp_runs_per_second", "Derived BP-run throughput (four iterations)"))
    finish_figure(fig, data)
    output = RESULTS / "rt_dispatch_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


def read_results() -> dict[int, dict[str, Measurement]]:
    raw = json.loads((RESULTS / "rt_compaction.json").read_text())
    return {int(runs): measurements(raw, int(runs)) for runs in raw["batch_sizes"]}


def measurements(raw: dict[str, object], runs: int) -> dict[str, Measurement]:
    group = cast(dict[str, dict[str, dict[str, float]]], raw["fixed_bits"])
    values = group["64"][str(runs)]
    return {"RTU": Measurement(runs, values["baseline_ms"]),
            "Cached edge IDs": Measurement(runs, values["cached_ms"]),
            "Cached + compacted": Measurement(runs, values["compacted_ms"])}


def draw_panel(ax: Axes, data: dict[int, dict[str, Measurement]], panel: Panel) -> None:
    for name in SERIES:
        points = [data[runs][name] for runs in data]
        y = [point.milliseconds if panel.metric == "milliseconds" else point.bp_runs_per_second for point in points]
        ax.plot([point.runs for point in points], y, marker="o", ms=4.5, lw=2, color=COLORS[name], label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("BP runs launched together")
    ax.set_ylabel("Milliseconds" if panel.metric == "milliseconds" else "BP runs/s")
    ax.set_title(panel.title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(ax)


def finish_figure(fig: plt.Figure, data: dict[int, dict[str, Measurement]]) -> None:
    gpu = data[1024]["RTU"].runs
    fig.suptitle("RTU dispatch optimization: edge caching and active-task compaction", x=.01, y=.985, ha="left", fontsize=15)
    fig.text(.01, .012, f"BB[[144,12,12]] · memory-Z · FP32 · RTX 5070 · {gpu:,} max BP runs · "
             "four BP iterations · 64 synthetic fixed-bit masks/run (modeling assigned beam-path bits).\n"
             "All variants use identical seeded syndromes and masks; BP outputs matched exactly. "
             "Right panel is derived from this microbenchmark, not full-decoder shots/s.", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, .06, 1, .94))


def style_axes(ax: Axes) -> None:
    ax.grid(True, which="major", axis="y", color="#dedede", lw=.8, zorder=0)
    ax.grid(True, which="minor", axis="y", color="#eeeeee", lw=.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#555", length=0)


if __name__ == "__main__":
    main()
