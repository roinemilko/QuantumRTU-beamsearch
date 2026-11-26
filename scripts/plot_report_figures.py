"""Every report figure, redrawn under one convention. Writes results/report_figures/*.png.

The original per-experiment figures (results/*.png, docs/rt_rays.png) are left untouched.

Convention (same in every figure):
- Two levels of measurement, never mixed on one axis:
  * Decoder (whole 12-round shots): latency = ms per shot at batch 1 (bar = mean, ◆ = p99.9);
    throughput = shots / s vs shots per batch. Both from the same 10,000 seeded p = 0.001 shots.
  * BP kernel (inside the decoder): µs per BP iteration vs BP runs processed together.
    S3 rounds are shown per iteration too (round ÷ 20, setup included).
- Panel titles start with the level: "Decoder latency", "Decoder throughput", "BP kernel", "Accuracy".
- Colour = implementation: CPU blue, CUDA orange, RT baseline green, the RT variant under test purple,
  setup / host / references grey. Shades of one colour = sub-parts or beam width (light = 8, dark = 64).
- Latency panels carry two real-time budgets for a 12-round shot: trapped ions (1 ms per round -> 12 ms,
  dashed) and superconducting qubits (1 µs per round -> 12 µs, dash-dot).
- Throughput panels mark the one published GPU number for this code, BP+OSD at batch 256 on an A100
  (arXiv:2605.30765), as a single point: it is one measurement, not a curve.
- CPU = IonQ C++ on one core. Latency: the laptop core of the Task A baseline (results/cpu_baseline.json,
  with percentiles). Throughput: the measured single-thread sweep on the GPU host (results/throughput_cpu.json).
  ``--cpu-throughput laptop`` takes the throughput sweep from the same laptop core instead
  (results/throughput_cpu_laptop.json) and writes the whole set to results/report_figures_laptop_cpu/.
- Decoder throughput for every GPU backend comes from one batch sweep (results/throughput_*.json, same
  session, beam8), so batch-1 throughput agrees with batch-1 latency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "report_figures"
CPU_THROUGHPUT = "throughput_cpu"  # main(): "--cpu-throughput laptop" switches to throughput_cpu_laptop
CPU_THROUGHPUT_HOST = "GPU-host"
Json = dict[str, Any]

CPU, CUDA, RT, RTV = "#2a78d6", "#eb6834", "#1baf7a", "#8b5fbf"
SHADES = {CPU: ("#9cc3ef", CPU, "#174a8b"), CUDA: ("#f6b08d", CUDA, "#a8431b"), RTV: ("#c9b3e6", RTV, "#57307f")}
GREY, INK, MUTED, GRID = "#b9b8b3", "#0b0b0b", "#52514e", "#e7e6e2"
NAMES = {"cpu": "CPU (IonQ C++, 1 laptop core)", "cuda": "CUDA", "rt": "RT baseline", "rt-cache": "RT, cached edge IDs"}
COLOR = {"cpu": CPU, "cuda": CUDA, "rt": RT, "rt-cache": RTV}
CASE = "bb144_z_p0.001"
PRESETS = ("beam8_230", "beam32_340", "beam64_640")
BEAM = {"beam8_230": "beam8", "beam32_340": "beam32", "beam64_640": "beam64", "bp30+osd": "BP+OSD"}
BUDGETS_MS = ((12.0, "--", "trapped\nions"), (0.012, "-.", "super-\nconducting"))  # 1 ms / 1 µs per round × 12
BPOSD_SHOTS_PER_S = 1e6 / (42.5 * 12)  # arXiv 2605.30765: BP+OSD, 42.5 µs/round amortised, batch 256, A100
BPOSD_BATCH = 256
LATENCY_YLIM = (0.005, 30_000.0)
THROUGHPUT_YLIM = (100.0, 40_000.0)
KERNEL_YLIM = (10.0, 30_000.0)
SETUP = "BB[[144,12,12]] memory-Z · 12 rounds · float32 · RTX 5070"

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#9a9995",
                     "legend.frameon": False, "legend.fontsize": 8.5, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})


def load(name: str) -> Json:
    data: Json = json.loads((RESULTS / f"{name}.json").read_text())
    return data


# ---------------------------------------------------------------- shared panels

def figure(ncols: int, footer: str, width: float | None = None, ratios: tuple[float, ...] | None = None
           ) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(1, ncols, figsize=(width or 6.2 * ncols, 5.2),
                             gridspec_kw={"width_ratios": ratios} if ratios else None)
    fig.text(0.005, -0.02, footer, ha="left", va="top", fontsize=8, color=MUTED)
    return fig, list(axes) if ncols > 1 else [axes]


def tidy(ax: Axes, grid: Literal["both", "x", "y"] = "y") -> None:
    ax.grid(axis=grid, color=GRID, lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def log_ticks(ax: Axes, ticks: list[int], axis: str = "x") -> None:
    labels = [f"{t // 1000}k" if t >= 1000 and t % 1000 == 0 and t >= 4000 else f"{t:,}" for t in ticks]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks, labels)
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(matplotlib.ticker.NullLocator())


def decoder_latency(ax: Axes, backends: tuple[str, ...], title: str = "Decoder latency, p = 0.001") -> None:
    reports = {"cpu": load("cpu_baseline")[CASE], "cuda": load("gpu_baseline"), "rt": load("rt_baseline"),
               "rt-cache": load("rt-cache_baseline")}
    n = len(backends)
    width = 0.8 / n
    for i, preset in enumerate(PRESETS):
        for j, name in enumerate(backends):
            v = reports[name][preset] if name == "cpu" else reports[name][f"{CASE}/{preset}/float32"]["latency"]
            x = i + (j - (n - 1) / 2) * width
            ax.bar(x, v["mean_ms"], width=width * 0.88, color=COLOR[name], zorder=2)
            ax.scatter(x, v["p999_ms"], marker="D", s=34, color=COLOR[name], edgecolor="white", lw=1.2, zorder=3)
            ax.text(x, v["mean_ms"] * 1.1, f"{v['mean_ms']:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    budget(ax)
    ax.set_yscale("log")
    ax.set_ylim(*LATENCY_YLIM)
    ax.set_xticks(range(len(PRESETS)), [BEAM[p] for p in PRESETS], color=INK)
    ax.set_ylabel("ms per shot, batch 1 (log)")
    ax.set_title(title)
    ax.legend(handles=[*(Patch(color=COLOR[b], label=NAMES[b]) for b in backends), *latency_keys()], loc="upper left")
    tidy(ax)


def latency_keys() -> list[Line2D]:
    return [Line2D([], [], marker="D", ls="", color=MUTED, label="99.9th percentile (bar = mean)"),
            Line2D([], [], ls="--", color=MUTED, label="real-time budget, trapped ions: 12 ms (1 ms per round)"),
            Line2D([], [], ls="-.", color=MUTED, label="real-time budget, superconducting: 12 µs (1 µs per round)")]


def budget(ax: Axes) -> None:
    for value, ls, label in BUDGETS_MS:
        ax.axhline(value, color=MUTED, lw=1, ls=ls, zorder=1)
        ax.text(1.01, value, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8,
                color=MUTED, clip_on=False)


def decoder_throughput(ax: Axes, backends: tuple[str, ...]) -> None:
    ax.scatter(BPOSD_BATCH, BPOSD_SHOTS_PER_S, marker="*", s=160, color=INK, zorder=4)
    ax.text(BPOSD_BATCH * 1.4, BPOSD_SHOTS_PER_S / 1.12, f"BP+OSD (arXiv:2605.30765):\n{BPOSD_SHOTS_PER_S:,.0f} shots/s",
            fontsize=8.5, color=INK, va="top")
    for name in ("cpu", *backends):
        points = load(CPU_THROUGHPUT if name == "cpu" else f"throughput_{name}")["batch_throughput_shots_per_s"]
        batches = sorted(int(b) for b in points)
        rates = [points[str(b)] for b in batches]
        label = (NAMES["cpu"] if CPU_THROUGHPUT_HOST == "laptop" else "CPU (IonQ C++, 1 thread, GPU host)") \
            if name == "cpu" else NAMES[name]
        ax.plot(batches, rates, "o-", color=COLOR[name], lw=2, ms=5, label=label, zorder=3)
        peak = max(rates)
        ax.text(batches[-1] * 1.25, rates[-1], f"peak\n{peak / 1000:.1f}k" if peak >= 1000 else f"peak\n{peak:,.0f}",
                va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_ticks(ax, [1, 16, 64, 256, 1024, 4096, 10_000, 100_000])
    ax.set_xlim(0.7, 4e5)
    ax.set_ylim(*THROUGHPUT_YLIM)
    ax.set_xlabel("shots decoded together (batch size, log)")
    ax.set_ylabel("shots / s (log)")
    ax.set_title("Decoder throughput, p = 0.001, beam8")
    ax.legend(loc="lower right")
    tidy(ax, "both")


def kernel_axes(ax: Axes, runs: list[int], title: str, ylabel: str = "µs per BP iteration (log)") -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    log_ticks(ax, runs)
    ax.set_ylim(*KERNEL_YLIM)
    ax.set_xlabel("BP runs processed together (log)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    tidy(ax, "both")


def kernel_line(ax: Axes, runs: list[int], values: list[float], color: str, label: str, ls: str = "-") -> None:
    ax.plot(runs, values, color=color, ls=ls, lw=2, marker="o" if ls == "-" else None, ms=6, label=label, zorder=3)


def save(fig: Figure, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(path)


# ---------------------------------------------------------------- figures

def e0_cpu_baseline() -> None:
    cpu = load("cpu_baseline")
    # IonQ paper (arXiv 2512.07057) Tables 2-3: one Apple M3 Pro core, mean & p99.9 ms per 12-round decode.
    paper = {"0.0005": {"beam8_230": (1.627, 8.704), "beam32_340": (3.202, 11.26), "bp30+osd": (3.55, 272.5)},
             "0.001": {"beam8_230": (2.318, 11.01), "beam32_340": (3.837, 14.18), "beam64_640": (6.698, 17.70),
                       "bp30+osd": (10.59, 289.0)}}
    colors = {"beam8_230": SHADES[CPU][0], "beam32_340": SHADES[CPU][1], "beam64_640": SHADES[CPU][2], "bp30+osd": GREY}
    fig, axes = figure(2, f"{SETUP.replace(' · float32 · RTX 5070', '')} · CPU: IonQ C++ on one Intel i7-1355U core "
                          "(laptop), 10,000 shots per p · grey ticks: IonQ paper (Apple M3 Pro core) · "
                          "data: results/cpu_baseline.json", ratios=(3, 4))
    for ax, p in zip(axes, ("0.0005", "0.001")):
        run = cpu[f"bb144_z_p{p}"]
        names = [d for d in ("beam8_230", "beam32_340", "beam64_640", "bp30+osd") if d in run]
        for i, d in enumerate(names):
            ax.bar(i, run[d]["mean_ms"], width=0.6, color=colors[d], zorder=2)
            ax.scatter(i, run[d]["p999_ms"], marker="D", s=40, color=colors[d], edgecolor="white", lw=1.2, zorder=3)
            ax.text(i, run[d]["mean_ms"] * 1.1, f"{run[d]['mean_ms']:.2f}", ha="center", va="bottom", fontsize=8.5)
            ax.scatter([i - 0.42] * 2, paper[p][d], marker="_", s=120, lw=2, color=MUTED, zorder=3)
        budget(ax)
        ax.set_yscale("log")
        ax.set_ylim(*LATENCY_YLIM)
        ax.set_xticks(range(len(names)), [BEAM[d] for d in names], color=INK)
        ax.set_ylabel("ms per shot, batch 1 (log)")
        ax.set_title(f"Decoder latency, CPU, p = {p}")
        tidy(ax)
    axes[0].legend(handles=[*latency_keys(), Line2D([], [], marker="_", ms=10, mew=2, ls="", color=MUTED,
                                                    label="IonQ paper: mean & p99.9")], loc="upper left")
    save(fig, "e0_cpu_baseline")


def a1_cuda_vs_cpu() -> None:
    fig, axes = figure(2, f"{SETUP} · p = 0.001 · latency: same 10,000 seeded shots, batch 1 · throughput: "
                          "batch sweep over the same shots · CPU: "
                          + ("one laptop core" if CPU_THROUGHPUT_HOST == "laptop" else "laptop core (latency), GPU-host core (throughput)")
                          + f" · data: results/{{cpu,gpu}}_baseline.json, results/{{{CPU_THROUGHPUT},throughput_cuda}}.json")
    decoder_latency(axes[0], ("cpu", "cuda"))
    decoder_throughput(axes[1], ("cuda",))
    save(fig, "a1_cuda_vs_cpu")


def a2_ler() -> None:
    summary = load("ler")
    # IonQ curves read off external/BeamSearchDecoder/simulation_results/beam_144_12.pdf (±~10%).
    paper = {"beam8_230": {0.001: 2.8e-7, 0.002: 1.9e-5, 0.003: 3.3e-4, 0.004: 3.1e-3, 0.005: 1.8e-2, 0.006: 6.0e-2},
             "beam32_340": {0.001: 6.8e-8, 0.002: 5.6e-6, 0.003: 1.1e-4, 0.004: 1.4e-3, 0.005: 1.1e-2, 0.006: 4.7e-2},
             "beam64_640": {0.001: 5.4e-8, 0.002: 4.0e-6, 0.003: 8.0e-5, 0.004: 8.5e-4, 0.005: 5.2e-3, 0.006: 2.5e-2}}
    osd = {0.001: 3.7e-7, 0.002: 2.7e-5, 0.003: 5.1e-4, 0.004: 4.8e-3, 0.005: 2.5e-2, 0.006: 7.5e-2}
    fig, (ax,) = figure(1, "BB[[144,12,12]] memory-X + memory-Z · 12 rounds · CUDA float32, RTX 5070 · up to 100 "
                           "failures per basis · error bars: binomial likelihood-ratio intervals (1000×) · "
                           "data: results/ler.json", width=8.5)
    for preset, color in zip(PRESETS, SHADES[CUDA]):
        pts = sorted((float(k.split("/")[0]), v) for k, v in summary.items() if k.split("/")[1:] == [preset, "gpu-fp32"])
        xs, ys = [x for x, _ in pts], [v["ler"] for _, v in pts]
        err = [[v["ler"] - v["low"] for _, v in pts], [v["high"] - v["ler"] for _, v in pts]]
        ax.errorbar(xs, ys, yerr=err, color=color, lw=2, marker="o", ms=7, capsize=3, mec="white", mew=1.2, zorder=3,
                    label=f"{BEAM[preset]}, CUDA (this work)")
        ref = sorted(paper[preset].items())
        ax.plot([x for x, _ in ref], [y for _, y in ref], color=color, lw=1.2, ls="--", marker="o", ms=5, mfc="white",
                zorder=2)
    cpu = sorted((float(k.split("/")[0]), v["ler"]) for k, v in summary.items() if k.endswith("beam8_230/ionq"))
    ax.scatter([x for x, _ in cpu], [y for _, y in cpu], marker="x", s=60, color=CPU, lw=2, zorder=4)
    ref = sorted(osd.items())
    ax.plot([x for x, _ in ref], [y for _, y in ref], color=GREY, lw=1.5, ls=":", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.00085, 0.0072)
    ax.set_xlabel("physical error rate p (log)")
    ax.set_ylabel("logical error rate per syndrome round, X + Z (log)")
    ax.set_title("Accuracy: logical error rate vs physical error rate")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[*handles, Line2D([], [], color=MUTED, lw=1.2, ls="--", marker="o", mfc="white",
                                        label="IonQ paper (C++), same colour per beam"),
                       Line2D([], [], color=CPU, marker="x", ls="", mew=2, label="IonQ C++ re-run here, beam8"),
                       Line2D([], [], color=GREY, lw=1.5, ls=":", label="BP+OSD (IonQ paper)")], loc="upper left")
    tidy(ax, "both")
    save(fig, "a2_ler")


def b_rt_baseline(split: bool = True, name: str = "b_rt_baseline") -> None:
    """``split=False``: slide version without the check-pass / bit-pass lines (kernel panel = CUDA vs RT only)."""
    it = load("bp_iteration")
    runs = sorted(int(r) for r in it)
    timing = "GPU time per iteration" if split else "wall time per iteration"  # the slide version fixes the old footer wording
    fig, axes = figure(3, f"{SETUP} · kernel: random syndromes, {timing} (results/bp_iteration.json) · "
                          f"decoder: p = 0.001; CPU latency on a laptop core, CPU throughput on a {CPU_THROUGHPUT_HOST} core (results/{{cpu,gpu,rt}}_baseline.json, results/{{{CPU_THROUGHPUT},throughput_cuda,throughput_rt}}.json)",
                       width=19)
    kernel_line(axes[0], runs, [it[str(r)]["cuda_us_per_iter"] for r in runs], CUDA, "CUDA")
    kernel_line(axes[0], runs, [it[str(r)]["rt_us_per_iter"] for r in runs], RT,
                "RT baseline, whole iteration" if split else "RT baseline")
    if split:
        kernel_line(axes[0], runs, [it[str(r)]["rt_check_pass_us"] for r in runs], RT, "  of which check pass (row rays)", "--")
        kernel_line(axes[0], runs, [it[str(r)]["rt_bit_pass_us"] for r in runs], RT, "  of which bit pass (column rays)", ":")
    kernel_axes(axes[0], runs, "BP kernel: one iteration, wall time")
    axes[0].legend(loc="upper left")
    decoder_latency(axes[1], ("cpu", "cuda", "rt"))
    decoder_throughput(axes[2], ("cuda", "rt"))
    save(fig, name)


def b_rt_baseline_slides() -> None:
    b_rt_baseline(split=False, name="b_rt_baseline_slides")


def c_rt_tuning() -> None:
    it = load("bp_iteration")
    runs = sorted(int(r) for r in it)
    fig, axes = figure(3, f"{SETUP} · kernel: results/bp_iteration.json · decoder: p = 0.001; CPU latency on a laptop core, CPU throughput on a {CPU_THROUGHPUT_HOST} core "
                          f"(results/{{cpu,gpu,rt,rt-cache}}_baseline.json, results/{{{CPU_THROUGHPUT},throughput_cuda,throughput_rt,throughput_rt-cache}}.json)",
                       width=19)
    for key, name in (("cuda_us_per_iter", "cuda"), ("rt_baseline_us_per_iter", "rt"),
                      ("rt_cached_neighbors_us_per_iter", "rt-cache")):
        kernel_line(axes[0], runs, [it[str(r)][key] for r in runs], COLOR[name], NAMES[name])
    kernel_axes(axes[0], runs, "BP kernel: one iteration, wall time")
    axes[0].legend(loc="upper left")
    decoder_latency(axes[1], ("cpu", "cuda", "rt", "rt-cache"))
    decoder_throughput(axes[2], ("cuda", "rt", "rt-cache"))
    save(fig, "c_rt_tuning")


def r_closest_hit() -> None:
    data = load("ch_iteration")
    runs = sorted(int(r) for r in data)
    fig, axes = figure(2, f"{SETUP} · random syndromes, quiet-GPU re-run (other session than B: CUDA 172 vs 141 µs "
                          "at 1 run is session-to-session spread) · data: results/ch_iteration.json")
    kernel_line(axes[0], runs, [data[str(r)]["cuda_us_per_iter"] for r in runs], CUDA, "CUDA")
    kernel_line(axes[0], runs, [data[str(r)]["rt_us_per_iter"] for r in runs], RT, "RT baseline")
    kernel_line(axes[0], runs, [data[str(r)]["ch_refit_us_per_iter"] for r in runs], RTV, "RT closest-hit min-sum")
    kernel_axes(axes[0], runs, "BP kernel: one iteration, wall time")
    axes[0].legend(loc="upper left")
    parts = (("ch_min_trace_gpu_us", SHADES[RTV][0], "closest-hit: RT trace of check minimums"),
             ("ch_refit_gpu_us", SHADES[RTV][2], "closest-hit: BVH refit (move the geometry)"),
             ("cuda_us_per_iter", CUDA, "CUDA: whole iteration (for scale)"))
    for i, r in enumerate(runs):
        for j, (key, color, _) in enumerate(parts):
            axes[1].bar(i + (j - 1) * 0.27, data[str(r)][key], color=color, width=0.25, zorder=2)
    axes[1].set_yscale("log")
    axes[1].set_ylim(*KERNEL_YLIM)
    axes[1].set_xticks(range(len(runs)), [str(r) for r in runs])
    axes[1].set_xlabel("BP runs processed together")
    axes[1].set_ylabel("µs per BP iteration (log)")
    axes[1].set_title("BP kernel: closest-hit split, GPU time")
    axes[1].legend(handles=[Patch(color=c, label=label) for _, c, label in parts], loc="upper left")
    tidy(axes[1])
    save(fig, "r_closest_hit")


def s3_frontier_scene(data: str = "frontier_scene", name: str = "s3_frontier_scene") -> None:
    """With ``data = "frontier_scene_v2"`` (same-session rerun, 2026-11-26) the left panel also shows the unpruned
    RT baseline, so what pruning saves can be read for RT as well as for CUDA."""
    k, iters = 3, 20
    rows = {int(key.split("/")[0]): row for key, row in load(data).items() if key.endswith(f"/{k}")}
    runs = sorted(rows)
    fig, axes = figure(2, f"{SETUP} · one 20-iteration beam round after k = 3 full iterations, shown per iteration "
                          f"(round / 20, per-round setup included) · data: results/{data}.json")
    kernel_line(axes[0], runs, [rows[r]["round_unpruned_cuda_us"] / iters for r in runs], CUDA, "CUDA, unpruned", "--")
    if all("round_unpruned_rt_us" in rows[r] for r in runs):
        kernel_line(axes[0], runs, [rows[r]["round_unpruned_rt_us"] / iters for r in runs], RTV, "RT, unpruned", "--")
    kernel_line(axes[0], runs, [rows[r]["round_cuda_us"] / iters for r in runs], CUDA, "CUDA, pruned (index lists)")
    kernel_line(axes[0], runs, [rows[r]["round_rt_us"] / iters for r in runs], RTV, "RT, pruned (per-round scene)")
    kernel_axes(axes[0], runs, "BP kernel: one iteration in a beam round, GPU time")
    axes[0].legend(loc="upper left")
    parts = (("setup: choose frontier, build lists", GREY), ("setup: build RT scene (BVH)", SHADES[RTV][2]),
             ("check pass, CUDA", CUDA), ("check pass, RT rays", RTV), ("bit pass (CUDA in both)", SHADES[CUDA][0]))
    labels = []
    for i, r in enumerate(runs):
        row = rows[r]
        for j, rt in enumerate((False, True)):
            values = ((row["setup_shared_us"] / iters, GREY),
                      (row["setup_rt_scene_us"] / iters if rt else 0.0, SHADES[RTV][2]),
                      (row["check_rt_us" if rt else "check_cuda_us"], RTV if rt else CUDA),
                      (row["bit_pass_us"], SHADES[CUDA][0]))
            bottom, x = 0.0, i * 3.2 + j
            for value, color in values:
                axes[1].bar(x, value, bottom=bottom, color=color, edgecolor="white", lw=1.5, width=0.85, zorder=2)
                bottom += value
            labels.append((x, f"{'RT' if rt else 'CUDA'}\n{r} run{'s' if r > 1 else ''}"))
    axes[1].set_xticks([x for x, _ in labels], [label for _, label in labels], fontsize=8)
    axes[1].set_ylabel("µs per BP iteration (linear)")
    axes[1].set_title("BP kernel: inside a pruned iteration, GPU time")
    axes[1].legend(handles=[Patch(color=c, label=label) for label, c in parts], loc="upper left")
    tidy(axes[1])
    save(fig, name)


def s3_frontier_scene_v2() -> None:
    s3_frontier_scene("frontier_scene_v2", "s3_frontier_scene_v2")


def s1_freezing(glossary_key: bool = False) -> None:
    """``glossary_key``: add a key explaining τ / warm / refresh / initial-BP freeze under the panels
    (off: the slide carries its own explanation)."""
    configs = {"fr8w0": "τ = 8", "fr8w3": "τ = 8, warm = 3", "fr16w0": "τ = 16", "fr32w0": "τ = 32", "fr32w3": "τ = 32, warm = 3",
               "fr64w0": "τ = 64", "fr64w3": "τ = 64, warm = 3", "fr32w0r2": "τ = 32, refresh = 2",
               "fr32w3r5": "τ = 32, warm = 3, refresh = 5", "fr32w0r5": "τ = 32, refresh = 5",
               "fr32w0i5r5": "τ = 32, refresh = 5,\n+ initial-BP freeze"}
    report, latency = load("s1_ler"), load("latency_breakdown")
    fig, axes = figure(2, "BB[[144,12,12]] memory-X + memory-Z · 12 rounds · CUDA float32, RTX 5070 · beam8 · "
                          "left: same shots decoded with and without freezing · right: batch 1, 300 non-trivial shots · "
                          "data: results/s1_ler.json, results/latency_breakdown.json", ratios=(1.1, 1))
    ax = axes[0]
    styles = {"0.003": (SHADES[CUDA][2], "o", -0.15), "0.002": (SHADES[CUDA][0], "s", 0.15)}
    for p, (color, marker, offset) in styles.items():
        for row, name in enumerate(configs):
            r = report.get(f"{p}/beam8_230/{name}")
            if r is not None:
                filled = r["sign_test_p"] < 0.05
                ax.plot(r["ler_ratio"], row + offset, marker, ms=8, color=color, mfc=color if filled else "white", mew=2)
    ax.axvline(1.0, color=MUTED, lw=1, zorder=0)
    ax.set_yticks(range(len(configs)), list(configs.values()))
    ax.set_xlabel("failures with freezing / failures without (same shots)")
    ax.set_title("Accuracy: paired logical-error ratio")
    ax.legend(handles=[Line2D([], [], marker=m, ls="", ms=8, color=c, label=f"p = {p}") for p, (c, m, _) in styles.items()]
              + [Line2D([], [], marker="o", ls="", ms=8, color=MUTED, mfc="white", mew=2,
                        label="hollow = not significant (sign test p ≥ 0.05)")], loc="upper right")
    tidy(ax, "x")
    ax = axes[1]
    parts = (("initial_gpu_mean_ms", SHADES[CUDA][2], "first BP run (GPU)"),
             ("rounds_gpu_mean_ms", SHADES[CUDA][0], "beam rounds (GPU)"), ("other_mean_ms", GREY, "host / copies"))
    names = {"unpruned": "no freezing", "tau32": "τ = 32", "tau32_w3_init5": "τ = 32, warm = 3\n+ initial-BP freeze"}
    rows = [(f"z/{p}/{key}", f"p = {p}: {label}") for p in ("0.001", "0.003") for key, label in names.items()]
    for i, (key, _) in enumerate(rows):
        left = 0.0
        for part, color, _ in parts:
            ax.barh(i, latency[key][part], left=left, color=color, edgecolor="white", lw=1.5, height=0.7, zorder=2)
            left += latency[key][part]
        ax.text(left + 0.08, i, f"{latency[key]['wall_mean_ms']:.2f} ms", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 6.5)
    ax.set_xlabel("ms per shot, batch 1 (mean, linear)")
    ax.set_title("Decoder latency: where the time goes")
    ax.legend(handles=[Patch(color=c, label=label) for _, c, label in parts], loc="upper right")
    tidy(ax, "x")
    if glossary_key:
        # What the setting names mean (decision 31), between the panels and the provenance footer.
        glossary = (("Settings", "a frozen bit keeps its messages fixed and is skipped in later BP iterations"),
               ("τ (tau)", "a bit freezes when all its messages agree in sign with |LLR| ≥ τ and none of its checks is "
                           "unsatisfied; smaller τ freezes more bits"),
               ("warm k", "each beam-round BP run first does k ordinary iterations before any bit may freeze (not listed = 0)"),
               ("refresh R", "after every R frozen iterations, one full iteration updates every bit and the frozen set is "
                             "chosen again"),
               ("initial-BP freeze", "freezing also runs in the first BP run of a shot (after 5 ordinary iterations), "
                                     "not only in the beam rounds"))
        for i, (term, meaning) in enumerate(glossary):
            y = -0.035 - 0.045 * i
            fig.text(0.005, y, term, ha="left", va="top", fontsize=9, color=INK, fontweight="bold")
            fig.text(0.1, y, meaning, ha="left", va="top", fontsize=9, color=INK)
        fig.texts[0].set_y(-0.035 - 0.045 * len(glossary) - 0.03)  # provenance footer below the key
    save(fig, "s1_freezing")


# ---------------------------------------------------------------- higher p (2026-11-26)

HP_LEVELS = ("0.001", "0.002", "0.003")
HP_IMPLS = (("cpu", "CPU (IonQ C++, 1 laptop core)", CPU), ("cuda", "CUDA", CUDA),
            ("cuda-fr", "CUDA, pruned (freezing)", SHADES[CUDA][2]), ("rt", "RT baseline", RT))


def hp_latency(p: str, impl: str) -> Json:
    """Batch-1 latency report (beam8) for one p; RT at p = 0.001 is from the earlier session (rt_baseline.json)."""
    case = "bb144_z_p0.001" if p == "0.001" else f"bb144_z_p{p}_10k"
    if impl == "cpu":
        report: Json = load("cpu_baseline")[case]["beam8_230"]
        return report
    name = {"cuda": "gpu_baseline", "cuda-fr": "cuda-fr_baseline", "rt": "rt_baseline"}[impl]
    tag = "" if (p == "0.001" and impl == "rt") else ("_p0.001rerun" if p == "0.001" else "_hp")
    result: Json = load(name + tag)[f"{case}/beam8_230/float32"]["latency"]
    return result


def hp_sweep(p: str, impl: str) -> dict[int, float]:
    """Throughput sweep (beam8); CPU on the laptop; RT at p = 0.001 from the earlier session."""
    if impl == "cpu":
        name = "throughput_cpu_laptop" if p == "0.001" else f"throughput_cpu_laptop_bb144_z_p{p}_10k"
    elif p == "0.001":
        name = "throughput_rt" if impl == "rt" else f"throughput_{impl}_p0.001rerun"
    else:
        name = f"throughput_{impl}_bb144_z_p{p}_10k"
    points = load(name)["batch_throughput_shots_per_s"]
    return {int(b): float(v) for b, v in sorted(points.items(), key=lambda kv: int(kv[0]))}


def hp_comparison() -> None:
    """Left: batch-1 latency per p (bar = mean, ◆ = p99.9). Right: peak throughput per p."""
    fig, axes = figure(2, f"{SETUP} · beam8 · same 10k shots per p · GPU: one session 2026-11-26 (RT at p = 0.001: "
                          "earlier session) · CPU: one laptop core · data: results/*_hp.json, *_p0.001rerun.json, throughput_*.json")
    ax = axes[0]
    width = 0.8 / len(HP_IMPLS)
    for i, p in enumerate(HP_LEVELS):
        for j, (impl, _, color) in enumerate(HP_IMPLS):
            v = hp_latency(p, impl)
            x = i + (j - (len(HP_IMPLS) - 1) / 2) * width
            ax.bar(x, v["mean_ms"], width=width * 0.88, color=color, zorder=2)
            ax.scatter(x, v["p999_ms"], marker="D", s=30, color=color, edgecolor="white", lw=1.2, zorder=3)
            ax.text(x, v["mean_ms"] * 1.1, f"{v['mean_ms']:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    budget(ax)
    ax.set_yscale("log")
    ax.set_ylim(LATENCY_YLIM[0], 3e5)  # room above the 604 ms CPU tail for the legend
    ax.set_xticks(range(len(HP_LEVELS)), [f"p = {p}" for p in HP_LEVELS], color=INK)
    ax.set_ylabel("ms per shot, batch 1 (log)")
    ax.set_title("Decoder latency by noise level, beam8")
    ax.legend(handles=[*(Patch(color=c, label=label) for _, label, c in HP_IMPLS), *latency_keys()], loc="upper left",
              ncol=2, fontsize=7.5)
    tidy(ax)
    ax = axes[1]
    xs = [float(p) for p in HP_LEVELS]
    for impl, label, color in HP_IMPLS:
        peaks = [max(hp_sweep(p, impl).values()) for p in HP_LEVELS]
        ax.plot(xs, peaks, "o-", color=color, lw=2, ms=6, label=label, zorder=3)
        ax.text(xs[-1] * 1.04, peaks[-1], f"{peaks[-1] / 1000:.1f}k" if peaks[-1] >= 1000 else f"{peaks[-1]:.0f}",
                va="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(30.0, THROUGHPUT_YLIM[1])  # CPU drops to 78 shots/s at p = 0.003
    ax.set_xticks(xs, [f"p = {p}" for p in HP_LEVELS], color=INK)
    ax.set_xlim(0.0008, 0.0033)
    ax.set_ylabel("peak shots / s (log)")
    ax.set_title("Decoder throughput by noise level, beam8 (peak over batch sizes)")
    ax.legend(loc="upper right")
    tidy(ax, "both")
    save(fig, "hp_comparison")


def hp_sweeps() -> None:
    """Throughput vs batch size, one panel per p, same axes."""
    fig, axes = figure(3, f"{SETUP} · beam8 · batch sweeps over the same 10k seeded shots per p (cycled for larger "
                          "batches) · CPU: one laptop core, swept to batch 10k at p ≥ 0.002 · RT at p = 0.001 from an "
                          "earlier session · data: results/throughput_*.json", width=19)
    for ax, p in zip(axes, HP_LEVELS, strict=True):
        for impl, label, color in HP_IMPLS:
            points = hp_sweep(p, impl)
            ax.plot(list(points), list(points.values()), "o-", color=color, lw=2, ms=4, label=label, zorder=3)
            peak = max(points.values())
            ax.text(max(points) * 1.25, list(points.values())[-1],
                    f"{peak / 1000:.1f}k" if peak >= 1000 else f"{peak:.0f}", va="center", fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        log_ticks(ax, [1, 16, 256, 4096, 100_000])
        ax.set_xlim(0.7, 4e5)
        ax.set_ylim(30.0, 40_000.0)
        ax.set_xlabel("shots decoded together (batch size, log)")
        ax.set_ylabel("shots / s (log)")
        ax.set_title(f"Decoder throughput, p = {p}, beam8")
        tidy(ax, "both")
    axes[0].legend(loc="upper left")
    save(fig, "hp_sweeps")


def rt_rays() -> None:
    """The concept diagram, recoloured green (RT) so orange keeps meaning CUDA."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import plot_rt_rays as rays

    rays.CHECK, rays.BIT, rays.CHECK_FILL, rays.BIT_FILL, rays.SURFACE = RT, "#0f6e4c", "#d6f3e7", "#c4e9d9", "white"
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": (1, 1.15)})
    rays.tanner(left)
    rays.scene(right)
    fig.tight_layout(w_pad=4)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "rt_rays.png", dpi=170)
    plt.close(fig)
    print(OUT / "rt_rays.png")


def main() -> None:
    global OUT, CPU_THROUGHPUT, CPU_THROUGHPUT_HOST
    if sys.argv[1:] == ["--cpu-throughput", "laptop"]:
        OUT, CPU_THROUGHPUT, CPU_THROUGHPUT_HOST = RESULTS / "report_figures_laptop_cpu", "throughput_cpu_laptop", "laptop"
    elif sys.argv[1:]:
        raise SystemExit("usage: plot_report_figures.py [--cpu-throughput laptop]")
    for draw in (e0_cpu_baseline, a1_cuda_vs_cpu, a2_ler, rt_rays, b_rt_baseline, b_rt_baseline_slides, c_rt_tuning,
                 r_closest_hit, s3_frontier_scene, s3_frontier_scene_v2, s1_freezing, hp_comparison, hp_sweeps):
        draw()


if __name__ == "__main__":
    main()
