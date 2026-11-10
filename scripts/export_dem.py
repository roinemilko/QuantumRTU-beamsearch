"""Export detector error models of IonQ's circuits to data/dem/*.npz plus a stats table."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rtbeam.problem import DecodingProblem
from rtbeam.tanner import TannerGraph

ROOT = Path(__file__).resolve().parents[1]
CIRCUITS = ROOT / "external" / "BeamSearchDecoder" / "StimCircuit"
OUT = ROOT / "data" / "dem"
SELECTION = (
    "BB[[144,12,12]],memory_{basis},error_rate={p},syndrome_rounds=12",
    "BB[[90,8,10]],memory_{basis},error_rate={p},syndrome_rounds=10",
    "HGP[[450,32,8]],memory_{basis},error_rate={p},syndrome_rounds=8",
)
RATES = ("0.0005", "0.001", "0.002", "0.003")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {stem: export(stem) for stem in available_stems()}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    for stem, s in stats.items():
        print(f"{stem}: {s['checks']}x{s['bits']} nnz={s['edges']} "
              f"deg_c={s['check_degree_min']}-{s['check_degree_max']} "
              f"deg_b={s['bit_degree_min']}-{s['bit_degree_max']} bits_deg<=2={s['bits_degree_le2']}")


def available_stems() -> list[str]:
    candidates = [t.format(basis=b, p=p) for t in SELECTION for b in "XZ" for p in RATES]
    return [c for c in candidates if (CIRCUITS / f"{c}.stim").exists()]


def export(stem: str) -> dict[str, float]:
    problem = DecodingProblem.from_stim_file(CIRCUITS / f"{stem}.stim")
    problem.save(OUT / f"{stem}.npz")
    return asdict(TannerGraph.from_check_matrix(problem.check_matrix).stats())


if __name__ == "__main__":
    main()
