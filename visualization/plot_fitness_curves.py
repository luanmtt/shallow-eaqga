"""Curvas de fitness por geração, um gráfico e três linhas.

Roda baseline (EAQGA fiel), ours (híbrido) e original (QGA) na mesma instância
e desenha uma linha por algoritmo. O default é ``best`` (best-so-far); use
``--metric avg`` ou ``--metric generation_best`` para ver a população cair.

Uso:
    python visualization/plot_fitness_curves.py
    python visualization/plot_fitness_curves.py --size 10 --population 5 --generations 10
    python visualization/plot_fitness_curves.py --metric avg
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _path in (_ROOT, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from eaqga.baseline import run_baseline
from eaqga.config import (
    BackendKind,
    BaselineConfig,
    ForestShape,
    GateSet,
    HybridConfig,
)
from eaqga.original_adapter import run_original
from eaqga.runner import run_hybrid
from eaqga.utils.embellish import sepComment, stylePrint
from plot_best_gates import _load_instance

C = "#f5c2e7"
DEFAULT_OUTPUT = _HERE / "fitness-curves"

ALGORITHMS = ("baseline", "ours", "original")
LABELS = {
    "baseline": "baseline (EAQGA)",
    "ours": "ours (híbrido)",
    "original": "original (QGA)",
}
COLORS = {
    "baseline": "#222222",
    "ours": "#4C78A8",
    "original": "#F58518",
}


def _metric_values(result: object, metric: str) -> list[float]:
    history = getattr(result, "history")
    if metric == "best":
        values = history.get("best", [])
    elif metric == "generation_best":
        values = history.get("generation_best") or history.get("best", [])
    elif metric == "avg":
        values = history.get("avg") or history.get("best", [])
    else:
        raise ValueError(f"metric inválida: {metric!r}.")

    return [float(value) for value in values]


def _plot(
    series: dict[str, list[float]],
    meta: dict[str, object],
    target: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9.2, 5.2),
        constrained_layout=True,
    )

    for algorithm in ALGORITHMS:
        values = series.get(algorithm, [])
        if not values:
            continue
        generations = list(range(len(values)))
        axis.plot(
            generations,
            [value * 100.0 for value in values],
            color=COLORS[algorithm],
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=LABELS[algorithm],
        )

    axis.set_xlabel("geração")
    axis.set_ylabel("fitness ×100")
    axis.set_xticks(range(int(meta["T"]) + 1))
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(fontsize=8, framealpha=0.9)
    axis.set_title(
        f"Fitness por geração · {meta['metric']} · "
        f"S={meta['S']} · N={meta['N']} · T={meta['T']} · "
        f"shots={meta['shots']} · initial_h={meta['initial_h']} · "
        f"fixed_h={meta['fixed_h']}",
        loc="left",
        fontsize=10,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _write_csv(
    series: dict[str, list[float]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "generation", "fitness"])
        for algorithm in ALGORITHMS:
            for generation, value in enumerate(series.get(algorithm, [])):
                writer.writerow([algorithm, generation, value])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Curvas de fitness por geração para os três algoritmos.",
    )
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--universe",
        choices=("sp500", "default"),
        default="sp500",
    )
    parser.add_argument(
        "--metric",
        choices=("best", "generation_best", "avg"),
        default="best",
        help="'best' é monotônico; 'avg'/'generation_best' mostram quedas.",
    )
    parser.add_argument("--shots-baseline", type=int, default=1)
    parser.add_argument("--shots-ours", type=int, default=50)
    parser.add_argument("--shots-original", type=int, default=50)
    parser.add_argument(
        "--initial-h",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="H^S na população inicial do híbrido.",
    )
    parser.add_argument(
        "--fixed-h",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="H^S nas gerações 1..T do híbrido.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    instance = _load_instance(
        args.size,
        args.subset,
        args.seed,
        universe=args.universe,
    )

    print(sepComment(40, stylePrint("Curvas de fitness", C, bold=True)))
    print(
        f"  S={args.size} · pop={args.population} · T={args.generations} · "
        f"subset={args.subset} · seed={args.seed} · metric={args.metric} · "
        f"shots={args.shots_baseline}/{args.shots_ours}/{args.shots_original} · "
        f"initial_h={args.initial_h} · fixed_h={args.fixed_h}"
    )
    print()

    print("  rodando baseline...", flush=True)
    baseline_result = run_baseline(
        instance,
        BaselineConfig(
            population_size=args.population,
            generations=args.generations,
            shots=args.shots_baseline,
            p_a=0.95,
            p_s=0.6,
            forest_shape=ForestShape.STAR,
            max_degree=None,
            seed=args.seed,
            backend=BackendKind.MPS,
            verbose=False,
        ),
    )

    print("  rodando ours...", flush=True)
    hybrid_result = run_hybrid(
        instance,
        HybridConfig(
            population_size=args.population,
            generations=args.generations,
            shots=args.shots_ours,
            num_columns=2,
            gate_set=GateSet.HILBERT_1Q,
            forest_shape=ForestShape.PATH,
            max_degree=2,
            mutation_rate=0.10,
            p_s=0.6,
            initial_h=args.initial_h,
            fixed_h=args.fixed_h,
            seed=args.seed,
            backend=BackendKind.MPS,
            verbose=False,
        ),
    )

    print("  rodando original...", flush=True)
    original_result = run_original(
        instance,
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
        shots=args.shots_original,
        num_portas=6,
        door_type=2,
        elite_size=0.20,
        crossover_rate=0.70,
        mutation_rate=0.30,
    )

    results = {
        "baseline": baseline_result,
        "ours": hybrid_result,
        "original": original_result,
    }
    series = {
        algorithm: _metric_values(results[algorithm], args.metric)
        for algorithm in ALGORITHMS
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    png_path = args.output_dir / f"fitness-curves-{stamp}.png"
    csv_path = args.output_dir / f"fitness-curves-{stamp}.csv"

    _plot(
        series,
        {
            "metric": args.metric,
            "S": args.size,
            "N": args.population,
            "T": args.generations,
            "initial_h": args.initial_h,
            "fixed_h": args.fixed_h,
            "shots": (
                f"{args.shots_baseline}/"
                f"{args.shots_ours}/"
                f"{args.shots_original}"
            ),
        },
        png_path,
    )
    _write_csv(series, csv_path)

    print()
    for algorithm in ALGORITHMS:
        values = series[algorithm]
        if values:
            print(
                f"  {algorithm:<9} "
                f"inicial={values[0] * 100:>7.3f} · "
                f"final={values[-1] * 100:>7.3f} · "
                f"n={len(values)}"
            )
    print()
    print(f"  png: {png_path}")
    print(f"  csv: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
