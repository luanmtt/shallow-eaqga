"""Portas usadas pelo melhor indivíduo de cada algoritmo.

Roda baseline, híbrido e original no mesmo QUBO, reconstrói o circuito do
melhor indivíduo (ou usa as contagens já coletadas no original) e desenha a
composição de portas em barras empilhadas. ``T``/``T†`` ficam em vermelho.

Uso:
    python visualization/plot_best_gates.py
    python visualization/plot_best_gates.py --size 10 --population 5 --generations 3
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from eaqga.baseline import (
    build_baseline_circuit,
    build_uniform_circuit,
    run_baseline,
)
from eaqga.config import (
    BackendKind,
    BaselineConfig,
    ForestShape,
    GateSet,
    HybridConfig,
)
from eaqga.individuo import build_hybrid_circuit
from eaqga.instance import QUBOInstance
from eaqga.original_adapter import run_original
from eaqga.portfolio import (
    DEFAULT_UNIVERSE,
    PAPER_END,
    PAPER_START,
    RISK_AVERSION,
    build_portfolio,
    disjoint_universes,
    fetch_sp500_tickers,
)
from eaqga.runner import run_hybrid
from eaqga.utils.embellish import sepComment, stylePrint

C = "#f5c2e7"
OUTPUT_DIR = Path(__file__).resolve().parent

_IGNORED = frozenset({"measure", "barrier", "id"})

GATE_ORDER = (
    "h",
    "ry",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "cx",
    "swap",
    "ccx",
    "cswap",
)

GATE_LABELS = {
    "h": "H",
    "ry": "RY",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "sdg": "S†",
    "t": "T",
    "tdg": "T†",
    "cx": "CX",
    "swap": "SWAP",
    "ccx": "CCX",
    "cswap": "CSWAP",
}

GATE_COLORS = {
    "h": "#4C78A8",
    "ry": "#9D755D",
    "x": "#54A24B",
    "y": "#B279A2",
    "z": "#F58518",
    "s": "#72B7B2",
    "sdg": "#499894",
    "t": "#E45756",
    "tdg": "#A83232",
    "cx": "#7F7F7F",
    "swap": "#BAB0AC",
    "ccx": "#5A5A5A",
    "cswap": "#8C6D31",
}

_EXTRA_COLORS = plt.get_cmap("tab20").colors

ALGORITHMS = ("baseline", "ours", "original")
LABELS = {
    "baseline": "baseline (EAQGA)",
    "ours": "ours (híbrido)",
    "original": "original (QGA)",
}


def _count_gates(circuit) -> dict[str, int]:
    return {
        str(name): int(value)
        for name, value in circuit.count_ops().items()
        if str(name) not in _IGNORED
    }


def _load_instance(
    size: int,
    subset: int,
    seed: int,
    *,
    universe: str,
) -> QUBOInstance:
    symbols = (
        fetch_sp500_tickers(as_of="2024-09-30")
        if universe == "sp500"
        else DEFAULT_UNIVERSE
    )
    data = build_portfolio(
        symbols,
        PAPER_START,
        PAPER_END,
        strict=False,
    )
    position = {ticker: index for index, ticker in enumerate(data.tickers)}
    universes = disjoint_universes(
        data.tickers,
        size,
        subset + 1,
        random.Random(seed),
    )
    indices = [position[ticker] for ticker in universes[subset]]
    return QUBOInstance.from_returns(
        data.mu[indices],
        data.sigma[np.ix_(indices, indices)],
        q=RISK_AVERSION,
        name=f"{universe}-{size}-{subset}",
    )


def _baseline_best_circuit(
    result,
    config: BaselineConfig,
):
    best = result.best_individual
    if best.forest is None:
        return build_uniform_circuit(len(best.bits))
    return build_baseline_circuit(
        best.bits,
        best.forest,
        config.p_a,
        paper_x_rule=config.paper_x_rule,
    )


def _hybrid_best_circuit(
    result,
    config: HybridConfig,
):
    best = result.best_individual
    initial_h = (
        config.initial_h
        if best.forest is None
        else config.fixed_h
    )
    return build_hybrid_circuit(
        best.genome,
        best.forest,
        initial_h=initial_h,
    )


def _plot(
    counts: dict[str, dict[str, int]],
    fitness: dict[str, float],
    meta: dict[str, object],
    target: Path,
) -> None:
    present = {
        gate
        for data in counts.values()
        for gate in data
    }
    gates = [gate for gate in GATE_ORDER if gate in present]
    gates += sorted(present - set(gates))

    algorithms = [algorithm for algorithm in ALGORITHMS if algorithm in counts]
    figure, axis = plt.subplots(
        figsize=(9.5, 3.9),
        constrained_layout=True,
    )

    y = np.arange(len(algorithms))
    left = np.zeros(len(algorithms), dtype=float)

    for index, gate in enumerate(gates):
        values = np.array(
            [counts[algorithm].get(gate, 0) for algorithm in algorithms],
            dtype=float,
        )
        color = GATE_COLORS.get(
            gate,
            _EXTRA_COLORS[index % len(_EXTRA_COLORS)],
        )
        axis.barh(
            y,
            values,
            left=left,
            color=color,
            edgecolor="darkred" if gate in {"t", "tdg"} else "none",
            linewidth=0.5 if gate in {"t", "tdg"} else 0.0,
            height=0.62,
        )
        left += values

    axis.set_yticks(
        y,
        [
            (
                f"{LABELS[algorithm]}\n"
                f"best={fitness[algorithm] * 100:.3f} · "
                f"{sum(counts[algorithm].values())} portas"
            )
            for algorithm in algorithms
        ],
    )
    axis.set_xlabel("portas no melhor indivíduo")
    axis.grid(axis="x", alpha=0.25, linewidth=0.6)
    axis.invert_yaxis()

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=GATE_COLORS.get(
                gate,
                _EXTRA_COLORS[index % len(_EXTRA_COLORS)],
            ),
            edgecolor="darkred" if gate in {"t", "tdg"} else "none",
            label=GATE_LABELS.get(gate, gate),
        )
        for index, gate in enumerate(gates)
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 9),
        fontsize=8,
        title="portas (T/T† em vermelho)",
    )
    axis.set_title(
        "Portas do melhor indivíduo · "
        f"S={meta['S']} · N={meta['N']} · T={meta['T']} · "
        f"initial_h={meta['initial_h']} · fixed_h={meta['fixed_h']} · "
        f"shots={meta['shots']}",
        loc="left",
        fontsize=10,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _write_csv(
    counts: dict[str, dict[str, int]],
    fitness: dict[str, float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["algorithm", "best_fitness", "total", "gate", "count"]
        )
        for algorithm, data in counts.items():
            total = sum(data.values())
            for gate, count in sorted(data.items()):
                writer.writerow(
                    [
                        algorithm,
                        fitness[algorithm],
                        total,
                        gate,
                        count,
                    ]
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Portas do melhor indivíduo de cada algoritmo.",
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
        help="H^S nas gerações 1..T do híbrido (default: desligado).",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    instance = _load_instance(
        args.size,
        args.subset,
        args.seed,
        universe=args.universe,
    )

    print(sepComment(40, stylePrint("Melhor indivíduo", C, bold=True)))
    print(
        f"  S={args.size} · pop={args.population} · T={args.generations} · "
        f"initial_h={args.initial_h} · fixed_h={args.fixed_h} · "
        f"shots={args.shots_baseline}/{args.shots_ours}/{args.shots_original}"
    )
    print()

    baseline_config = BaselineConfig(
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
    )
    hybrid_config = HybridConfig(
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
    )

    print("  rodando baseline...", flush=True)
    baseline_result = run_baseline(instance, baseline_config)
    print("  rodando híbrido...", flush=True)
    hybrid_result = run_hybrid(instance, hybrid_config)
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

    counts = {
        "baseline": _count_gates(
            _baseline_best_circuit(baseline_result, baseline_config)
        ),
        "ours": _count_gates(
            _hybrid_best_circuit(hybrid_result, hybrid_config)
        ),
        "original": dict(original_result.best_individual.gate_counts or {}),
    }
    fitness = {
        "baseline": float(baseline_result.best_fitness),
        "ours": float(hybrid_result.best_fitness),
        "original": float(original_result.best_fitness),
    }

    if not counts["original"]:
        raise RuntimeError(
            "o original não devolveu gate_counts para o melhor indivíduo."
        )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    png_path = args.output_dir / f"best-gates-{stamp}.png"
    csv_path = args.output_dir / f"best-gates-{stamp}.csv"

    _plot(
        counts,
        fitness,
        {
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
    _write_csv(counts, fitness, csv_path)

    print()
    header = f"{'algoritmo':<10} {'best':>9} {'portas':>8}"
    print(f"  {stylePrint(header, C, bold=True)}")
    for algorithm in ALGORITHMS:
        total = sum(counts[algorithm].values())
        print(
            f"  {algorithm:<10} "
            f"{fitness[algorithm] * 100:>9.3f} "
            f"{total:>8}"
        )

    print()
    print(f"  gráfico: {png_path}")
    print(f"  dados:   {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
