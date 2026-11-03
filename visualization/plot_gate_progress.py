"""Gera o gráfico geração × portas por tipo para os três algoritmos.

Cada porta é uma **linha** de cor própria, com a média de portas por indivíduo
naquela geração; ``T``/``T†`` ficam em vermelho. O fitness é a **área
preenchida** no eixo direito (melhor fitness), com a média da população em
linha tracejada.

O gráfico inclui a camada ``H^S`` fixa do híbrido (e a ``H^S`` inicial do
baseline). Para ver só a parte evoluída, o resumo impresso também calcula as
portas sem essa camada fixa.

Uso:
    python tests/plot_gate_progress.py
    python tests/plot_gate_progress.py --size 30 --population 10 --generations 20
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import eaqga.baseline as baseline_module
import eaqga.individuo as individuo_module
from eaqga.baseline import run_baseline
from eaqga.config import (
    BackendKind,
    BaselineConfig,
    ForestShape,
    GateSet,
    HybridConfig,
)
from eaqga.instance import QUBOInstance
from eaqga.original_adapter import load_engine, run_original
from eaqga.portfolio import (
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
OUTPUT_DIR = _ROOT / "outputs" / "plots"

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


def _count_gates(circuit) -> Counter[str]:
    counts = circuit.count_ops()
    return Counter(
        {
            str(name): int(value)
            for name, value in counts.items()
            if str(name) not in _IGNORED
        }
    )


def _load_instance(size: int, subset: int, seed: int) -> QUBOInstance:
    symbols = fetch_sp500_tickers(as_of="2024-09-30")
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
        name=f"sp500-{size}-{subset}",
    )


def _run_baseline_recorded(
    instance: QUBOInstance,
    config: BaselineConfig,
) -> tuple[object, list[Counter[str]]]:
    records: list[Counter[str]] = []
    original = baseline_module._measure_circuit  # type: ignore[attr-defined]

    def wrapper(circuit, simulator, instance_, config_, *, forest, seed_simulator):
        records.append(_count_gates(circuit))
        return original(
            circuit,
            simulator,
            instance_,
            config_,
            forest=forest,
            seed_simulator=seed_simulator,
        )

    baseline_module._measure_circuit = wrapper  # type: ignore[attr-defined]
    try:
        result = run_baseline(instance, config)
    finally:
        baseline_module._measure_circuit = original  # type: ignore[attr-defined]

    return result, records


def _run_hybrid_recorded(
    instance: QUBOInstance,
    config: HybridConfig,
) -> tuple[object, list[Counter[str]]]:
    records: list[Counter[str]] = []
    original = individuo_module.build_hybrid_circuit

    def wrapper(genome, forest=None, *, measure=True, initial_h=True):
        circuit = original(
            genome,
            forest,
            measure=measure,
            initial_h=initial_h,
        )
        if measure:
            records.append(_count_gates(circuit))
        return circuit

    individuo_module.build_hybrid_circuit = wrapper
    try:
        result = run_hybrid(instance, config)
    finally:
        individuo_module.build_hybrid_circuit = original

    return result, records


def _run_original_recorded(
    instance: QUBOInstance,
    *,
    population: int,
    generations: int,
    shots: int,
    seed: int,
) -> tuple[object, list[list[Counter[str]]]]:
    engine_module, _ = load_engine()
    original = engine_module.ParallelQuantumGA.evaluate_population
    records: list[list[Counter[str]]] = []

    def wrapper(self):
        original(self)
        records.append(
            [
                _count_gates(
                    individual.construir_circuito_qiskit(measure=False)
                )
                for individual in self.population
            ]
        )

    engine_module.ParallelQuantumGA.evaluate_population = wrapper
    try:
        result = run_original(
            instance,
            population_size=population,
            generations=generations,
            seed=seed,
            shots=shots,
            num_portas=6,
            door_type=2,
            elite_size=0.20,
            crossover_rate=0.70,
            mutation_rate=0.30,
        )
    finally:
        engine_module.ParallelQuantumGA.evaluate_population = original

    return result, records


def _to_blocks(
    records: list[Counter[str]] | list[list[Counter[str]]],
    population: int,
    generations: int,
) -> list[list[Counter[str]]]:
    if records and isinstance(records[0], list):
        # O original pode variar o tamanho efetivo da população (elitismo/
        # pareamento); as médias usam o tamanho observado.
        blocks = records
        if len(blocks) != generations + 1:
            raise ValueError(
                f"esperava {generations + 1} gerações, recebi {len(blocks)}."
            )
        if any(not block for block in blocks):
            raise ValueError("geração vazia no histórico do original.")
        return blocks

    flat = records
    blocks = [
        flat[index * population : (index + 1) * population]
        for index in range(generations + 1)
    ]

    if len(blocks) != generations + 1:
        raise ValueError(
            f"esperava {generations + 1} gerações, recebi {len(blocks)}."
        )
    for index, block in enumerate(blocks):
        if len(block) != population:
            raise ValueError(
                f"geração {index} tem {len(block)} circuitos, "
                f"esperava {population}."
            )

    return blocks


def _mean_by_gate(block: list[Counter[str]]) -> dict[str, float]:
    total: Counter[str] = Counter()
    for counts in block:
        total.update(counts)
    return {
        gate: value / len(block)
        for gate, value in total.items()
    }


def _series(
    blocks: list[list[Counter[str]]],
) -> tuple[list[str], dict[str, list[float]]]:
    means = [_mean_by_gate(block) for block in blocks]
    gates = [
        gate
        for gate in GATE_ORDER
        if any(gate in row for row in means)
    ]
    for row in means:
        for gate in row:
            if gate not in gates:
                gates.append(gate)

    return gates, {
        gate: [row.get(gate, 0.0) for row in means]
        for gate in gates
    }


def _plot(
    title: str,
    series: dict[str, dict[str, list[float]]],
    fitness: dict[str, dict[str, list[float]]],
    *,
    target_path: Path,
) -> None:
    algorithms = list(series)
    generations = len(next(iter(series.values()))["total"])

    figure, axes = plt.subplots(
        len(algorithms),
        1,
        figsize=(11, 3.2 * len(algorithms)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(algorithms) == 1:
        axes = [axes]

    present = {
        gate
        for data in series.values()
        for gate in data
        if gate != "total"
    }
    all_gates = [gate for gate in GATE_ORDER if gate in present]
    all_gates += sorted(present - set(all_gates))

    max_gate = max(
        (
            max(data.get(gate, [0.0]))
            for data in series.values()
            for gate in all_gates
        ),
        default=1.0,
    )
    fit_values = [
        value
        for algorithm in algorithms
        for key in ("best", "mean")
        for value in fitness[algorithm][key]
    ]
    fit_low = max(0.0, min(fit_values) * 100.0 * 0.97)
    fit_high = max(fit_values) * 100.0 * 1.03

    for axis, algorithm in zip(axes, algorithms):
        data = series[algorithm]
        x = np.arange(generations)

        for gate in all_gates:
            values = np.array(
                data.get(gate, [0.0] * generations),
                dtype=float,
            )
            if not values.any():
                continue

            color = GATE_COLORS.get(
                gate,
                _EXTRA_COLORS[len(gate) % len(_EXTRA_COLORS)],
            )
            axis.plot(
                x,
                values,
                color=color,
                linewidth=2.6 if gate in {"t", "tdg"} else 1.5,
                marker="o",
                markersize=2.6,
                alpha=0.95,
                zorder=3,
                label=GATE_LABELS.get(gate, gate),
            )

        axis.set_ylabel("portas / indivíduo")
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_ylim(0.0, max_gate * 1.08)

        twin = axis.twinx()
        twin.patch.set_visible(False)
        best = np.array(fitness[algorithm]["best"]) * 100.0
        mean = np.array(fitness[algorithm]["mean"]) * 100.0

        twin.fill_between(
            x,
            fit_low,
            best,
            color="#7F7F7F",
            alpha=0.10,
            zorder=0,
            label="fitness (área)",
        )
        twin.plot(
            x,
            best,
            color="black",
            marker="o",
            markersize=3,
            linewidth=1.8,
            zorder=4,
            label="best fitness",
        )
        twin.plot(
            x,
            mean,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            zorder=4,
            label="média",
        )
        twin.set_ylim(fit_low, fit_high)
        twin.set_ylabel("fitness ×100")

        mean_gates = float(np.mean(data["total"][1:]))
        axis.set_title(
            f"{algorithm} · best={max(fitness[algorithm]['best']) * 100:.3f} "
            f"· {mean_gates:.1f} portas/indiv (g1..g{generations - 1})",
            loc="left",
            fontsize=10,
        )

    axes[-1].set_xlabel("geração")
    axes[-1].set_xticks(range(generations))

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=GATE_COLORS.get(
                gate,
                _EXTRA_COLORS[index % len(_EXTRA_COLORS)],
            ),
            linewidth=2.6 if gate in {"t", "tdg"} else 1.5,
            marker="o",
            markersize=4,
            label=GATE_LABELS.get(gate, gate),
        )
        for index, gate in enumerate(all_gates)
    ] + [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#7F7F7F",
            alpha=0.10,
            label="fitness (área)",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.8,
            marker="o",
            markersize=4,
            label="best fitness",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.0,
            linestyle="--",
            label="média",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 8),
        fontsize=8,
        title="portas (T/T† em vermelho) · fitness (área)",
    )
    figure.suptitle(title, fontsize=11)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _write_csv(
    series: dict[str, dict[str, list[float]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "generation", "gate", "count_mean"])
        for algorithm, data in series.items():
            total_generations = len(data["total"])
            for generation in range(total_generations):
                for gate, values in data.items():
                    if gate == "total":
                        continue
                    writer.writerow(
                        [algorithm, generation, gate, values[generation]]
                    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gráfico geração × portas por tipo para os algoritmos.",
    )
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument(
        "--initial-h",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="H^S na população inicial do híbrido.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    instance = _load_instance(args.size, args.subset, args.seed)

    print(sepComment(40, stylePrint("Progresso de portas", C, bold=True)))
    print(
        f"  S={args.size} · pop={args.population} · T={args.generations} · "
        f"shots={args.shots} · initial_h={args.initial_h} · seed={args.seed}"
    )
    print()

    baseline_config = BaselineConfig(
        population_size=args.population,
        generations=args.generations,
        shots=args.shots,
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
        shots=args.shots,
        num_columns=2,
        gate_set=GateSet.HILBERT_1Q,
        forest_shape=ForestShape.PATH,
        max_degree=2,
        mutation_rate=0.10,
        p_s=0.6,
        initial_h=args.initial_h,
        seed=args.seed,
        backend=BackendKind.MPS,
        verbose=False,
    )

    print("  rodando baseline...", flush=True)
    baseline_result, baseline_records = _run_baseline_recorded(
        instance,
        baseline_config,
    )
    print("  rodando híbrido...", flush=True)
    hybrid_result, hybrid_records = _run_hybrid_recorded(
        instance,
        hybrid_config,
    )
    print("  rodando original...", flush=True)
    original_result, original_records = _run_original_recorded(
        instance,
        population=args.population,
        generations=args.generations,
        shots=args.shots,
        seed=args.seed,
    )

    results = {
        "baseline": baseline_result,
        "ours": hybrid_result,
        "original": original_result,
    }
    raw_records = {
        "baseline": baseline_records,
        "ours": hybrid_records,
        "original": original_records,
    }

    blocks = {
        algorithm: _to_blocks(
            raw_records[algorithm],
            args.population,
            args.generations,
        )
        for algorithm in results
    }

    series: dict[str, dict[str, list[float]]] = {}
    fitness: dict[str, dict[str, list[float]]] = {}

    for algorithm, generation_blocks in blocks.items():
        gates, data = _series(generation_blocks)
        totals = [
            sum(data[gate][generation] for gate in gates)
            for generation in range(args.generations + 1)
        ]
        series[algorithm] = {**data, "total": totals}
        result = results[algorithm]
        fitness[algorithm] = {
            "best": [float(value) for value in result.history["best"]],
            "mean": [float(value) for value in result.history["avg"]],
        }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    png_path = args.output / f"gate-progress-{stamp}.png"
    csv_path = args.output / f"gate-progress-{stamp}.csv"

    _plot(
        (
            f"EAQGA · S={args.size} · pop={args.population} · "
            f"T={args.generations} · shots={args.shots} · "
            f"initial_h={args.initial_h} · seed={args.seed}"
        ),
        series,
        fitness,
        target_path=png_path,
    )
    _write_csv(series, csv_path)

    print()
    header = (
        f"{'algoritmo':<10} {'portas/indiv':>13} "
        f"{'sem H^S fixo':>13} {'best (×100)':>12}"
    )
    print(f"  {stylePrint(header, C, bold=True)}")

    for algorithm, data in series.items():
        totals = data["total"][1:]
        variable = []
        for generation in range(1, args.generations + 1):
            value = data["total"][generation]
            if algorithm == "ours":
                value -= args.size  # camada H^S fixa
            variable.append(value)

        print(
            f"  {algorithm:<10} "
            f"{np.mean(totals):>13.2f} "
            f"{np.mean(variable):>13.2f} "
            f"{max(fitness[algorithm]['best']) * 100:>12.3f}"
        )

    print()
    print(f"  gráfico: {png_path}")
    print(f"  dados:   {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
