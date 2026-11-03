"""Desenha o circuito do melhor indivíduo de cada algoritmo.

Usa a mesma configuração do ``--quick`` por default (S=10, pop 5, T=3,
universo default, shots 1/50/1024), para que os circuitos fiquem legíveis.
Cada algoritmo vira uma página do PDF e um PNG separado.

Uso:
    python visualization/draw_best_circuits.py
    python visualization/draw_best_circuits.py \
        --size 30 --population 10 --generations 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _path in (_ROOT, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
from plot_best_gates import (
    _baseline_best_circuit,
    _count_gates,
    _hybrid_best_circuit,
    _load_instance,
)

C = "#f5c2e7"
DEFAULT_OUTPUT = _HERE / "best-circuits"

ALGORITHMS = ("baseline", "ours", "original")
LABELS = {
    "baseline": "baseline (EAQGA)",
    "ours": "ours (híbrido)",
    "original": "original (QGA)",
}


def _draw(
    circuit,
    label: str,
    fitness: float,
    total: int,
):
    figure = circuit.draw(
        output="mpl",
        style="iqp",
        scale=0.7,
        fold=40,
        plot_barriers=False,
        justify="left",
    )
    figure.suptitle(
        f"{label} · best={fitness * 100:.3f} · {total} portas",
        fontsize=11,
    )
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Desenha o circuito do melhor indivíduo de cada algoritmo.",
    )
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--population", type=int, default=5)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--universe",
        choices=("sp500", "default"),
        default="default",
    )
    parser.add_argument("--shots-baseline", type=int, default=1)
    parser.add_argument("--shots-ours", type=int, default=50)
    parser.add_argument("--shots-original", type=int, default=1024)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    instance = _load_instance(
        args.size,
        args.subset,
        args.seed,
        universe=args.universe,
    )

    print(sepComment(40, stylePrint("Circuitos do melhor indivíduo", C, bold=True)))
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

    baseline_circuit = _baseline_best_circuit(
        baseline_result,
        baseline_config,
    )
    hybrid_circuit = _hybrid_best_circuit(
        hybrid_result,
        hybrid_config,
    )
    original_circuit = original_result.best_individual.circuit
    if original_circuit is None:
        raise RuntimeError(
            "o original não devolveu o circuito do melhor indivíduo."
        )

    circuits = {
        "baseline": baseline_circuit,
        "ours": hybrid_circuit,
        "original": original_circuit,
    }
    fitness = {
        "baseline": float(baseline_result.best_fitness),
        "ours": float(hybrid_result.best_fitness),
        "original": float(original_result.best_fitness),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "best-circuits.pdf"

    print()
    with PdfPages(pdf_path) as pdf:
        for algorithm in ALGORITHMS:
            circuit = circuits[algorithm]
            total = sum(_count_gates(circuit).values())
            figure = _draw(
                circuit,
                LABELS[algorithm],
                fitness[algorithm],
                total,
            )
            png_path = output_dir / f"best-circuit-{algorithm}.png"
            figure.savefig(png_path, dpi=170, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            print(
                f"  {algorithm:<10} best={fitness[algorithm] * 100:>7.3f} "
                f"portas={total:>3} → {png_path.name}"
            )

    print()
    print(f"  pdf:  {pdf_path}")
    print(f"  dir:  {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
