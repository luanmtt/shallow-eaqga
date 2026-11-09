"""Estado final do grafo de conexões (floresta de CNOTs).

Cada nó é um qubit/ativo (ticker). As arestas são CNOTs orientados
``controle → alvo``, coloridos por correlação P/N. O grafo é o da última
geração por default; ``--generation`` escolhe outra.

Uso:
    python visualization/plot_forest_final.py
    python visualization/plot_forest_final.py --shape star --size 30
    python visualization/plot_forest_final.py --mode union --generation 20
"""

from __future__ import annotations

import argparse
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

from eaqga.config import ForestShape
from eaqga.utils.embellish import sepComment, stylePrint
from forest_common import (
    capture_frames,
    draw_frame,
    load_instance_with_tickers,
    make_config,
    write_edge_csv,
)

C = "#f5c2e7"
DEFAULT_OUTPUT = _HERE / "forests"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Desenha o estado final do grafo de conexões do híbrido.",
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
        "--shape",
        choices=("star", "path"),
        default="star",
        help="topologia da floresta (default: star).",
    )
    parser.add_argument("--num-columns", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--p-s", type=float, default=0.6)
    parser.add_argument("--shots", type=int, default=50)
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
        "--mode",
        choices=("best", "union"),
        default="best",
        help="'best': floresta do melhor filho; 'union': união dos N filhos.",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=None,
        help="geração a desenhar (default: a última).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    shape = ForestShape(args.shape)

    instance, tickers = load_instance_with_tickers(
        args.size,
        args.subset,
        args.seed,
        universe=args.universe,
    )
    config = make_config(
        shape=shape,
        population=args.population,
        generations=args.generations,
        shots=args.shots,
        num_columns=args.num_columns,
        mutation_rate=args.mutation_rate,
        p_s=args.p_s,
        initial_h=args.initial_h,
        fixed_h=args.fixed_h,
        seed=args.seed,
    )

    print(sepComment(40, stylePrint("Grafo de conexões", C, bold=True)))
    print(
        f"  S={args.size} · pop={args.population} · T={args.generations} · "
        f"shape={args.shape} · subset={args.subset} · seed={args.seed} · "
        f"mode={args.mode} · shots={args.shots}"
    )
    print("  rodando o híbrido...", flush=True)

    _, frames = capture_frames(instance, config, mode=args.mode)
    if not frames:
        raise RuntimeError("nenhuma floresta capturada.")

    if args.generation is None:
        frame = frames[-1]
    else:
        matches = [item for item in frames if item.generation == args.generation]
        if not matches:
            raise SystemExit(
                f"geração {args.generation} não encontrada "
                f"(disponíveis: 0..{frames[-1].generation})."
            )
        frame = matches[0]

    title = (
        f"Floresta {args.shape.upper()} · S={args.size} · "
        f"subset={args.subset} · seed={args.seed} · "
        f"geração {frame.generation} · best={frame.best_fitness * 100:.3f} · "
        f"{frame.edge_count} arestas (P={frame.positive}, N={frame.negative})"
    )

    figure, axis = plt.subplots(figsize=(10.0, 10.0), constrained_layout=True)
    draw_frame(axis, frame, tickers, title=title)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = (
        f"forest-{args.shape}-S{args.size}-subset{args.subset}-"
        f"seed{args.seed}-gen{frame.generation}-{stamp}"
    )
    png_path = args.output_dir / f"{stem}.png"
    csv_path = args.output_dir / f"{stem}-edges.csv"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    write_edge_csv(csv_path, frame, tickers)

    print()
    print(
        f"  geração {frame.generation}: {frame.edge_count} arestas "
        f"(P={frame.positive}, N={frame.negative}) · "
        f"best={frame.best_fitness * 100:.3f}"
    )
    print(f"  png: {png_path}")
    print(f"  csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
