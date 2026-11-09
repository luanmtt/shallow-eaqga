"""Anima a progressão do grafo de conexões (floresta de CNOTs).

Cada frame é uma geração; cada nó é um qubit/ativo (ticker). As arestas são
CNOTs orientados, coloridos por correlação P/N. Gera um GIF.

Uso:
    python visualization/animate_forest.py
    python visualization/animate_forest.py --shape star --fps 2
    python visualization/animate_forest.py --mode union --generations 20
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

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from eaqga.config import ForestShape
from eaqga.utils.embellish import sepComment, stylePrint
from forest_common import (
    capture_frames,
    draw_frame,
    load_instance_with_tickers,
    make_config,
)

C = "#f5c2e7"
DEFAULT_OUTPUT = _HERE / "forests"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anima a progressão do grafo de conexões do híbrido.",
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
    parser.add_argument("--fps", type=int, default=2)
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

    print(sepComment(40, stylePrint("Animação do grafo", C, bold=True)))
    print(
        f"  S={args.size} · pop={args.population} · T={args.generations} · "
        f"shape={args.shape} · subset={args.subset} · seed={args.seed} · "
        f"mode={args.mode} · shots={args.shots}"
    )
    print("  rodando o híbrido...", flush=True)

    _, frames = capture_frames(instance, config, mode=args.mode)
    if not frames:
        raise RuntimeError("nenhuma floresta capturada.")

    figure, axis = plt.subplots(figsize=(10.0, 10.0), constrained_layout=True)

    def update(index: int) -> None:
        frame = frames[index]
        title = (
            f"Floresta {args.shape.upper()} · S={args.size} · "
            f"subset={args.subset} · seed={args.seed} · "
            f"geração {frame.generation}/{frames[-1].generation} · "
            f"best={frame.best_fitness * 100:.3f} · "
            f"{frame.edge_count} arestas "
            f"(P={frame.positive}, N={frame.negative})"
        )
        draw_frame(axis, frame, tickers, title=title)
        return None

    anim = animation.FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=1000.0 / max(1, args.fps),
        repeat=False,
        blit=False,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    gif_path = args.output_dir / (
        f"forest-{args.shape}-S{args.size}-subset{args.subset}-"
        f"seed{args.seed}-{args.mode}-{stamp}.gif"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anim.save(gif_path, writer=animation.PillowWriter(fps=args.fps))
    plt.close(figure)

    print()
    print(f"  frames: {len(frames)} · fps={args.fps}")
    print(f"  gif: {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
