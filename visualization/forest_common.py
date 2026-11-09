"""Captura e desenho das florestas de CNOT do híbrido.

A floresta não faz parte do genoma: ela é reconstruída a cada geração a partir
dos dois melhores bitstrings. Este módulo instrumenta ``runner.measure_genome``
para guardar o ``Forest`` de cada indivíduo medido e oferece utilitários de
layout/desenho para os scripts ``plot_forest_final.py`` e ``animate_forest.py``.

Cada nó é um qubit e, no problema de portfólio, também um ativo (ticker).
"""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import eaqga.runner as runner_module
from eaqga.config import BackendKind, ForestShape, GateSet, HybridConfig
from eaqga.instance import QUBOInstance
from eaqga.portfolio import (
    DEFAULT_UNIVERSE,
    PAPER_END,
    PAPER_START,
    RISK_AVERSION,
    build_portfolio,
    disjoint_universes,
    fetch_sp500_tickers,
)

POSITIVE_COLOR = "#4C78A8"
NEGATIVE_COLOR = "#F58518"
NODE_COLOR = "#F4F4F4"
EDGE_COLOR = "#444444"

Mode = Literal["best", "union"]


@dataclass(frozen=True, slots=True)
class Frame:
    """Uma geração do grafo de conexões.

    ``edges`` guarda ``(control, target, kind, weight)``. No modo ``best`` o
    peso é sempre 1; no modo ``union`` ele é a frequência da aresta entre os
    ``N`` filhos da geração.
    """

    generation: int
    best_fitness: float
    best_bits: str
    edges: tuple[tuple[int, int, str, int], ...]
    nodes: int

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def positive(self) -> int:
        return sum(1 for _, _, kind, _ in self.edges if kind == "P")

    @property
    def negative(self) -> int:
        return sum(1 for _, _, kind, _ in self.edges if kind == "N")


def load_instance_with_tickers(
    size: int,
    subset: int,
    seed: int,
    *,
    universe: str = "sp500",
    as_of: str = "2024-09-30",
) -> tuple[QUBOInstance, tuple[str, ...]]:
    """Carrega um subset QUBO e devolve também os tickers dos nós."""

    if universe == "sp500":
        symbols = fetch_sp500_tickers(as_of=as_of)
    elif universe == "default":
        symbols = DEFAULT_UNIVERSE
    else:
        raise ValueError(f"universe deve ser 'sp500' ou 'default', recebeu {universe!r}.")

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
    tickers = universes[subset]
    indices = [position[ticker] for ticker in tickers]

    instance = QUBOInstance.from_returns(
        data.mu[indices],
        data.sigma[np.ix_(indices, indices)],
        q=RISK_AVERSION,
        name=f"{universe}-{size}-{subset}",
    )
    return instance, tickers


def make_config(
    *,
    shape: ForestShape,
    population: int,
    generations: int,
    shots: int,
    num_columns: int,
    mutation_rate: float,
    p_s: float,
    initial_h: bool,
    fixed_h: bool,
    seed: int,
) -> HybridConfig:
    """Configuração do híbrido usada pelos visualizadores."""

    return HybridConfig(
        population_size=population,
        generations=generations,
        shots=shots,
        num_columns=num_columns,
        gate_set=GateSet.HILBERT_1Q,
        forest_shape=shape,
        max_degree=None if shape is ForestShape.STAR else 2,
        mutation_rate=mutation_rate,
        p_s=p_s,
        initial_h=initial_h,
        fixed_h=fixed_h,
        seed=seed,
        backend=BackendKind.MPS,
        verbose=False,
    )


def _forest_edges(individual: object) -> tuple[tuple[int, int, str], ...]:
    forest = getattr(individual, "forest", None)
    if forest is None:
        return ()
    return tuple(
        (edge.control, edge.target, edge.kind.value)
        for edge in forest.edges
    )


def capture_frames(
    instance: QUBOInstance,
    config: HybridConfig,
    *,
    mode: Mode = "best",
) -> tuple[object, tuple[Frame, ...]]:
    """Roda o híbrido e captura uma floresta por geração.

    ``mode='best'`` usa a floresta do melhor filho da geração; ``mode='union'``
    agrega as arestas de todos os filhos e usa a frequência como peso.
    """

    if mode not in {"best", "union"}:
        raise ValueError(f"mode deve ser 'best' ou 'union', recebeu {mode!r}.")

    original = runner_module.measure_genome
    records: list[tuple[int, object]] = []
    counter = 0

    def wrapper(*args: object, **kwargs: object) -> object:
        nonlocal counter
        individual = original(*args, **kwargs)
        generation = counter // config.population_size
        counter += 1
        records.append((generation, individual))
        return individual

    runner_module.measure_genome = wrapper
    try:
        result = runner_module.run_hybrid(instance, config)
    finally:
        runner_module.measure_genome = original

    grouped: dict[int, list[object]] = defaultdict(list)
    for generation, individual in records:
        grouped[generation].append(individual)

    frames: list[Frame] = []

    for generation in range(config.generations + 1):
        individuals = grouped.get(generation, [])
        if not individuals:
            continue

        best = max(
            individuals,
            key=lambda individual: (
                float(individual.fitness),
                str(individual.bits),
            ),
        )

        if generation == 0:
            selected: tuple[object, ...] = ()
        elif mode == "best":
            selected = (best,)
        else:
            selected = tuple(individuals)

        counts: Counter[tuple[int, int, str]] = Counter()
        for individual in selected:
            counts.update(_forest_edges(individual))

        frames.append(
            Frame(
                generation=generation,
                best_fitness=float(best.fitness),
                best_bits=str(best.bits),
                edges=tuple(
                    (control, target, kind, weight)
                    for (control, target, kind), weight in sorted(counts.items())
                ),
                nodes=instance.S,
            )
        )

    return result, tuple(frames)


def circular_positions(size: int) -> dict[int, tuple[float, float]]:
    """Layout circular determinístico: um nó por qubit/ativo."""

    angles = np.linspace(
        np.pi / 2.0,
        np.pi / 2.0 + 2.0 * np.pi,
        size,
        endpoint=False,
    )
    return {
        index: (float(np.cos(angle)), float(np.sin(angle)))
        for index, angle in enumerate(angles)
    }


def _edge_width(weight: int) -> float:
    return 0.9 + 0.55 * max(1, weight)


def draw_frame(
    axis: object,
    frame: Frame,
    tickers: tuple[str, ...],
    *,
    title: str,
    label_fontsize: float = 6.5,
) -> None:
    """Desenha o grafo de uma geração no eixo ``axis``."""

    positions = circular_positions(frame.nodes)
    degree: Counter[int] = Counter()
    for control, target, _, _ in frame.edges:
        degree[control] += 1
        degree[target] += 1

    axis.clear()
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_xlim(-1.38, 1.38)
    axis.set_ylim(-1.38, 1.38)

    for control, target, kind, weight in frame.edges:
        first = positions[control]
        second = positions[target]
        color = POSITIVE_COLOR if kind == "P" else NEGATIVE_COLOR
        axis.add_patch(
            FancyArrowPatch(
                first,
                second,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=_edge_width(weight),
                color=color,
                alpha=0.82,
                shrinkA=11,
                shrinkB=11,
                connectionstyle="arc3,rad=0.06",
                zorder=2,
            )
        )

    xs = [positions[index][0] for index in range(frame.nodes)]
    ys = [positions[index][1] for index in range(frame.nodes)]
    colors = [
        NODE_COLOR if degree[index] else "#E8E8E8"
        for index in range(frame.nodes)
    ]
    sizes = [210 if degree[index] else 150 for index in range(frame.nodes)]

    axis.scatter(
        xs,
        ys,
        s=sizes,
        facecolors=colors,
        edgecolors=EDGE_COLOR,
        linewidths=0.8,
        zorder=3,
    )

    for index, (x, y) in positions.items():
        ticker = tickers[index] if index < len(tickers) else str(index)
        axis.text(
            x * 1.10,
            y * 1.10,
            f"{index} {ticker}",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            color="#222222",
            zorder=4,
        )

    handles = [
        Line2D([0], [0], color=POSITIVE_COLOR, lw=2.2, label="P (positiva)"),
        Line2D([0], [0], color=NEGATIVE_COLOR, lw=2.2, label="N (negativa)"),
    ]
    axis.legend(
        handles=handles,
        loc="lower right",
        fontsize=7,
        frameon=True,
        framealpha=0.9,
    )
    axis.set_title(title, loc="left", fontsize=10)
    axis.text(
        0.0,
        -1.30,
        "nó = qubit/ativo · aresta = CNOT controle → alvo",
        ha="center",
        va="center",
        fontsize=7,
        color="#666666",
    )


def edge_csv_rows(
    frame: Frame,
    tickers: tuple[str, ...],
) -> list[dict[str, object]]:
    """Linhas de CSV para inspecionar as arestas da geração."""

    rows: list[dict[str, object]] = []
    for control, target, kind, weight in frame.edges:
        rows.append(
            {
                "generation": frame.generation,
                "control": control,
                "target": target,
                "control_ticker": tickers[control],
                "target_ticker": tickers[target],
                "kind": kind,
                "weight": weight,
            }
        )
    return rows


def write_edge_csv(
    path: Path,
    frame: Frame,
    tickers: tuple[str, ...],
) -> Path:
    """Grava as arestas da geração em CSV."""

    rows = edge_csv_rows(frame, tickers)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "generation",
        "control",
        "target",
        "control_ticker",
        "target_ticker",
        "kind",
        "weight",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
