"""Laço evolutivo do EAQGA híbrido e exportação de resultados.

O laço é o do EAQGA, com o nosso indivíduo no lugar do circuito ``RY``:

1. população inicial: ``N`` genomas aleatórios, sem floresta;
2. avaliar cada um com ``k`` shots e guardar o melhor shot;
3. top-2 por fitness (empate por ``bits``);
4. para cada um dos ``N`` filhos:
   - floresta sorteada a partir dos **bitstrings** dos dois pais;
   - genoma = crossover uniforme dos genomas dos pais + mutação 1Q;
   - circuito ``H^S → genoma → CNOTs → medida`` com ``k`` shots;
5. atualizar histórico e repetir.

Nenhum pai é copiado intacto: todo filho passa por mutação com
``min_changes=1``. A floresta muda a cada filho (sorteio próprio), mas o par de
pais é o mesmo dentro da geração.

O histórico é colunar (``dict[str, list]``), igual ao do baseline, e pode ser
exportado com :func:`write_history_csv`. A tabela de execuções (uma linha por
run) sai de :func:`run_row`.
"""

from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from qiskit_aer import AerSimulator

from .config import BackendKind, HybridConfig
from .crossover import build_crossover_forest, forest_metrics
from .individuo import HybridIndividual, measure_genome, random_genome
from .instance import QUBOInstance
from .population import (
    PopulationMetrics,
    make_children,
    population_metrics,
    top_two,
)
from .utils.prints import (
    GENERATION_HEADER,
    format_generation_line,
    make_generation_bars,
)

__all__ = [
    "HybridResult",
    "history_rows",
    "run_hybrid",
    "run_row",
    "write_history_csv",
    "write_runs_csv",
]

_COLUMNS: Final[tuple[str, ...]] = (
    "generation",
    "best",
    "generation_best",
    "avg",
    "median",
    "std",
    "unique_bits",
    "unique_genomes",
    "mean_hamming",
    "bit_entropy",
    "non_identity",
    "h_gates",
    "t_gates",
    "forest_edges",
    "forest_layers",
    "forest_chains",
    "best_bits",
)


@dataclass(frozen=True, slots=True)
class HybridResult:
    """Resultado de uma execução do híbrido."""

    best_individual: HybridIndividual
    best_generation: int
    history: dict[str, list[object]]
    metrics: tuple[PopulationMetrics, ...]
    final_population: tuple[HybridIndividual, ...]
    evaluations: int
    elapsed_s: float

    @property
    def best_bits(self) -> str:
        return self.best_individual.bits

    @property
    def best_fitness(self) -> float:
        return self.best_individual.fitness


def _make_simulator(backend: BackendKind) -> AerSimulator:
    method = (
        "matrix_product_state"
        if backend is BackendKind.MPS
        else "statevector"
    )
    return AerSimulator(method=method)


def _initial_population(
    instance: QUBOInstance,
    config: HybridConfig,
    simulator: AerSimulator,
    rng: random.Random,
) -> list[HybridIndividual]:
    population: list[HybridIndividual] = []

    for index in range(config.population_size):
        genome = random_genome(
            instance.S,
            config.num_columns,
            config.gate_set,
            rng,
        )
        individual = measure_genome(
            genome,
            None,
            instance,
            shots=config.shots,
            seed=config.seed + index,
            simulator=simulator,
            backend=config.backend,
            evaluation_mode=config.evaluation_mode,
        )
        population.append(individual)

    return population


def run_hybrid(
    instance: QUBOInstance,
    config: HybridConfig,
    *,
    simulator: AerSimulator | None = None,
    progress: bool = False,
    reference: float | None = None,
) -> HybridResult:
    """Roda o EAQGA híbrido e devolve o histórico completo.

    ``progress=True`` imprime uma linha por geração (barras de geração e de
    fitness). Só faz sentido em processo único: com ``joblib`` paralelo, os
    workers escreveriam no mesmo terminal. ``reference`` é o alvo da barra de
    fitness (no experimento, a busca local multi-start).
    """

    if not isinstance(instance, QUBOInstance):
        raise TypeError(
            f"instance deve ser QUBOInstance, recebeu {type(instance).__name__}."
        )
    if not isinstance(config, HybridConfig):
        raise TypeError(
            f"config deve ser HybridConfig, recebeu {type(config).__name__}."
        )
    if simulator is not None and not isinstance(simulator, AerSimulator):
        raise TypeError(
            "simulator deve ser AerSimulator ou None, recebeu "
            f"{type(simulator).__name__}."
        )
    if not isinstance(progress, bool):
        raise TypeError(
            f"progress deve ser bool, recebeu {type(progress).__name__}."
        )
    if reference is not None and (
        isinstance(reference, bool) or not isinstance(reference, (int, float))
    ):
        raise TypeError(
            "reference deve ser float ou None, recebeu "
            f"{type(reference).__name__}."
        )

    target = float(reference) if reference is not None else None
    if target is not None and not math.isfinite(target):
        raise ValueError(f"reference deve ser finito, recebeu {reference}.")

    engine = simulator if simulator is not None else _make_simulator(config.backend)
    rng = random.Random(config.seed)
    start = time.perf_counter()

    circuit_index = config.population_size
    population = _initial_population(instance, config, engine, rng)
    evaluations = len(population)

    best_so_far = max(
        population,
        key=lambda individual: (individual.fitness, individual.bits),
    )
    best_generation = 0

    history: dict[str, list[object]] = {column: [] for column in _COLUMNS}
    metrics: list[PopulationMetrics] = []
    bars = make_generation_bars() if progress else None

    if progress:
        print(GENERATION_HEADER, flush=True)

    def record(
        generation: int,
        current: PopulationMetrics,
        forest: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        metrics.append(current)
        history["generation"].append(generation)
        history["best"].append(best_so_far.fitness)
        history["generation_best"].append(current.best)
        history["avg"].append(current.mean)
        history["median"].append(current.median)
        history["std"].append(current.std)
        history["unique_bits"].append(current.unique_bits)
        history["unique_genomes"].append(current.unique_genomes)
        history["mean_hamming"].append(current.mean_hamming)
        history["bit_entropy"].append(current.bit_entropy)
        history["non_identity"].append(current.non_identity)
        history["h_gates"].append(current.h_gates)
        history["t_gates"].append(current.t_gates)
        history["forest_edges"].append(forest[0])
        history["forest_layers"].append(forest[1])
        history["forest_chains"].append(forest[2])
        history["best_bits"].append(best_so_far.bits)

    first_metrics = population_metrics(population, 0)
    record(0, first_metrics)

    if bars is not None:
        print(
            format_generation_line(
                generation=0,
                generations=config.generations,
                elapsed=time.perf_counter() - start,
                best=first_metrics.best,
                average=first_metrics.mean,
                unique=first_metrics.unique_bits,
                generation_bar=bars[0],
                fitness_bar=bars[1],
                reference=target,
            ),
            flush=True,
        )

    for generation in range(1, config.generations + 1):
        first, second = top_two(population)
        genomes = make_children(first, second, config, rng)

        new_population: list[HybridIndividual] = []
        edge_counts: list[int] = []
        layer_counts: list[int] = []
        chain_counts: list[int] = []

        for genome in genomes:
            forest = build_crossover_forest(
                first.bits,
                second.bits,
                instance.coupling,
                t=generation,
                t_max=config.generations,
                p_s=config.p_s,
                shape=config.forest_shape,
                max_degree=config.max_degree,
                rng=rng,
            )
            individual = measure_genome(
                genome,
                forest,
                instance,
                shots=config.shots,
                seed=config.seed + circuit_index,
                simulator=engine,
                backend=config.backend,
                evaluation_mode=config.evaluation_mode,
            )
            circuit_index += 1
            evaluations += 1
            new_population.append(individual)

            summary = forest_metrics(forest)
            edge_counts.append(summary.edges)
            layer_counts.append(summary.layers)
            chain_counts.append(summary.chains)

        population = new_population

        current_best = max(
            population,
            key=lambda individual: (individual.fitness, individual.bits),
        )
        if current_best.fitness > best_so_far.fitness:
            best_so_far = current_best
            best_generation = generation

        current = population_metrics(population, generation)
        record(
            generation,
            current,
            (
                sum(edge_counts) / len(edge_counts),
                sum(layer_counts) / len(layer_counts),
                sum(chain_counts) / len(chain_counts),
            ),
        )

        if bars is not None:
            print(
                format_generation_line(
                    generation=generation,
                    generations=config.generations,
                    elapsed=time.perf_counter() - start,
                    best=current.best,
                    average=current.mean,
                    unique=current.unique_bits,
                    generation_bar=bars[0],
                    fitness_bar=bars[1],
                    reference=target,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - start

    return HybridResult(
        best_individual=best_so_far,
        best_generation=best_generation,
        history=history,
        metrics=tuple(metrics),
        final_population=tuple(population),
        evaluations=evaluations,
        elapsed_s=elapsed,
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# CSV


def history_rows(result: HybridResult) -> list[dict[str, object]]:
    """Transpõe o histórico colunar em linhas de CSV."""

    if not isinstance(result, HybridResult):
        raise TypeError(
            f"result deve ser HybridResult, recebeu {type(result).__name__}."
        )

    length = len(result.history["generation"])
    for column, values in result.history.items():
        if len(values) != length:
            raise ValueError(
                f"histórico inconsistente: coluna {column!r} tem "
                f"{len(values)} valores, esperado {length}."
            )

    return [
        {column: result.history[column][row] for column in _COLUMNS}
        for row in range(length)
    ]


def write_history_csv(result: HybridResult, path: str | Path) -> Path:
    """Escreve ``history_rows(result)`` em ``path`` e devolve o caminho."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    rows = history_rows(result)
    fieldnames: list[str] = list(_COLUMNS)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return destination


def run_row(
    result: HybridResult,
    instance: QUBOInstance,
    config: HybridConfig,
) -> dict[str, object]:
    """Linha resumo de uma execução, para a tabela de experimentos."""

    if not isinstance(result, HybridResult):
        raise TypeError(
            f"result deve ser HybridResult, recebeu {type(result).__name__}."
        )
    if not isinstance(instance, QUBOInstance):
        raise TypeError(
            f"instance deve ser QUBOInstance, recebeu {type(instance).__name__}."
        )
    if not isinstance(config, HybridConfig):
        raise TypeError(
            f"config deve ser HybridConfig, recebeu {type(config).__name__}."
        )

    final_values = [individual.fitness for individual in result.final_population]
    final_mean = sum(final_values) / len(final_values)

    return {
        "instance": instance.name,
        "S": instance.S,
        "population_size": config.population_size,
        "generations": config.generations,
        "shots": config.shots,
        "num_columns": config.num_columns,
        "gate_set": config.gate_set.value,
        "mutation_rate": config.mutation_rate,
        "p_s": config.p_s,
        "forest_shape": config.forest_shape.value,
        "seed": config.seed,
        "best_bits": result.best_bits,
        "best_fitness": result.best_fitness,
        "best_generation": result.best_generation,
        "final_mean": final_mean,
        "final_unique_bits": len(
            {individual.bits for individual in result.final_population}
        ),
        "evaluations": result.evaluations,
        "elapsed_s": result.elapsed_s,
    }


def write_runs_csv(rows: list[dict[str, object]], path: str | Path) -> Path:
    """Escreve uma lista de linhas (``run_row``) em CSV."""

    if not isinstance(rows, list) or not rows:
        raise ValueError("rows deve ser uma lista não vazia de dicionários.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(
                f"rows[{index}] deve ser dict, recebeu {type(row).__name__}."
            )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0])
    for index, row in enumerate(rows[1:], start=1):
        if list(row) != fieldnames:
            raise ValueError(
                f"rows[{index}] tem colunas diferentes de rows[0]."
            )

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return destination
