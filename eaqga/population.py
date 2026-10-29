"""População do EAQGA híbrido: ranking, herança e métricas.

A população é uma sequência de :class:`~eaqga.individuo.HybridIndividual`. Este
módulo é a camada "genética" do híbrido:

- ranking e top-2 (o EAQGA usa exatamente dois pais);
- geração de filhos por crossover uniforme dos genomas + mutação 1Q;
- métricas de exploração por geração (unicidade, Hamming, entropia de bits).

O que **não** mora aqui: avaliação (Qiskit), floresta (depende do `coupling` e de
`t`) e CSV. Isso é ``runner.py``.

Regra de herança (Leitura A): o genoma é herdado, a floresta é reconstruída. O
filho nasce de ``crossover_genomes(pai_bom, pai_ruim)`` seguido de
``mutate_genome(..., min_changes=1)``. O ``min_changes=1`` garante que nenhum
filho é cópia intacta de um pai — o EAQGA original também não copia pais.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from .config import HybridConfig
from .individuo import (
    Genome,
    HybridIndividual,
    crossover_genomes,
    genome_metrics,
    mutate_genome,
)

__all__ = [
    "PopulationMetrics",
    "make_children",
    "population_metrics",
    "rank_population",
    "top_two",
]

_MIX: float = 0.5
"""Probabilidade de um gene vir do segundo pai no crossover uniforme."""


def _require_instance(name: str, value: object) -> None:
    if not isinstance(value, HybridIndividual):
        raise TypeError(
            f"{name} deve ser HybridIndividual, recebeu "
            f"{type(value).__name__}."
        )


def _as_population(population: Sequence[HybridIndividual]) -> tuple[HybridIndividual, ...]:
    if isinstance(population, (str, bytes)):
        raise TypeError("population deve ser uma sequência de HybridIndividual.")

    try:
        items = tuple(population)
    except TypeError as exc:
        raise TypeError(
            "population deve ser iterável de HybridIndividual."
        ) from exc

    if not items:
        raise ValueError("population não pode ser vazia.")

    for index, individual in enumerate(items):
        _require_instance(f"population[{index}]", individual)

    return items


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Ranking


def rank_population(
    population: Sequence[HybridIndividual],
) -> tuple[HybridIndividual, ...]:
    """Ordena por fitness decrescente.

    Empate é resolvido por ``bits`` lexicograficamente crescente. O tie-break
    precisa ser determinístico: com ``k = 50`` shots, dois indivíduos diferentes
    empatam com frequência, e um pai sorteado de forma instável tornaria o
    experimento irreprodutível.
    """

    items = _as_population(population)
    return tuple(sorted(items, key=lambda individual: (-individual.fitness, individual.bits)))


def top_two(
    population: Sequence[HybridIndividual],
) -> tuple[HybridIndividual, HybridIndividual]:
    """Devolve os dois melhores (o pool de elitismo do EAQGA)."""

    ranked = rank_population(population)
    if len(ranked) < 2:
        raise ValueError(
            "população precisa de pelo menos 2 indivíduos para o top-2."
        )
    return ranked[0], ranked[1]


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Herança


def make_children(
    first: HybridIndividual,
    second: HybridIndividual,
    config: HybridConfig,
    rng: random.Random,
) -> tuple[Genome, ...]:
    """Gera os ``N`` genomas da próxima geração a partir do top-2.

    Cada filho é ``crossover uniforme`` dos dois genomas seguido de mutação 1Q
    com pelo menos uma troca. A mutação é aplicada **depois** do crossover, para
    não ser desfeita por ele. Nenhum CNOT é tocado: a floresta nem está aqui.
    """

    _require_instance("first", first)
    _require_instance("second", second)

    if not isinstance(config, HybridConfig):
        raise TypeError(
            f"config deve ser HybridConfig, recebeu {type(config).__name__}."
        )
    if not isinstance(rng, random.Random):
        raise TypeError(
            f"rng deve ser random.Random, recebeu {type(rng).__name__}."
        )

    if (first.genome.num_columns, first.genome.size) != (
        second.genome.num_columns,
        second.genome.size,
    ):
        raise ValueError(
            "genomas dos pais com formatos diferentes: "
            f"{first.genome.num_columns}×{first.genome.size} e "
            f"{second.genome.num_columns}×{second.genome.size}."
        )

    children: list[Genome] = []

    for _ in range(config.population_size):
        mixed = crossover_genomes(first.genome, second.genome, rng, mix=_MIX)
        mutated = mutate_genome(
            mixed,
            config.mutation_rate,
            config.gate_set,
            rng,
            min_changes=1,
        )
        children.append(mutated)

    return tuple(children)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Métricas


@dataclass(frozen=True, slots=True)
class PopulationMetrics:
    """Resumo de uma geração, pronto para histórico e CSV.

    ``bit_entropy`` é a média, sobre os qubits, da entropia de Shannon da
    frequência de 1 na população, em bits (0 = todos os bits determinísticos,
    1 = todos em 50/50). ``mean_hamming`` é a distância de Hamming média entre
    todos os pares, normalizada por ``S``.
    """

    generation: int
    size: int
    best: float
    mean: float
    median: float
    std: float
    unique_bits: int
    unique_genomes: int
    mean_hamming: float
    bit_entropy: float
    non_identity: int
    h_gates: int
    t_gates: int
    identity: int

    def __post_init__(self) -> None:
        for name in ("generation", "size", "unique_bits", "unique_genomes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} deve ser int, recebeu {type(value).__name__}.")
            if int(value) < 0:
                raise ValueError(f"{name} deve ser >= 0, recebeu {value}.")

        for name in ("best", "mean", "median", "std", "mean_hamming", "bit_entropy"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} deve ser float, recebeu {type(value).__name__}.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} deve ser finito, recebeu {value}.")

        for name in ("non_identity", "h_gates", "t_gates", "identity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} deve ser int, recebeu {type(value).__name__}.")
            if int(value) < 0:
                raise ValueError(f"{name} deve ser >= 0, recebeu {value}.")

    @property
    def total_genes(self) -> int:
        return self.non_identity + self.identity

    @property
    def genome_fill(self) -> float:
        """Fração de genes não-identidade na população inteira."""

        total = self.total_genes
        return self.non_identity / total if total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "size": self.size,
            "best": self.best,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "unique_bits": self.unique_bits,
            "unique_genomes": self.unique_genomes,
            "mean_hamming": self.mean_hamming,
            "bit_entropy": self.bit_entropy,
            "non_identity": self.non_identity,
            "h_gates": self.h_gates,
            "t_gates": self.t_gates,
            "identity": self.identity,
            "genome_fill": self.genome_fill,
        }

    def __str__(self) -> str:
        return (
            f"g{self.generation:<3d} best={self.best:.4f} mean={self.mean:.4f} "
            f"std={self.std:.4f} únicos={self.unique_bits} "
            f"hamming={self.mean_hamming:.3f} H={self.bit_entropy:.3f} "
            f"H_gates={self.h_gates} T={self.t_gates}"
        )


def _require_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} deve ser int, recebeu {type(value).__name__}.")
    if int(value) < minimum:
        raise ValueError(f"{name} deve ser >= {minimum}, recebeu {value}.")
    return int(value)


def _binary_entropy(probability: float) -> float:
    """Entropia de Shannon de uma Bernoulli, em bits."""

    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (
        1.0 - probability
    ) * math.log2(1.0 - probability)


def _mean_hamming(bitstrings: Sequence[str], size: int) -> float:
    if len(bitstrings) < 2:
        return 0.0

    total = 0
    pairs = 0

    for first in range(len(bitstrings)):
        for second in range(first + 1, len(bitstrings)):
            total += sum(
                1
                for left, right in zip(bitstrings[first], bitstrings[second])
                if left != right
            )
            pairs += 1

    return total / (pairs * size)


def population_metrics(
    population: Sequence[HybridIndividual],
    generation: int = 0,
) -> PopulationMetrics:
    """Agrega fitness, diversidade e uso do genoma de uma geração."""

    items = _as_population(population)
    label = _require_int("generation", generation, minimum=0)

    size = items[0].genome.size
    for index, individual in enumerate(items):
        if individual.genome.size != size:
            raise ValueError(
                f"population[{index}] tem {individual.genome.size} qubits, "
                f"esperado {size}."
            )

    values = [individual.fitness for individual in items]
    bitstrings = [individual.bits for individual in items]

    one_frequency = [
        sum(1 for bits in bitstrings if bits[qubit] == "1") / len(bitstrings)
        for qubit in range(size)
    ]

    summaries = [genome_metrics(individual.genome) for individual in items]

    return PopulationMetrics(
        generation=label,
        size=len(items),
        best=max(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        unique_bits=len(set(bitstrings)),
        unique_genomes=len({individual.genome for individual in items}),
        mean_hamming=_mean_hamming(bitstrings, size),
        bit_entropy=statistics.fmean(
            _binary_entropy(probability) for probability in one_frequency
        ),
        non_identity=sum(summary.non_identity for summary in summaries),
        h_gates=sum(summary.h_gates for summary in summaries),
        t_gates=sum(summary.t_gates for summary in summaries),
        identity=sum(summary.identity for summary in summaries),
    )
