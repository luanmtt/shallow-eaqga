"""Compara o pareamento P/N do crossover entre baseline e híbrido.

Duas perguntas:

1. **Controlada:** dados os mesmos pais, o mesmo ``coupling``, ``t`` e RNG, os
   dois algoritmos selecionam os mesmos pares correlacionados? A seleção é a
   mesma; a topologia (``STAR`` × ``PATH``) muda as arestas finais.
2. **No protocolo:** rodando os dois com o mesmo seed, os pais de cada geração
   coincidem? Se não, os pares sorteados também não — a seleção é idêntica,
   mas opera sobre bitstrings diferentes.

Uso:
    python tests/compare_pairing.py
    python tests/compare_pairing.py --size 30 --population 10 --generations 20
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

import eaqga.baseline as baseline_module
import eaqga.runner as runner_module
from eaqga.baseline import run_baseline
from eaqga.config import (
    BackendKind,
    BaselineConfig,
    ForestShape,
    GateSet,
    HybridConfig,
)
from eaqga.crossover import Forest, build_crossover_forest
from eaqga.instance import QUBOInstance
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

EdgeSet = frozenset[tuple[int, int]]
PairSet = frozenset[tuple[int, int, str]]


@dataclass(frozen=True, slots=True)
class Record:
    generation: int
    first: str
    second: str
    shape: ForestShape
    pairs: PairSet
    edges: EdgeSet
    nodes: frozenset[int]


def _record_forest(
    forest: Forest,
    first: str,
    second: str,
    generation: int,
    shape: ForestShape,
) -> Record:
    pairs = frozenset(
        (pair.i, pair.j, pair.kind.value) for pair in forest.selected_pairs
    )
    edges = frozenset(
        (min(edge.control, edge.target), max(edge.control, edge.target))
        for edge in forest.edges
    )
    nodes = frozenset(
        node
        for edge in forest.edges
        for node in (edge.control, edge.target)
    )
    return Record(
        generation=generation,
        first=first,
        second=second,
        shape=shape,
        pairs=pairs,
        edges=edges,
        nodes=nodes,
    )


def _install_recorder(module: object, records: list[Record]) -> object:
    original = getattr(module, "build_crossover_forest")

    def wrapper(
        first_bits: str,
        second_bits: str,
        coupling: object,
        t: int,
        t_max: int,
        p_s: float,
        *,
        shape: ForestShape = ForestShape.PATH,
        max_degree: int | None = 2,
        rng: random.Random | None = None,
    ) -> Forest:
        forest = original(
            first_bits,
            second_bits,
            coupling,
            t,
            t_max,
            p_s,
            shape=shape,
            max_degree=max_degree,
            rng=rng,
        )
        records.append(
            _record_forest(forest, first_bits, second_bits, t, shape)
        )
        return forest

    setattr(module, "build_crossover_forest", wrapper)
    return original


def _jaccard(first: frozenset[object], second: frozenset[object]) -> float:
    if not first and not second:
        return 1.0
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _hamming(first: str, second: str) -> int:
    return sum(left != right for left, right in zip(first, second))


def _load_instance(
    size: int,
    subset: int,
    seed: int,
) -> QUBOInstance:
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


def _run_both(
    instance: QUBOInstance,
    *,
    population: int,
    generations: int,
    shots: int,
    seed: int,
) -> tuple[list[Record], list[Record]]:
    baseline_records: list[Record] = []
    hybrid_records: list[Record] = []

    baseline_original = _install_recorder(
        baseline_module,
        baseline_records,
    )
    runner_original = _install_recorder(
        runner_module,
        hybrid_records,
    )

    try:
        run_baseline(
            instance,
            BaselineConfig(
                population_size=population,
                generations=generations,
                shots=shots,
                p_a=0.95,
                p_s=0.6,
                forest_shape=ForestShape.STAR,
                max_degree=None,
                seed=seed,
                backend=BackendKind.MPS,
                verbose=False,
            ),
        )
        run_hybrid(
            instance,
            HybridConfig(
                population_size=population,
                generations=generations,
                shots=shots,
                num_columns=2,
                gate_set=GateSet.HILBERT_1Q,
                forest_shape=ForestShape.PATH,
                max_degree=2,
                mutation_rate=0.10,
                p_s=0.6,
                seed=seed,
                backend=BackendKind.MPS,
                verbose=False,
            ),
        )
    finally:
        setattr(baseline_module, "build_crossover_forest", baseline_original)
        setattr(runner_module, "build_crossover_forest", runner_original)

    return baseline_records, hybrid_records


def _by_generation(records: list[Record]) -> dict[int, list[Record]]:
    grouped: dict[int, list[Record]] = {}
    for record in records:
        grouped.setdefault(record.generation, []).append(record)
    return grouped


def _union(records: list[Record], attribute: str) -> frozenset[object]:
    values: set[object] = set()
    for record in records:
        values.update(getattr(record, attribute))
    return frozenset(values)


def _controlled_test(
    instance: QUBOInstance,
    records: list[Record],
    *,
    generations: int,
    seed: int,
) -> None:
    """Mesmos pais + mesmo RNG: seleção de pares e arestas STAR × PATH."""

    pair_scores: list[float] = []
    edge_scores: list[float] = []
    same_selection = 0
    total = 0

    for generation, entries in sorted(_by_generation(records).items()):
        if not entries:
            continue
        first = entries[0].first
        second = entries[0].second

        star = build_crossover_forest(
            first,
            second,
            instance.coupling,
            t=generation,
            t_max=generations,
            p_s=0.6,
            shape=ForestShape.STAR,
            max_degree=None,
            rng=random.Random(seed),
        )
        path = build_crossover_forest(
            first,
            second,
            instance.coupling,
            t=generation,
            t_max=generations,
            p_s=0.6,
            shape=ForestShape.PATH,
            max_degree=2,
            rng=random.Random(seed),
        )

        star_record = _record_forest(
            star, first, second, generation, ForestShape.STAR
        )
        path_record = _record_forest(
            path, first, second, generation, ForestShape.PATH
        )

        pair_scores.append(
            _jaccard(
                frozenset(star_record.pairs),
                frozenset(path_record.pairs),
            )
        )
        edge_scores.append(
            _jaccard(
                frozenset(star_record.edges),
                frozenset(path_record.edges),
            )
        )
        same_selection += int(star_record.pairs == path_record.pairs)
        total += 1

    print(sepComment(40, stylePrint("1. Controlado (mesmos pais)", C, bold=True)))
    print(f"  gerações comparadas: {total}")
    print(
        "  pares selecionados idênticos: "
        f"{same_selection}/{total} "
        f"(Jaccard médio {np.mean(pair_scores):.3f})"
    )
    print(
        "  arestas finais iguais: "
        f"Jaccard médio {np.mean(edge_scores):.3f}"
    )
    print(
        "  leitura: a seleção P/N é a mesma; STAR e PATH orientam/limitam "
        "as arestas de forma diferente."
    )


def _protocol_test(
    baseline_records: list[Record],
    hybrid_records: list[Record],
    *,
    population: int,
) -> None:
    baseline_groups = _by_generation(baseline_records)
    hybrid_groups = _by_generation(hybrid_records)

    print()
    print(sepComment(40, stylePrint("2. No protocolo (mesmo seed)", C, bold=True)))
    header = (
        f"{'ger':>4} {'pais':>6} {'dist':>8} "
        f"{'pares':>7} {'arestas':>8} {'qubits':>7}"
    )
    print(f"  {stylePrint(header, C, bold=True)}")

    pair_scores: list[float] = []
    edge_scores: list[float] = []
    node_scores: list[float] = []
    same_parents = 0
    same_pairs = 0
    generations = sorted(set(baseline_groups) & set(hybrid_groups))

    for generation in generations:
        base_entries = baseline_groups[generation]
        hybrid_entries = hybrid_groups[generation]

        if len(base_entries) != population or len(hybrid_entries) != population:
            continue

        base_first = base_entries[0].first
        hybrid_first = hybrid_entries[0].first
        base_second = base_entries[0].second
        hybrid_second = hybrid_entries[0].second

        parents_equal = {base_first, base_second} == {
            hybrid_first,
            hybrid_second,
        }
        same_parents += int(parents_equal)
        distance = min(
            _hamming(base_first, hybrid_first)
            + _hamming(base_second, hybrid_second),
            _hamming(base_first, hybrid_second)
            + _hamming(base_second, hybrid_first),
        )

        base_pairs = _union(base_entries, "pairs")
        hybrid_pairs = _union(hybrid_entries, "pairs")
        base_edges = _union(base_entries, "edges")
        hybrid_edges = _union(hybrid_entries, "edges")
        base_nodes = _union(base_entries, "nodes")
        hybrid_nodes = _union(hybrid_entries, "nodes")

        pair_score = _jaccard(base_pairs, hybrid_pairs)
        edge_score = _jaccard(base_edges, hybrid_edges)
        node_score = _jaccard(base_nodes, hybrid_nodes)
        pair_scores.append(pair_score)
        edge_scores.append(edge_score)
        node_scores.append(node_score)
        same_pairs += int(base_pairs == hybrid_pairs)

        print(
            f"  {generation:>4} "
            f"{'sim' if parents_equal else 'não':>6} "
            f"{distance:>8} "
            f"{pair_score:>7.3f} "
            f"{edge_score:>8.3f} "
            f"{node_score:>7.3f}"
        )

    print()
    print(
        f"  pais idênticos: {same_parents}/{len(generations)} gerações · "
        f"pares idênticos: {same_pairs}/{len(generations)}"
    )
    print(
        "  Jaccard médio (união dos N filhos): "
        f"pares {np.mean(pair_scores):.3f} · "
        f"arestas {np.mean(edge_scores):.3f} · "
        f"qubits {np.mean(node_scores):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compara o pareamento P/N entre baseline e híbrido.",
    )
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shots", type=int, default=1)
    args = parser.parse_args()

    instance = _load_instance(args.size, args.subset, args.seed)
    print(sepComment(40, stylePrint("Pareamento P/N", C, bold=True)))
    print(
        f"  S={args.size} · pop={args.population} · T={args.generations} · "
        f"shots={args.shots} · seed={args.seed}"
    )
    print()

    baseline_records, hybrid_records = _run_both(
        instance,
        population=args.population,
        generations=args.generations,
        shots=args.shots,
        seed=args.seed,
    )

    _controlled_test(
        instance,
        baseline_records,
        generations=args.generations,
        seed=args.seed,
    )
    _protocol_test(
        baseline_records,
        hybrid_records,
        population=args.population,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
