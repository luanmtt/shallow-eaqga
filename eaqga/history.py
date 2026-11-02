"""Histórico por geração dos três algoritmos em um CSV único.

O CSV de execuções (``runs-*.csv``) guarda uma linha por run. Este módulo
transpõe o histórico de cada run para uma linha por geração, com o mesmo
schema para baseline, híbrido e original. Como as três implementações
registram colunas diferentes, o CSV é a união delas: o que um algoritmo não
mede fica vazio.

Semântica das colunas:

- ``best``: melhor fitness visto até a geração (best-so-far);
- ``generation_best``: melhor fitness da população na geração;
- ``avg``/``median``/``std``: estatísticas do fitness da população;
- ``unique_bits``/``unique_genomes``/``mean_hamming``/``bit_entropy``:
  diversidade (só o híbrido registra);
- ``non_identity``/``h_gates``/``t_gates``: uso do genoma (só o híbrido);
- ``forest_edges``/``forest_layers``/``forest_chains``: no híbrido são médias
  da geração; no baseline são do melhor indivíduo da geração;
- ``best_bits``: bitstring do melhor fitness visto.

Os três algoritmos incluem ``generation=0`` (população inicial) e terminam em
``generation=T``. No original, a avaliação final do motor (feita depois do
laço) é anexada como ``generation=T``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from .baseline import BaselineResult
from .crossover import ForestMetrics
from .original_adapter import OriginalResult
from .runner import HybridResult

__all__ = ["HISTORY_COLUMNS", "history_rows"]

_META_COLUMNS: Final[tuple[str, ...]] = (
    "size",
    "population",
    "generations",
    "shots",
    "initial_h",
    "algorithm",
    "subset",
    "run",
    "seed",
)

_METRIC_COLUMNS: Final[tuple[str, ...]] = (
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

HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    *_META_COLUMNS,
    "generation",
    *_METRIC_COLUMNS,
)

_ALGORITHMS: Final[tuple[str, ...]] = ("baseline", "ours", "original")


def _base_row(
    meta: Mapping[str, object],
    generation: int,
) -> dict[str, object]:
    """Linha com metadados, geração e todas as métricas vazias."""

    row: dict[str, object] = dict(meta)
    row["generation"] = generation
    for column in _METRIC_COLUMNS:
        row[column] = None
    return row


def _hybrid_rows(
    result: HybridResult,
    meta: Mapping[str, object],
) -> list[dict[str, object]]:
    history = result.history
    length = len(history["generation"])
    rows: list[dict[str, object]] = []

    for index in range(length):
        row = _base_row(meta, int(history["generation"][index]))
        for column in _METRIC_COLUMNS:
            if column in history:
                row[column] = history[column][index]
        rows.append(row)

    return rows


def _baseline_rows(
    result: BaselineResult,
    meta: Mapping[str, object],
) -> list[dict[str, object]]:
    history = result.history
    length = len(history["generation"])
    rows: list[dict[str, object]] = []

    for index in range(length):
        row = _base_row(meta, int(history["generation"][index]))
        for column in ("best", "generation_best", "avg", "median", "best_bits"):
            if column in history:
                row[column] = history[column][index]

        metrics = history.get("forest_metrics", [])
        summary = metrics[index] if index < len(metrics) else None
        if isinstance(summary, ForestMetrics):
            row["forest_edges"] = summary.edges
            row["forest_layers"] = summary.layers
            row["forest_chains"] = summary.chains

        rows.append(row)

    return rows


def _original_rows(
    result: OriginalResult,
    meta: Mapping[str, object],
) -> list[dict[str, object]]:
    history = result.history
    best = history.get("best", [])
    rows: list[dict[str, object]] = []

    for index in range(len(best)):
        row = _base_row(meta, index)
        row["best"] = best[index]

        for column in ("avg", "median", "best_bits"):
            values = history.get(column, [])
            if index < len(values):
                row[column] = values[index]

        rows.append(row)

    return rows


def history_rows(
    result: HybridResult | BaselineResult | OriginalResult,
    *,
    algorithm: str,
    size: int,
    population: int,
    generations: int,
    shots: int,
    initial_h: bool,
    subset: int,
    run: int,
    seed: int,
) -> list[dict[str, object]]:
    """Transpõe o histórico de uma execução em linhas (uma por geração).

    ``algorithm`` precisa casar com o tipo de ``result``; isso evita exportar
    um histórico com o rótulo errado no CSV.
    """

    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"algorithm deve ser um de {_ALGORITHMS}, recebeu {algorithm!r}."
        )
    if not isinstance(initial_h, bool):
        raise TypeError(
            "initial_h deve ser bool, recebeu "
            f"{type(initial_h).__name__}."
        )

    meta: dict[str, object] = {
        "size": size,
        "population": population,
        "generations": generations,
        "shots": shots,
        "initial_h": initial_h,
        "algorithm": algorithm,
        "subset": subset,
        "run": run,
        "seed": seed,
    }

    if algorithm == "ours":
        if not isinstance(result, HybridResult):
            raise TypeError(
                "algorithm='ours' exige HybridResult, recebeu "
                f"{type(result).__name__}."
            )
        return _hybrid_rows(result, meta)

    if algorithm == "baseline":
        if not isinstance(result, BaselineResult):
            raise TypeError(
                "algorithm='baseline' exige BaselineResult, recebeu "
                f"{type(result).__name__}."
            )
        return _baseline_rows(result, meta)

    if not isinstance(result, OriginalResult):
        raise TypeError(
            "algorithm='original' exige OriginalResult, recebeu "
            f"{type(result).__name__}."
        )
    return _original_rows(result, meta)
