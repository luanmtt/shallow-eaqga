"""Checagens de integridade do pacote ``eaqga``.

Este script é propositalmente leve: não simula circuitos nem baixa dados.
Ele valida imports, guards das configs, construção da instância QUBO,
fitness escalar/vetorizado e serialização. Serve como smoke test antes de
rodar experimentos.

Uso:
    python tests/check_integrity.py
    python tests/check_integrity.py --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from eaqga.config import (
    BaselineConfig,
    EvaluationMode,
    ForestShape,
    HybridConfig,
)
from eaqga.crossover import (
    CorrelationKind,
    Pair,
    build_crossover_forest,
    correlation_pairs,
    decay_factor,
    forest_metrics,
    make_forest,
    orient_forest,
    pair_probability,
)
from eaqga.fitness import evaluate_counts
from eaqga.instance import QUBOInstance, synthetic_qubo
from eaqga.utils.embellish import sepComment, stylePrint
from eaqga.utils.formats import formatValue

C = "#f5c2e7"
PASS_COLOR = "#3a5716"
FAIL_COLOR = "#ff6b6b"

Check = Callable[[], str | None]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect(
    exc_type: type[Exception],
    fn: Callable[[], object],
    label: str,
) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"{label}: esperava {exc_type.__name__}, veio "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"{label}: esperava {exc_type.__name__}, nada foi levantado."
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Checks


def check_config_defaults() -> None:
    hybrid = HybridConfig(verbose=False)
    baseline = BaselineConfig(verbose=False)

    _check(hybrid.population_size >= 2, "population_size default inválido")
    _check(hybrid.n_elites == 2, "n_elites default deve ser 2")
    _check(
        hybrid.evaluation_mode is EvaluationMode.SCALAR,
        "avaliação default deve ser SCALAR",
    )
    _check(baseline.shots == 1, "baseline default deve ter 1 shot")


def check_config_guards() -> None:
    _expect(
        ValueError,
        lambda: HybridConfig(population_size=1),
        "population_size=1",
    )
    _expect(
        ValueError,
        lambda: HybridConfig(n_elites=3),
        "n_elites=3",
    )
    _expect(
        ValueError,
        lambda: HybridConfig(mutation_rate=1.5),
        "mutation_rate=1.5",
    )
    _expect(
        TypeError,
        lambda: HybridConfig(gate_set="hilbert_1q"),  # type: ignore[arg-type]
        "gate_set string",
    )
    _expect(
        ValueError,
        lambda: BaselineConfig(p_a=1.0),
        "p_a=1.0",
    )


def check_config_roundtrip() -> None:
    config = HybridConfig(
        evaluation_mode=EvaluationMode.VECTORIZED,
        verbose=False,
    )
    data = config.to_dict()

    _check(
        data["evaluation_mode"] == "vectorized",
        "evaluation_mode não serializou como string",
    )
    _check(
        HybridConfig.from_dict(data) == config,
        "round-trip de HybridConfig falhou",
    )


def check_instance_synthetic() -> None:
    instance = synthetic_qubo(S=4, seed=7)

    _check(instance.Q.shape == (4, 4), "Q com shape errado")
    _check(instance.coupling.shape == (4, 4), "coupling com shape errado")
    _check(not instance.Q.flags.writeable, "Q deveria ser read-only")
    _check(
        not instance.coupling.flags.writeable,
        "coupling deveria ser read-only",
    )


def check_instance_fitness() -> None:
    mu = np.array([1.0, 2.0])
    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
    instance = QUBOInstance.from_returns(mu, sigma, q=0.5)

    for bits in ("00", "10", "01", "11"):
        x = np.array([int(char) for char in bits], dtype=float)
        expected = float(mu @ x - 0.5 * x @ sigma @ x)
        got = instance.fitness(bits)
        _check(
            math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12),
            f"fitness({bits})={got} != {expected}",
        )


def check_instance_roundtrip() -> None:
    instance = synthetic_qubo(S=4, seed=7)
    data = instance.to_dict()

    json.dumps(data)  # deve ser JSON-friendly
    restored = QUBOInstance.from_dict(data)

    _check(
        math.isclose(
            instance.fitness("1010"),
            restored.fitness("1010"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "round-trip da instância mudou o fitness",
    )


def check_instance_guards() -> None:
    _expect(
        ValueError,
        lambda: QUBOInstance("", 2, np.eye(2), np.eye(2)),
        "nome vazio",
    )
    _expect(
        ValueError,
        lambda: QUBOInstance("x", 3, np.eye(2), np.eye(2)),
        "shape incompatível",
    )
    _expect(
        TypeError,
        lambda: QUBOInstance.from_dict([]),  # type: ignore[arg-type]
        "from_dict com lista",
    )


def check_fitness_counts() -> None:
    instance = QUBOInstance.from_qubo(np.eye(2), name="identity")
    counts = {"00": 2, "10": 6, "01": 1, "11": 1}
    evaluation = evaluate_counts(counts, instance)

    _check(evaluation.shots == 10, "shots errado")
    _check(
        math.isclose(evaluation.mean, 0.9, rel_tol=0.0, abs_tol=1e-12),
        f"mean={evaluation.mean} != 0.9",
    )
    _check(
        evaluation.best_bitstring == "11",
        f"best_bitstring={evaluation.best_bitstring!r} != '11'",
    )
    _check(
        evaluation.to_dict()["shots"] == 10,
        "Evaluation.to_dict não incluiu shots",
    )


def check_fitness_vectorized() -> None:
    instance = synthetic_qubo(S=6, seed=7)
    bitstrings = ["000000", "111111", "101010", "010101"]

    scalar = instance.fitness_many(bitstrings, vectorized=False)
    vector = instance.fitness_many(bitstrings, vectorized=True)
    _check(
        np.allclose(scalar, vector),
        "fitness_many escalar e vetorizado divergiram",
    )

    counts = {
        bits: index + 1
        for index, bits in enumerate(bitstrings)
    }
    scalar_eval = evaluate_counts(
        counts,
        instance,
        reverse_bits=False,
        vectorized=False,
    )
    vector_eval = evaluate_counts(
        counts,
        instance,
        reverse_bits=False,
        vectorized=True,
    )

    _check(
        math.isclose(
            scalar_eval.mean,
            vector_eval.mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "mean escalar e vetorizado divergiram",
    )
    _check(
        scalar_eval.best_bitstring == vector_eval.best_bitstring,
        "best_bitstring escalar e vetorizado divergiram",
    )


def check_brute_force() -> str:
    instance = synthetic_qubo(S=4, seed=7)
    best_bits, best_value = instance.brute_force()

    _check(
        math.isclose(
            instance.fitness(best_bits),
            best_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "brute_force devolveu fitness inconsistente",
    )

    return f"best={best_bits} fitness={formatValue(best_value)}"


def check_crossover_pairs() -> None:
    pairs = correlation_pairs("00110", "01011")
    got = {(pair.i, pair.j, pair.kind) for pair in pairs}

    expected = {
        (0, 3, CorrelationKind.NEGATIVE),
        (1, 2, CorrelationKind.NEGATIVE),
        (1, 4, CorrelationKind.POSITIVE),
        (2, 4, CorrelationKind.NEGATIVE),
    }

    _check(got == expected, f"pares P/N inesperados: {got}")


def check_crossover_probability() -> None:
    _check(
        math.isclose(decay_factor(1, 10), 0.55, rel_tol=0.0, abs_tol=1e-12),
        "df(1, 10) != 0.55",
    )

    cases = (
        (CorrelationKind.POSITIVE, 0.5, 0.165),
        (CorrelationKind.POSITIVE, -0.5, 0.300),
        (CorrelationKind.NEGATIVE, -0.5, 0.300),
        (CorrelationKind.NEGATIVE, 0.5, 0.165),
    )

    for kind, coupling, expected in cases:
        got = pair_probability(kind, coupling, 1, 10, 0.6)
        _check(
            math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12),
            f"p_s({kind.value}, {coupling})={got} != {expected}",
        )


def check_crossover_forest() -> None:
    path_pairs = (
        Pair(0, 1, CorrelationKind.POSITIVE),
        Pair(1, 2, CorrelationKind.NEGATIVE),
        Pair(2, 3, CorrelationKind.POSITIVE),
    )

    path_edges = orient_forest(path_pairs, 4, ForestShape.PATH)
    path_forest = make_forest(path_edges, path_pairs)
    path_metrics = forest_metrics(path_forest)

    _check(path_metrics.edges == 3, "path: edges != 3")
    _check(path_metrics.layers == 2, "path: layers != 2")
    _check(path_metrics.chains == 1, "path: chains != 1")
    _check(path_metrics.max_chain_length == 4, "path: max_chain_length != 4")
    _check(path_metrics.max_degree == 2, "path: max_degree != 2")
    _check(path_metrics.positive_edges == 2, "path: positive_edges != 2")
    _check(path_metrics.negative_edges == 1, "path: negative_edges != 1")

    star_edges = orient_forest(path_pairs, 4, ForestShape.STAR)
    star_forest = make_forest(star_edges, path_pairs)
    star_metrics = forest_metrics(star_forest)

    _check(star_metrics.layers == 3, "star: layers != 3")
    _check(star_metrics.max_degree == 3, "star: max_degree != 3")

    empty_metrics = forest_metrics(make_forest(()))
    _check(empty_metrics.edges == 0, "empty: edges != 0")
    _check(empty_metrics.chains == 0, "empty: chains != 0")

    coupling = np.full((4, 4), 0.5)
    np.fill_diagonal(coupling, 0.0)
    full_forest = build_crossover_forest(
        "0011",
        "0101",
        coupling,
        t=10,
        t_max=10,
        p_s=1.0,
        shape=ForestShape.PATH,
        max_degree=2,
    )
    full_metrics = forest_metrics(full_forest)

    _check(full_metrics.edges <= 3, "full: edges > S - 1")
    _check(
        full_metrics.selected_pairs == len(full_forest.selected_pairs),
        "full: selected_pairs inconsistente",
    )


def check_crossover_guards() -> None:
    cycle = (
        Pair(0, 1, CorrelationKind.POSITIVE),
        Pair(1, 2, CorrelationKind.POSITIVE),
        Pair(0, 2, CorrelationKind.POSITIVE),
    )

    _expect(
        ValueError,
        lambda: orient_forest(cycle, 3, ForestShape.PATH),
        "ciclo em orient_forest",
    )
    _expect(
        ValueError,
        lambda: correlation_pairs("0a", "01"),
        "bit inválido",
    )
    _expect(
        ValueError,
        lambda: build_crossover_forest(
            "01",
            "01",
            np.zeros((2, 2)),
            1,
            1,
            0.5,
            shape=ForestShape.PATH,
            max_degree=3,
        ),
        "PATH com max_degree != 2",
    )
    _expect(
        ValueError,
        lambda: build_crossover_forest(
            "01",
            "01",
            np.zeros((2, 2)),
            1,
            1,
            0.5,
            shape=ForestShape.STAR,
            max_degree=2,
        ),
        "STAR com max_degree != None",
    )


CHECKS: list[tuple[str, Check]] = [
    ("config.defaults", check_config_defaults),
    ("config.guards", check_config_guards),
    ("config.roundtrip", check_config_roundtrip),
    ("instance.synthetic", check_instance_synthetic),
    ("instance.fitness", check_instance_fitness),
    ("instance.roundtrip", check_instance_roundtrip),
    ("instance.guards", check_instance_guards),
    ("fitness.counts", check_fitness_counts),
    ("fitness.vectorized", check_fitness_vectorized),
    ("instance.brute_force", check_brute_force),
    ("crossover.pairs", check_crossover_pairs),
    ("crossover.probability", check_crossover_probability),
    ("crossover.forest", check_crossover_forest),
    ("crossover.guards", check_crossover_guards),
]


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Runner


def run_checks(*, verbose: bool = False) -> int:
    print(sepComment(40, stylePrint("Integrity checks", C, bold=True)))
    print(f"  {stylePrint('• root', C)}: {_ROOT}")

    results: list[tuple[str, bool, str | None, Exception | None]] = []

    for name, check in CHECKS:
        detail: str | None = None
        error: Exception | None = None

        try:
            detail = check()
        except Exception as exc:  # noqa: BLE001 - o runner agrega qualquer falha
            error = exc

        results.append((name, error is None, detail, error))

    print()
    for name, passed, detail, error in results:
        status = (
            stylePrint("PASS", PASS_COLOR, bold=True)
            if passed
            else stylePrint("FAIL", FAIL_COLOR, bold=True)
        )
        line = f"  {status}  {name}"
        if detail:
            line += f"  ({stylePrint(detail, C)})"
        print(line)

        if error is not None:
            print(f"        {stylePrint(str(error), FAIL_COLOR)}")
            if verbose:
                print(stylePrint(traceback.format_exc(), FAIL_COLOR))

    passed_count = sum(1 for _, passed, _, _ in results if passed)
    total = len(results)
    summary_color = PASS_COLOR if passed_count == total else FAIL_COLOR

    print()
    print(
        sepComment(
            40,
            stylePrint(
                f"{passed_count}/{total} checks passed",
                summary_color,
                bold=True,
            ),
        )
    )

    return 0 if passed_count == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Checagens leves de integridade do pacote eaqga.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="imprime traceback completo das falhas.",
    )
    args = parser.parse_args()

    return run_checks(verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
