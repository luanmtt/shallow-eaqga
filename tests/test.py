"""Smoke test do pipeline do EAQGA com um dataset dummy.

O baseline usa o runner real em ``eaqga.baseline`` (Qiskit/Aer). O "ours"
ainda é um surrogate clássico de bitstrings, porque ``individuo.py`` e
``runner.py`` reais ainda não existem. O objetivo é comparar o encanamento:

    dataset dummy → fitness QUBO → top-2 → crossover → mutação → métricas

O baseline é fiel ao paper: ``H^S → RY → CNOTs → medida``, um shot por
circuito, top-2 da geração. O "ours" usa uma versão bit-a-bit da ideia de
entrelaçamento (pares P/N dos dois pais) e serve só como placeholder.

Uso:
    python tests/test.py              # baseline real + ours surrogate
    python tests/test.py --baseline   # só baseline real (precisa de Qiskit)
    python tests/test.py --ours       # só ours surrogate
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eaqga.config import (
    BackendKind,
    BaselineConfig,
    EvaluationMode,
    ForestShape,
)
from eaqga.instance import QUBOInstance, synthetic_qubo
from eaqga.utils.embellish import sepComment, stylePrint
from eaqga.utils.formats import formatValue

C = "#f5c2e7"
PASS_COLOR = "#3a5716"
FAIL_COLOR = "#ff6b6b"

POPULATION_SIZE = 5
GENERATIONS = 5
MUTATION_RATE = 0.10
P_A = 0.95
P_S = 0.60
PARENT_BIAS = 0.75
SEED = 42

DUMMY_S = 4
DUMMY_SEED = 7


@dataclass(frozen=True, slots=True)
class Individual:
    bits: str
    fitness: float


@dataclass(frozen=True, slots=True)
class RunResult:
    name: str
    initial_best: float
    best: float
    best_bits: str
    final_mean: float
    final_std: float
    evaluations: int
    elapsed_s: float
    best_generation: int | None = None

    def gap_to(self, optimum: float) -> float:
        return optimum - self.best

    def found_optimum(self, optimum: float, tol: float = 1e-9) -> bool:
        return math.isclose(self.best, optimum, rel_tol=0.0, abs_tol=tol)


def _random_bits(rng: random.Random, size: int) -> str:
    return "".join(rng.choice("01") for _ in range(size))


def _evaluate(bits: str, instance: QUBOInstance) -> Individual:
    return Individual(bits=bits, fitness=instance.fitness(bits))


def _evaluate_population(
    bitstrings: list[str],
    instance: QUBOInstance,
) -> list[Individual]:
    return [_evaluate(bits, instance) for bits in bitstrings]


def _select_elites(population: list[Individual]) -> list[Individual]:
    ranked = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    return ranked[:2]


def _mutate(
    bits: str,
    rng: random.Random,
    mutation_rate: float,
) -> str:
    flipped = [
        ("1" if char == "0" else "0") if rng.random() < mutation_rate else char
        for char in bits
    ]
    return "".join(flipped)


def _correlation_pairs(
    first: str,
    second: str,
) -> list[tuple[int, int, str]]:
    """Pares P/N entre os dois pais, no espírito do crossover do EAQGA."""

    pairs: list[tuple[int, int, str]] = []

    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            if first[i] == first[j] and second[i] == second[j]:
                pairs.append((i, j, "P"))
            elif first[i] != first[j] and second[i] != second[j]:
                pairs.append((i, j, "N"))

    return pairs


def _apply_correlations(
    bits: str,
    pairs: list[tuple[int, int, str]],
    rng: random.Random,
    p_s: float,
) -> str:
    result = list(bits)

    for i, j, kind in pairs:
        if rng.random() >= p_s:
            continue

        if kind == "P":
            result[j] = result[i]
        else:
            result[j] = "1" if result[i] == "0" else "0"

    return "".join(result)


def _mean_std(population: list[Individual]) -> tuple[float, float]:
    values = [ind.fitness for ind in population]
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def run_baseline_real(instance: QUBOInstance) -> RunResult:
    """Roda o baseline real (Qiskit/Aer) e adapta para a tabela do teste."""

    try:
        from eaqga.baseline import run_baseline as run_baseline_qiskit
    except ImportError as exc:
        print(
            stylePrint(
                "Qiskit/Aer indisponível para --baseline. "
                "Instale requirements.txt ou use --ours.",
                FAIL_COLOR,
                bold=True,
            )
        )
        raise SystemExit(2) from exc

    config = BaselineConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        shots=1,
        p_a=P_A,
        p_s=P_S,
        forest_shape=ForestShape.STAR,
        max_degree=None,
        accumulated_pool=False,
        seed=SEED,
        backend=BackendKind.STATEVECTOR,
        evaluation_mode=EvaluationMode.SCALAR,
        verbose=False,
    )

    result = run_baseline_qiskit(instance, config)

    final_values = [ind.fitness for ind in result.final_population]
    initial_value = result.history["best"][0]
    if isinstance(initial_value, bool) or not isinstance(initial_value, (int, float)):
        raise TypeError(
            "history['best'][0] deveria ser numérico, recebeu "
            f"{type(initial_value).__name__}."
        )

    return RunResult(
        name="baseline",
        initial_best=float(initial_value),
        best=result.best_fitness,
        best_bits=result.best_bits,
        final_mean=statistics.fmean(final_values),
        final_std=(
            statistics.stdev(final_values)
            if len(final_values) > 1
            else 0.0
        ),
        evaluations=result.evaluations,
        elapsed_s=result.elapsed_s,
        best_generation=result.best_generation,
    )


def run_ours_surrogate(
    instance: QUBOInstance,
    *,
    seed: int = SEED,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    mutation_rate: float = MUTATION_RATE,
    p_s: float = P_S,
) -> RunResult:
    """Surrogate do nosso crossover: copia um pai, reforça P/N e muta."""

    rng = random.Random(seed)
    population = _evaluate_population(
        [_random_bits(rng, instance.S) for _ in range(population_size)],
        instance,
    )

    initial_best = max(ind.fitness for ind in population)
    best = max(population, key=lambda ind: ind.fitness)
    best_generation = 0
    evaluations = population_size
    start = time.perf_counter()

    for generation in range(1, generations + 1):
        parents = _select_elites(population)
        pairs = _correlation_pairs(parents[0].bits, parents[1].bits)
        children: list[str] = []

        for _ in range(population_size):
            parent = parents[0] if rng.random() < PARENT_BIAS else parents[1]

            # No circuito real a mutação acontece nas 1Q antes dos CNOTs.
            # Aqui, como surrogate de bitstring, aplicamos as correlações
            # P/N e depois mutamos; o alvo é exercitar a lógica de
            # entrelaçamento, não reproduzir o circuito.
            bits = _apply_correlations(parent.bits, pairs, rng, p_s)
            bits = _mutate(bits, rng, mutation_rate)
            children.append(bits)

        population = _evaluate_population(children, instance)
        evaluations += population_size

        current = max(population, key=lambda ind: ind.fitness)
        if current.fitness > best.fitness:
            best = current
            best_generation = generation

    final_mean, final_std = _mean_std(population)
    elapsed = time.perf_counter() - start

    return RunResult(
        name="ours (sur.)",
        initial_best=initial_best,
        best=best.fitness,
        best_bits=best.bits,
        final_mean=final_mean,
        final_std=final_std,
        evaluations=evaluations,
        elapsed_s=elapsed,
        best_generation=best_generation,
    )


def _check_result(
    result: RunResult,
    instance: QUBOInstance,
    optimum: float,
    tol: float = 1e-9,
) -> tuple[bool, str]:
    if not math.isfinite(result.best):
        return False, f"{result.name}: best não finito"

    if result.best > optimum + tol:
        return False, (
            f"{result.name}: best={result.best} acima do brute force={optimum}"
        )

    if result.best + tol < result.initial_best:
        return False, (
            f"{result.name}: piorou em relação à população inicial "
            f"({result.best} < {result.initial_best})"
        )

    if not result.found_optimum(optimum, tol):
        return False, (
            f"{result.name}: melhor fitness {result.best} != "
            f"brute force {optimum}"
        )

    if not math.isclose(
        instance.fitness(result.best_bits),
        result.best,
        rel_tol=0.0,
        abs_tol=tol,
    ):
        return False, f"{result.name}: fitness do best_bits inconsistente"

    return True, ""


def _print_header(
    instance: QUBOInstance,
    brute_bits: str,
    brute_value: float,
    *,
    do_baseline: bool,
    do_ours: bool,
) -> None:
    print(sepComment(40, stylePrint("EAQGA smoke test", C, bold=True)))
    print(f"  {stylePrint('• instância', C)}: {instance.name} (S={instance.S})")
    print(f"  {stylePrint('• população', C)}: {POPULATION_SIZE}")
    print(f"  {stylePrint('• gerações', C)}: {GENERATIONS}")
    print(f"  {stylePrint('• seed', C)}: {SEED}")
    print(f"  {stylePrint('• p_a', C)}: {P_A}   {stylePrint('p_s', C)}: {P_S}")
    print(
        f"  {stylePrint('• brute force', C)}: {brute_bits} "
        f"fitness={formatValue(brute_value)}"
    )
    if do_baseline:
        print(f"  {stylePrint('• baseline', C)}: Qiskit EAQGA (real)")
    if do_ours:
        print(f"  {stylePrint('• ours', C)}: surrogate clássico")
    if do_baseline and do_ours:
        print(
            f"  {stylePrint('• nota', C)}: baseline real × ours surrogate "
            "(comparação não é apples-to-apples)"
        )
    print()


def _print_results(
    results: list[RunResult],
    instance: QUBOInstance,
    brute_value: float,
) -> None:
    print(sepComment(40, stylePrint("Resultados", C, bold=True)))

    header = (
        f"{'algoritmo':<12} {'best':>16} {'gap':>16} "
        f"{'mean final':>16} {'std':>16} {'evals':>6} {'gen':>5} {'ótimo':>6}"
    )
    print(f"  {stylePrint(header, C, bold=True)}")

    for result in results:
        found = "sim" if result.found_optimum(brute_value) else "não"
        generation = (
            "-"
            if result.best_generation is None
            else str(result.best_generation)
        )
        print(
            f"  {result.name:<12} "
            f"{formatValue(result.best):>16} "
            f"{formatValue(result.gap_to(brute_value)):>16} "
            f"{formatValue(result.final_mean):>16} "
            f"{formatValue(result.final_std):>16} "
            f"{result.evaluations:>6} "
            f"{generation:>5} "
            f"{found:>6}"
        )

    if len(results) == 2:
        baseline, ours = results
        delta_best = ours.best - baseline.best
        delta_gap = baseline.gap_to(brute_value) - ours.gap_to(brute_value)
        print()
        print(
            f"  {stylePrint('ours - baseline (best)', C)}: "
            f"{formatValue(delta_best)}"
        )
        print(
            f"  {stylePrint('redução de gap (ours)', C)}: "
            f"{formatValue(delta_gap)}"
        )
        print(
            f"  {stylePrint('nota', C)}: ours é surrogate; "
            "baseline é Qiskit EAQGA."
        )

    print(
        f"  {stylePrint('brute force', C)}: "
        f"{brute_value:.6f} em {instance.S} bits"
    )


def _check_all(
    results: list[RunResult],
    instance: QUBOInstance,
    brute_value: float,
) -> int:
    checks = [
        _check_result(result, instance, brute_value)
        for result in results
    ]
    failed = [message for ok, message in checks if not ok]

    print()
    print(sepComment(40, stylePrint("Check", C, bold=True)))

    if not failed:
        label = "baseline + ours" if len(results) == 2 else results[0].name
        print(
            f"  {stylePrint('• algoritmo', C)}: "
            f"{label} {stylePrint('OK', PASS_COLOR, bold=True)}"
        )
        return 0

    print(
        f"  {stylePrint('• algoritmo', C)}: "
        f"{stylePrint('FAIL', FAIL_COLOR, bold=True)}"
    )
    for message in failed:
        print(f"    {stylePrint(message, FAIL_COLOR)}")

    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test do pipeline do EAQGA com dataset dummy.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="roda o baseline real (Qiskit) e compara.",
    )
    parser.add_argument(
        "--ours",
        action="store_true",
        help="roda o surrogate do nosso crossover.",
    )
    args = parser.parse_args()

    selected = args.baseline or args.ours
    do_baseline = args.baseline or not selected
    do_ours = args.ours or not selected

    instance = synthetic_qubo(S=DUMMY_S, seed=DUMMY_SEED)
    brute_bits, brute_value = instance.brute_force()

    _print_header(
        instance,
        brute_bits,
        brute_value,
        do_baseline=do_baseline,
        do_ours=do_ours,
    )

    results: list[RunResult] = []

    if do_baseline:
        results.append(run_baseline_real(instance))

    if do_ours:
        results.append(run_ours_surrogate(instance))

    _print_results(results, instance, brute_value)
    return _check_all(results, instance, brute_value)


if __name__ == "__main__":
    raise SystemExit(main())
