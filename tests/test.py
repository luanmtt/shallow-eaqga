"""Smoke test do pipeline do EAQGA com um dataset dummy.

Os dois algoritmos são reais agora:

- **baseline** (``eaqga.baseline``): fiel ao Algorithm 1, ``RY`` nos controles,
  1 shot por circuito, topologia ``STAR``;
- **ours** (``eaqga.runner``): genoma 1Q (``H^S`` fixo + colunas Hilbert),
  floresta ``PATH`` reconstruída dos pais, 50 shots por circuito, mutação 1Q.

O objetivo é comparar o **encanamento** no mesmo dummy:

    dataset dummy → fitness QUBO → top-2 → floresta → genoma/mutação → métricas

Baseline e híbrido não são apples-to-apples: mudam shots (1 × 50) e topologia
(``STAR`` × ``PATH``). A comparação de mérito vem depois, com dados reais e
mesma configuração.

Uso:
    python tests/test.py              # baseline + híbrido (precisa de Qiskit)
    python tests/test.py --baseline   # só o baseline fiel
    python tests/test.py --ours       # só o híbrido
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
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
    GateSet,
    HybridConfig,
)
from eaqga.instance import QUBOInstance, synthetic_qubo
from eaqga.utils.embellish import sepComment, stylePrint
from eaqga.utils.formats import formatValue

C = "#f5c2e7"
PASS_COLOR = "#3a5716"
FAIL_COLOR = "#ff6b6b"

POPULATION_SIZE = 5
GENERATIONS = 5
P_A = 0.95
P_S = 0.60
SEED = 42

HYBRID_SHOTS = 50
HYBRID_COLUMNS = 3
HYBRID_MUTATION = 0.10

DUMMY_S = 4
DUMMY_SEED = 7


@dataclass(frozen=True, slots=True)
class RunResult:
    name: str
    initial_best: float
    best: float
    best_bits: str
    final_mean: float
    final_std: float
    final_unique: int
    evaluations: int
    elapsed_s: float
    best_generation: int | None = None

    def gap_to(self, optimum: float) -> float:
        return optimum - self.best

    def found_optimum(self, optimum: float, tol: float = 1e-9) -> bool:
        return math.isclose(self.best, optimum, rel_tol=0.0, abs_tol=tol)


def _initial_value(result: object) -> float:
    """Extrai ``history['best'][0]`` de um resultado, com guard de tipo."""

    history = getattr(result, "history")
    value = history["best"][0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "history['best'][0] deveria ser numérico, recebeu "
            f"{type(value).__name__}."
        )
    return float(value)


def _as_run_result(
    name: str,
    result: object,
) -> RunResult:
    """Adapta um resultado do baseline/híbrido à tabela do teste."""

    final_population = getattr(result, "final_population")
    final_values = [individual.fitness for individual in final_population]
    final_unique = len({individual.bits for individual in final_population})

    return RunResult(
        name=name,
        initial_best=_initial_value(result),
        best=float(getattr(result, "best_fitness")),
        best_bits=str(getattr(result, "best_bits")),
        final_mean=statistics.fmean(final_values),
        final_std=(
            statistics.stdev(final_values) if len(final_values) > 1 else 0.0
        ),
        final_unique=final_unique,
        evaluations=int(getattr(result, "evaluations")),
        elapsed_s=float(getattr(result, "elapsed_s")),
        best_generation=int(getattr(result, "best_generation")),
    )


def run_baseline_real(instance: QUBOInstance) -> RunResult:
    """Roda o baseline fiel (Qiskit/Aer) e adapta para a tabela."""

    try:
        from eaqga.baseline import run_baseline as run_baseline_qiskit
    except ImportError as exc:
        print(
            stylePrint(
                "Qiskit/Aer indisponível. Instale requirements.txt.",
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

    return _as_run_result("baseline", run_baseline_qiskit(instance, config))


def run_hybrid_real(instance: QUBOInstance) -> RunResult:
    """Roda o nosso indivíduo no laço do EAQGA e adapta para a tabela."""

    try:
        from eaqga.runner import run_hybrid
    except ImportError as exc:
        print(
            stylePrint(
                "Qiskit/Aer indisponível. Instale requirements.txt.",
                FAIL_COLOR,
                bold=True,
            )
        )
        raise SystemExit(2) from exc

    config = HybridConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        shots=HYBRID_SHOTS,
        num_columns=HYBRID_COLUMNS,
        gate_set=GateSet.HILBERT_1Q,
        forest_shape=ForestShape.PATH,
        max_degree=2,
        mutation_rate=HYBRID_MUTATION,
        p_s=P_S,
        seed=SEED,
        backend=BackendKind.STATEVECTOR,
        evaluation_mode=EvaluationMode.SCALAR,
        verbose=False,
    )

    return _as_run_result("ours", run_hybrid(instance, config))


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
    print(
        f"  {stylePrint('• brute force', C)}: {brute_bits} "
        f"fitness={formatValue(brute_value)}"
    )
    if do_baseline:
        print(
            f"  {stylePrint('• baseline', C)}: EAQGA fiel · 1 shot · "
            f"STAR · p_a={P_A}"
        )
    if do_ours:
        print(
            f"  {stylePrint('• ours', C)}: híbrido · {HYBRID_SHOTS} shots · "
            f"PATH d≤2 · {HYBRID_COLUMNS} colunas Hilbert · "
            f"mut={HYBRID_MUTATION}"
        )
    if do_baseline and do_ours:
        print(
            f"  {stylePrint('• nota', C)}: shots e topologia diferem; "
            "compara o encanamento, não o mérito."
        )
    print()


def _print_results(
    results: list[RunResult],
    instance: QUBOInstance,
    brute_value: float,
) -> None:
    print(sepComment(40, stylePrint("Resultados", C, bold=True)))

    header = (
        f"{'algoritmo':<12} {'best':>12} {'gap':>12} "
        f"{'mean final':>12} {'std':>12} {'únicos':>7} "
        f"{'evals':>6} {'gen':>5} {'ótimo':>6}"
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
            f"{formatValue(result.best):>12} "
            f"{formatValue(result.gap_to(brute_value)):>12} "
            f"{formatValue(result.final_mean):>12} "
            f"{formatValue(result.final_std):>12} "
            f"{result.final_unique:>7} "
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
            f"  {stylePrint('tempo', C)}: baseline "
            f"{baseline.elapsed_s:.2f}s · ours {ours.elapsed_s:.2f}s"
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
        help="roda o baseline fiel (Qiskit) e compara.",
    )
    parser.add_argument(
        "--ours",
        action="store_true",
        help="roda o híbrido (Qiskit) e compara.",
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
        results.append(run_hybrid_real(instance))

    _print_results(results, instance, brute_value)
    return _check_all(results, instance, brute_value)


if __name__ == "__main__":
    raise SystemExit(main())
