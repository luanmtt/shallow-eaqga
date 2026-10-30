"""Experimento principal: EAQGA fiel × nosso híbrido, com dados reais.

O protocolo é o do paper: 10 subconjuntos disjuntos de 30 (e 40) ativos do
S&P 500, período 01/10/2023–30/09/2024, `T_max = 20`. Cada unidade de trabalho é
**uma execução** (subset, seed, algoritmo); o paralelismo é por execução, com
`joblib`, não por circuito.

Os dois algoritmos sempre rodam juntos: a comparação baseline × ours é o
produto do script, não um extra. A tabela agregada sai no terminal e, com
`--save-table`, também em CSV.

Por que por execução e não por circuito:

- cada `run()` do Aer tem um custo fixo alto (~100-170 ms em 16 qubits, medido
  na codebase antiga); paralelizar circuito a circuito só multiplica workers
  disputando CPU;
- um worker que roda a execução inteira reaproveita o interpretador e o cache de
  imports, e devolve uma linha de CSV pequena;
- o CSV é escrito **conforme as execuções terminam** (`generator_unordered`),
  então `Ctrl-C` não perde o que já rodou.

Memória: `statevector` é `16 B × 2^S`. Em `S=30` isso é 16 GiB; em `S=40`,
16 TiB. O paper usou MPS e nós também usamos MPS por default. Como cada worker
tem o próprio processo, a memória total é `n_jobs × custo(MPS)` — outra razão
para não exagerar em `--jobs`.

GPU (`statevector_gpu`) fica para depois: os benchmarks antigos deram 1.79× em
16 qubits e 0.98× em 24. Ver `docs/04-aer-memoria-e-gpu.md`.

Uso:
    python experiments/eaqga.py --quick
    python experiments/eaqga.py --sizes 30 --population 20 --runs 100
    python experiments/eaqga.py --sizes 30 40 --population 10 20 --save-table
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import TypedDict, cast

_ROOT = Path(__file__).resolve().parents[1]

# O script se chama ``eaqga.py``; se ``experiments/`` ficar antes da raiz no
# sys.path, ``import eaqga`` acha este arquivo em vez do pacote. Forçamos a raiz
# na frente — sempre, não só quando ausente.
if str(_ROOT) in sys.path:
    sys.path.remove(str(_ROOT))
sys.path.insert(0, str(_ROOT))

import numpy as np
from joblib import Parallel, delayed

from eaqga.baseline import run_baseline
from eaqga.config import (
    BackendKind,
    BaselineConfig,
    ForestShape,
    GateSet,
    HybridConfig,
)
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
from eaqga.runner import run_hybrid
from eaqga.utils.embellish import sepComment, stylePrint
from eaqga.utils.formats import formatValue

C = "#f5c2e7"
RUN_COLUMNS = (
    "size",
    "population",
    "generations",
    "shots",
    "algorithm",
    "subset",
    "run",
    "seed",
    "best_bits",
    "best_fitness",
    "best_generation",
    "evaluations",
    "final_unique",
    "elapsed_s",
)


class RunRow(TypedDict):
    """Linha do CSV de execuções (tipada para o pyright e para o leitor)."""

    size: int
    population: int
    generations: int
    shots: int
    algorithm: str
    subset: int
    run: int
    seed: int
    best_bits: str
    best_fitness: float
    best_generation: int
    evaluations: int
    final_unique: int
    elapsed_s: float


class SummaryRow(TypedDict):
    """Linha da tabela agregada (terminal e CSV)."""

    size: int
    population: int
    algorithm: str
    subsets: int
    runs: int
    fitness_mean: float
    fitness_std: float
    gap_ref: float
    generation_mean: float
    elapsed_mean: float
    delta_pct_vs_baseline: float


SUMMARY_COLUMNS = (
    "size",
    "population",
    "algorithm",
    "subsets",
    "runs",
    "fitness_mean",
    "fitness_std",
    "gap_ref",
    "generation_mean",
    "elapsed_mean",
    "delta_pct_vs_baseline",
)


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Uma execução independente: (tamanho, subset, run, algoritmo)."""

    size: int
    population: int
    generations: int
    shots: int
    algorithm: str
    subset: int
    run: int
    seed: int
    mu: np.ndarray
    sigma: np.ndarray
    progress: bool = False
    reference: float | None = None


def _run_one(spec: RunSpec) -> RunRow:
    """Roda uma execução e devolve uma linha de CSV (sem objetos grandes)."""

    instance = QUBOInstance.from_returns(
        spec.mu,
        spec.sigma,
        q=RISK_AVERSION,
        name=f"sp500-{spec.size}-{spec.subset}",
    )

    start = time.perf_counter()

    if spec.algorithm == "baseline":
        config = BaselineConfig(
            population_size=spec.population,
            generations=spec.generations,
            shots=spec.shots,
            p_a=0.95,
            p_s=0.6,
            forest_shape=ForestShape.STAR,
            max_degree=None,
            paper_x_rule=False,
            seed=spec.seed,
            backend=BackendKind.MPS,
            verbose=False,
        )
        result = run_baseline(
            instance,
            config,
            progress=spec.progress,
            reference=spec.reference,
        )
    else:
        config = HybridConfig(
            population_size=spec.population,
            generations=spec.generations,
            shots=spec.shots,
            num_columns=2,
            gate_set=GateSet.HILBERT_1Q,
            forest_shape=ForestShape.PATH,
            max_degree=2,
            mutation_rate=0.1,
            p_s=0.6,
            seed=spec.seed,
            backend=BackendKind.MPS,
            verbose=False,
        )
        result = run_hybrid(
            instance,
            config,
            progress=spec.progress,
            reference=spec.reference,
        )

    row: RunRow = {
        "size": spec.size,
        "population": spec.population,
        "generations": spec.generations,
        "shots": spec.shots,
        "algorithm": spec.algorithm,
        "subset": spec.subset,
        "run": spec.run,
        "seed": spec.seed,
        "best_bits": result.best_bits,
        "best_fitness": float(result.best_fitness),
        "best_generation": int(result.best_generation),
        "evaluations": int(result.evaluations),
        "final_unique": len({ind.bits for ind in result.final_population}),
        "elapsed_s": time.perf_counter() - start,
    }
    return row


def _statevector_memory(size: int) -> str:
    """Memória de um statevector ``complex128`` (sem overhead do Aer)."""

    total = 16.0 * (2**size)

    for unit, scale in (("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if total >= scale:
            return f"{total / scale:.1f} {unit}"

    return f"{total:.0f} B"


def _local_search_reference(
    instance: QUBOInstance,
    *,
    restarts: int = 8,
    seed: int = 0,
) -> tuple[str, float]:
    """Referência barata: multi-start + 1-flip até não melhorar.

    **Não é ótimo global.** Para `S=30/40` o brute force do paper é inviável
    (`2^40 ≈ 10^12`); esta referência serve só para dar escala ao gap.
    """

    rng = np.random.default_rng(seed)
    matrix = instance.Q
    best_value = 0.0
    best_bits = "0" * instance.S

    for _ in range(restarts):
        x = rng.integers(0, 2, size=instance.S).astype(float)
        current = float(x @ matrix @ x)

        improved = True
        while improved:
            improved = False
            for index in range(instance.S):
                x[index] = 1.0 - x[index]
                candidate = float(x @ matrix @ x)
                if candidate > current + 1e-12:
                    current = candidate
                    improved = True
                else:
                    x[index] = 1.0 - x[index]

        if current > best_value:
            best_value = current
            best_bits = "".join("1" if bit else "0" for bit in x)

    return best_bits, best_value


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Tarefas


def _build_tasks(
    mu: np.ndarray,
    sigma: np.ndarray,
    universes: dict[int, list[tuple[int, ...]]],
    *,
    algorithms: list[str],
    populations: list[int],
    runs: int,
    generations: int,
    shots: dict[str, int],
    seed: int,
    references: dict[int, float] | None = None,
    progress: bool = False,
) -> list[RunSpec]:
    tasks: list[RunSpec] = []

    for size, subsets in universes.items():
        for subset_index, indices in enumerate(subsets):
            subset_mu = mu[list(indices)].copy()
            subset_sigma = sigma[np.ix_(indices, indices)].copy()

            for population in populations:
                for algorithm in algorithms:
                    for run in range(runs):
                        tasks.append(
                            RunSpec(
                                size=size,
                                population=population,
                                generations=generations,
                                shots=shots[algorithm],
                                algorithm=algorithm,
                                subset=subset_index,
                                run=run,
                                seed=seed + 10_000 * subset_index + run,
                                mu=subset_mu,
                                sigma=subset_sigma,
                                progress=progress,
                                reference=(references or {}).get(size),
                            )
                        )

    return tasks


def _run_all(
    tasks: list[RunSpec],
    path: Path,
    *,
    jobs: int,
    backend: str = "loky",
    timings: dict[str, float] | None = None,
) -> list[RunRow]:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[RunRow] = []
    started = time.perf_counter()
    single_process = jobs == 1
    interactive = sys.stdout.isatty() and not (
        tasks and tasks[0].progress
    )
    stride = max(1, len(tasks) // 20)

    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = list(RUN_COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        stream = cast(
            Iterator[RunRow],
            Parallel(
                n_jobs=jobs,
                backend=backend,
                return_as="generator_unordered",
                verbose=0,
            )(delayed(_run_one)(task) for task in tasks),
        )

        for index, row in enumerate(stream, start=1):
            if index == 1 and timings is not None:
                timings["primeiro"] = time.perf_counter() - started
            writer.writerow(row)
            handle.flush()
            rows.append(row)

            if index % stride and index != len(tasks):
                continue

            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            eta = ((len(tasks) - index) / rate if rate else 0.0) / 60
            line = (
                f"  {index}/{len(tasks)} execuções · {rate:.2f}/s · "
                f"ETA {eta:.1f} min"
            )
            print(f"\r{line}" if interactive else line, end="" if interactive else "\n", flush=True)

    print()
    if timings is not None:
        timings["execucao"] = time.perf_counter() - started
    return rows


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Agregação


def _subset_stats(
    rows: list[RunRow],
    size: int,
    population: int,
    algorithm: str,
) -> dict[int, tuple[float, float]]:
    values: dict[int, list[float]] = {}

    for row in rows:
        if (
            int(row["size"]) == size
            and row["population"] == population
            and row["algorithm"] == algorithm
        ):
            values.setdefault(row["subset"], []).append(
                row["best_fitness"]
            )

    return {
        subset: (
            statistics.fmean(items),
            statistics.stdev(items) if len(items) > 1 else 0.0,
        )
        for subset, items in values.items()
    }


def _summary_rows(
    rows: list[RunRow],
    sizes: list[int],
    populations: list[int],
    references: dict[int, float],
) -> list[SummaryRow]:
    """Agrega as execuções no formato das Tabelas I/II.

    A média é a média das médias por subset (o paper reporta assim); o desvio é
    a média dos desvios por subset. ``delta_pct_vs_baseline`` só existe na
    linha do ``ours``.
    """

    summary: list[SummaryRow] = []

    for size in sizes:
        for population in populations:
            means: dict[str, float] = {}
            block: list[SummaryRow] = []

            for algorithm in ("baseline", "ours"):
                stats = _subset_stats(rows, size, population, algorithm)
                if not stats:
                    continue

                average = statistics.fmean(mean for mean, _ in stats.values())
                spread = statistics.fmean(std for _, std in stats.values())
                means[algorithm] = average

                selected = [
                    row
                    for row in rows
                    if row["size"] == size
                    and row["population"] == population
                    and row["algorithm"] == algorithm
                ]

                block.append(
                    SummaryRow(
                        size=size,
                        population=population,
                        algorithm=algorithm,
                        subsets=len(stats),
                        runs=len(selected),
                        fitness_mean=average,
                        fitness_std=spread,
                        gap_ref=references.get(size, float("nan")) - average,
                        generation_mean=statistics.fmean(
                            row["best_generation"] for row in selected
                        ),
                        elapsed_mean=statistics.fmean(
                            row["elapsed_s"] for row in selected
                        ),
                        delta_pct_vs_baseline=float("nan"),
                    )
                )

            if "baseline" in means and "ours" in means and means["baseline"]:
                delta = (
                    (means["ours"] - means["baseline"]) / means["baseline"] * 100.0
                )
                for row in block:
                    if row["algorithm"] == "ours":
                        row["delta_pct_vs_baseline"] = delta

            summary.extend(block)

    return summary


def _print_table(summary: list[SummaryRow]) -> None:
    """Imprime as duas linhas sempre; algoritmos ausentes viram ``—``.

    A comparação é o produto do script, então a tabela não muda de forma
    quando só um algoritmo roda: ela mostra o que faltou e marca o Δ% como
    indisponível.
    """

    header = (
        f"{'algoritmo':<10} {'fitness':>10} {'std':>10} "
        f"{'gap_ref':>10} {'gen méd':>8} {'tempo/run':>10}"
    )
    blank = f"  {'—':>10} {'—':>10} {'—':>10} {'—':>8} {'—':>9}"

    lookup = {
        (row["size"], row["population"], row["algorithm"]): row
        for row in summary
    }
    groups = sorted({(row["size"], row["population"]) for row in summary})

    for size, population in groups:
        runs = sum(
            lookup[(size, population, algorithm)]["runs"]
            for algorithm in ("baseline", "ours")
            if (size, population, algorithm) in lookup
        )

        print()
        print(
            sepComment(
                40,
                stylePrint(
                    f"S={size} · pop={population} · {runs} execuções",
                    C,
                    bold=True,
                ),
            )
        )
        print(f"  {stylePrint(header, C, bold=True)}")

        present: list[str] = []

        for algorithm in ("baseline", "ours"):
            row = lookup.get((size, population, algorithm))
            if row is None:
                print(f"  {algorithm:<10}{blank}")
                continue

            present.append(algorithm)
            print(
                f"  {algorithm:<10} "
                f"{formatValue(row['fitness_mean'] * 100):>10} "
                f"{formatValue(row['fitness_std'] * 100):>10} "
                f"{formatValue(row['gap_ref'] * 100):>10} "
                f"{row['generation_mean']:>8.1f} "
                f"{row['elapsed_mean']:>9.1f}s"
            )

        if len(present) == 2:
            print(
                f"  {stylePrint('Δ% ours vs baseline', C)}: "
                f"{lookup[(size, population, 'ours')]['delta_pct_vs_baseline']:+.2f}%"
            )
        else:
            print(
                f"  {stylePrint('Δ% ours vs baseline', C)}: — "
                "(rode os dois algoritmos)"
            )

    print()
    print(
        f"  {stylePrint('nota', C)}: fitness e std multiplicados por 100, "
        "como nas Tabelas I/II."
    )
    print(
        f"  {stylePrint('nota', C)}: 'gap_ref' usa busca local multi-start; "
        "não é o ótimo global."
    )


def _write_summary_csv(summary: list[SummaryRow], path: Path) -> Path:
    """Grava a tabela agregada (valores crus, sem o ×100 da tela)."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = list(SUMMARY_COLUMNS)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary:
            payload: dict[str, object] = {
                column: row[column] for column in SUMMARY_COLUMNS
            }
            if math.isnan(row["delta_pct_vs_baseline"]):
                payload["delta_pct_vs_baseline"] = ""
            writer.writerow(payload)

    return path


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduz as Tabelas I/II do EAQGA com dados reais.",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[30, 40])
    parser.add_argument("--population", nargs="+", type=int, default=[20])
    parser.add_argument("--subsets", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("baseline", "ours"),
        default=["baseline", "ours"],
        help="quais algoritmos rodam; a tabela sempre compara os dois.",
    )
    parser.add_argument(
        "--universe",
        choices=("sp500", "default"),
        default="sp500",
        help="'sp500' baixa/carrega o índice; 'default' usa 30 large caps.",
    )
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "outputs" / "eaqga",
    )
    parser.add_argument(
        "--backend",
        choices=("loky", "threading"),
        default="loky",
        help="'loky' isola processos; 'threading' evita o import por worker.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="barra por geração (default: ligada só com --jobs 1).",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="imprime o tempo de cada fase (dados, referência, 1º resultado).",
    )
    parser.add_argument(
        "--save-table",
        action="store_true",
        help="grava a tabela agregada em summary-<timestamp>.csv.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: S=10, 2 subsets, 2 runs, 3 gerações, pop 5.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.quick:
        args.sizes = [10]
        args.subsets = 2
        args.runs = 2
        args.generations = 3
        args.population = [5]
        args.universe = "default"

    if args.jobs < 1 and args.jobs != -1:
        raise SystemExit("--jobs deve ser >= 1 ou -1.")

    if args.universe == "sp500":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            symbols = fetch_sp500_tickers(refresh=args.refresh)
    else:
        symbols = DEFAULT_UNIVERSE

    timings: dict[str, float] = {}
    clock = time.perf_counter()

    print(sepComment(40, stylePrint("EAQGA", C, bold=True)))
    print(f"  {stylePrint('• universo', C)}: {len(symbols)} tickers ({args.universe})")
    print(f"  {stylePrint('• período', C)}: {PAPER_START}..{PAPER_END}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = build_portfolio(
            symbols,
            PAPER_START,
            PAPER_END,
            refresh=args.refresh,
            strict=False,
        )

    timings["dados"] = time.perf_counter() - clock
    print(f"  {stylePrint('• dados', C)}: {data.summary()}")

    position = {ticker: index for index, ticker in enumerate(data.tickers)}
    universes: dict[int, list[tuple[int, ...]]] = {}

    for size in args.sizes:
        needed = size * args.subsets
        if needed > data.S:
            print(
                stylePrint(
                    f"  S={size} exige {needed} tickers e só há {data.S}.",
                    "#ff6b6b",
                    bold=True,
                )
            )
            return 2

        universes[size] = [
            tuple(position[ticker] for ticker in tickers)
            for tickers in disjoint_universes(
                data.tickers,
                size,
                args.subsets,
                random.Random(args.seed),
            )
        ]

    print(
        f"  {stylePrint('• memória', C)}: statevector em S="
        f"{max(args.sizes)} exigiria {_statevector_memory(max(args.sizes))}; "
        "usando MPS."
    )

    clock = time.perf_counter()
    references: dict[int, float] = {}

    for size, subsets in universes.items():
        values = []

        for indices in subsets:
            instance = QUBOInstance.from_returns(
                data.mu[list(indices)],
                data.sigma[np.ix_(indices, indices)],
                q=RISK_AVERSION,
                name=f"ref-{size}",
            )
            _, reference = _local_search_reference(instance, seed=args.seed)
            values.append(reference)

        references[size] = statistics.fmean(values)

    timings["referencia"] = time.perf_counter() - clock

    show_progress = args.progress if args.progress is not None else args.jobs == 1
    if show_progress and args.jobs != 1:
        print(
            f"  {stylePrint('• progresso', C)}: desligado (jobs={args.jobs}; "
            "use --jobs 1 para a barra por geração)."
        )
        show_progress = False

    tasks = _build_tasks(
        data.mu,
        data.sigma,
        universes,
        algorithms=args.algorithms,
        populations=args.population,
        runs=args.runs,
        generations=args.generations,
        shots={"baseline": 1, "ours": 50},
        seed=args.seed,
        references=references,
        progress=show_progress,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output / f"runs-{stamp}.csv"

    print(f"  {stylePrint('• tarefas', C)}: {len(tasks)} · jobs={args.jobs} · backend={args.backend}")
    print(f"  {stylePrint('• saída', C)}: {output}")

    if args.jobs == 1:
        print(f"  {stylePrint('• workers', C)}: processo único (sem spawn)")
    else:
        print(
            f"  {stylePrint('• workers', C)}: {args.jobs} processos — o 1º "
            "resultado demora ~1-2 s (import do Qiskit em cada worker)"
        )
    print()

    rows = _run_all(
        tasks,
        output,
        jobs=args.jobs,
        backend=args.backend,
        timings=timings,
    )

    summary = _summary_rows(rows, args.sizes, args.population, references)
    _print_table(summary)

    if args.save_table:
        table_path = _write_summary_csv(
            summary,
            args.output / f"summary-{stamp}.csv",
        )
        print(f"\n  {stylePrint('tabela.csv', C)}: {table_path}")

    if args.timings:
        print()
        print(f"  {stylePrint('• tempos', C)}:")
        print(f"      dados         {timings.get('dados', 0.0):6.2f}s")
        print(f"      referência    {timings.get('referencia', 0.0):6.2f}s")
        print(f"      1º resultado  {timings.get('primeiro', 0.0):6.2f}s")
        print(f"      execução      {timings.get('execucao', 0.0):6.2f}s")
    print(f"\n  {stylePrint('runs.csv', C)}: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
