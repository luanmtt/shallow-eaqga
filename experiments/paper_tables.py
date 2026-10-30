"""Reproduz as Tabelas I/II do paper (EAQGA × nosso híbrido).

O protocolo é o do paper: 10 subconjuntos disjuntos de 30 (e 40) ativos do
S&P 500, período 01/10/2023–30/09/2024, `T_max = 20`. Cada unidade de trabalho é
**uma execução** (subset, seed, algoritmo); o paralelismo é por execução, com
`joblib`, não por circuito.

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
    python experiments/paper_tables.py --quick
    python experiments/paper_tables.py --sizes 30 --population 20 --runs 100
    python experiments/paper_tables.py --sizes 30 40 --population 10 20
"""

from __future__ import annotations

import argparse
import csv
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
if str(_ROOT) not in sys.path:
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
        result = run_baseline(instance, config)
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
        result = run_hybrid(instance, config)

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
                            )
                        )

    return tasks


def _run_all(
    tasks: list[RunSpec],
    path: Path,
    *,
    jobs: int,
) -> list[RunRow]:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[RunRow] = []
    started = time.perf_counter()

    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = list(RUN_COLUMNS)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        stream = cast(
            Iterator[RunRow],
            Parallel(
                n_jobs=jobs,
                return_as="generator_unordered",
                verbose=0,
            )(delayed(_run_one)(task) for task in tasks),
        )

        for index, row in enumerate(stream, start=1):
            writer.writerow(row)
            handle.flush()
            rows.append(row)

            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            print(
                f"\r  {index}/{len(tasks)} execuções · {rate:.2f}/s · "
                f"ETA {((len(tasks) - index) / rate if rate else 0) / 60:.1f} min",
                end="",
                flush=True,
            )

    print()
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


def _print_table(
    rows: list[RunRow],
    sizes: list[int],
    populations: list[int],
    references: dict[int, float],
) -> None:
    for size in sizes:
        for population in populations:
            print()
            print(
                sepComment(
                    40,
                    stylePrint(
                        f"S={size} · pop={population} · {len(rows)} execuções",
                        C,
                        bold=True,
                    ),
                )
            )

            header = (
                f"{'algoritmo':<10} {'fitness':>10} {'std':>10} "
                f"{'gap_ref':>10} {'gen méd':>8} {'tempo/run':>10}"
            )
            print(f"  {stylePrint(header, C, bold=True)}")

            means: dict[str, float] = {}

            for algorithm in ("baseline", "ours"):
                stats = _subset_stats(rows, size, population, algorithm)
                if not stats:
                    continue

                subset_means = [mean for mean, _ in stats.values()]
                subset_stds = [std for _, std in stats.values()]
                average = statistics.fmean(subset_means)
                spread = statistics.fmean(subset_stds)
                means[algorithm] = average

                reference = references.get(size, float("nan"))
                gap = reference - average

                generation = statistics.fmean(
                    row["best_generation"]
                    for row in rows
                    if int(row["size"]) == size
                    and row["population"] == population
                    and row["algorithm"] == algorithm
                )
                seconds = statistics.fmean(
                    row["elapsed_s"]
                    for row in rows
                    if int(row["size"]) == size
                    and row["population"] == population
                    and row["algorithm"] == algorithm
                )

                print(
                    f"  {algorithm:<10} "
                    f"{formatValue(average * 100):>10} "
                    f"{formatValue(spread * 100):>10} "
                    f"{formatValue(gap * 100):>10} "
                    f"{generation:>8.1f} "
                    f"{seconds:>9.1f}s"
                )

            if "baseline" in means and "ours" in means and means["baseline"]:
                delta = (means["ours"] - means["baseline"]) / means["baseline"] * 100
                print(
                    f"  {stylePrint('Δ% ours vs baseline', C)}: "
                    f"{delta:+.2f}%"
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
        default=_ROOT / "outputs" / "paper_tables",
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

    print(sepComment(40, stylePrint("paper_tables", C, bold=True)))
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
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output / f"runs-{stamp}.csv"

    print(f"  {stylePrint('• tarefas', C)}: {len(tasks)} · jobs={args.jobs}")
    print(f"  {stylePrint('• saída', C)}: {output}")
    print()

    rows = _run_all(tasks, output, jobs=args.jobs)

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

    _print_table(rows, args.sizes, args.population, references)
    print(f"\n  {stylePrint('runs.csv', C)}: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
