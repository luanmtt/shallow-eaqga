"""Completa runs faltantes do original S40 Hilbert e gera o summary.

Lê o runs-*.csv existente, descobre as combinações (population, subset, run)
faltantes, roda só elas e acrescenta ao CSV. Depois recalcula o summary no
mesmo formato do eaqga.py.

Uso:
    python scripts/resume_s40_hilbert.py --dry-run
    python scripts/resume_s40_hilbert.py --jobs 3
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from joblib import Parallel, delayed

from eaqga.instance import QUBOInstance
from eaqga.original_adapter import run_original
from eaqga.portfolio import (
    PAPER_END,
    PAPER_START,
    RISK_AVERSION,
    build_portfolio,
    disjoint_universes,
    fetch_sp500_tickers,
)

SIZE = 40
GENERATIONS = 20
SHOTS = 20
RUNS = 5
SUBSETS = 10
SEED = 0


@dataclass(frozen=True, slots=True)
class Task:
    population: int
    subset: int
    run: int
    seed: int
    mu: np.ndarray
    sigma: np.ndarray


def _load_instances() -> list[tuple[int, np.ndarray, np.ndarray]]:
    symbols = fetch_sp500_tickers(as_of="2024-09-30")
    data = build_portfolio(symbols, PAPER_START, PAPER_END, strict=False)
    position = {ticker: index for index, ticker in enumerate(data.tickers)}
    universes = disjoint_universes(data.tickers, SIZE, SUBSETS, random.Random(SEED))
    out = []
    for subset, tickers in enumerate(universes):
        indices = [position[ticker] for ticker in tickers]
        out.append((subset, data.mu[indices].copy(), data.sigma[np.ix_(indices, indices)].copy()))
    return out


def _run(task: Task) -> dict[str, object]:
    instance = QUBOInstance.from_returns(
        task.mu,
        task.sigma,
        q=RISK_AVERSION,
        name=f"sp500-{SIZE}-{task.subset}",
    )
    started = time.perf_counter()
    result = run_original(
        instance,
        population_size=task.population,
        generations=GENERATIONS,
        seed=task.seed,
        shots=SHOTS,
        num_portas=6,
        door_type=2,
        elite_size=0.2,
        crossover_rate=0.7,
        mutation_rate=0.3,
        selection_method="tournament",
        tournament_size=3,
        crossover_selection="vertical",
        sigma_es=False,
        annealing_window=False,
    )
    elapsed = time.perf_counter() - started
    return {
        "size": SIZE,
        "population": task.population,
        "generations": GENERATIONS,
        "shots": SHOTS,
        "initial_h": True,
        "fixed_h": True,
        "adaptive_mutation": False,
        "selection_metric": "best",
        "bias_layer": "none",
        "algorithm": "original",
        "subset": task.subset,
        "run": task.run,
        "seed": task.seed,
        "best_bits": result.best_bits,
        "best_fitness": float(result.best_fitness_bits),
        "best_generation": int(result.best_generation),
        "evaluations": int(result.evaluations),
        "final_unique": len({member.bits for member in result.final_population}),
        "elapsed_s": elapsed,
        "best_fitness_bits": float(result.best_fitness_bits),
        "best_fitness_internal": float(result.best_fitness),
        "original_crossover_selection": "vertical",
        "original_selection": "tournament",
        "original_tournament": 3,
        "original_sigma_es": False,
        "original_annealing": False,
        "original_portas": 6,
        "original_door_type": 2,
    }


def _local_reference(instance: QUBOInstance, restarts: int = 8, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    matrix = instance.Q
    best = 0.0
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
        best = max(best, current)
    return best


def _summary(instances: list[tuple[int, np.ndarray, np.ndarray]], rows: list[dict[str, str]], output: Path) -> None:
    refs = []
    for subset, mu, sigma in instances:
        instance = QUBOInstance.from_returns(mu, sigma, q=RISK_AVERSION, name=f"ref-{subset}")
        refs.append(_local_reference(instance, seed=SEED + subset))
    reference = statistics.fmean(refs)

    out_rows = []
    for population in (10, 20):
        by_subset: dict[int, list[float]] = {}
        selected = []
        for row in rows:
            if row["algorithm"] == "original" and int(row["population"]) == population:
                by_subset.setdefault(int(row["subset"]), []).append(float(row["best_fitness_bits"]))
                selected.append(row)
        means = [statistics.fmean(values) for values in by_subset.values()]
        stds = [statistics.stdev(values) if len(values) > 1 else 0.0 for values in by_subset.values()]
        out_rows.append(
            {
                "size": SIZE,
                "population": population,
                "algorithm": "original",
                "subsets": len(by_subset),
                "runs": len(selected),
                "fitness_mean": statistics.fmean(means),
                "fitness_std": statistics.fmean(stds),
                "reference": reference,
                "gap_ref": reference - statistics.fmean(means),
                "generation_mean": statistics.fmean(float(row["best_generation"]) for row in selected),
                "elapsed_mean": statistics.fmean(float(row["elapsed_s"]) for row in selected),
                "delta_pct_vs_baseline": "",
            }
        )

    path = output / f"summary-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"summary: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Completa o Hilbert S40 faltante.")
    parser.add_argument("--runs-csv", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    output = _ROOT / "outputs" / "plannedComparisons" / "E1-S40" / "original-hilbert"
    runs_csv = args.runs_csv or sorted(output.glob("runs-*.csv"))[-1]
    with runs_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    have = {(row["population"], row["subset"], row["run"]) for row in rows}
    instances = _load_instances()
    tasks = []
    for population in (10, 20):
        for subset, mu, sigma in instances:
            for run in range(RUNS):
                key = (str(population), str(subset), str(run))
                if key not in have:
                    tasks.append(
                        Task(
                            population=population,
                            subset=subset,
                            run=run,
                            seed=SEED + 10_000 * subset + run,
                            mu=mu,
                            sigma=sigma,
                        )
                    )

    print(f"runs_csv: {runs_csv}")
    if args.summary_only:
        instances = _load_instances()
        _summary(instances, rows, output)
        return 0
    print(f"faltando: {len(tasks)}")
    for task in tasks:
        print(f"  N={task.population} subset={task.subset} run={task.run} seed={task.seed}")
    if args.dry_run or not tasks:
        return 0

    new_rows = Parallel(n_jobs=args.jobs)(delayed(_run)(task) for task in tasks)
    with runs_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerows(new_rows)

    rows.extend({key: str(value) for key, value in row.items()} for row in new_rows)
    _summary(instances, rows, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
