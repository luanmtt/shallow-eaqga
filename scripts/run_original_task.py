"""Executa uma única tarefa do original e imprime uma linha CSV em JSON.

Invocado pelo supervisor `resume_with_watchdog.py` em um subprocesso isolado.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--subset", type=int, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--door-type", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--subsets", type=int, default=10)
    parser.add_argument("--as-of", default="2024-09-30")
    args = parser.parse_args()

    symbols = fetch_sp500_tickers(as_of=args.as_of)
    data = build_portfolio(symbols, PAPER_START, PAPER_END, strict=False)
    position = {ticker: index for index, ticker in enumerate(data.tickers)}
    universes = disjoint_universes(
        data.tickers,
        args.size,
        args.subsets,
        random.Random(args.dataset_seed),
    )
    tickers = universes[args.subset]
    indices = [position[ticker] for ticker in tickers]
    instance = QUBOInstance.from_returns(
        data.mu[indices],
        data.sigma[np.ix_(indices, indices)],
        q=RISK_AVERSION,
        name=f"sp500-{args.size}-{args.subset}",
    )

    started = time.perf_counter()
    result = run_original(
        instance,
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
        shots=args.shots,
        num_portas=6,
        door_type=args.door_type,
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

    row = {
        "size": args.size,
        "population": args.population,
        "generations": args.generations,
        "shots": args.shots,
        "initial_h": True,
        "fixed_h": True,
        "adaptive_mutation": False,
        "selection_metric": "best",
        "bias_layer": "none",
        "algorithm": "original",
        "subset": args.subset,
        "run": args.run,
        "seed": args.seed,
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
        "original_door_type": args.door_type,
    }
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
