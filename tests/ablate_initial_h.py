"""Ablação da camada ``H^S`` do híbrido.

Três variantes, mesmo seed e mesmos hiperparâmetros:

- ``H``: comportamento atual (``H^S`` antes do genoma em todas as gerações);
- ``sem H inicial``: remove o ``H^S`` só da população inicial (``forest=None``);
- ``sem H``: remove o ``H^S`` de todas as gerações.

A comparação é pareada por seed: cada variante usa o mesmo ``HybridConfig`` e
o mesmo RNG de genomas/florestas. O que muda é só o circuito medido.

Uso:
    python tests/ablate_initial_h.py
    python tests/ablate_initial_h.py --seeds 10 --shots 1 50 --jobs 8
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from joblib import Parallel, delayed

import eaqga.individuo as individuo_module
from eaqga.config import (
    BackendKind,
    ForestShape,
    GateSet,
    HybridConfig,
)
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

Mode = Literal["current", "no_h_initial", "no_h_all"]
MODES: tuple[Mode, ...] = ("current", "no_h_initial", "no_h_all")
MODE_LABELS = {
    "current": "H^S (atual)",
    "no_h_initial": "sem H inicial",
    "no_h_all": "sem H (todas)",
}


def _load_instance(size: int, subset: int, seed: int) -> QUBOInstance:
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


def _run_one(
    mode: Mode,
    seed: int,
    shots: int,
    instance: QUBOInstance,
    *,
    population: int,
    generations: int,
) -> dict[str, object]:
    original = individuo_module.build_hybrid_circuit

    def wrapper(genome, forest=None, *, measure=True, initial_h=True):
        if mode == "no_h_all":
            initial_h = False
        return original(genome, forest, measure=measure, initial_h=initial_h)

    config = HybridConfig(
        population_size=population,
        generations=generations,
        shots=shots,
        num_columns=2,
        gate_set=GateSet.HILBERT_1Q,
        forest_shape=ForestShape.PATH,
        max_degree=2,
        mutation_rate=0.10,
        p_s=0.6,
        initial_h=(mode != "no_h_initial"),
        seed=seed,
        backend=BackendKind.MPS,
        verbose=False,
    )

    individuo_module.build_hybrid_circuit = wrapper
    try:
        result = run_hybrid(instance, config)
    finally:
        individuo_module.build_hybrid_circuit = original

    return {
        "mode": mode,
        "seed": seed,
        "shots": shots,
        "initial_best": float(result.history["best"][0]),
        "final_best": float(result.best_fitness),
        "final_mean": float(result.history["avg"][-1]),
        "best_generation": int(result.best_generation),
    }


def _summarize(
    rows: list[dict[str, object]],
    shots: int,
    seeds: int,
) -> None:
    print()
    print(
        sepComment(
            40,
            stylePrint(f"shots={shots} · {seeds} seeds pareadas", C, bold=True),
        )
    )
    header = (
        f"{'variante':<16} {'best inicial':>13} {'best final':>12} "
        f"{'Δ vs atual':>12} {'vitórias':>9} {'gen méd':>8}"
    )
    print(f"  {stylePrint(header, C, bold=True)}")

    selected = [row for row in rows if int(row["shots"]) == shots]
    by_mode = {
        mode: {
            int(row["seed"]): row
            for row in selected
            if row["mode"] == mode
        }
        for mode in MODES
    }

    for mode in MODES:
        entries = list(by_mode[mode].values())
        if not entries:
            continue

        best_initial = statistics.fmean(
            float(row["initial_best"]) for row in entries
        ) * 100
        best_final = statistics.fmean(
            float(row["final_best"]) for row in entries
        ) * 100
        generation = statistics.fmean(
            int(row["best_generation"]) for row in entries
        )

        if mode == "current":
            delta_text = "—"
            wins_text = "—"
        else:
            diffs = [
                (
                    float(by_mode[mode][seed]["final_best"])
                    - float(by_mode["current"][seed]["final_best"])
                )
                * 100
                for seed in sorted(by_mode[mode])
            ]
            wins = sum(diff > 0 for diff in diffs)
            delta_text = f"{statistics.fmean(diffs):+.3f}"
            if len(diffs) > 1:
                delta_text += f" ± {statistics.stdev(diffs):.3f}"
            wins_text = f"{wins}/{len(diffs)}"

        print(
            f"  {MODE_LABELS[mode]:<16} "
            f"{best_initial:>13.3f} "
            f"{best_final:>12.3f} "
            f"{delta_text:>12} "
            f"{wins_text:>9} "
            f"{generation:>8.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ablação da camada H^S do híbrido.",
    )
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[1, 50],
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    instance = _load_instance(args.size, args.subset, args.seed)
    seeds = list(range(args.seed, args.seed + args.seeds))

    print(sepComment(40, stylePrint("Ablação H^S", C, bold=True)))
    print(
        f"  S={args.size} · pop={args.population} · T={args.generations} · "
        f"seeds={seeds[0]}..{seeds[-1]} · shots={args.shots}"
    )

    rows = Parallel(n_jobs=args.jobs)(
        delayed(_run_one)(
            mode,
            seed,
            shots,
            instance,
            population=args.population,
            generations=args.generations,
        )
        for shots in args.shots
        for seed in seeds
        for mode in MODES
    )

    for shots in args.shots:
        _summarize(rows, shots, len(seeds))

    print()
    print(
        "  nota: 'sem H inicial' remove H^S só da geração 0; "
        "'sem H (todas)' remove de todas."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
