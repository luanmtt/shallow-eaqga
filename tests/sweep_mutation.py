"""Progressão de ``p_mut`` para ours e original no modo ``--quick``.

Para cada valor de ``p_mut`` e cada seed, roda:

    python eaqga.py --quick --mutation-rate P --original-mutation P --seed S

O baseline não usa mutação, então é o mesmo em todos os ``P``; a comparação é
pareada por seed. O resultado é uma tabela e um gráfico de
``Δ fitness vs baseline`` por ``p_mut``.

Uso:
    python tests/sweep_mutation.py
    python tests/sweep_mutation.py --p-values 0.10 0.15 0.20 0.25 --seeds 5
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PYTHON = _ROOT / ".venv" / "bin" / "python"
OUTPUT = _ROOT / "empirical" / "mutation-sweep"
C = "#f5c2e7"


def _run_quick(
    p_mut: float,
    seed: int,
    output: Path,
) -> dict[str, dict[str, str]]:
    cmd = [
        str(PYTHON),
        str(_ROOT / "eaqga.py"),
        "--quick",
        "--mutation-rate",
        f"{p_mut}",
        "--original-mutation",
        f"{p_mut}",
        "--seed",
        str(seed),
        "--jobs",
        "1",
        "--save-table",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        cmd,
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"falhou p_mut={p_mut} seed={seed}:\n{result.stderr}"
        )

    summaries = sorted(output.glob("summary-*.csv"))
    if not summaries:
        raise RuntimeError(
            f"sem summary em {output} (p_mut={p_mut}, seed={seed})."
        )

    with summaries[-1].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    return {row["algorithm"]: row for row in rows}


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _write_csv(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    p_values: list[float],
    ours_delta: list[float],
    ours_std: list[float],
    original_delta: list[float],
    original_std: list[float],
    path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8.5, 4.8),
        constrained_layout=True,
    )
    axis.errorbar(
        p_values,
        ours_delta,
        yerr=ours_std,
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#4C78A8",
        label="ours",
    )
    axis.errorbar(
        p_values,
        original_delta,
        yerr=original_std,
        marker="s",
        linewidth=1.8,
        capsize=3,
        color="#F58518",
        label="original",
    )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel("p_mut")
    axis.set_ylabel("Δ% fitness vs baseline")
    axis.set_xticks(p_values)
    axis.grid(alpha=0.3, linewidth=0.6)
    axis.legend()
    axis.set_title(
        "Quick · progressão de p_mut · baseline = 0",
        loc="left",
        fontsize=10,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Progressão de p_mut no modo --quick.",
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25],
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seeds = list(range(args.seed, args.seed + args.seeds))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = OUTPUT / f"sweep-{stamp}"

    print(f"p_mut: {args.p_values}")
    print(f"seeds: {seeds} · rápido (S=10, 2 subsets, 2 runs, T=3, pop=5)")

    per_point: dict[float, dict[str, list[float]]] = {}

    started = time.perf_counter()
    for p_mut in args.p_values:
        ours_values: list[float] = []
        original_values: list[float] = []
        baseline_values: list[float] = []

        for seed in seeds:
            output = root / f"p{p_mut:.2f}" / f"seed{seed}"
            rows = _run_quick(p_mut, seed, output)
            baseline = float(rows["baseline"]["fitness_mean"])
            ours = float(rows["ours"]["fitness_mean"])
            original = float(rows["original"]["fitness_mean"])
            baseline_values.append(baseline)
            ours_values.append(ours)
            original_values.append(original)
            print(
                f"  p={p_mut:.2f} seed={seed} "
                f"base={baseline * 100:.3f} "
                f"ours={ours * 100:.3f} "
                f"orig={original * 100:.3f}",
                flush=True,
            )

        per_point[p_mut] = {
            "baseline": baseline_values,
            "ours": ours_values,
            "original": original_values,
        }

    elapsed = time.perf_counter() - started

    summary_rows: list[dict[str, object]] = []
    ours_delta_pct: list[float] = []
    ours_delta_pct_std: list[float] = []
    original_delta_pct: list[float] = []
    original_delta_pct_std: list[float] = []

    print()
    header = (
        f"{'p_mut':>6} {'base':>8} "
        f"{'ours':>8} {'Δ%ours':>8} {'±':>7} {'wins':>6} "
        f"{'orig':>8} {'Δ%orig':>8} {'±':>7} {'wins':>6}"
    )
    print(header)
    print("-" * len(header))

    for p_mut in args.p_values:
        point = per_point[p_mut]
        baseline = point["baseline"]
        ours = point["ours"]
        original = point["original"]

        ours_diffs = [
            (value - base) * 100.0
            for value, base in zip(ours, baseline)
        ]
        original_diffs = [
            (value - base) * 100.0
            for value, base in zip(original, baseline)
        ]
        ours_pct = [
            (value - base) / base * 100.0
            for value, base in zip(ours, baseline)
        ]
        original_pct = [
            (value - base) / base * 100.0
            for value, base in zip(original, baseline)
        ]

        ours_mean = _mean(ours) * 100.0
        original_mean = _mean(original) * 100.0
        ours_delta_mean = _mean(ours_diffs)
        original_delta_mean = _mean(original_diffs)
        ours_delta_std = _stdev(ours_diffs)
        original_delta_std = _stdev(original_diffs)
        ours_pct_mean = _mean(ours_pct)
        original_pct_mean = _mean(original_pct)
        ours_pct_std = _stdev(ours_pct)
        original_pct_std = _stdev(original_pct)
        ours_wins = sum(value > 0 for value in ours_pct)
        original_wins = sum(value > 0 for value in original_pct)

        ours_delta_pct.append(ours_pct_mean)
        ours_delta_pct_std.append(ours_pct_std)
        original_delta_pct.append(original_pct_mean)
        original_delta_pct_std.append(original_pct_std)

        summary_rows.append(
            {
                "p_mut": p_mut,
                "baseline_mean": _mean(baseline) * 100.0,
                "ours_mean": ours_mean,
                "ours_delta": ours_delta_mean,
                "ours_delta_std": ours_delta_std,
                "ours_delta_pct": ours_pct_mean,
                "ours_delta_pct_std": ours_pct_std,
                "ours_wins": ours_wins,
                "original_mean": original_mean,
                "original_delta": original_delta_mean,
                "original_delta_std": original_delta_std,
                "original_delta_pct": original_pct_mean,
                "original_delta_pct_std": original_pct_std,
                "original_wins": original_wins,
                "seeds": len(seeds),
            }
        )

        print(
            f"{p_mut:>6.2f} "
            f"{_mean(baseline) * 100:>8.3f} "
            f"{ours_mean:>8.3f} "
            f"{ours_pct_mean:>+8.2f} {ours_pct_std:>7.2f} "
            f"{ours_wins:>3}/{len(seeds):<2} "
            f"{original_mean:>8.3f} "
            f"{original_pct_mean:>+8.2f} {original_pct_std:>7.2f} "
            f"{original_wins:>3}/{len(seeds):<2}"
        )

    print()
    print(f"tempo: {elapsed:.1f}s")

    csv_path = root / "summary.csv"
    png_path = root / "mutation-sweep.png"
    _write_csv(summary_rows, csv_path)
    _plot(
        args.p_values,
        ours_delta_pct,
        ours_delta_pct_std,
        original_delta_pct,
        original_delta_pct_std,
        png_path,
    )

    print(f"csv:   {csv_path}")
    print(f"plot:  {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
