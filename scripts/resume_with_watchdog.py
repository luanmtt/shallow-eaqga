"""Supervisor de runs faltantes: executa, mata timeout e tenta de novo.

Cada run roda em um subprocesso isolado (`run_original_task.py`). Se passar do
timeout, o grupo do processo é morto. O resultado é acrescentado ao CSV
imediatamente. No fim, só restam no CSV as runs que concluíram.

Uso:
    python scripts/resume_with_watchdog.py \
      --runs-csv outputs/plannedComparisons/E1-S40/original-hilbert/runs-XXXX.csv \
      --size 40 --populations 10 20 --subsets 10 --runs 5 \
      --generations 20 --shots 20 --door-type 2 \
      --timeout 900 --max-iterations 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
WORKER = _ROOT / "scripts" / "run_original_task.py"


def _tasks(populations: list[int], subsets: int, runs: int) -> list[tuple[int, int, int]]:
    return [
        (population, subset, run)
        for population in populations
        for subset in range(subsets)
        for run in range(runs)
    ]


def _run_one(
    task: tuple[int, int, int],
    *,
    args: argparse.Namespace,
    attempt: int,
) -> dict[str, object] | None:
    population, subset, run = task
    seed = (
        args.dataset_seed
        + args.seed_offset
        + 10_000 * subset
        + run
        + (attempt - 1) * 1_000_000
    )
    cmd = [
        sys.executable,
        str(WORKER),
        "--size",
        str(args.size),
        "--population",
        str(population),
        "--subset",
        str(subset),
        "--run",
        str(run),
        "--generations",
        str(args.generations),
        "--shots",
        str(args.shots),
        "--door-type",
        str(args.door_type),
        "--seed",
        str(seed),
        "--dataset-seed",
        str(args.dataset_seed),
        "--subsets",
        str(args.subsets),
    ]

    print(
        f"[watchdog] start N={population} subset={subset} run={run} "
        f"attempt={attempt} seed={seed}",
        flush=True,
    )
    started = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.communicate()
        print(
            f"[watchdog] TIMEOUT {elapsed:.0f}s N={population} "
            f"subset={subset} run={run} — morto",
            flush=True,
        )
        return None

    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        print(
            f"[watchdog] ERRO N={population} subset={subset} run={run} "
            f"code={process.returncode}\n{stderr[-2000:]}",
            flush=True,
        )
        return None

    line = next((item for item in reversed(stdout.splitlines()) if item.strip()), "")
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        print(
            f"[watchdog] JSON inválido N={population} subset={subset} "
            f"run={run}\n{stdout[-1000:]}\n{stderr[-1000:]}",
            flush=True,
        )
        return None

    print(
        f"[watchdog] OK {elapsed:.0f}s N={population} subset={subset} "
        f"run={run} fitness={float(row['best_fitness']) * 100:.4f}",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--populations", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--subsets", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--door-type", type=int, default=2)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=5.0)
    args = parser.parse_args()

    with args.runs_csv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise SystemExit("CSV sem header.")

    have = {(row["population"], row["subset"], row["run"]) for row in rows}
    missing = [
        task
        for task in _tasks(args.populations, args.subsets, args.runs)
        if (str(task[0]), str(task[1]), str(task[2])) not in have
    ]

    print(
        f"[watchdog] CSV: {args.runs_csv}\n"
        f"[watchdog] faltando: {len(missing)} · timeout={args.timeout}s · "
        f"iterações={args.max_iterations}",
        flush=True,
    )

    attempts: dict[tuple[int, int, int], int] = {task: 0 for task in missing}
    abandoned: set[tuple[int, int, int]] = set()

    for iteration in range(1, args.max_iterations + 1):
        if not missing:
            break
        print(f"\n[watchdog] === iteração {iteration} ({len(missing)} restantes) ===", flush=True)
        for task in list(missing):
            if task in abandoned:
                continue
            if attempts[task] >= args.max_attempts:
                abandoned.add(task)
                print(
                    f"[watchdog] ABANDONADO após {attempts[task]} falhas: "
                    f"N={task[0]} subset={task[1]} run={task[2]}",
                    flush=True,
                )
                continue

            row = _run_one(task, args=args, attempt=attempts[task] + 1)
            if row is None:
                attempts[task] += 1
                if attempts[task] >= args.max_attempts:
                    abandoned.add(task)
                    print(
                        f"[watchdog] ABANDONADO após {attempts[task]} falhas: "
                        f"N={task[0]} subset={task[1]} run={task[2]}",
                        flush=True,
                    )
                continue

            with args.runs_csv.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            missing.remove(task)
            print(f"[watchdog] gravado; restantes={len(missing)}", flush=True)

        if missing and iteration < args.max_iterations:
            time.sleep(args.sleep)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if missing:
        path = args.runs_csv.parent / f"missing-{stamp}.txt"
        path.write_text(
            "\n".join(f"N={p} subset={s} run={r}" for p, s, r in missing) + "\n",
            encoding="utf-8",
        )
        print(f"[watchdog] ainda faltam {len(missing)}; lista: {path}", flush=True)
        return 1

    print("[watchdog] todas as runs faltantes concluídas.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
