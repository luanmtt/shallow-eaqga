"""Monta as Tabelas I/II do paper a partir de um summary CSV.

Lê o `summary-*.csv` gerado por `eaqga.py` em `outputs/plannedComparisons/`
e escreve:

- um CSV pivô por tamanho (`paper-table-S30.csv`, `paper-table-S40.csv`);
- um `.tex` com as duas tabelas no formato do paper (fitness ×100).

Uso:
    python visualization/paper_table.py
    python visualization/paper_table.py --experiment E1
    python visualization/paper_table.py --summary outputs/plannedComparisons/E1/summary-*.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eaqga.utils.embellish import sepComment, stylePrint

C = "#f5c2e7"
DEFAULT_INPUT = _ROOT / "outputs" / "plannedComparisons"

ALGORITHMS = ("baseline", "ours", "original")
LABELS = {
    "baseline": "EAQGA (baseline)",
    "ours": "ours (híbrido)",
    "original": "original (QGA)",
}


def _latest_summary(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = sorted(path.rglob("summary-*.csv"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"nenhum summary-*.csv em {path}")
    return candidates[-1]


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _value(row: dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    return float(raw) if raw not in ("", None) else float("nan")


def _cell(rows: list[dict[str, str]], size: int, population: int, algorithm: str) -> tuple[float, float]:
    for row in rows:
        if (
            int(row["size"]) == size
            and int(row["population"]) == population
            and row["algorithm"] == algorithm
        ):
            return _value(row, "fitness_mean") * 100.0, _value(row, "fitness_std") * 100.0
    return float("nan"), float("nan")


def _reference(rows: list[dict[str, str]], size: int) -> float:
    for row in rows:
        if int(row["size"]) == size and row.get("reference", "") not in ("", None):
            return _value(row, "reference") * 100.0
    return float("nan")


def _best(
    rows: list[dict[str, str]],
    size: int,
    population: int,
) -> float:
    values = [
        _cell(rows, size, population, algorithm)[0]
        for algorithm in ALGORITHMS
    ]
    values = [value for value in values if value == value]
    return max(values) if values else float("nan")


def _fmt(value: float) -> str:
    return "—" if value != value else f"{value:.4f}"


def _write_pivot(
    rows: list[dict[str, str]],
    size: int,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "algorithm",
                "population",
                "fitness_mean_x100",
                "fitness_std_x100",
                "reference_x100",
            ]
        )
        reference = _reference(rows, size)
        for algorithm in ALGORITHMS:
            for population in (10, 20):
                mean, std = _cell(rows, size, population, algorithm)
                if mean != mean:
                    continue
                writer.writerow([algorithm, population, mean, std, reference])


def _latex_table(
    rows: list[dict[str, str]],
    size: int,
) -> str:
    reference = _reference(rows, size)
    best10 = _best(rows, size, 10)
    best20 = _best(rows, size, 20)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Fitness value of the three algorithms with problem size {size} "
        r"and population sizes 10 and 20 over 20 iterations. "
        r"Values multiplied by 100.}",
        rf"\label{{tab:fitness-{size}}}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Population = 10} & \multicolumn{2}{c}{Population = 20} & Ref.$^\dagger$ \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Algorithm & Avg & Std & Avg & Std & \\",
        r"\midrule",
    ]

    for algorithm in ALGORITHMS:
        mean10, std10 = _cell(rows, size, 10, algorithm)
        mean20, std20 = _cell(rows, size, 20, algorithm)
        if mean10 != mean10 and mean20 != mean20:
            continue

        mean10_text = _fmt(mean10)
        mean20_text = _fmt(mean20)
        if mean10 == best10:
            mean10_text = rf"\textbf{{{mean10_text}}}"
        if mean20 == best20:
            mean20_text = rf"\textbf{{{mean20_text}}}"

        lines.append(
            f"{LABELS[algorithm]} & {mean10_text} & {_fmt(std10)} & "
            f"{mean20_text} & {_fmt(std20)} & {_fmt(reference)} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\footnotesize $^\dagger$Ref. = melhor valor da busca local "
            r"multi-start no tamanho; não é o ótimo global exato.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera Tabelas I/II (LaTeX + CSV pivô) do protocolo do paper.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="diretório com summary-*.csv (default: outputs/plannedComparisons).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="subdiretório do experimento, ex.: E1.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        nargs="+",
        default=None,
        help="um ou mais summary-*.csv explícitos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="diretório de saída (default: <input-dir>/latex).",
    )
    args = parser.parse_args()

    base = args.input_dir / args.experiment if args.experiment else args.input_dir
    summaries = args.summary if args.summary else [_latest_summary(base)]
    rows: list[dict[str, str]] = []
    for summary_path in summaries:
        rows.extend(_load(summary_path))
    if not rows:
        raise SystemExit("nenhum summary com linhas.")

    sizes = sorted({int(row["size"]) for row in rows})
    output_dir = args.output_dir or (base / "latex")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(sepComment(40, stylePrint("Tabelas do paper", C, bold=True)))
    for summary_path in summaries:
        print(f"  summary: {summary_path}")
    print(f"  sizes:   {sizes}")

    tables: list[str] = []
    for size in sizes:
        pivot = output_dir / f"paper-table-S{size}.csv"
        _write_pivot(rows, size, pivot)
        tables.append(_latex_table(rows, size))
        print(f"  S={size}: {pivot}")

    tex_path = output_dir / "paper-tables.tex"
    tex_path.write_text(
        "% requer \\usepackage{booktabs}\n\n" + "\n\n".join(tables) + "\n",
        encoding="utf-8",
    )

    print(f"  tex:     {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
