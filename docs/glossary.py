"""Glossário dos endpoints executáveis do repositório.

Um **endpoint** é um arquivo que se roda direto (``python arquivo.py``) e faz
alguma coisa: teste, checagem ou experimento. Módulos de biblioteca
(``eaqga/runner.py``, ``eaqga/portfolio.py``, ...) não são endpoints: eles são
importados por alguém.

O glossário é estático, mas ``--check`` compara as flags listadas com as
declaradas no código — se alguém adicionar ``--nova-flag`` e esquecer daqui, o
check acusa.

Uso:
    python docs/glossary.py                 # tudo
    python docs/glossary.py eaqga           # só endpoints que contêm "eaqga"
    python docs/glossary.py --check         # valida flags contra o código
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) in sys.path:
    sys.path.remove(str(_ROOT))
sys.path.insert(0, str(_ROOT))

from eaqga.utils.embellish import stylePrint

__all__ = ["ENDPOINTS", "Endpoint", "Flag", "main"]

TITLE = "#f5c2e7"
COMMAND = "#f9e2af"
PATH_COLOR = "#89b4fa"
FLAG_COLOR = "#a6e3a1"
MUTED = "#6c7086"
OK_COLOR = "#a6e3a1"
FAIL_COLOR = "#ff6b6b"

_RULE_WIDTH = 78


def _rule(title: str, color: str = TITLE) -> str:
    """Separador de largura fixa, independente do tamanho do título."""

    visible = f" {title} "
    free = max(0, _RULE_WIDTH - len(visible))
    left = free // 2
    right = free - left
    return f"{'─' * left}{stylePrint(visible, color, bold=True)}{'─' * right}"


@dataclass(frozen=True, slots=True)
class Flag:
    """Uma flag e o que ela faz (inclui o metavalor, ex.: ``--sizes N...``)."""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Um arquivo executável, com exemplo de uso e flags."""

    path: str
    summary: str
    command: str
    flags: tuple[Flag, ...] = ()
    note: str = ""


ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint(
        path="tests/check_integrity.py",
        summary="Checagens leves: guards, QUBO, fitness, crossover e baseline.",
        command=".venv/bin/python tests/check_integrity.py",
        flags=(
            Flag("--verbose", "traceback completo quando uma checagem falha."),
        ),
        note="roda sem Qiskit (a checagem de circuito vira 'skipped').",
    ),
    Endpoint(
        path="tests/test.py",
        summary="Smoke test: EAQGA fiel × híbrido no dataset dummy.",
        command=".venv/bin/python tests/test.py",
        flags=(
            Flag("--baseline", "roda só o EAQGA fiel (Qiskit)."),
            Flag("--ours", "roda só o híbrido (Qiskit)."),
        ),
        note="sem flag roda os dois; exige Qiskit/Aer.",
    ),
    Endpoint(
        path="eaqga.py",
        summary="Experimento principal: baseline × ours × original, com CSV.",
        command=".venv/bin/python eaqga.py --quick",
        flags=(
            Flag("--preset {default,paper}", "paper = protocolo do artigo; default = comparativo usual."),
            Flag("--quick", "smoke: S=10, 2 subsets, 2 runs, 3 gerações, pop 5."),
            Flag("--sizes N [N ...]", "tamanhos (paper: 30 40)."),
            Flag("--population N [N ...]", "populações (paper: 10 20; default: 20)."),
            Flag("--subsets N", "subconjuntos disjuntos por tamanho (10)."),
            Flag("--runs N", "execuções por subset (paper: 100)."),
            Flag("--generations N", "T_max, gerações (paper: 20)."),
            Flag("--algorithms {baseline,ours,original}", "quais rodam (default: os três)."),
            Flag("--original-runs N", "runs do original (paper: 100; default: 5)."),
            Flag("--universe {sp500,b3,default}", "S&P 500, B3 draft ou 30 large caps."),
            Flag("--as-of DATE", "constituintes do S&P na data (paper: 2024-09-30)."),
            Flag("--jobs N", "workers do joblib (default: min(8, cpus))."),
            Flag("--backend {loky,threading}", "loky isola processos; threading é ~4× pior."),
            Flag("--progress / --no-progress", "barra por geração (auto: só com --jobs 1)."),
            Flag("--timings", "tempo de cada fase até o 1º resultado."),
            Flag("--save-table", "grava summary-<timestamp>.csv."),
            Flag("--refresh", "ignora o cache de preços e baixa de novo."),
            Flag("--seed N", "semente dos subsets e das runs (default: 0)."),
            Flag("--output PATH", "diretório de saída (default: outputs/eaqga)."),
        ),
        note=(
            "paper: baseline com 1 shot (como no artigo), ours/original com 50; "
            "p_a=0.95, p_s=0.6, T=20, 10 subsets, 100 runs, S&P 500 de "
            "2024-09-30."
        ),
    ),
    Endpoint(
        path="docs/glossary.py",
        summary="Este arquivo: endpoints, flags e exemplos.",
        command="python docs/glossary.py",
        flags=(
            Flag("[filtro]", "mostra só endpoints que contêm o texto."),
            Flag("--check", "compara as flags listadas com o código."),
        ),
    ),
)

LIBRARY = (
    ("eaqga/portfolio.py", "preços, μ/Σ, QUBO; cache em data/prices/"),
    ("eaqga/b3.py", "draft B3/Ibovespa via yfinance (.SA)"),
    ("eaqga/instance.py", "QUBOInstance, fitness, brute force"),
    ("eaqga/individuo.py", "genoma 1Q, circuito, medição"),
    ("eaqga/population.py", "ranking, top-2, crossover/mutação, métricas"),
    ("eaqga/runner.py", "laço run_hybrid() — não é executável"),
    ("eaqga/baseline.py", "EAQGA fiel, run_baseline()"),
    ("eaqga/crossover.py", "pares P/N, floresta, agendamento"),
    ("eaqga/config.py", "HybridConfig / BaselineConfig"),
    ("eaqga/original_adapter.py", "roda o motor de original/ no QUBO"),
    ("original/src/", "QGA antigo vendorizado, sem edição"),
)

_SETUP = (
    "uv venv .venv",
    "uv pip install --python .venv/bin/python -r requirements.txt",
)


def _wrap(text: str, width: int) -> list[str]:
    """Quebra um texto em linhas de até ``width`` caracteres."""

    words = text.split()
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def _render(endpoint: Endpoint) -> None:
    print(_rule(endpoint.path, PATH_COLOR))
    for line in _wrap(endpoint.summary, 70):
        print(f"  {line}")
    print(f"  {stylePrint('$', MUTED)} {stylePrint(endpoint.command, COMMAND)}")

    if endpoint.flags:
        width = max(len(flag.name) for flag in endpoint.flags) + 2
        print()
        print(f"  {stylePrint('FLAG'.ljust(width), TITLE, bold=True)}DESCRIÇÃO")

        for flag in endpoint.flags:
            padding = " " * max(0, width - len(flag.name))
            print(
                f"  {stylePrint(flag.name, FLAG_COLOR)}{padding}"
                f"{flag.description}"
            )

    if endpoint.note:
        print()
        for line in _wrap(endpoint.note, 68):
            print(f"  {stylePrint(line, MUTED)}")

    print()


def _listed_flags(endpoint: Endpoint) -> set[str]:
    found: set[str] = set()
    for flag in endpoint.flags:
        found.update(re.findall(r"--[a-z][a-z0-9-]*", flag.name))
    return found


def _declared_flags(path: Path) -> set[str]:
    """Flags declaradas no ``_parse_args`` (ou no arquivo inteiro).

    Limitar ao ``_parse_args`` evita que o próprio glossário, que cita flags
    nas descrições, se auto-detecte como desatualizado.
    """

    text = path.read_text(encoding="utf-8")
    block = re.search(r"def _parse_args\(.*?(?=\ndef |\Z)", text, re.S)

    return set(re.findall(r'"(--[a-z][a-z0-9-]*)"', block.group(0) if block else text))


def _check() -> int:
    print(_rule("check"))
    failures = 0

    for endpoint in ENDPOINTS:
        path = _ROOT / endpoint.path

        if not path.exists():
            print(f"  {stylePrint('FALHA', FAIL_COLOR, bold=True)}  {endpoint.path}: não existe")
            failures += 1
            continue

        declared = _declared_flags(path)
        if "--progress" in declared:
            # BooleanOptionalAction gera --no-progress sem aparecer no fonte.
            declared.add("--no-progress")

        listed = _listed_flags(endpoint)
        missing = sorted(declared - listed)
        extra = sorted(listed - declared)

        if not missing and not extra:
            print(
                f"  {stylePrint('OK', OK_COLOR, bold=True)}      "
                f"{endpoint.path} ({len(listed)} flags)"
            )
            continue

        failures += 1
        print(f"  {stylePrint('FALHA', FAIL_COLOR, bold=True)}   {endpoint.path}")
        if missing:
            print(f"          no código, fora do glossário: {', '.join(missing)}")
        if extra:
            print(f"          no glossário, fora do código: {', '.join(extra)}")

    print()
    if failures:
        print(
            f"  {stylePrint(str(failures) + ' endpoint(s) desatualizado(s)', FAIL_COLOR)}"
        )
        return 1

    print(f"  {stylePrint('glossário em dia', OK_COLOR, bold=True)}")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Glossário dos endpoints executáveis do repositório.",
    )
    parser.add_argument(
        "filter",
        nargs="?",
        default="",
        help="mostra só endpoints que contêm o texto.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compara as flags listadas com o código.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.check:
        return _check()

    selected = [
        endpoint
        for endpoint in ENDPOINTS
        if args.filter.lower() in endpoint.path.lower()
        or args.filter.lower() in endpoint.summary.lower()
    ]

    print(_rule("EAQGA · endpoints"))
    print(f"  {stylePrint('raiz', MUTED)}: {_ROOT}")
    print(f"  {stylePrint('setup', MUTED)}:")
    for step in _SETUP:
        print(f"    {stylePrint('$', MUTED)} {step}")
    print()

    if not selected:
        print(f"  {stylePrint('nenhum endpoint com', FAIL_COLOR)} {args.filter!r}")
        return 1

    for endpoint in selected:
        _render(endpoint)

    print(_rule("biblioteca · não executável"))
    width = max(len(path) for path, _ in LIBRARY) + 2
    for path, purpose in LIBRARY:
        padding = " " * max(0, width - len(path))
        print(f"  {stylePrint(path, PATH_COLOR)}{padding}{stylePrint(purpose, MUTED)}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
