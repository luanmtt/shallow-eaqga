"""Linhas de progresso por geração, adaptadas do QGA antigo.

O QGA antigo (`oldCodebasePrints.py`) tinha três barras: geração, fitness e
batch. Aqui ficam só as duas que fazem sentido no EAQGA: a **geração**
(`g/T`) e a **proximidade da referência** (`best / referência`, quando existe).

Diferenças deliberadas do original:

- fitness usa :func:`~eaqga.utils.formats.formatValue`, não mantissa. Nossos
  valores ficam em `~0.02`; `2.0 × 10^-2` é pior de ler que `0.0200`;
- a barra de fitness fica **vazia** quando não há referência. Inventar um
  denominador (ex.: `best/1`) daria uma barra que não significa nada.

Uma linha por geração, sem `\\r`: assim funciona em `nohup`/log e não briga com
o progresso do `joblib`.
"""

from __future__ import annotations

from .embellish import stylePrint
from .formats import formatValue
from .progressbar import ProgressBar

__all__ = [
    "GENERATION_HEADER",
    "format_duration",
    "format_generation_line",
    "make_generation_bars",
]

C = "#f5c2e7"

GENERATION_HEADER = (
    "                       geração                fitness          "
    "  gen/s       best         avg    únicos"
)


def make_generation_bars() -> tuple[ProgressBar, ProgressBar]:
    """As duas barras de uma execução: geração e referência."""

    generation = ProgressBar(
        total=1,
        style="keras",
        fg_color="#CC3363",
        bg_color="#5b5f66",
    )
    fitness = ProgressBar(
        total=1,
        style="keras",
        fg_color="#f6cdb8",
        bg_color="#5b5f66",
    )
    return generation, fitness


def format_duration(seconds: float) -> str:
    """Formata segundos em dias/horas/min/seg, omitindo unidades zero."""

    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    rest = seconds % 60

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}min")
    if rest or not parts:
        parts.append(f"{rest:.0f}sec")

    return " ".join(parts)


def format_generation_line(
    *,
    generation: int,
    generations: int,
    elapsed: float,
    best: float,
    average: float,
    unique: int,
    generation_bar: ProgressBar,
    fitness_bar: ProgressBar,
    reference: float | None = None,
) -> str:
    """Uma linha de progresso: geração, barras, taxa, best, avg e diversidade.

    ``reference`` (opcional) é o alvo da barra de fitness — no experimento, a
    busca local multi-start. Sem referência, a barra fica vazia.
    """

    total = max(generations, 1)
    width = len(str(total))
    ratio = min(1.0, max(0.0, generation / total))

    if reference is not None and reference > 0.0:
        fitness_ratio = min(1.0, max(0.0, best / reference))
    else:
        fitness_ratio = 0.0

    rate = generation / elapsed if elapsed > 0.0 else 0.0
    label = stylePrint(
        f"Generation {generation:>{width}}/{total}",
        C,
        bold=True,
    )

    return (
        f"{label}   {generation_bar.bar(ratio)}   "
        f"{fitness_bar.bar(fitness_ratio)}   "
        f"{rate:>6.2f}g/s  {formatValue(best, 4):>10}  "
        f"{formatValue(average, 4):>10}  {unique:>6}"
    )
