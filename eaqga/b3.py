"""Draft: dados do Ibovespa (B3) via ``yfinance``, no mesmo pipeline do S&P.

A B3 usa o sufixo ``.SA`` no Yahoo Finance (``PETR4.SA``). Fora isso, nada muda:
preço → retorno → ``μ``/``Σ`` → QUBO, tudo por ``eaqga.portfolio``.

Por que ``yfinance``: é a mesma base que o Qiskit Finance usa (``YahooDataProvider``),
o cache e o loader já existem, e não entra dependência nova. Alternativas ficam
para quando faltar cobertura:

- **brapi.dev**: API brasileira, gratuita com token; melhor para small caps.
- **COTAHIST** (B3): arquivos anuais/mensais oficiais, sem rede instável, mas
  exigem parser próprio e download pesado.

A lista ``B3_TICKERS`` é um **draft**, não a composição oficial do Ibovespa —
o índice é revisado periodicamente. Para o artigo, o universo é o S&P 500; a B3
serve para validar o pipeline em outro mercado.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .portfolio import (
    DEFAULT_CACHE_DIR,
    PAPER_END,
    PAPER_START,
    PortfolioData,
    build_portfolio,
    load_prices,
)

__all__ = [
    "B3_TICKERS",
    "build_b3_portfolio",
    "load_b3_prices",
    "to_yahoo",
]

B3_TICKERS: tuple[str, ...] = (
    "ABEV3",
    "B3SA3",
    "BBAS3",
    "BBDC4",
    "BBSE3",
    "BPAC11",
    "BRFS3",
    "CMIG4",
    "CPLE6",
    "CSAN3",
    "CSNA3",
    "CYRE3",
    "EGIE3",
    "ELET3",
    "EMBR3",
    "EQTL3",
    "GGBR4",
    "HAPV3",
    "ITUB4",
    "JBSS3",
    "KLBN11",
    "LREN3",
    "MRFG3",
    "MULT3",
    "NTCO3",
    "PETR4",
    "PRIO3",
    "PSSA3",
    "RADL3",
    "RAIL3",
    "RENT3",
    "SANB11",
    "SBSP3",
    "SUZB3",
    "TIMS3",
    "TOTS3",
    "UGPA3",
    "VALE3",
    "VIVT3",
    "WEGE3",
)
"""Amostra líquida da B3 (draft).

Símbolos sem sufixo; :func:`to_yahoo` adiciona ``.SA``. Não é a composição
oficial do Ibovespa e não cobre o período do paper (que é S&P 500).
"""

_B3_SUFFIX = ".SA"


def to_yahoo(symbol: str) -> str:
    """Converte ``PETR4`` em ``PETR4.SA`` (idempotente)."""

    if not isinstance(symbol, str):
        raise TypeError(f"symbol deve ser str, recebeu {type(symbol).__name__}.")

    clean = symbol.strip().upper()
    if not clean:
        raise ValueError("symbol não pode ser vazio.")
    if clean.endswith(_B3_SUFFIX):
        return clean
    return f"{clean}{_B3_SUFFIX}"


def load_b3_prices(
    symbols: Sequence[str] | None = None,
    start: str = PAPER_START,
    end: str = PAPER_END,
    *,
    refresh: bool = False,
    strict: bool = False,
    directory: str | Path = DEFAULT_CACHE_DIR,
):
    """Preços diários ajustados de ações da B3 (``.SA`` aplicado)."""

    chosen = B3_TICKERS if symbols is None else tuple(symbols)
    tickers = tuple(to_yahoo(symbol) for symbol in chosen)

    return load_prices(
        tickers,
        start,
        end,
        refresh=refresh,
        strict=strict,
        directory=directory,
    )


def build_b3_portfolio(
    symbols: Sequence[str] | None = None,
    start: str = PAPER_START,
    end: str = PAPER_END,
    *,
    refresh: bool = False,
    strict: bool = False,
    directory: str | Path = DEFAULT_CACHE_DIR,
) -> PortfolioData:
    """``PortfolioData`` da B3, pronto para ``.qubo()`` como o S&P.

    ``strict=False`` por default: tickers renomeados ou deslistados são
    descartados em vez de derrubar o download.
    """

    chosen = B3_TICKERS if symbols is None else tuple(symbols)
    tickers = tuple(to_yahoo(symbol) for symbol in chosen)

    return build_portfolio(
        tickers,
        start,
        end,
        refresh=refresh,
        strict=strict,
        directory=directory,
    )
