"""Dados de portfólio: preços, ``μ``/``Σ`` e a instância QUBO.

O paper usa ações do S&P 500, período de **01/10/2023 a 30/09/2024**, intervalo
diário, e resolve

```
max_x  μᵀx − q·xᵀΣx      x_i ∈ {0,1},  q = 0.5
```

As fórmulas são as do paper (``docs/01-eaqga-baseline.md`` §1.1):

```
R_it   = P_it / P_i,t-1 − 1
μ_i    = (1/T) Σ_t R_it
Σ_ij   = (1/(T−1)) Σ_t (R_it − μ_i)(R_jt − μ_j)
```

O fitness fica na escala de retorno diário decimal (``~0.02`` para 30 ativos).
As Tabelas I/II do paper multiplicam por 100; não confundir as duas escalas.

Dados vêm de ``yfinance`` com ``auto_adjust=False``: o default novo
(``auto_adjust=True``) remove a coluna ``Adj Close``, que é a que o
``YahooDataProvider`` do Qiskit Finance espera. Aqui lemos ``Adj Close``
quando existe e caímos para ``Close`` quando não.

Tudo é cacheado em CSV. Sem rede, ``load_prices`` usa o cache; com
``refresh=True`` baixa de novo.
"""

from __future__ import annotations

import hashlib
import io
import random
import urllib.error
import urllib.request
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from .instance import QUBOInstance

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_UNIVERSE",
    "PAPER_END",
    "PAPER_START",
    "RISK_AVERSION",
    "PortfolioData",
    "build_portfolio",
    "cache_path",
    "disjoint_universes",
    "download_prices",
    "fetch_sp500_tickers",
    "load_prices",
    "mean_covariance",
    "returns_from_prices",
    "sample_tickers",
]

PAPER_START: Final[str] = "2023-10-01"
PAPER_END: Final[str] = "2024-09-30"
RISK_AVERSION: Final[float] = 0.5

DEFAULT_CACHE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[1] / "data" / "prices"
)

DEFAULT_UNIVERSE: Final[tuple[str, ...]] = (
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "META",
    "JPM",
    "V",
    "MA",
    "XOM",
    "JNJ",
    "WMT",
    "PG",
    "UNH",
    "HD",
    "CVX",
    "MRK",
    "ABBV",
    "KO",
    "PEP",
    "AVGO",
    "COST",
    "ADBE",
    "CSCO",
    "TMO",
    "MCD",
    "ACN",
    "LIN",
    "AMD",
    "NFLX",
)
"""Amostra fixa de large caps para smoke tests.

Não é o universo do paper: o paper sorteia do S&P 500 inteiro. Serve para rodar
offline e para testes de encanamento.
"""


def _require_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    if isinstance(tickers, (str, bytes)):
        raise TypeError("tickers deve ser uma sequência de strings, não uma string.")

    try:
        items = tuple(tickers)
    except TypeError as exc:
        raise TypeError("tickers deve ser iterável de strings.") from exc

    if len(items) < 1:
        raise ValueError("tickers não pode ser vazio.")

    cleaned: list[str] = []
    for index, ticker in enumerate(items):
        if not isinstance(ticker, str):
            raise TypeError(
                f"tickers[{index}] deve ser str, recebeu {type(ticker).__name__}."
            )
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError(f"tickers[{index}] é vazio.")
        cleaned.append(symbol)

    if len(set(cleaned)) != len(cleaned):
        raise ValueError("tickers contém duplicatas.")

    return tuple(cleaned)


def _require_dates(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        first = pd.Timestamp(start)
        last = pd.Timestamp(end)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"start/end inválidos: {start!r}, {end!r}."
        ) from exc

    if pd.isna(first) or pd.isna(last):
        raise ValueError(f"start/end inválidos: {start!r}, {end!r}.")
    if first >= last:
        raise ValueError(
            f"start deve ser anterior a end, recebeu {start} e {end}."
        )

    return cast(pd.Timestamp, first), cast(pd.Timestamp, last)


def cache_path(
    tickers: Sequence[str],
    start: str,
    end: str,
    directory: str | Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Caminho determinístico do CSV de preços para (tickers, período)."""

    symbols = _require_tickers(tickers)
    first, last = _require_dates(start, end)

    digest = hashlib.sha1("|".join(sorted(symbols)).encode("utf-8")).hexdigest()[:10]
    return Path(directory) / (
        f"{first.date()}_{last.date()}_{len(symbols)}_{digest}.csv"
    )


def download_prices(
    tickers: Sequence[str],
    start: str = PAPER_START,
    end: str = PAPER_END,
) -> pd.DataFrame:
    """Baixa preços ajustados diários via ``yfinance``.

    ``end`` é **inclusivo** aqui (o Yahoo trata o argumento como exclusivo, então
    somamos um dia). Devolve um DataFrame ``datas × tickers``.
    """

    symbols = _require_tickers(tickers)
    first, last = _require_dates(start, end)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance não está instalado; rode `pip install yfinance`."
        ) from exc

    with warnings.catch_warnings():
        # yfinance 1.7 ainda usa `inplace=True`, que o pandas 3 deprecia em
        # cada download. O aviso é do fornecedor, não nosso, e polui o log.
        warnings.filterwarnings("ignore", message=".*inplace keyword.*")
        data = yf.download(
            list(symbols),
            start=first.date().isoformat(),
            end=(last + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )

    if data is None or len(data) == 0:
        raise RuntimeError(
            "yfinance devolveu vazio. Verifique conexão/tickers ou use o cache."
        )

    return _extract_close(data, symbols)


def _extract_close(data: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Extrai a matriz de fechamento, aceitando 1 ou vários tickers."""

    if isinstance(data.columns, pd.MultiIndex):
        fields = data.columns.get_level_values(0)
        field = "Adj Close" if "Adj Close" in fields else "Close"
        close = data[field]
    elif "Adj Close" in data.columns:
        close = data["Adj Close"].to_frame()
    elif "Close" in data.columns:
        close = data["Close"].to_frame()
    else:
        raise RuntimeError(
            "yfinance não devolveu 'Adj Close' nem 'Close'; "
            f"colunas: {list(data.columns)[:5]}."
        )

    if close.shape[1] == 1 and len(tickers) == 1:
        close.columns = [tickers[0]]

    missing = [ticker for ticker in tickers if ticker not in close.columns]
    if missing:
        raise RuntimeError(f"tickers sem dados no período: {missing}.")

    close = close.loc[:, list(tickers)]
    close = cast(pd.DataFrame, close)
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    return close


def _clean_prices(prices: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Remove colunas majoritariamente vazias, preenche 1 dia e corta NaN."""

    frame = cast(pd.DataFrame, prices.apply(pd.to_numeric, errors="coerce"))
    frame = frame.reindex(columns=list(tickers))

    fractions = cast(pd.Series, frame.isna().mean()).to_numpy(dtype=float)
    keep = [
        ticker
        for ticker, fraction in zip(tickers, fractions)
        if float(fraction) <= 0.01
    ]
    dropped = [ticker for ticker in tickers if ticker not in keep]
    if dropped:
        frame = frame.loc[:, keep]

    frame = frame.ffill().dropna()

    if frame.shape[1] < 2:
        raise ValueError(
            "menos de 2 ativos com dados suficientes no período; "
            f"descartados: {dropped}."
        )
    if len(frame) < 30:
        raise ValueError(
            f"apenas {len(frame)} dias úteis com dados; mínimo 30."
        )

    return frame


def load_prices(
    tickers: Sequence[str],
    start: str = PAPER_START,
    end: str = PAPER_END,
    *,
    refresh: bool = False,
    directory: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Carrega preços do cache; baixa e cacheia quando necessário."""

    if not isinstance(refresh, bool):
        raise TypeError(f"refresh deve ser bool, recebeu {type(refresh).__name__}.")

    symbols = _require_tickers(tickers)
    path = cache_path(symbols, start, end, directory)

    if path.exists() and not refresh:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return _clean_prices(frame, symbols)

    frame = download_prices(symbols, start, end)
    cleaned = _clean_prices(frame, symbols)

    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(path, index_label="date")
    return cleaned


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos simples diários ``P_t / P_{t-1} − 1``."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            f"prices deve ser DataFrame, recebeu {type(prices).__name__}."
        )
    if prices.shape[1] < 1 or len(prices) < 3:
        raise ValueError(
            f"prices precisa de >= 3 linhas e >= 1 coluna, recebeu "
            f"{prices.shape}."
        )

    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) < 2:
        raise ValueError("retornos insuficientes após remover NaN/inf.")

    return returns


def mean_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``μ`` (média) e ``Σ`` (covariância amostral, ``ddof=1``)."""

    if not isinstance(returns, pd.DataFrame):
        raise TypeError(
            f"returns deve ser DataFrame, recebeu {type(returns).__name__}."
        )
    if returns.shape[1] < 2:
        raise ValueError("returns precisa de pelo menos 2 ativos.")
    if len(returns) < 2:
        raise ValueError("returns precisa de pelo menos 2 observações.")

    mu = returns.mean().to_numpy(dtype=float)
    sigma = returns.cov().to_numpy(dtype=float)
    return mu, sigma


@dataclass(frozen=True, slots=True, eq=False)
class PortfolioData:
    """Preços, retornos e matrizes de um portfólio.

    ``prices``/``returns`` são DataFrames (datas × tickers); ``mu``/``sigma`` são
    arrays na ordem de ``tickers``.
    """

    tickers: tuple[str, ...]
    start: str
    end: str
    prices: pd.DataFrame
    returns: pd.DataFrame
    mu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.tickers, tuple) or len(self.tickers) < 2:
            raise ValueError("tickers deve ser tuple com pelo menos 2 símbolos.")
        if len(self.tickers) != self.mu.size:
            raise ValueError(
                f"tickers tem {len(self.tickers)} símbolos e mu {self.mu.size}."
            )
        if self.sigma.shape != (self.mu.size, self.mu.size):
            raise ValueError(
                f"sigma tem formato {self.sigma.shape}, esperado "
                f"({self.mu.size}, {self.mu.size})."
            )

    @property
    def S(self) -> int:
        return len(self.tickers)

    def qubo(
        self,
        risk_aversion: float = RISK_AVERSION,
        *,
        name: str | None = None,
    ) -> QUBOInstance:
        """Instância QUBO no formato que o baseline/híbrido consomem."""

        return QUBOInstance.from_returns(
            self.mu,
            self.sigma,
            q=risk_aversion,
            name=name or f"portfolio-{self.S}",
        )

    def summary(self) -> str:
        return (
            f"{self.S} ativos · {self.start}..{self.end} · "
            f"{len(self.returns)} dias · "
            f"μ∈[{self.mu.min():.5f}, {self.mu.max():.5f}]"
        )


def build_portfolio(
    tickers: Sequence[str],
    start: str = PAPER_START,
    end: str = PAPER_END,
    *,
    refresh: bool = False,
    directory: str | Path = DEFAULT_CACHE_DIR,
) -> PortfolioData:
    """Baixa/carrega preços e devolve ``PortfolioData`` com ``μ`` e ``Σ``."""

    symbols = _require_tickers(tickers)
    prices = load_prices(symbols, start, end, refresh=refresh, directory=directory)
    returns = returns_from_prices(prices)
    mu, sigma = mean_covariance(returns)

    return PortfolioData(
        tickers=tuple(str(column) for column in prices.columns),
        start=start,
        end=end,
        prices=prices,
        returns=returns,
        mu=mu,
        sigma=sigma,
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Universo e subconjuntos


def sample_tickers(
    pool: Sequence[str],
    size: int,
    rng: random.Random,
) -> tuple[str, ...]:
    """Sorteia ``size`` tickers distintos do pool."""

    symbols = _require_tickers(pool)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"size deve ser int >= 1, recebeu {size!r}.")
    if size > len(symbols):
        raise ValueError(
            f"size={size} maior que o pool de {len(symbols)} tickers."
        )
    if not isinstance(rng, random.Random):
        raise TypeError(f"rng deve ser random.Random, recebeu {type(rng).__name__}.")

    return tuple(rng.sample(symbols, size))


def disjoint_universes(
    pool: Sequence[str],
    size: int,
    count: int,
    rng: random.Random,
) -> tuple[tuple[str, ...], ...]:
    """Divide um sorteio em ``count`` subconjuntos disjuntos de ``size``.

    É o protocolo do paper no simulador: 10 subconjuntos de 30 (ou 40), sem
    sobreposição.
    """

    symbols = _require_tickers(pool)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"count deve ser int >= 1, recebeu {count!r}.")
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError(f"size deve ser int >= 2, recebeu {size!r}.")
    if size * count > len(symbols):
        raise ValueError(
            f"{count} subconjuntos de {size} exigem {size * count} tickers; "
            f"o pool tem {len(symbols)}."
        )
    if not isinstance(rng, random.Random):
        raise TypeError(f"rng deve ser random.Random, recebeu {type(rng).__name__}.")

    drawn = rng.sample(symbols, size * count)
    return tuple(
        tuple(drawn[offset : offset + size])
        for offset in range(0, size * count, size)
    )


def fetch_sp500_tickers(
    *,
    refresh: bool = False,
    directory: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[str, ...]:
    """Lista atual do S&P 500 (Wikipedia), cacheada em CSV.

    Requer ``lxml`` para o ``pandas.read_html``. O universo do paper é o S&P 500
    da época (2023-2024); esta lista é a de hoje, então não reproduz exatamente
    os mesmos subconjuntos.
    """

    if not isinstance(refresh, bool):
        raise TypeError(f"refresh deve ser bool, recebeu {type(refresh).__name__}.")

    path = Path(directory) / "sp500_tickers.csv"

    if path.exists() and not refresh:
        frame = pd.read_csv(path)
        return tuple(str(item) for item in frame["ticker"])

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (eaqga)"})

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "não foi possível baixar a lista do S&P 500; passe os tickers "
            "manualmente ou tente de novo."
        ) from exc

    try:
        tables = pd.read_html(io.StringIO(html))
    except ImportError as exc:
        raise ImportError(
            "pandas.read_html precisa de lxml; rode `pip install lxml` "
            "ou passe a lista de tickers manualmente."
        ) from exc

    table = tables[0]
    symbols = [
        str(symbol).strip().replace(".", "-")
        for symbol in table["Symbol"].tolist()
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": symbols}).to_csv(path, index=False)

    return tuple(symbols)
