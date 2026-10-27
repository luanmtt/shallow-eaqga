"""Configurações do EAQGA e do baseline.

Este módulo centraliza os hiperparâmetros e valida tudo na construção.
A regra é falhar cedo: nenhum módulo posterior deve precisar repetir
checagens como ``population_size > 0`` ou ``0 <= mutation_rate <= 1``.

Os tipos são fechados em enums sempre que possível. Isso evita o padrão
"string mágica" (por exemplo, ``gate_set="hilbert"``) e deixa os erros
aparecerem no ponto de configuração, não no meio da simulação.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any

__all__ = [
    "BackendKind",
    "BaselineConfig",
    "EvaluationMode",
    "ForestShape",
    "GateSet",
    "HybridConfig",
]


class GateSet(str, Enum):
    """Conjunto de portas de 1 qubit usado no genoma do híbrido."""

    HILBERT_1Q = "hilbert_1q"
    CLIFFORD_1Q = "clifford_1q"


class ForestShape(str, Enum):
    """Topologia permitida para a floresta de CNOTs."""

    PATH = "path"
    """Coleção de caminhos; grau máximo 2 e profundidade máxima 2."""

    STAR = "star"
    """Um controle por componente; profundidade igual ao fan-out."""

    BOUNDED = "bounded"
    """Floresta com grau máximo configurável."""


class BackendKind(str, Enum):
    """Backend de simulação usado para medir os circuitos."""

    STATEVECTOR = "statevector"
    MPS = "mps"


class EvaluationMode(str, Enum):
    """Como o fitness de um lote de bitstrings é calculado.

    ``SCALAR`` é o caminho seguro: uma chamada a ``QUBOInstance.fitness`` por
    estado único. ``VECTORIZED`` usa ``QUBOInstance.fitness_many`` com
    ``einsum``; é mais rápido para lotes grandes, mas fica atrás de uma flag
    para poder ser desligado em caso de divergência numérica.
    """

    SCALAR = "scalar"
    VECTORIZED = "vectorized"


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Guards


def _require_int(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} deve ser int, recebeu {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} deve ser >= {minimum}, recebeu {value}.")


def _require_probability(name: str, value: object, *, open_interval: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} deve ser float, recebeu {type(value).__name__}.")

    if open_interval:
        valid = 0.0 < float(value) < 1.0
        interval = "(0, 1)"
    else:
        valid = 0.0 <= float(value) <= 1.0
        interval = "[0, 1]"

    if not valid:
        raise ValueError(f"{name} deve estar em {interval}, recebeu {value}.")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} deve ser bool, recebeu {type(value).__name__}.")


def _require_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(
            f"{name} deve ser {enum_type.__name__}, recebeu {type(value).__name__}."
        )


def _require_forest_shape(
    shape: object,
    max_degree: object,
) -> None:
    _require_enum("forest_shape", shape, ForestShape)

    if shape is ForestShape.STAR:
        if max_degree is not None:
            raise ValueError(
                "ForestShape.STAR não usa max_degree; passe max_degree=None."
            )
        return

    if isinstance(max_degree, bool) or not isinstance(max_degree, Integral):
        raise TypeError(
            "max_degree deve ser int ou None, "
            f"recebeu {type(max_degree).__name__}."
        )

    if shape is ForestShape.PATH:
        if max_degree != 2:
            raise ValueError(
                "ForestShape.PATH exige max_degree=2; "
                f"recebeu {max_degree}."
            )
        return

    # ForestShape.BOUNDED
    if max_degree < 2:
        raise ValueError(
            "ForestShape.BOUNDED exige max_degree >= 2; "
            f"recebeu {max_degree}."
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Configurações


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Configuração do nosso indivíduo dentro do laço do EAQGA.

    O híbrido mantém ``n_elites=2`` porque o crossover do EAQGA é definido
    a partir de exatamente dois bitstrings parentais. Todos os ``N`` filhos
    são gerados a partir desses dois elites; os pais não são copiados
    intactos para a próxima geração.
    """

    population_size: int = 20
    generations: int = 20
    n_elites: int = 2
    shots: int = 50
    num_columns: int = 4
    gate_set: GateSet = GateSet.HILBERT_1Q
    forest_shape: ForestShape = ForestShape.PATH
    max_degree: int | None = 2
    mutation_rate: float = 0.05
    p_s: float = 0.6
    seed: int = 42
    backend: BackendKind = BackendKind.STATEVECTOR
    evaluation_mode: EvaluationMode = EvaluationMode.SCALAR
    verbose: bool = True

    def __post_init__(self) -> None:
        _require_int("population_size", self.population_size, minimum=2)
        _require_int("generations", self.generations, minimum=1)
        _require_int("n_elites", self.n_elites, minimum=2)
        _require_int("shots", self.shots, minimum=1)
        _require_int("num_columns", self.num_columns, minimum=1)
        _require_int("seed", self.seed, minimum=0)

        if self.n_elites != 2:
            raise ValueError(
                "n_elites deve ser exatamente 2: o crossover do EAQGA "
                f"usa dois pais, recebeu {self.n_elites}."
            )

        _require_enum("gate_set", self.gate_set, GateSet)
        _require_enum("backend", self.backend, BackendKind)
        _require_enum("evaluation_mode", self.evaluation_mode, EvaluationMode)
        _require_bool("verbose", self.verbose)
        _require_probability("mutation_rate", self.mutation_rate)
        _require_probability("p_s", self.p_s)
        _require_forest_shape(self.forest_shape, self.max_degree)

    def to_dict(self, *, verbose: bool | None = None) -> dict[str, object]:
        """Serializa a configuração para um dicionário JSON-friendly.

        ``verbose=None`` usa o campo ``self.verbose``; ``True``/``False``
        força o comportamento. Enums são convertidos para seus valores.
        """

        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value

        show = self.verbose if verbose is None else verbose
        if show:
            print(f"[{type(self).__name__}] {data}")

        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        *,
        verbose: bool = False,
    ) -> "HybridConfig":
        """Reconstrói a configuração a partir de um dicionário.

        Aceita enums já convertidos ou seus valores string. Chaves ausentes
        usam os defaults da dataclass; chaves desconhecidas viram ``TypeError``.
        """

        if not isinstance(data, Mapping):
            raise TypeError(
                f"data deve ser Mapping, recebeu {type(data).__name__}."
            )

        payload: dict[str, Any] = dict(data)
        enum_fields = {
            "gate_set": GateSet,
            "forest_shape": ForestShape,
            "backend": BackendKind,
            "evaluation_mode": EvaluationMode,
        }
        for key, enum_type in enum_fields.items():
            if key in payload and not isinstance(payload[key], enum_type):
                payload[key] = enum_type(payload[key])

        config = cls(**payload)

        if verbose:
            print(f"[{cls.__name__}] {config}")

        return config


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Configuração do EAQGA fiel, com portas de rotação ``RY``.

    O baseline usa um shot por circuito por padrão, como no paper. Para uma
    comparação justa com o híbrido (que usa ``shots`` maiores), pode-se
    configurar o mesmo número de shots aqui.
    """

    population_size: int = 20
    generations: int = 20
    shots: int = 1
    p_a: float = 0.95
    p_s: float = 0.6
    forest_shape: ForestShape = ForestShape.STAR
    max_degree: int | None = None
    accumulated_pool: bool = False
    seed: int = 42
    backend: BackendKind = BackendKind.STATEVECTOR
    evaluation_mode: EvaluationMode = EvaluationMode.SCALAR
    verbose: bool = True

    def __post_init__(self) -> None:
        _require_int("population_size", self.population_size, minimum=2)
        _require_int("generations", self.generations, minimum=1)
        _require_int("shots", self.shots, minimum=1)
        _require_int("seed", self.seed, minimum=0)

        _require_probability("p_a", self.p_a, open_interval=True)
        _require_probability("p_s", self.p_s)

        _require_bool("accumulated_pool", self.accumulated_pool)
        _require_bool("verbose", self.verbose)
        _require_enum("backend", self.backend, BackendKind)
        _require_enum("evaluation_mode", self.evaluation_mode, EvaluationMode)
        _require_forest_shape(self.forest_shape, self.max_degree)

    def to_dict(self, *, verbose: bool | None = None) -> dict[str, object]:
        """Serializa a configuração para um dicionário JSON-friendly.

        ``verbose=None`` usa o campo ``self.verbose``; ``True``/``False``
        força o comportamento. Enums são convertidos para seus valores.
        """

        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value

        show = self.verbose if verbose is None else verbose
        if show:
            print(f"[{type(self).__name__}] {data}")

        return data

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        *,
        verbose: bool = False,
    ) -> "BaselineConfig":
        """Reconstrói a configuração a partir de um dicionário.

        Aceita enums já convertidos ou seus valores string. Chaves ausentes
        usam os defaults da dataclass; chaves desconhecidas viram ``TypeError``.
        """

        if not isinstance(data, Mapping):
            raise TypeError(
                f"data deve ser Mapping, recebeu {type(data).__name__}."
            )

        payload: dict[str, Any] = dict(data)
        enum_fields = {
            "forest_shape": ForestShape,
            "backend": BackendKind,
            "evaluation_mode": EvaluationMode,
        }
        for key, enum_type in enum_fields.items():
            if key in payload and not isinstance(payload[key], enum_type):
                payload[key] = enum_type(payload[key])

        config = cls(**payload)

        if verbose:
            print(f"[{cls.__name__}] {config}")

        return config


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
