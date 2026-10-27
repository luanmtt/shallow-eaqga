"""Avaliação de bitstrings e contagem de shots.

O EAQGA mede o circuito e avalia o fitness da bitstring. No nosso híbrido,
usamos ``k`` shots para estimar o fitness médio do indivíduo; o melhor shot
é guardado para o crossover. Este módulo transforma os ``counts`` do Qiskit
em uma :class:`Evaluation` com:

- contagens canônicas (bit 0 = variável 0);
- fitness por estado;
- média e desvio do fitness;
- melhor bitstring observado e seu fitness.

A ordem importa: o Qiskit devolve bitstrings com o qubit de maior índice à
esquerda. ``evaluate_counts`` reverte por padrão (``reverse_bits=True``) para
que o índice 0 da string seja a variável 0, igual à convenção de
:class:`~eaqga.instance.QUBOInstance`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real

from .instance import QUBOInstance

__all__ = ["Evaluation", "evaluate_bitstrings", "evaluate_counts"]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Resultado agregado de uma avaliação por shots.

    Attributes
    ----------
    S:
        Número de variáveis/qubits.
    counts:
        Contagens canônicas ``{bitstring: count}``, com bit 0 = variável 0.
    fitness_by_state:
        Fitness de cada bitstring observado.
    bitstrings:
        Bitstrings únicos, ordenados por contagem decrescente e depois
        lexicograficamente.
    mean:
        Média ponderada do fitness pelos shots.
    std:
        Desvio padrão amostral do fitness (``n - 1`` no denominador).
    best_bitstring:
        Bitstring observado com maior fitness. Empates são resolvidos por
        maior contagem e, depois, lexicograficamente.
    best_fitness:
        Fitness de ``best_bitstring``.
    shots:
        Total de shots (soma das contagens).

    Notes
    -----
    O desvio é uma estimativa amostral. Com ``shots=1``, ``std=0``.
    """

    S: int
    counts: dict[str, int]
    fitness_by_state: dict[str, float]
    bitstrings: tuple[str, ...]
    mean: float
    std: float
    best_bitstring: str
    best_fitness: float
    shots: int

    def __post_init__(self) -> None:
        if isinstance(self.S, bool) or not isinstance(self.S, int) or self.S < 1:
            raise ValueError(f"S deve ser int >= 1, recebeu {self.S!r}.")

        if (
            isinstance(self.shots, bool)
            or not isinstance(self.shots, int)
            or self.shots < 1
        ):
            raise ValueError(
                f"shots deve ser int >= 1, recebeu {self.shots!r}."
            )

        if not isinstance(self.counts, dict):
            raise TypeError(
                f"counts deve ser dict, recebeu {type(self.counts).__name__}."
            )
        if not isinstance(self.fitness_by_state, dict):
            raise TypeError(
                "fitness_by_state deve ser dict, recebeu "
                f"{type(self.fitness_by_state).__name__}."
            )
        if not isinstance(self.bitstrings, tuple):
            raise TypeError(
                "bitstrings deve ser tuple, recebeu "
                f"{type(self.bitstrings).__name__}."
            )

        if not self.bitstrings:
            raise ValueError("bitstrings não pode ser vazio.")
        if len(set(self.bitstrings)) != len(self.bitstrings):
            raise ValueError("bitstrings contém duplicatas.")
        if any(len(bits) != self.S for bits in self.bitstrings):
            raise ValueError(
                f"todos os bitstrings devem ter {self.S} caracteres."
            )
        if set(self.bitstrings) != set(self.counts):
            raise ValueError(
                "bitstrings e counts devem ter as mesmas chaves."
            )
        if set(self.counts) != set(self.fitness_by_state):
            raise ValueError(
                "counts e fitness_by_state devem ter as mesmas chaves."
            )

        for bits, count in self.counts.items():
            if (
                isinstance(count, bool)
                or not isinstance(count, Integral)
                or count < 0
            ):
                raise ValueError(
                    f"contagem inválida para {bits!r}: {count!r}."
                )

        for bits, value in self.fitness_by_state.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"fitness inválido para {bits!r}: {value!r}."
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"fitness não finito para {bits!r}: {value!r}."
                )

        if sum(self.counts.values()) != self.shots:
            raise ValueError(
                "soma das contagens "
                f"({sum(self.counts.values())}) != shots ({self.shots})."
            )

        if len(self.best_bitstring) != self.S:
            raise ValueError(
                f"best_bitstring deve ter {self.S} caracteres, "
                f"recebeu {len(self.best_bitstring)}."
            )
        if self.best_bitstring not in self.counts:
            raise ValueError(
                f"best_bitstring {self.best_bitstring!r} não está em counts."
            )

        if (
            not math.isfinite(self.mean)
            or not math.isfinite(self.std)
            or not math.isfinite(self.best_fitness)
        ):
            raise ValueError("mean, std e best_fitness devem ser finitos.")
        if self.std < 0.0:
            raise ValueError(f"std deve ser >= 0, recebeu {self.std}.")
        if not math.isclose(
            self.best_fitness,
            float(self.fitness_by_state[self.best_bitstring]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "best_fitness não corresponde ao fitness do best_bitstring."
            )

        # Cópia defensiva: o resultado não deve mudar se o chamador
        # reutilizar os dicionários de entrada.
        object.__setattr__(self, "counts", dict(self.counts))
        object.__setattr__(self, "fitness_by_state", dict(self.fitness_by_state))

    def probability(self, bits: str) -> float:
        """Probabilidade empírica de um bitstring observado."""

        if not isinstance(bits, str) or len(bits) != self.S:
            raise TypeError(
                f"bits deve ser str de tamanho {self.S}."
            )
        return self.counts.get(bits, 0) / self.shots

    @property
    def best_count(self) -> int:
        """Número de vezes que o melhor bitstring foi observado."""

        return self.counts[self.best_bitstring]

    def __repr__(self) -> str:
        return (
            f"Evaluation(S={self.S}, shots={self.shots}, "
            f"mean={self.mean:.6g}, best={self.best_bitstring!r}, "
            f"best_fitness={self.best_fitness:.6g})"
        )


def evaluate_counts(
    counts: Mapping[str, int],
    instance: QUBOInstance,
    *,
    reverse_bits: bool = True,
) -> Evaluation:
    """Agrega ``counts`` do Qiskit em uma :class:`Evaluation`.

    Parameters
    ----------
    counts:
        Dicionário ``{bitstring: contagem}``, como o retornado por
        ``AerSimulator.run(...).result().get_counts()``.
    instance:
        Instância QUBO que define ``S`` e o fitness.
    reverse_bits:
        Se ``True`` (default), inverte cada chave para converter do padrão
        do Qiskit (qubit de maior índice à esquerda) para o padrão do
        repositório (bit 0 = variável 0). Use ``False`` se as chaves já
        estiverem na ordem canônica.

    Raises
    ------
    TypeError
        Se ``instance``, ``counts``, chaves ou contagens tiverem tipo errado.
    ValueError
        Se alguma chave tiver tamanho/alfabeto inválido, se não houver
        shots positivos ou se a soma das contagens for zero.
    """

    if not isinstance(instance, QUBOInstance):
        raise TypeError(
            f"instance deve ser QUBOInstance, recebeu {type(instance).__name__}."
        )
    if not isinstance(counts, Mapping):
        raise TypeError(
            f"counts deve ser Mapping, recebeu {type(counts).__name__}."
        )
    if not counts:
        raise ValueError("counts não pode ser vazio.")
    if not isinstance(reverse_bits, bool):
        raise TypeError(
            f"reverse_bits deve ser bool, recebeu {type(reverse_bits).__name__}."
        )

    canonical: dict[str, int] = {}

    for raw_bits, raw_count in counts.items():
        if not isinstance(raw_bits, str):
            raise TypeError(
                f"chave de counts deve ser str, recebeu "
                f"{type(raw_bits).__name__}."
            )
        if len(raw_bits) != instance.S:
            raise ValueError(
                f"chave {raw_bits!r} tem {len(raw_bits)} bits; "
                f"esperado {instance.S}."
            )

        invalid = set(raw_bits) - {"0", "1"}
        if invalid:
            raise ValueError(
                f"chave {raw_bits!r} contém caracteres inválidos: "
                f"{sorted(invalid)}."
            )

        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, Integral)
            or raw_count < 0
        ):
            raise ValueError(
                f"contagem inválida para {raw_bits!r}: {raw_count!r}."
            )

        # Estado com contagem zero não foi observado; não deve competir
        # pelo melhor bitstring nem entrar nas métricas.
        if raw_count == 0:
            continue

        bits = raw_bits[::-1] if reverse_bits else raw_bits
        canonical[bits] = canonical.get(bits, 0) + int(raw_count)

    total = sum(canonical.values())
    if total <= 0:
        raise ValueError("a soma das contagens deve ser positiva.")

    fitness_by_state = {
        bits: instance.fitness(bits) for bits in canonical
    }

    mean = (
        sum(
            canonical[bits] * fitness_by_state[bits]
            for bits in canonical
        )
        / total
    )

    if total > 1:
        variance = (
            sum(
                canonical[bits] * (fitness_by_state[bits] - mean) ** 2
                for bits in canonical
            )
            / (total - 1)
        )
        std = math.sqrt(max(0.0, variance))
    else:
        std = 0.0

    # Empate: maior fitness, depois maior contagem, depois lexicográfico.
    best_bitstring = max(
        fitness_by_state,
        key=lambda bits: (
            fitness_by_state[bits],
            canonical[bits],
            bits,
        ),
    )

    bitstrings = tuple(
        sorted(canonical, key=lambda bits: (-canonical[bits], bits))
    )

    return Evaluation(
        S=instance.S,
        counts=canonical,
        fitness_by_state=fitness_by_state,
        bitstrings=bitstrings,
        mean=float(mean),
        std=float(std),
        best_bitstring=best_bitstring,
        best_fitness=float(fitness_by_state[best_bitstring]),
        shots=total,
    )


def evaluate_bitstrings(
    bitstrings: Iterable[str],
    instance: QUBOInstance,
) -> Evaluation:
    """Agrega uma sequência de bitstrings já na ordem canônica.

    Útil para testes e para quando o chamador já tem as amostras, sem passar
    pelos ``counts`` do Qiskit.
    """

    if not isinstance(instance, QUBOInstance):
        raise TypeError(
            f"instance deve ser QUBOInstance, recebeu {type(instance).__name__}."
        )
    if isinstance(bitstrings, str):
        raise TypeError(
            "bitstrings deve ser uma coleção de strings, não uma string única."
        )

    try:
        samples = list(bitstrings)
    except TypeError as exc:
        raise TypeError(
            "bitstrings deve ser iterável de strings."
        ) from exc

    if not samples:
        raise ValueError("bitstrings não pode ser vazio.")

    counts = Counter(samples)
    return evaluate_counts(counts, instance, reverse_bits=False)
