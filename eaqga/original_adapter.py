"""Adapter para rodar o indivíduo original (QGA contínuo) no nosso QUBO.

O código original vive em ``original/`` e **não é editado**. Este módulo apenas
o configura para o problema binário do portfólio e devolve o resultado no
formato que o experimento consome.

Três decisões de adaptação (nenhuma altera o algoritmo):

1. **1 qubit por variável.** O original fixa ``N_QUBITS = 8`` (8 qubits por
   variável contínua). Para um QUBO binário usamos ``N_QUBITS = 1`` via
   atributo do módulo, então o domínio ``(0, 1)`` decodifica exatamente para
   ``0``/``1``. Sem isso, ``S = 30`` viraria 240 qubits.
2. **Backend.** ``Individuo`` instancia ``AerSimulator()`` (automático), que em
   30 qubits com portas ``T`` tentaria statevector (16 GiB). Trocamos o símbolo
   do módulo por um simulador MPS — a mesma escolha que fazemos nos nossos
   experimentos. O algoritmo não muda.
3. **Fitness.** O original avalia ``f(𝔼[x])``, a média dos valores decodificados.
   Como o QUBO é binário, aplicamos ``f`` ao ponto binarizado (limiar 0.5).
   Assim o fitness reportado é comparável ao do baseline/híbrido.

O que **não** mudamos: seleção, crossover, mutação, elitismo, número de portas,
conjunto de portas e o loop de gerações — tudo do original.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from qiskit_aer import AerSimulator

from .config import BackendKind
from .instance import QUBOInstance

__all__ = [
    "OriginalMember",
    "OriginalResult",
    "load_engine",
    "run_original",
]

_ORIGINAL_ROOT = Path(__file__).resolve().parents[1] / "original"
_ORIGINAL_SRC = _ORIGINAL_ROOT / "src"

_ENGINE: ModuleType | None = None
_INDIVIDUO: ModuleType | None = None


def load_engine() -> tuple[ModuleType, ModuleType]:
    """Importa o motor original, isolando ``sys.path`` no diretório ``original/``.

    O código original usa imports absolutos (``from config import ...``,
    ``from src.Individuo import *``, ``from utils.progressbar import ...``), então
    ``original/`` e ``original/src`` precisam estar no ``sys.path``.
    """

    global _ENGINE, _INDIVIDUO

    if _ENGINE is not None and _INDIVIDUO is not None:
        return _ENGINE, _INDIVIDUO

    if not _ORIGINAL_SRC.is_dir():
        raise FileNotFoundError(
            f"motor original não encontrado em {_ORIGINAL_SRC}."
        )

    for path in (str(_ORIGINAL_SRC), str(_ORIGINAL_ROOT)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    # ``ParallelQuantumGA`` faz ``from src.Individuo import *``: precisamos do
    # MESMO objeto de módulo (``src.Individuo``), senão o patch de N_QUBITS não
    # chega na classe usada pelo motor.
    individuo_module = importlib.import_module("src.Individuo")
    engine_module = importlib.import_module("ParallelQuantumGA")

    _INDIVIDUO = individuo_module
    _ENGINE = engine_module
    return _ENGINE, _INDIVIDUO


@dataclass(frozen=True, slots=True)
class OriginalMember:
    """Um indivíduo do original, reduzido ao que o experimento usa."""

    bits: str
    fitness: float


@dataclass(frozen=True, slots=True)
class OriginalResult:
    """Resultado do original no mesmo formato que baseline/híbrido."""

    best_individual: OriginalMember
    best_generation: int
    history: dict[str, list[float]]
    final_population: tuple[OriginalMember, ...]
    evaluations: int
    elapsed_s: float

    @property
    def best_bits(self) -> str:
        return self.best_individual.bits

    @property
    def best_fitness(self) -> float:
        return self.best_individual.fitness


def _bits_from_location(location: list[float] | None, size: int) -> str:
    if location is None or len(location) != size:
        raise ValueError(
            "indivíduo original sem localização coletada; "
            "rode com dataCollection=True."
        )
    return "".join("1" if value >= 0.5 else "0" for value in location)


def _make_simulator(backend: BackendKind):
    if backend is BackendKind.MPS:
        return lambda: AerSimulator(method="matrix_product_state")
    return AerSimulator


def run_original(
    instance: QUBOInstance,
    *,
    population_size: int,
    generations: int,
    seed: int,
    num_portas: int = 6,
    door_type: int = 2,
    backend: BackendKind = BackendKind.MPS,
    elite_size: float = 0.2,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.3,
    tournament_size: int = 3,
) -> OriginalResult:
    """Roda o QGA original no QUBO e devolve o melhor bitstring.

    ``door_type`` e ``num_portas`` são os defaults do repositório original
    (``qga.py``): Hilbert, 6 camadas. ``n_jobs`` fica em 1 de propósito: o
    experimento já paraleliza por execução, e o motor original serializa o
    ``self`` inteiro no joblib quando ``n_jobs > 1``.
    """

    if not isinstance(instance, QUBOInstance):
        raise TypeError(
            f"instance deve ser QUBOInstance, recebeu {type(instance).__name__}."
        )
    if isinstance(population_size, bool) or population_size < 2:
        raise ValueError("population_size deve ser int >= 2.")
    if isinstance(generations, bool) or generations < 1:
        raise ValueError("generations deve ser int >= 1.")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed deve ser int >= 0.")
    if door_type not in (0, 1, 2, 3, -1):
        raise ValueError(f"door_type inválido: {door_type}.")
    if not isinstance(backend, BackendKind):
        raise TypeError(
            f"backend deve ser BackendKind, recebeu {type(backend).__name__}."
        )
    if tournament_size > population_size:
        tournament_size = max(1, population_size)

    engine_module, individuo_module = load_engine()

    # N_QUBITS é global do módulo; 1 qubit por variável = modo binário.
    setattr(individuo_module, "N_QUBITS", 1)
    setattr(individuo_module, "AerSimulator", _make_simulator(backend))

    size = instance.S

    def fitness(values: list[float]) -> float:
        bits = "".join("1" if value >= 0.5 else "0" for value in values)
        return instance.fitness(bits)

    engine = engine_module.ParallelQuantumGA(
        fitness_function=fitness,
        fitness_function_name="portfolio-qubo",
        num_variaveis=size,
        num_portas=num_portas,
        door_type=door_type,
        domain=[(0.0, 1.0)] * size,
        minimize=False,
        population_size=population_size,
        generations=generations,
        elite_size=elite_size,
        selection_method="tournament",
        tournament_size=tournament_size,
        crossover_selection="vertical",
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        n_jobs=1,
        seed=seed,
        verbose=False,
        dataCollection=True,
    )

    start = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        raw = engine.run()

    elapsed = time.perf_counter() - start

    best = raw["best_individual"]
    if best is None:
        raise RuntimeError("o motor original não devolveu best_individual.")

    history = raw["history"]["best"]
    best_value = float(raw["best_fitness"])
    best_generation = next(
        (
            index
            for index, value in enumerate(history)
            if abs(float(value) - best_value) <= 1e-12
        ),
        len(history) - 1,
    )

    final_population = tuple(
        OriginalMember(
            bits=_bits_from_location(individual.location, size),
            fitness=float(individual.fitness),
        )
        for individual in raw["final_population"]
    )

    return OriginalResult(
        best_individual=OriginalMember(
            bits=_bits_from_location(best.location, size),
            fitness=best_value,
        ),
        best_generation=max(0, int(best_generation)),
        history={"best": [float(value) for value in history]},
        final_population=final_population,
        evaluations=population_size * (generations + 1),
        elapsed_s=elapsed,
    )
