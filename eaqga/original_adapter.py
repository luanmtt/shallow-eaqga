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
4. **Seed.** O original chama ``simulator.run(...)`` sem ``seed_simulator`` e
   ``mutacao`` sem ``rng`` (cai em ``secrets.randbits``). O adapter injeta uma
   sequência determinística de seeds por execução, tanto na simulação quanto na
   mutação, sem alterar a lógica do algoritmo.

O que **não** mudamos: seleção, crossover, mutação, elitismo, número de portas,
conjunto de portas e o loop de gerações — tudo do original.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import itertools
import random
import statistics
import sys
import time
from collections.abc import Iterator
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
_PRISTINE_CALCULAR = None
_PRISTINE_MUTACAO = None


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

    global _PRISTINE_CALCULAR, _PRISTINE_MUTACAO
    _PRISTINE_CALCULAR = individuo_module.Individuo.calcular_fitness
    _PRISTINE_MUTACAO = individuo_module.Individuo.mutacao

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
    history: dict[str, list[object]]
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


def _make_simulator(
    backend: BackendKind,
    seeds: Iterator[int],
):
    """Fábrica de ``AerSimulator`` que injeta um ``seed_simulator`` por run.

    O código original chama ``simulator.run(qc, shots=...)`` sem seed, então
    duas execuções com o mesmo seed do algoritmo divergem. Aqui cada chamada
    ao simulador recebe o próximo inteiro da sequência, que é reiniciada a
    cada :func:`run_original`.
    """

    method = (
        "matrix_product_state"
        if backend is BackendKind.MPS
        else None
    )

    def factory():
        simulator = (
            AerSimulator(method=method)
            if method is not None
            else AerSimulator()
        )
        run = simulator.run

        def seeded_run(circuit, **kwargs):
            kwargs.setdefault("seed_simulator", next(seeds) % (2**31 - 1))
            return run(circuit, **kwargs)

        simulator.run = seeded_run
        return simulator

    return factory


def run_original(
    instance: QUBOInstance,
    *,
    population_size: int,
    generations: int,
    seed: int,
    num_portas: int = 6,
    door_type: int = 2,
    shots: int | None = None,
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

    ``shots=None`` mantém o default do original (1024 por circuito). Um inteiro
    troca **só** o número de shots, via wrapper do método ``calcular_fitness``;
    a lógica de avaliação continua a mesma.
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
    if shots is not None and (isinstance(shots, bool) or shots < 1):
        raise ValueError("shots deve ser int >= 1 ou None.")

    engine_module, individuo_module = load_engine()

    # N_QUBITS é global do módulo; 1 qubit por variável = modo binário.
    setattr(individuo_module, "N_QUBITS", 1)
    simulation_seeds = itertools.count(seed * 1_000_003 + 1)
    setattr(
        individuo_module,
        "AerSimulator",
        _make_simulator(backend, simulation_seeds),
    )

    base_calcular = _PRISTINE_CALCULAR
    if base_calcular is None:  # pragma: no cover - load_engine sempre define
        raise RuntimeError("motor original não carregado corretamente.")

    base_mutacao = _PRISTINE_MUTACAO
    if base_mutacao is None:  # pragma: no cover - load_engine sempre define
        raise RuntimeError("motor original não carregado corretamente.")

    mutation_seeds = itertools.count(seed * 2_000_003 + 1)

    def mutacao(self, taxa_mutacao_, geracao_atual, max_geracoes, rng=None):
        # ``_create_offspring_task`` chama sem rng; o original usa
        # ``secrets.randbits``, que ignora o seed do experimento.
        if rng is None:
            rng = random.Random(next(mutation_seeds))
        return base_mutacao(
            self,
            taxa_mutacao_,
            geracao_atual,
            max_geracoes,
            rng,
        )

    setattr(individuo_module.Individuo, "mutacao", mutacao)

    if shots is None:
        setattr(individuo_module.Individuo, "calcular_fitness", base_calcular)
    else:
        def calcular(self, fitness_function, shots=shots, dataCollection=False, exact=False):
            return base_calcular(
                self,
                fitness_function,
                shots=shots,
                dataCollection=dataCollection,
                exact=exact,
            )

        setattr(individuo_module.Individuo, "calcular_fitness", calcular)

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

    raw_history = raw["history"]
    best_history = [float(value) for value in raw_history["best"]]
    length = len(best_history)

    def _as_float_column(key: str) -> list[object]:
        values = [float(value) for value in (raw_history.get(key) or [])[:length]]
        return [*values, *([None] * (length - len(values)))]

    avg_history = _as_float_column("avg")
    median_history = _as_float_column("median")

    locations = raw_history.get("best_location") or []
    best_bits: list[object] = [
        (
            _bits_from_location(list(location), size)
            if location is not None
            else None
        )
        for location in locations[:length]
    ]
    best_bits += [None] * (length - len(best_bits))

    # O `run()` avalia a população final depois do laço, mas não registra essa
    # avaliação no histórico. Ela é a geração `T`: sem ela, o melhor fitness
    # final não aparece no CSV e `best_generation` fica preso em `T-1`.
    final_values = [
        float(individual.fitness)
        for individual in raw["final_population"]
        if individual.fitness is not None
    ]
    best_value = float(raw["best_fitness"])
    best_history.append(best_value)
    avg_history.append(
        statistics.fmean(final_values) if final_values else None
    )
    median_history.append(
        statistics.median(final_values) if final_values else None
    )
    best_bits.append(_bits_from_location(best.location, size))

    best_generation = next(
        (
            index
            for index, value in enumerate(best_history)
            if abs(value - best_value) <= 1e-12
        ),
        len(best_history) - 1,
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
        history={
            "best": best_history,
            "avg": avg_history,
            "median": median_history,
            "best_bits": best_bits,
        },
        final_population=final_population,
        evaluations=population_size * (generations + 1),
        elapsed_s=elapsed,
    )
