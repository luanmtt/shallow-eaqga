"""Baseline EAQGA fiel ao paper.

O baseline mantém a população como bitstrings: cada circuito é medido e a
bitstring resultante é avaliada pelo QUBO. Os dois melhores bitstrings viram
o pool; todos os ``N`` circuitos da próxima geração são gerados a partir deles.

Fidelidade ao Algorithm 1:

- ``H`` aparece **só na população inicial** (linhas 2-8). A partir da primeira
  geração, todo circuito nasce de ``|0>^S`` (``QC^p ← QC0``) e usa apenas
  ``RY``/``X``/``CX``. Não há camada ``H`` por geração;
- o ``RY`` do controle e o ``CX`` do par são aplicados **na ordem dos pares**
  (``SelectedEntangledQubits``), não em duas camadas separadas;
- qubits fora dos pares selecionados recebem ``RY`` no fim;
- ``X`` no alvo segue o tipo do par (``P``/``N``). A regra literal do
  pseudocódigo (``if xb1[n1] == 1: X(n2)``) fica atrás de
  :attr:`BaselineConfig.paper_x_rule`, porque ela inverte o alvo sempre que o
  bit do pai naquele alvo vale 1.

Diferenças em relação ao nosso híbrido:

- o baseline usa ``RY`` para enviesar os bits; não há genoma 1Q herdado;
- não há mutação;
- a população é toda nova a cada geração (sem cópia explícita dos pais).

O ``Forest`` do crossover é o mesmo dos dois algoritmos. A única diferença é
como o circuito é construído depois da floresta.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .config import BackendKind, BaselineConfig, EvaluationMode
from .crossover import (
    CorrelationKind,
    Forest,
    build_crossover_forest,
    forest_metrics,
)
from .fitness import Evaluation, evaluate_counts
from .instance import QUBOInstance

__all__ = [
    "BaselineIndividual",
    "BaselineResult",
    "build_baseline_circuit",
    "build_uniform_circuit",
    "run_baseline",
]


@dataclass(frozen=True, slots=True)
class BaselineIndividual:
    """Bitstring representativo de um circuito do baseline.

    ``bits`` é o melhor shot observado; ``fitness`` é o fitness desse shot.
    ``evaluation`` guarda os counts, a média e o desvio dos shots.
    """

    bits: str
    fitness: float
    evaluation: Evaluation
    forest: Forest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, Evaluation):
            raise TypeError(
                "evaluation deve ser Evaluation, recebeu "
                f"{type(self.evaluation).__name__}."
            )
        if not isinstance(self.bits, str):
            raise TypeError(
                f"bits deve ser str, recebeu {type(self.bits).__name__}."
            )
        if len(self.bits) != self.evaluation.S:
            raise ValueError(
                f"bits deve ter {self.evaluation.S} caracteres, "
                f"recebeu {len(self.bits)}."
            )
        if len(self.bits) != len(self.evaluation.best_bitstring):
            raise ValueError("bits e evaluation.best_bitstring divergem.")
        if self.bits != self.evaluation.best_bitstring:
            raise ValueError(
                "bits deve ser o best_bitstring da evaluation."
            )
        if not math.isfinite(self.fitness):
            raise ValueError("fitness deve ser finito.")
        if not math.isclose(
            self.fitness,
            self.evaluation.best_fitness,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "fitness deve ser igual a evaluation.best_fitness."
            )
        if self.forest is not None and not isinstance(self.forest, Forest):
            raise TypeError(
                "forest deve ser Forest ou None, recebeu "
                f"{type(self.forest).__name__}."
            )

        object.__setattr__(self, "fitness", float(self.fitness))


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Resultado de uma execução do baseline."""

    best_individual: BaselineIndividual
    best_generation: int
    history: dict[str, list[object]]
    final_population: tuple[BaselineIndividual, ...]
    evaluations: int
    elapsed_s: float

    @property
    def best_bits(self) -> str:
        return self.best_individual.bits

    @property
    def best_fitness(self) -> float:
        return self.best_individual.fitness


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Circuitos


def _require_bits(value: object, name: str = "bits") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} deve ser str, recebeu {type(value).__name__}.")
    if not value:
        raise ValueError(f"{name} não pode ser vazio.")

    invalid = set(value) - {"0", "1"}
    if invalid:
        raise ValueError(
            f"{name} deve conter apenas '0' e '1'; inválidos: {sorted(invalid)}."
        )

    return value


def _ry_angle(bit: str, p_a: float) -> float:
    """Ângulo que dá ``P(bit) = p_a`` no qubit medido."""

    _require_bits(bit)
    if len(bit) != 1:
        raise ValueError("bit deve ter exatamente 1 caractere.")

    if isinstance(p_a, bool) or not isinstance(p_a, (int, float)):
        raise TypeError(f"p_a deve ser float, recebeu {type(p_a).__name__}.")

    probability = float(p_a)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"p_a deve estar em (0, 1), recebeu {p_a}.")

    target = probability if bit == "0" else 1.0 - probability
    return 2.0 * math.acos(math.sqrt(target))


def build_uniform_circuit(size: int) -> QuantumCircuit:
    """Circuito da **população inicial**: ``H`` em todos os qubits e medição.

    Corresponde às linhas 2-8 do Algorithm 1. É o único ponto do baseline com
    ``H``; as gerações seguintes nascem de ``|0>^S``.
    """

    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"size deve ser int >= 1, recebeu {size!r}.")

    circuit = QuantumCircuit(size, size)

    for qubit in range(size):
        circuit.h(qubit)

    circuit.barrier()
    circuit.measure(range(size), range(size))
    return circuit


def build_baseline_circuit(
    best_bits: str,
    forest: Forest,
    p_a: float,
    *,
    paper_x_rule: bool = False,
) -> QuantumCircuit:
    """Monta o circuito de crossover do baseline fiel ao Algorithm 1.

    O circuito **não** tem ``H``: ele é construído a partir de ``|0>^S``, como
    na linha ``QC^p ← QC0`` do paper. A camada ``H`` existe apenas na população
    inicial (:func:`build_uniform_circuit`).

    Para cada par ``(n1, n2)``, na ordem em que foi selecionado:

    - ``RY`` no controle ``n1`` com o ângulo do bit de ``best_bits[n1]``;
    - ``X`` no alvo se o par for ``NEGATIVE`` (ou se ``paper_x_rule``);
    - ``CX(n1, n2)``.

    Um qubit que aparece como controle em vários pares recebe ``RY`` **uma
    vez**, antes do primeiro ``CX`` que ele controla; repetir a mesma rotação
    k vezes equivaleria a ``RY(k·θ)``, o que afastaria a probabilidade de
    ``p_a``. Qubits fora de todos os pares recebem ``RY`` no fim.

    ``paper_x_rule=True`` reproduz a literalidade do pseudocódigo:
    ``if xb1[n1] == 1: X(n2)``. Isso **inverte** o alvo quando o bit do pai no
    alvo é 1, então o default é a regra por tipo de par (``P`` não inverte,
    ``N`` inverte), que é a que conserva a correlação observada nos pais.
    """

    bits = _require_bits(best_bits, "best_bits")
    if not isinstance(forest, Forest):
        raise TypeError(
            f"forest deve ser Forest, recebeu {type(forest).__name__}."
        )

    size = len(bits)
    if isinstance(p_a, bool) or not isinstance(p_a, (int, float)):
        raise TypeError(f"p_a deve ser float, recebeu {type(p_a).__name__}.")
    if not 0.0 < float(p_a) < 1.0:
        raise ValueError(f"p_a deve estar em (0, 1), recebido {p_a}.")

    if not isinstance(paper_x_rule, bool):
        raise TypeError(
            "paper_x_rule deve ser bool, recebeu "
            f"{type(paper_x_rule).__name__}."
        )

    incoming: set[int] = set()

    for edge in forest.edges:
        if edge.control >= size or edge.target >= size:
            raise ValueError(
                f"aresta {edge.control}->{edge.target} fora do intervalo "
                f"S={size}."
            )
        if edge.control == edge.target:
            raise ValueError(
                f"aresta {edge.control}->{edge.target} é um laço."
            )
        if edge.target in incoming:
            raise ValueError(
                "floresta com alvo múltiplo: "
                f"{edge.target} recebe mais de um CNOT."
            )
        incoming.add(edge.target)

    # |0>^S: o H do paper é só a população inicial, não a crossover.
    circuit = QuantumCircuit(size, size)

    rotated: set[int] = set()
    entangled: set[int] = set()

    for edge in forest.edges:
        entangled.add(edge.control)
        entangled.add(edge.target)

        if edge.control not in rotated:
            circuit.ry(_ry_angle(bits[edge.control], float(p_a)), edge.control)
            rotated.add(edge.control)

        flip = (
            bits[edge.control] == "1"
            if paper_x_rule
            else edge.kind is CorrelationKind.NEGATIVE
        )
        if flip:
            circuit.x(edge.target)

        circuit.cx(edge.control, edge.target)

    for qubit in range(size):
        if qubit not in entangled:
            circuit.ry(_ry_angle(bits[qubit], float(p_a)), qubit)

    circuit.barrier()
    circuit.measure(range(size), range(size))
    return circuit


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Execução


def _make_simulator(config: BaselineConfig) -> AerSimulator:
    method = (
        "matrix_product_state"
        if config.backend is BackendKind.MPS
        else "statevector"
    )
    return AerSimulator(method=method)


def _top_two(
    population: Sequence[BaselineIndividual],
) -> tuple[BaselineIndividual, BaselineIndividual]:
    if len(population) < 2:
        raise ValueError(
            "população precisa de pelo menos 2 indivíduos para o top-2."
        )

    ranked = sorted(
        population,
        key=lambda individual: individual.fitness,
        reverse=True,
    )
    return ranked[0], ranked[1]


def _measure_circuit(
    circuit: QuantumCircuit,
    simulator: AerSimulator,
    instance: QUBOInstance,
    config: BaselineConfig,
    *,
    forest: Forest | None,
    seed_simulator: int,
) -> BaselineIndividual:
    result = simulator.run(
        circuit,
        shots=config.shots,
        seed_simulator=seed_simulator,
    ).result()
    counts = result.get_counts()

    evaluation = evaluate_counts(
        counts,
        instance,
        reverse_bits=True,
        vectorized=config.evaluation_mode is EvaluationMode.VECTORIZED,
    )

    return BaselineIndividual(
        bits=evaluation.best_bitstring,
        fitness=evaluation.best_fitness,
        evaluation=evaluation,
        forest=forest,
    )


def run_baseline(
    instance: QUBOInstance,
    config: BaselineConfig,
) -> BaselineResult:
    """Executa o baseline completo e devolve histórico e melhor indivíduo.

    O representante de cada circuito é o melhor shot. Com ``shots=1`` (default
    do paper), isso é simplesmente o único bitstring medido. Com ``shots>1``,
    a média/desvio dos shots ficam em ``evaluation``, mas o ranking usa o
    melhor shot.
    """

    if not isinstance(instance, QUBOInstance):
        raise TypeError(
            f"instance deve ser QUBOInstance, recebeu {type(instance).__name__}."
        )
    if not isinstance(config, BaselineConfig):
        raise TypeError(
            f"config deve ser BaselineConfig, recebeu {type(config).__name__}."
        )

    simulator = _make_simulator(config)
    rng = random.Random(config.seed)

    circuit_index = 0
    evaluations = 0
    start = time.perf_counter()

    uniform = build_uniform_circuit(instance.S)
    initial_population: list[BaselineIndividual] = []

    for _ in range(config.population_size):
        individual = _measure_circuit(
            uniform,
            simulator,
            instance,
            config,
            forest=None,
            seed_simulator=config.seed + circuit_index,
        )
        circuit_index += 1
        evaluations += 1
        initial_population.append(individual)

    initial_best = max(
        initial_population,
        key=lambda individual: individual.fitness,
    )
    best_so_far = initial_best
    best_generation = 0

    initial_values = [ind.fitness for ind in initial_population]
    history: dict[str, list[object]] = {
        "generation": [0],
        "best": [best_so_far.fitness],
        "generation_best": [initial_best.fitness],
        "avg": [statistics.fmean(initial_values)],
        "median": [statistics.median(initial_values)],
        "best_bits": [best_so_far.bits],
        "forest_metrics": [None],
    }

    pool = _top_two(initial_population)
    population = initial_population

    for generation in range(1, config.generations + 1):
        first, second = pool
        new_population: list[BaselineIndividual] = []

        for _ in range(config.population_size):
            forest = build_crossover_forest(
                first.bits,
                second.bits,
                instance.coupling,
                t=generation,
                t_max=config.generations,
                p_s=config.p_s,
                shape=config.forest_shape,
                max_degree=config.max_degree,
                rng=rng,
            )
            circuit = build_baseline_circuit(
                first.bits,
                forest,
                config.p_a,
                paper_x_rule=config.paper_x_rule,
            )
            individual = _measure_circuit(
                circuit,
                simulator,
                instance,
                config,
                forest=forest,
                seed_simulator=config.seed + circuit_index,
            )
            circuit_index += 1
            evaluations += 1
            new_population.append(individual)

        population = new_population

        current_best = max(
            population,
            key=lambda individual: individual.fitness,
        )
        if current_best.fitness > best_so_far.fitness:
            best_so_far = current_best
            best_generation = generation

        values = [ind.fitness for ind in population]
        history["generation"].append(generation)
        history["best"].append(best_so_far.fitness)
        history["generation_best"].append(current_best.fitness)
        history["avg"].append(statistics.fmean(values))
        history["median"].append(statistics.median(values))
        history["best_bits"].append(best_so_far.bits)
        history["forest_metrics"].append(
            forest_metrics(current_best.forest)
            if current_best.forest is not None
            else None
        )

        next_pool = _top_two(population)
        pool = (
            _top_two(pool + next_pool)
            if config.accumulated_pool
            else next_pool
        )

    elapsed = time.perf_counter() - start

    return BaselineResult(
        best_individual=best_so_far,
        best_generation=best_generation,
        history=history,
        final_population=tuple(population),
        evaluations=evaluations,
        elapsed_s=elapsed,
    )
