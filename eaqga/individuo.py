"""Indivíduo híbrido: genoma 1Q herdável + floresta de CNOTs.

O circuito é sempre o mesmo esqueleto:

```
H^S  →  colunas do genoma (1Q)  →  CNOTs da floresta  →  medida
```

O que é **herdado** é o genoma. A floresta é **reconstruída** a cada geração a
partir dos dois melhores bitstrings (Leitura A do plano), então ela não faz
parte do genoma. A mutação troca portas 1Q; os CNOTs não são tocados porque nem
estão no genoma.

Consequências que valem registro:

- a camada ``H^S`` é fixa e não é evoluída. Ela existe para que o genoma possa
  criar viés por **interferência**: uma segunda ``H`` leva ``|+>`` para ``|0>``,
  um ``X`` leva para ``|1>``, e ``S``/``T`` mudam fase sem mudar a marginal em
  ``Z``;
- só portas que alteram amplitude em ``Z`` (``H``, ``X``, ``Y``) criam viés
  visível na medida. ``S``/``T`` só importam se depois vier outra ``H`` ou um
  ``CNOT`` que converta fase em correlação;
- como **todas** as colunas 1Q vêm antes de todos os CNOTs, um qubit que é alvo
  de um CNOT perde a marginal que o genoma definiu (o CNOT sobrescreve), mas
  não perde a fase. Isso é o trade-off raso documentado em
  ``docs/leftovers.md`` §11.7.

O módulo é puro Qiskit/Aer: não conhece o laço evolutivo. Quem orquestra é
``population.py``/``runner.py``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Final

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .config import BackendKind, EvaluationMode, GateSet
from .crossover import CorrelationKind, Forest
from .fitness import Evaluation, evaluate_counts
from .instance import QUBOInstance

__all__ = [
    "GATE_SETS",
    "Genome",
    "GenomeMetrics",
    "HybridIndividual",
    "OneQGate",
    "build_hybrid_circuit",
    "crossover_genomes",
    "genome_metrics",
    "measure_genome",
    "mutate_genome",
    "random_genome",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class OneQGate(str, Enum):
    """Portas de 1 qubit disponíveis no genoma.

    O conjunto ``HILBERT_1Q`` inclui ``T``/``Tdg``: ``H`` + ``S`` + ``T`` é
    universal para 1 qubit, então o genoma alcança todo o Hilbert de 1 qubit.
    O conjunto ``CLIFFORD_1Q`` para em ``S``/``Sdg`` e fica no grupo de Clifford.
    """

    I = "i"
    H = "h"
    X = "x"
    Y = "y"
    Z = "z"
    S = "s"
    SDG = "sdg"
    T = "t"
    TDG = "tdg"


GATE_SETS: Final[dict[GateSet, tuple[OneQGate, ...]]] = {
    GateSet.CLIFFORD_1Q: (
        OneQGate.I,
        OneQGate.H,
        OneQGate.X,
        OneQGate.Y,
        OneQGate.Z,
        OneQGate.S,
        OneQGate.SDG,
    ),
    GateSet.HILBERT_1Q: (
        OneQGate.I,
        OneQGate.H,
        OneQGate.X,
        OneQGate.Y,
        OneQGate.Z,
        OneQGate.S,
        OneQGate.SDG,
        OneQGate.T,
        OneQGate.TDG,
    ),
}

_MUTATING_GATES: Final[dict[GateSet, tuple[OneQGate, ...]]] = {
    gate_set: tuple(gate for gate in gates if gate is not OneQGate.I)
    for gate_set, gates in GATE_SETS.items()
}


def _require_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} deve ser int, recebeu {type(value).__name__}.")
    if int(value) < minimum:
        raise ValueError(f"{name} deve ser >= {minimum}, recebeu {value}.")
    return int(value)


def _require_rate(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} deve ser float, recebeu {type(value).__name__}.")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} deve estar em [0, 1], recebeu {value}.")
    return number


def _require_enum(name: str, value: object, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        raise TypeError(
            f"{name} deve ser {enum_type.__name__}, recebeu "
            f"{type(value).__name__}."
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True, slots=True)
class Genome:
    """Genoma 1Q: ``num_columns`` colunas, uma porta por qubit em cada coluna.

    ``gates[column][qubit]``. Cada coluna é aplicada como uma camada (as portas
    são em qubits distintos por construção), então a profundidade 1Q é
    exatamente ``num_columns``.

    O genoma não guarda CNOTs. A floresta entra no circuito na hora de medir.
    """

    gates: tuple[tuple[OneQGate, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.gates, tuple):
            raise TypeError(
                f"gates deve ser tuple, recebeu {type(self.gates).__name__}."
            )
        if not self.gates:
            raise ValueError("gates não pode ser vazio.")

        width: int | None = None
        for column in self.gates:
            if not isinstance(column, tuple):
                raise TypeError("cada coluna deve ser tuple.")
            if not column:
                raise ValueError("coluna vazia: todo qubit precisa de uma porta.")
            if width is None:
                width = len(column)
            elif len(column) != width:
                raise ValueError(
                    "colunas com larguras diferentes: "
                    f"{width} e {len(column)}."
                )
            for gate in column:
                if not isinstance(gate, OneQGate):
                    raise TypeError(
                        "gates deve conter OneQGate, recebeu "
                        f"{type(gate).__name__}."
                    )

    @property
    def num_columns(self) -> int:
        return len(self.gates)

    @property
    def size(self) -> int:
        return len(self.gates[0])

    def gate(self, column: int, qubit: int) -> OneQGate:
        return self.gates[column][qubit]

    def with_gate(self, column: int, qubit: int, gate: OneQGate) -> "Genome":
        """Devolve uma cópia com uma porta trocada (o genoma é imutável)."""

        if not isinstance(gate, OneQGate):
            raise TypeError(
                f"gate deve ser OneQGate, recebeu {type(gate).__name__}."
            )
        if not 0 <= column < self.num_columns:
            raise IndexError(
                f"coluna {column} fora do intervalo [0, {self.num_columns})."
            )
        if not 0 <= qubit < self.size:
            raise IndexError(f"qubit {qubit} fora do intervalo [0, {self.size}).")

        columns = [list(column_gates) for column_gates in self.gates]
        columns[column][qubit] = gate
        return Genome(tuple(tuple(item) for item in columns))

    def count(self, gate: OneQGate) -> int:
        return sum(column.count(gate) for column in self.gates)

    def non_identity(self) -> int:
        return sum(
            1 for column in self.gates for gate in column if gate is not OneQGate.I
        )

    def to_columns(self) -> list[list[str]]:
        """Representação ``[coluna][qubit]`` em strings, para CSV/JSON."""

        return [[gate.value for gate in column] for column in self.gates]

    def to_dict(self) -> dict[str, object]:
        return {
            "num_columns": self.num_columns,
            "size": self.size,
            "columns": self.to_columns(),
        }

    @classmethod
    def from_columns(
        cls,
        columns: Sequence[Sequence[str | OneQGate]],
    ) -> "Genome":
        """Reconstrói o genoma a partir de strings (``"h"``, ``"t"``, ...)."""

        if isinstance(columns, (str, bytes)):
            raise TypeError("columns deve ser uma sequência de colunas.")

        parsed: list[tuple[OneQGate, ...]] = []
        for index, column in enumerate(columns):
            gates: list[OneQGate] = []
            for item in column:
                try:
                    gates.append(item if isinstance(item, OneQGate) else OneQGate(item))
                except ValueError as exc:
                    raise ValueError(
                        f"coluna {index}: porta desconhecida {item!r}."
                    ) from exc
            parsed.append(tuple(gates))

        return cls(tuple(parsed))

    def __str__(self) -> str:
        header = "     " + " ".join(f"c{c:<2d}" for c in range(self.num_columns))
        lines = [header]
        for qubit in range(self.size):
            cells = " ".join(
                f"{self.gates[column][qubit].name:<3s}"
                for column in range(self.num_columns)
            )
            lines.append(f"q{qubit:<3d} {cells}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class GenomeMetrics:
    """Resumo de um genoma, para histórico e CSV."""

    columns: int
    qubits: int
    non_identity: int
    h_gates: int
    t_gates: int
    identity: int

    @property
    def fill_ratio(self) -> float:
        total = self.identity + self.non_identity
        return self.non_identity / total if total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "columns": self.columns,
            "qubits": self.qubits,
            "non_identity": self.non_identity,
            "h_gates": self.h_gates,
            "t_gates": self.t_gates,
            "identity": self.identity,
            "fill_ratio": self.fill_ratio,
        }

    def __str__(self) -> str:
        return (
            f"{self.columns}c×{self.qubits}q "
            f"{self.non_identity}≠I ({self.fill_ratio:.0%}) "
            f"H={self.h_gates} T={self.t_gates}"
        )


def genome_metrics(genome: Genome) -> GenomeMetrics:
    """Conta as portas do genoma e devolve o resumo."""

    if not isinstance(genome, Genome):
        raise TypeError(
            f"genome deve ser Genome, recebeu {type(genome).__name__}."
        )

    non_identity = genome.non_identity()
    return GenomeMetrics(
        columns=genome.num_columns,
        qubits=genome.size,
        non_identity=non_identity,
        h_gates=genome.count(OneQGate.H),
        t_gates=genome.count(OneQGate.T) + genome.count(OneQGate.TDG),
        identity=genome.num_columns * genome.size - non_identity,
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Geração e variação


def random_genome(
    size: int,
    num_columns: int,
    gate_set: GateSet,
    rng: random.Random,
) -> Genome:
    """Sorteia um genoma ``num_columns × size`` no conjunto de portas dado."""

    total = _require_int("size", size, minimum=1)
    columns = _require_int("num_columns", num_columns, minimum=1)
    _require_enum("gate_set", gate_set, GateSet)
    if not isinstance(rng, random.Random):
        raise TypeError(f"rng deve ser random.Random, recebeu {type(rng).__name__}.")

    choices = GATE_SETS[gate_set]
    return Genome(
        tuple(
            tuple(rng.choice(choices) for _ in range(total))
            for _ in range(columns)
        )
    )


def mutate_genome(
    genome: Genome,
    rate: float,
    gate_set: GateSet,
    rng: random.Random,
    *,
    min_changes: int = 0,
) -> Genome:
    """Troca portas 1Q com probabilidade ``rate`` por gene.

    A troca é sempre por **outra** porta do conjunto: um gene sorteado nunca
    fica igual. ``min_changes`` garante um número mínimo de trocas, útil para
    não devolver um filho idêntico ao pai quando ``rate`` é baixo.

    Não existe mutação de CNOT: a floresta não está no genoma.
    """

    if not isinstance(genome, Genome):
        raise TypeError(
            f"genome deve ser Genome, recebeu {type(genome).__name__}."
        )
    probability = _require_rate("rate", rate)
    _require_enum("gate_set", gate_set, GateSet)
    if not isinstance(rng, random.Random):
        raise TypeError(f"rng deve ser random.Random, recebeu {type(rng).__name__}.")
    minimum = _require_int("min_changes", min_changes, minimum=0)

    choices = _MUTATING_GATES[gate_set]
    if not choices:
        raise ValueError(f"gate_set {gate_set.value} não tem portas mutáveis.")

    columns = [list(column) for column in genome.gates]
    changed = 0

    for column in range(genome.num_columns):
        for qubit in range(genome.size):
            if rng.random() >= probability:
                continue

            current = columns[column][qubit]
            replacements = tuple(gate for gate in choices if gate is not current)
            if not replacements:
                continue

            columns[column][qubit] = rng.choice(replacements)
            changed += 1

    while changed < minimum:
        column = rng.randrange(genome.num_columns)
        qubit = rng.randrange(genome.size)
        current = columns[column][qubit]
        replacements = tuple(gate for gate in choices if gate is not current)
        if not replacements:
            continue
        columns[column][qubit] = rng.choice(replacements)
        changed += 1

    return Genome(tuple(tuple(column) for column in columns))


def crossover_genomes(
    first: Genome,
    second: Genome,
    rng: random.Random,
    *,
    mix: float = 0.5,
) -> Genome:
    """Crossover uniforme gene a gene entre dois genomas.

    ``mix`` é a probabilidade de pegar o gene do **segundo** pai. Os dois
    genomas precisam ter o mesmo formato (``num_columns × size``).
    """

    if not isinstance(first, Genome) or not isinstance(second, Genome):
        raise TypeError("first e second devem ser Genome.")
    if first.gates and second.gates and (
        first.num_columns != second.num_columns or first.size != second.size
    ):
        raise ValueError(
            "genomas de formatos diferentes: "
            f"{first.num_columns}×{first.size} e "
            f"{second.num_columns}×{second.size}."
        )
    probability = _require_rate("mix", mix)
    if not isinstance(rng, random.Random):
        raise TypeError(f"rng deve ser random.Random, recebeu {type(rng).__name__}.")

    return Genome(
        tuple(
            tuple(
                second.gates[column][qubit]
                if rng.random() < probability
                else first.gates[column][qubit]
                for qubit in range(first.size)
            )
            for column in range(first.num_columns)
        )
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Circuito


def _apply_gate(circuit: QuantumCircuit, gate: OneQGate, qubit: int) -> None:
    if gate is OneQGate.I:
        return
    if gate is OneQGate.H:
        circuit.h(qubit)
    elif gate is OneQGate.X:
        circuit.x(qubit)
    elif gate is OneQGate.Y:
        circuit.y(qubit)
    elif gate is OneQGate.Z:
        circuit.z(qubit)
    elif gate is OneQGate.S:
        circuit.s(qubit)
    elif gate is OneQGate.SDG:
        circuit.sdg(qubit)
    elif gate is OneQGate.T:
        circuit.t(qubit)
    elif gate is OneQGate.TDG:
        circuit.tdg(qubit)
    else:  # pragma: no cover - o enum é fechado
        raise ValueError(f"porta não suportada: {gate!r}.")


def build_hybrid_circuit(
    genome: Genome,
    forest: Forest | None = None,
    *,
    measure: bool = True,
) -> QuantumCircuit:
    """Monta ``H^S → genoma → CNOTs → medida`` para um indivíduo.

    ``forest=None`` (ou floresta vazia) monta só a parte 1Q; é o que a
    população inicial usa.
    """

    if not isinstance(genome, Genome):
        raise TypeError(
            f"genome deve ser Genome, recebeu {type(genome).__name__}."
        )
    if not isinstance(measure, bool):
        raise TypeError(f"measure deve ser bool, recebeu {type(measure).__name__}.")

    size = genome.size
    circuit = QuantumCircuit(size, size)

    for qubit in range(size):
        circuit.h(qubit)

    for column in range(genome.num_columns):
        for qubit in range(size):
            _apply_gate(circuit, genome.gates[column][qubit], qubit)

    if forest is not None:
        if not isinstance(forest, Forest):
            raise TypeError(
                f"forest deve ser Forest ou None, recebeu "
                f"{type(forest).__name__}."
            )

        incoming: set[int] = set()

        for edge in forest.edges:
            if edge.control >= size or edge.target >= size:
                raise ValueError(
                    f"aresta {edge.control}->{edge.target} fora do intervalo "
                    f"S={size}."
                )
            if edge.control == edge.target:
                raise ValueError(f"aresta {edge.control}->{edge.target} é um laço.")
            if edge.target in incoming:
                raise ValueError(
                    "floresta com alvo múltiplo: "
                    f"{edge.target} recebe mais de um CNOT."
                )
            incoming.add(edge.target)

        for edge in forest.edges:
            if edge.kind is CorrelationKind.NEGATIVE:
                circuit.x(edge.target)
            circuit.cx(edge.control, edge.target)

    if measure:
        circuit.barrier()
        circuit.measure(range(size), range(size))

    return circuit


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Indivíduo


@dataclass(frozen=True, slots=True)
class HybridIndividual:
    """Indivíduo híbrido já medido.

    ``bits``/``fitness`` são do **melhor shot** (``k = 50`` por padrão);
    ``evaluation`` guarda todos os counts, a média e o desvio.
    """

    genome: Genome
    bits: str
    fitness: float
    evaluation: Evaluation
    forest: Forest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.genome, Genome):
            raise TypeError(
                f"genome deve ser Genome, recebeu {type(self.genome).__name__}."
            )
        if not isinstance(self.evaluation, Evaluation):
            raise TypeError(
                "evaluation deve ser Evaluation, recebeu "
                f"{type(self.evaluation).__name__}."
            )
        if len(self.genome.gates[0]) != self.evaluation.S:
            raise ValueError(
                f"genoma tem {self.genome.size} qubits e a evaluation "
                f"{self.evaluation.S}."
            )
        if self.bits != self.evaluation.best_bitstring:
            raise ValueError("bits deve ser o best_bitstring da evaluation.")
        if not isinstance(self.fitness, (int, float)) or isinstance(self.fitness, bool):
            raise TypeError(
                f"fitness deve ser float, recebeu {type(self.fitness).__name__}."
            )
        if self.forest is not None and not isinstance(self.forest, Forest):
            raise TypeError(
                "forest deve ser Forest ou None, recebeu "
                f"{type(self.forest).__name__}."
            )

        object.__setattr__(self, "fitness", float(self.fitness))

    @property
    def bits_int(self) -> int:
        return int(self.bits, 2)

    def __str__(self) -> str:
        return (
            f"HybridIndividual(bits={self.bits} fitness={self.fitness:.4f} "
            f"genoma={self.genome.num_columns}×{self.genome.size})"
        )


def _make_simulator(backend: BackendKind) -> AerSimulator:
    method = (
        "matrix_product_state"
        if backend is BackendKind.MPS
        else "statevector"
    )
    return AerSimulator(method=method)


def measure_genome(
    genome: Genome,
    forest: Forest | None,
    instance: QUBOInstance,
    *,
    shots: int,
    seed: int,
    simulator: AerSimulator | None = None,
    backend: BackendKind = BackendKind.STATEVECTOR,
    evaluation_mode: EvaluationMode = EvaluationMode.SCALAR,
) -> HybridIndividual:
    """Mede o indivíduo ``shots`` vezes e devolve o melhor shot.

    É o equivalente híbrido de ``baseline._measure_circuit``: um circuito,
    ``k`` shots, o melhor deles vira o representante e alimenta o elitismo.
    """

    if not isinstance(genome, Genome):
        raise TypeError(
            f"genome deve ser Genome, recebeu {type(genome).__name__}."
        )
    if not isinstance(instance, QUBOInstance):
        raise TypeError(
            f"instance deve ser QUBOInstance, recebeu {type(instance).__name__}."
        )
    if genome.size != instance.S:
        raise ValueError(
            f"genoma tem {genome.size} qubits, instância tem S={instance.S}."
        )

    total_shots = _require_int("shots", shots, minimum=1)
    seed_value = _require_int("seed", seed, minimum=0)
    _require_enum("backend", backend, BackendKind)
    _require_enum("evaluation_mode", evaluation_mode, EvaluationMode)

    if simulator is not None and not isinstance(simulator, AerSimulator):
        raise TypeError(
            "simulator deve ser AerSimulator ou None, recebeu "
            f"{type(simulator).__name__}."
        )

    engine = simulator if simulator is not None else _make_simulator(backend)
    circuit = build_hybrid_circuit(genome, forest)

    result = engine.run(
        circuit,
        shots=total_shots,
        seed_simulator=seed_value,
    ).result()

    evaluation = evaluate_counts(
        result.get_counts(),
        instance,
        reverse_bits=True,
        vectorized=evaluation_mode is EvaluationMode.VECTORIZED,
    )

    return HybridIndividual(
        genome=genome,
        bits=evaluation.best_bitstring,
        fitness=evaluation.best_fitness,
        evaluation=evaluation,
        forest=forest,
    )
