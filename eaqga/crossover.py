"""Crossover consciente de entrelaçamento do EAQGA.

O fluxo é:

1. comparar os dois bitstrings parentais e classificar pares P/N;
2. sortear pares com probabilidade ``p_s`` e fator de decaimento ``df(t)``;
3. unir os pares sorteados em uma floresta (sem ciclos, no máximo ``S - 1``
   arestas);
4. limitar o grau, se a topologia exigir;
5. orientar as arestas (controle → alvo);
6. agendar os CNOTs em camadas disjuntas.

O resultado é um :class:`Forest`, que o indivíduo usa para montar o circuito.
Este módulo não depende de Qiskit; ele só produz a estrutura.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Final

import numpy as np

from .config import ForestShape

__all__ = [
    "CorrelationKind",
    "Edge",
    "Forest",
    "ForestMetrics",
    "Pair",
    "build_crossover_forest",
    "build_forest",
    "cap_degree_pairs",
    "correlation_pairs",
    "decay_factor",
    "forest_metrics",
    "make_forest",
    "orient_forest",
    "pair_probability",
    "sample_pairs",
    "schedule_forest",
]

_SYMMETRY_ATOL: Final[float] = 1e-10
_NORMALIZATION_ATOL: Final[float] = 1e-9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CorrelationKind(str, Enum):
    """Tipo de correlação entre dois bits nos dois pais."""

    POSITIVE = "P"
    NEGATIVE = "N"


@dataclass(frozen=True, slots=True)
class Pair:
    """Par candidato a entrelaçamento, com ``i < j``."""

    i: int
    j: int
    kind: CorrelationKind

    def __post_init__(self) -> None:
        if isinstance(self.i, bool) or not isinstance(self.i, Integral):
            raise TypeError(f"i deve ser int, recebeu {type(self.i).__name__}.")
        if isinstance(self.j, bool) or not isinstance(self.j, Integral):
            raise TypeError(f"j deve ser int, recebeu {type(self.j).__name__}.")

        if self.i < 0 or self.j < 0:
            raise ValueError("i e j devem ser >= 0.")
        if self.i >= self.j:
            raise ValueError(f"exige i < j, recebeu i={self.i}, j={self.j}.")
        if not isinstance(self.kind, CorrelationKind):
            raise TypeError(
                "kind deve ser CorrelationKind, recebeu "
                f"{type(self.kind).__name__}."
            )

        object.__setattr__(self, "i", int(self.i))
        object.__setattr__(self, "j", int(self.j))


@dataclass(frozen=True, slots=True)
class Edge:
    """CNOT direcionado: ``control`` → ``target``."""

    control: int
    target: int
    kind: CorrelationKind

    def __post_init__(self) -> None:
        if isinstance(self.control, bool) or not isinstance(self.control, Integral):
            raise TypeError(
                "control deve ser int, recebeu "
                f"{type(self.control).__name__}."
            )
        if isinstance(self.target, bool) or not isinstance(self.target, Integral):
            raise TypeError(
                f"target deve ser int, recebeu {type(self.target).__name__}."
            )

        if self.control < 0 or self.target < 0:
            raise ValueError("control e target devem ser >= 0.")
        if self.control == self.target:
            raise ValueError("control e target devem ser diferentes.")
        if not isinstance(self.kind, CorrelationKind):
            raise TypeError(
                "kind deve ser CorrelationKind, recebeu "
                f"{type(self.kind).__name__}."
            )

        object.__setattr__(self, "control", int(self.control))
        object.__setattr__(self, "target", int(self.target))


@dataclass(frozen=True, slots=True)
class Forest:
    """Floresta orientada e agendada em camadas.

    ``edges`` são os CNOTs que sobreviveram ao limite de grau; ``layers`` é
    uma partição de ``edges`` em camadas com qubits disjuntos; ``selected_pairs``
    guarda os pares sorteados antes do limite de grau, para métricas.
    """

    edges: tuple[Edge, ...]
    layers: tuple[tuple[Edge, ...], ...]
    selected_pairs: tuple[Pair, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.edges, tuple):
            raise TypeError(
                f"edges deve ser tuple, recebeu {type(self.edges).__name__}."
            )
        if not isinstance(self.layers, tuple):
            raise TypeError(
                f"layers deve ser tuple, recebeu {type(self.layers).__name__}."
            )
        if not isinstance(self.selected_pairs, tuple):
            raise TypeError(
                "selected_pairs deve ser tuple, recebeu "
                f"{type(self.selected_pairs).__name__}."
            )

        if any(not isinstance(edge, Edge) for edge in self.edges):
            raise TypeError("edges deve conter apenas Edge.")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("edges contém duplicatas.")
        if any(not isinstance(pair, Pair) for pair in self.selected_pairs):
            raise TypeError("selected_pairs deve conter apenas Pair.")
        if len(set(self.selected_pairs)) != len(self.selected_pairs):
            raise ValueError("selected_pairs contém duplicatas.")

        flat: list[Edge] = []
        for layer in self.layers:
            if not isinstance(layer, tuple):
                raise TypeError("cada camada deve ser tuple.")
            if any(not isinstance(edge, Edge) for edge in layer):
                raise TypeError("cada camada deve conter apenas Edge.")
            if len(set(layer)) != len(layer):
                raise ValueError("camada contém CNOTs duplicados.")

            used: set[int] = set()
            for edge in layer:
                if edge.control in used or edge.target in used:
                    raise ValueError(
                        "camada com qubit repetido: "
                        f"{edge.control}->{edge.target}."
                    )
                used.update((edge.control, edge.target))

            flat.extend(layer)

        if len(flat) != len(self.edges):
            raise ValueError(
                "layers deve conter cada edge exatamente uma vez."
            )
        if set(flat) != set(self.edges):
            raise ValueError("layers e edges não têm o mesmo conjunto.")

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        return (
            f"Forest(edges={self.edge_count}, layers={self.layer_count}, "
            f"pairs={len(self.selected_pairs)})"
        )


@dataclass(frozen=True, slots=True)
class ForestMetrics:
    """Métricas resumidas de uma floresta de CNOTs."""

    edges: int
    layers: int
    selected_pairs: int
    chains: int
    max_chain_length: int
    max_degree: int
    positive_edges: int
    negative_edges: int

    @property
    def two_qubit_gates(self) -> int:
        return self.edges

    @property
    def depth(self) -> int:
        return self.layers

    def __post_init__(self) -> None:
        values = {
            "edges": self.edges,
            "layers": self.layers,
            "selected_pairs": self.selected_pairs,
            "chains": self.chains,
            "max_chain_length": self.max_chain_length,
            "max_degree": self.max_degree,
            "positive_edges": self.positive_edges,
            "negative_edges": self.negative_edges,
        }

        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(
                    f"{name} deve ser int, recebeu {type(value).__name__}."
                )
            if value < 0:
                raise ValueError(f"{name} deve ser >= 0, recebeu {value}.")

        if self.positive_edges + self.negative_edges != self.edges:
            raise ValueError(
                "positive_edges + negative_edges deve ser igual a edges."
            )
        if self.edges > 0 and self.layers < 1:
            raise ValueError("floresta com edges > 0 deve ter layers >= 1.")

    def __repr__(self) -> str:
        return (
            f"ForestMetrics(edges={self.edges}, layers={self.layers}, "
            f"chains={self.chains}, max_degree={self.max_degree})"
        )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Guards locais


def _require_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} deve ser int, recebeu {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} deve ser >= {minimum}, recebeu {value}.")
    return int(value)


def _require_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} deve ser float, recebeu {type(value).__name__}.")

    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} deve ser finito, recebeu {value}.")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} deve estar em [0, 1], recebeu {value}.")

    return number


def _require_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} deve ser float, recebeu {type(value).__name__}.")

    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} deve ser finito, recebeu {value}.")

    return number


def _require_bitstring(name: str, value: object, size: int | None = None) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} deve ser str, recebeu {type(value).__name__}.")
    if size is not None and len(value) != size:
        raise ValueError(
            f"{name} deve ter {size} caracteres, recebeu {len(value)}."
        )

    invalid = set(value) - {"0", "1"}
    if invalid:
        raise ValueError(
            f"{name} deve conter apenas '0' e '1'; inválidos: {sorted(invalid)}."
        )

    return value


def _require_rng(rng: object) -> random.Random:
    if not hasattr(rng, "random") or not callable(getattr(rng, "random")):
        raise TypeError(
            "rng deve ter um método random(), como random.Random."
        )
    return rng  # type: ignore[return-value]


def _require_forest_options(
    shape: object,
    max_degree: object,
) -> None:
    if not isinstance(shape, ForestShape):
        raise TypeError(
            f"shape deve ser ForestShape, recebeu {type(shape).__name__}."
        )

    if shape is ForestShape.STAR:
        if max_degree is not None:
            raise ValueError(
                "ForestShape.STAR não usa max_degree; passe max_degree=None."
            )
        return

    degree = _require_int("max_degree", max_degree, minimum=2)

    if shape is ForestShape.PATH and degree != 2:
        raise ValueError(
            f"ForestShape.PATH exige max_degree=2, recebeu {degree}."
        )


def _as_coupling(coupling: object, size: int) -> np.ndarray:
    try:
        array = np.asarray(coupling, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("coupling deve ser numérico e retangular.") from exc

    if array.shape != (size, size):
        raise ValueError(
            f"coupling deve ter shape=({size}, {size}), "
            f"recebeu {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("coupling contém NaN ou infinito.")
    if not np.allclose(array, array.T, atol=_SYMMETRY_ATOL, rtol=0.0):
        raise ValueError("coupling deve ser simétrica.")

    max_abs = float(np.max(np.abs(array)))
    if max_abs > 1.0 + _NORMALIZATION_ATOL:
        raise ValueError(
            "coupling deve ser normalizada em [-1, 1]; "
            f"max|.|={max_abs:.6g}."
        )

    return array


def _require_pairs(pairs: Sequence[Pair], size: int) -> tuple[Pair, ...]:
    if isinstance(pairs, (str, bytes)):
        raise TypeError("pairs deve ser uma sequência de Pair.")
    try:
        items = tuple(pairs)
    except TypeError as exc:
        raise TypeError("pairs deve ser iterável de Pair.") from exc

    for pair in items:
        if not isinstance(pair, Pair):
            raise TypeError(
                f"pairs deve conter Pair, recebeu {type(pair).__name__}."
            )
        if pair.i >= size or pair.j >= size:
            raise ValueError(
                f"par ({pair.i}, {pair.j}) fora do intervalo S={size}."
            )

    return items


def _require_acyclic_pairs(
    pairs: Sequence[Pair],
    size: int,
) -> tuple[Pair, ...]:
    """Valida que os pares formam uma floresta (sem ciclos)."""

    items = _require_pairs(pairs, size)

    parent = list(range(size))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for pair in items:
        first = find(pair.i)
        second = find(pair.j)

        if first == second:
            raise ValueError(
                "pairs contém ciclo; use build_forest antes de orientar."
            )

        parent[second] = first

    return items


def _compose_kind(
    first: CorrelationKind,
    second: CorrelationKind,
) -> CorrelationKind:
    """Composição de correlações: P∘P=P, N∘N=P, P∘N=N, N∘P=N."""

    if not isinstance(first, CorrelationKind):
        raise TypeError("first deve ser CorrelationKind.")
    if not isinstance(second, CorrelationKind):
        raise TypeError("second deve ser CorrelationKind.")

    return (
        CorrelationKind.POSITIVE
        if first is second
        else CorrelationKind.NEGATIVE
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# P/N e probabilidades


def correlation_pairs(
    first_bits: str,
    second_bits: str,
) -> tuple[Pair, ...]:
    """Classifica todos os pares ``i < j`` como P, N ou nenhum.

    P: os bits são iguais no primeiro pai e iguais no segundo.
    N: os bits são diferentes no primeiro pai e diferentes no segundo.
    """

    first = _require_bitstring("first_bits", first_bits)
    second = _require_bitstring(
        "second_bits",
        second_bits,
        size=len(first),
    )

    pairs: list[Pair] = []

    for i in range(len(first)):
        for j in range(i + 1, len(first)):
            if first[i] == first[j] and second[i] == second[j]:
                pairs.append(
                    Pair(i=i, j=j, kind=CorrelationKind.POSITIVE)
                )
            elif first[i] != first[j] and second[i] != second[j]:
                pairs.append(
                    Pair(i=i, j=j, kind=CorrelationKind.NEGATIVE)
                )

    return tuple(pairs)


def decay_factor(t: int, t_max: int) -> float:
    """``df(t) = 0.5 + t / (2 * t_max)``, com ``1 <= t <= t_max``."""

    limit = _require_int("t_max", t_max, minimum=1)
    current = _require_int("t", t, minimum=1)

    if current > limit:
        raise ValueError(f"t deve ser <= t_max={limit}, recebeu {current}.")

    return 0.5 + current / (2.0 * limit)


def pair_probability(
    kind: CorrelationKind,
    coupling_ij: float,
    t: int,
    t_max: int,
    p_s: float,
) -> float:
    """Probabilidade de selecionar um par, seguindo a fórmula do paper.

    P e coupling > 0  -> p_s * df * |coupling|
    P e coupling <= 0 -> p_s * |coupling|
    N e coupling < 0  -> p_s * |coupling|
    N e coupling >= 0 -> p_s * df * |coupling|

    O ``df`` começa em ~0.5 e chega a 1 no fim. A leitura textual do paper
    é ambígua; implementamos a fórmula como escrita.
    """

    if not isinstance(kind, CorrelationKind):
        raise TypeError(
            f"kind deve ser CorrelationKind, recebido {type(kind).__name__}."
        )

    coupling = _require_real("coupling_ij", coupling_ij)
    probability = _require_probability("p_s", p_s)
    df = decay_factor(t, t_max)

    magnitude = abs(coupling)

    if kind is CorrelationKind.POSITIVE:
        factor = df if coupling > 0.0 else 1.0
    else:
        factor = 1.0 if coupling < 0.0 else df

    return float(np.clip(probability * factor * magnitude, 0.0, 1.0))


def sample_pairs(
    first_bits: str,
    second_bits: str,
    coupling: object,
    t: int,
    t_max: int,
    p_s: float,
    rng: random.Random,
) -> tuple[Pair, ...]:
    """Sorteia Bernoulli por par candidato P/N."""

    first = _require_bitstring("first_bits", first_bits)
    second = _require_bitstring(
        "second_bits",
        second_bits,
        size=len(first),
    )

    matrix = _as_coupling(coupling, len(first))
    selected_rng = _require_rng(rng)

    # Valida t/t_max/p_s mesmo quando não existe nenhum par candidato.
    decay_factor(t, t_max)
    _require_probability("p_s", p_s)

    selected: list[Pair] = []

    for pair in correlation_pairs(first, second):
        probability = pair_probability(
            pair.kind,
            float(matrix[pair.i, pair.j]),
            t,
            t_max,
            p_s,
        )
        if selected_rng.random() < probability:
            selected.append(pair)

    return tuple(selected)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Floresta


def build_forest(
    pairs: Sequence[Pair],
    size: int,
) -> tuple[Pair, ...]:
    """Une os pares em uma floresta sem ciclos, com no máximo ``size - 1`` arestas."""

    total = _require_int("size", size, minimum=1)
    items = _require_pairs(pairs, total)

    parent = list(range(total))
    rank = [0] * total

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: int, second: int) -> bool:
        root_first = find(first)
        root_second = find(second)

        if root_first == root_second:
            return False

        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first

        parent[root_second] = root_first
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1

        return True

    forest: list[Pair] = []

    for pair in items:
        if len(forest) >= total - 1:
            break
        if union(pair.i, pair.j):
            forest.append(pair)

    return tuple(forest)


def cap_degree_pairs(
    pairs: Sequence[Pair],
    size: int,
    max_degree: int,
) -> tuple[Pair, ...]:
    """Remove pares que fariam um nó exceder ``max_degree``."""

    total = _require_int("size", size, minimum=1)
    degree_limit = _require_int("max_degree", max_degree, minimum=1)
    items = _require_pairs(pairs, total)

    degree = [0] * total
    kept: list[Pair] = []

    for pair in items:
        if degree[pair.i] >= degree_limit or degree[pair.j] >= degree_limit:
            continue
        kept.append(pair)
        degree[pair.i] += 1
        degree[pair.j] += 1

    return tuple(kept)


def _adjacency(
    pairs: Sequence[Pair],
    size: int,
) -> list[list[tuple[int, CorrelationKind]]]:
    adjacency: list[list[tuple[int, CorrelationKind]]] = [
        [] for _ in range(size)
    ]

    for pair in pairs:
        adjacency[pair.i].append((pair.j, pair.kind))
        adjacency[pair.j].append((pair.i, pair.kind))

    return adjacency


def _components(
    adjacency: list[list[tuple[int, CorrelationKind]]],
) -> list[list[int]]:
    seen: set[int] = set()
    components: list[list[int]] = []

    for start in range(len(adjacency)):
        if start in seen or not adjacency[start]:
            continue

        component: list[int] = []
        queue: deque[int] = deque([start])
        seen.add(start)

        while queue:
            node = queue.popleft()
            component.append(node)

            for neighbor, _ in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        components.append(component)

    return components


def orient_star(
    pairs: Sequence[Pair],
    size: int,
) -> tuple[Edge, ...]:
    """Orienta cada componente como estrela: um controle, todos os outros alvos.

    O tipo de uma aresta sintetizada é a paridade dos tipos no caminho até o
    controle. O controle é o nó de maior grau; empate pelo menor índice.
    """

    total = _require_int("size", size, minimum=1)
    items = _require_acyclic_pairs(pairs, total)
    adjacency = _adjacency(items, total)
    edges: list[Edge] = []

    for component in _components(adjacency):
        center = max(
            component,
            key=lambda node: (len(adjacency[node]), -node),
        )

        parity: dict[int, CorrelationKind] = {
            center: CorrelationKind.POSITIVE
        }
        queue: deque[int] = deque([center])

        while queue:
            node = queue.popleft()
            for neighbor, kind in adjacency[node]:
                if neighbor in parity:
                    continue
                parity[neighbor] = _compose_kind(parity[node], kind)
                queue.append(neighbor)

        for node in component:
            if node == center:
                continue
            edges.append(
                Edge(control=center, target=node, kind=parity[node])
            )

    return tuple(edges)


def orient_tree(
    pairs: Sequence[Pair],
    size: int,
) -> tuple[Edge, ...]:
    """Orienta cada componente como árvore enraizada no nó de maior grau."""

    total = _require_int("size", size, minimum=1)
    items = _require_acyclic_pairs(pairs, total)
    adjacency = _adjacency(items, total)
    edges: list[Edge] = []

    for component in _components(adjacency):
        root = max(
            component,
            key=lambda node: (len(adjacency[node]), -node),
        )

        parent: dict[int, int | None] = {root: None}
        queue: deque[int] = deque([root])

        while queue:
            node = queue.popleft()
            for neighbor, kind in adjacency[node]:
                if neighbor in parent:
                    continue
                parent[neighbor] = node
                edges.append(
                    Edge(control=node, target=neighbor, kind=kind)
                )
                queue.append(neighbor)

    return tuple(edges)


def orient_forest(
    pairs: Sequence[Pair],
    size: int,
    shape: ForestShape,
) -> tuple[Edge, ...]:
    """Orienta a floresta conforme a topologia escolhida."""

    if not isinstance(shape, ForestShape):
        raise TypeError(
            f"shape deve ser ForestShape, recebeu {type(shape).__name__}."
        )

    if shape is ForestShape.STAR:
        return orient_star(pairs, size)

    return orient_tree(pairs, size)


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Agendamento


def schedule_forest(
    edges: Sequence[Edge],
) -> tuple[tuple[Edge, ...], ...]:
    """Agenda CNOTs em camadas com qubits disjuntos (greedy)."""

    if isinstance(edges, (str, bytes)):
        raise TypeError("edges deve ser uma sequência de Edge.")

    try:
        items = tuple(edges)
    except TypeError as exc:
        raise TypeError("edges deve ser iterável de Edge.") from exc

    for edge in items:
        if not isinstance(edge, Edge):
            raise TypeError(
                f"edges deve conter Edge, recebeu {type(edge).__name__}."
            )

    if len(set(items)) != len(items):
        raise ValueError("edges contém duplicatas.")

    layers: list[list[Edge]] = []

    for edge in items:
        placed = False

        for layer in layers:
            used: set[int] = set()
            for placed_edge in layer:
                used.add(placed_edge.control)
                used.add(placed_edge.target)

            if edge.control not in used and edge.target not in used:
                layer.append(edge)
                placed = True
                break

        if not placed:
            layers.append([edge])

    return tuple(tuple(layer) for layer in layers)


def make_forest(
    edges: Sequence[Edge],
    selected_pairs: Sequence[Pair] = (),
) -> Forest:
    """Monta um :class:`Forest` a partir das arestas orientadas."""

    layers = schedule_forest(edges)
    return Forest(
        edges=tuple(edges),
        layers=layers,
        selected_pairs=tuple(selected_pairs),
    )


def forest_metrics(forest: Forest) -> ForestMetrics:
    """Calcula métricas de entanglement/estrutura da floresta.

    ``chains`` conta componentes conexas com pelo menos uma aresta;
    ``max_chain_length`` é o número de nós da maior componente; ``max_degree``
    é o grau máximo no grafo não direcionado.
    """

    if not isinstance(forest, Forest):
        raise TypeError(
            f"forest deve ser Forest, recebeu {type(forest).__name__}."
        )

    if not forest.edges:
        return ForestMetrics(
            edges=0,
            layers=0,
            selected_pairs=len(forest.selected_pairs),
            chains=0,
            max_chain_length=0,
            max_degree=0,
            positive_edges=0,
            negative_edges=0,
        )

    adjacency: dict[int, set[int]] = {}

    for edge in forest.edges:
        adjacency.setdefault(edge.control, set()).add(edge.target)
        adjacency.setdefault(edge.target, set()).add(edge.control)

    seen: set[int] = set()
    chains = 0
    max_chain_length = 0

    for start in adjacency:
        if start in seen:
            continue

        queue: deque[int] = deque([start])
        seen.add(start)
        length = 0

        while queue:
            node = queue.popleft()
            length += 1

            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        chains += 1
        max_chain_length = max(max_chain_length, length)

    positive_edges = sum(
        1
        for edge in forest.edges
        if edge.kind is CorrelationKind.POSITIVE
    )
    negative_edges = forest.edge_count - positive_edges

    return ForestMetrics(
        edges=forest.edge_count,
        layers=forest.layer_count,
        selected_pairs=len(forest.selected_pairs),
        chains=chains,
        max_chain_length=max_chain_length,
        max_degree=max(len(neighbors) for neighbors in adjacency.values()),
        positive_edges=positive_edges,
        negative_edges=negative_edges,
    )


def build_crossover_forest(
    first_bits: str,
    second_bits: str,
    coupling: object,
    t: int,
    t_max: int,
    p_s: float,
    *,
    shape: ForestShape = ForestShape.PATH,
    max_degree: int | None = 2,
    rng: random.Random | None = None,
) -> Forest:
    """Executa o crossover completo e devolve a floresta agendada.

    ``selected_pairs`` no resultado contém todos os pares sorteados; ``edges``
    contém apenas os que sobreviveram ao limite de grau e à união sem ciclos.
    Se ``rng`` for ``None``, usa ``random.Random(0)`` para reprodutibilidade;
    em experimentos, passe um RNG semeado explicitamente.
    """

    _require_forest_options(shape, max_degree)

    first = _require_bitstring("first_bits", first_bits)
    _require_bitstring("second_bits", second_bits, size=len(first))

    if rng is None:
        rng = random.Random(0)
    selected_rng = _require_rng(rng)

    selected_pairs = sample_pairs(
        first,
        second_bits,
        coupling,
        t,
        t_max,
        p_s,
        selected_rng,
    )

    forest_pairs = build_forest(selected_pairs, len(first))

    if shape is not ForestShape.STAR:
        if max_degree is None:
            raise ValueError(
                "max_degree não pode ser None para PATH/BOUNDED."
            )
        forest_pairs = cap_degree_pairs(
            forest_pairs,
            len(first),
            max_degree,
        )

    edges = orient_forest(forest_pairs, len(first), shape)

    return make_forest(edges, selected_pairs=selected_pairs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
