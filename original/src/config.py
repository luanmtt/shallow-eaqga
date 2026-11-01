from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Conjuntos de portas (door_type) — nome → valor
#
# Usado pelo CLI (`--types classical,quantum`) e pelo nome dos subdiretórios
# de saída do batch. Mantenha em sincronia com Individuo.__init__.

DOOR_TYPES = {
    "classical_pure":   -1,   # 1Q apenas (X, I) — sem portas multi-qubit
    "classical":         0,   # reversível: + CNOT, SWAP, Toffoli, Fredkin (𝒢_cl do paper)
    "clifford":          1,   # + H, Y, Z, S, S† (sem T, sem 3Q)
    "quantum":           2,   # Hilbert/geral: + T, T† (𝒢_qu do paper)
    "no_entanglement":   3,   # portas quânticas 1Q apenas (superposição sem emaranhamento)
}

# valor → nome de exibição (para logs/relatórios)
DOOR_TYPE_NAMES = {
    -1: "Classical Pure (1Q only)",
     0: "Classical Gates",
     1: "Clifford Gates",
     2: "Hilbert Gates",
     3: "Quantum (no entanglement)",
}


@dataclass
class GAConfig:
    """
    Configuração centralizada para o Quantum Genetic Algorithm.
    
    Uso:
        config = GAConfig(
            num_variaveis=2,
            domain=[(-5.0, 5.0), (-1.0, 1.0)],
            sigma_es=True,
        )
        qga = ParallelQuantumGA(config=config, fitness_function=ackley)
    """

    # ── Fitness ────────────────────────────────────────────────────────────
    # True = avalia via statevector exato (𝔼[x] sem ruído de amostragem).
    # False = avalia via amostragem com shots (metodologia clássica do paper).
    exact_expectation: bool   = False

    # ── Entanglement inter-individual ──────────────────────────────────────
    # "none"    = indivíduos independentes (cada um avaliado isoladamente).
    # "pairwise" = indivíduos agrupados em pares que compartilham estados de
    #              Bell na inicialização; o par é simulado junto (2× qubits).
    #              Implica avaliação por shots (32 qubits inviável em statevector).
    entanglement_mode: str    = "none"

    # ── Indivíduo ──────────────────────────────────────────────────────────
    num_variaveis: int          = 2
    num_portas: int             = 6
    door_type: int              = 1                 # 0=Classic, 1=Clifford, 2=Hilbert
    domain: List[Tuple[float, float]] = field(default_factory=lambda: [(-1.0, 1.0)])

    # ── População ──────────────────────────────────────────────────────────
    population_size: int        = 50
    generations: int            = 50
    elite_size: float           = 0.2               # fração da população
    minimize: bool              = True

    # ── Seleção ────────────────────────────────────────────────────────────
    selection_method: str       = "tournament"       # tournament | roulette
    tournament_size: int        = 3

    # ── Crossover ──────────────────────────────────────────────────────────
    crossover_selection: str    = "vertical"         # vertical | horizontal | mixed | paper
    crossover_rate: float       = 0.7                # paper: 70%

    # ── Mutação ────────────────────────────────────────────────────────────
    mutation_rate: float        = 0.3                # paper: p_mut = 30% por porta
    # Janela progressiva de annealing na mutação (NÃO é do paper — variante própria).
    # False = paper-fiel: qualquer gene pode mutar. True = comportamento antigo
    # (congela colunas/qubits de alta significância conforme a evolução avança).
    annealing_window: bool      = False

    # ── Self-Adaptation (ES) ───────────────────────────────────────────────
    sigma_es: bool              = False
    sigma_es_initial: float     = 1.0

    # ── Paralelismo ────────────────────────────────────────────────────────
    n_jobs: int                 = -1
    seed: Optional[int]         = None

    # ── Display ────────────────────────────────────────────────────────────
    verbose: bool               = True
    continuous_bar: bool        = False             # Sobrescrever barra (para batch)
    log_best_each_generation: bool = False
    progress_label: str         = ""
    target_fitness: float       = 0.0

    # ── CSV Streaming ──────────────────────────────────────────────────────
    csv_flush: List             = field(default_factory=lambda: [True, 10])  # [ativo, frequência]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
