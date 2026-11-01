import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE
for _ in range(5):
    if (_PROJECT_ROOT / "utils" / "embellish.py").exists():
        break
    _PROJECT_ROOT = _PROJECT_ROOT.parent
if not (_PROJECT_ROOT / "utils" / "embellish.py").exists():
    _PROJECT_ROOT = _HERE.parents[1]

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_PROJECT_ROOT))

from src.Individuo import *
from config import DOOR_TYPE_NAMES, DOOR_TYPES
from metrics import measurement_entropy, gate_entropy, population_diversity

from typing import List, Tuple, Callable, Dict, Union, Optional, cast
from joblib import Parallel, delayed
from utils.progressbar import ProgressBar
from utils.formats import formatLargeNumber, formatMantissa
from utils.embellish import *
import numpy as np
import random
import secrets
import copy
import math
import time
import csv
import shutil
from datetime import datetime


C = "#f5c2e7"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _avaliar_par(a: 'Individuo', b: 'Individuo',
                 fitness_function: Callable[[List[float]], float],
                 dataCollection: bool) -> Tuple['Individuo', 'Individuo']:
    """Função de módulo (picklable p/ joblib): avalia um par entrelaçado junto."""
    if a.fitness is None and b.fitness is None:
        a.calcular_fitness_par(b, fitness_function, dataCollection=dataCollection)
    elif a.fitness is None:
        a.calcular_fitness(fitness_function, dataCollection=dataCollection)
    elif b.fitness is None:
        b.calcular_fitness(fitness_function, dataCollection=dataCollection)
    return a, b


def _create_offspring_task(parent1: 'Individuo', parent2: 'Individuo',
                           crossover_selection: str, do_crossover: bool,
                           annealing_window: bool, sigma_es: bool,
                           mutation_rate: float,
                           generations: int) -> Tuple['Individuo', 'Individuo', int]:
    """Função de módulo (picklable p/ joblib): cria um par de filhos.

    Recebe apenas dados imutáveis + os pais — nunca a instância do GA.
    (Serializar bound method arrastava o ParallelQuantumGA inteiro, com
    population/history/_rng mutáveis, e causou SEGV no feeder do loky.)
    """
    if do_crossover:
        child1, child2 = parent1.crossover(parent2, method=crossover_selection)
    else:
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        child1.geracao += 1
        child2.geracao += 1

    # Propagar a flag de annealing_window e sigma_es_ativo para os filhos
    # (crossover cria Individuo com defaults; são propriedades da evolução).
    child1.annealing_window = annealing_window
    child2.annealing_window = annealing_window
    child1.sigma_es_ativo = sigma_es
    child2.sigma_es_ativo = sigma_es

    mut1 = child1.mutacao(mutation_rate, parent1.geracao + 1, generations)
    mut2 = child2.mutacao(mutation_rate, parent2.geracao + 1, generations)

    return child1, child2, mut1 + mut2


def _evaluate_individual_task(individual: 'Individuo',
                              fitness_function: Callable[[List[float]], float],
                              dataCollection: bool,
                              exact: bool) -> 'Individuo':
    """Função de módulo (picklable p/ joblib): avalia um indivíduo."""
    if individual.fitness is None:
        individual.calcular_fitness(fitness_function,
                                    dataCollection=dataCollection,
                                    exact=exact)
    return individual


class ParallelQuantumGA:

    __slots__ = [

        'fitness_function',
        'fitness_function_name',
        'num_variaveis', 
        'num_portas', 
        'door_type', 
        'domain',
        'minimize', 
        'population_size', 
        'generations', 
        'elite_size', 
        'selection_method',
        'crossover_selection',
        'tournament_size', 
        'crossover_rate', 
        'mutation_rate', 
        'annealing_window',
        'sigma_es',
        'exact_expectation',
        'entanglement_mode',
        'pairs',
        'n_jobs',
        'population', 
        'best_individual', 
        'best_fitness', 
        'history', 
        '_rng', 
        'verbose',
        'log_best_each_generation',
        'progress_label',
        'seed',
        'target_fitness',
        'optimum',
        'optimum_label',
        'exp_retrieval',
        'csv_flush',
        'advanced_graphs',
        'dataCollection',
        'continuous_bar',
        '_gen_bar',
        '_fit_bar',
        '_batch_bar',
        '_run_dir'
    ]


    def __init__(self,  fitness_function: Callable[[List[float]], float],
                        fitness_function_name: str = "",
                        num_variaveis: int      = 1, 
                        num_portas: int         = 10, 
                        door_type:  int         = 1,
                        domain: Optional[List[Tuple[float, float]]] = None, 
                        minimize: bool          = True,
                        population_size: int    = 50, 
                        generations: int        = 50, 
                        elite_size: float       = 0.2,
                        selection_method: str   = 'tournament',
                        crossover_selection: str = 'vertical',
                        tournament_size: int    = 3,
                        crossover_rate: float   = 1.0, 
                        mutation_rate: float    = 0.1,
                        annealing_window: bool  = False,
                        sigma_es: bool          = False,
                        exact_expectation: bool = False,
                        entanglement_mode: str  = "none",
                        n_jobs: int             = -1, 
                        seed: Optional[int]    = None, 
                        dataCollection: bool    = False,

                        verbose: bool           = True,
                        continuous_bar: bool    = False,
                        log_best_each_generation: bool = False,
                        progress_label: str     = "",
                        target_fitness: float   = 0.0,
                        optimum: Optional[Tuple[float, ...]] = None,
                        optimum_label: str      = "",
                        exp_retrieval: Optional[List] = None,
                        csv_flush: Optional[List] = None,
                        advanced_graphs: bool   = False
                 ):

        self.seed               = seed if seed is not None else secrets.randbits(32)
        self.fitness_function   = fitness_function
        self.fitness_function_name  = fitness_function_name
        self.num_variaveis      = num_variaveis
        self.num_portas         = num_portas
        self.door_type          = door_type
        self.domain             = domain if domain else [(-1.0, 1.0)] * num_variaveis
        self.minimize           = minimize
        self.population_size    = population_size
        self.generations        = generations
        if not isinstance(elite_size, (int, float)) or not 0.0 < float(elite_size) <= 1.0:
            raise ValueError(f"elite_size deve ser fração em (0, 1], got {elite_size!r}")
        self.elite_size = max(1, int(elite_size * population_size))
        self.selection_method   = selection_method
        self.crossover_selection = crossover_selection
        if self.crossover_selection not in ('vertical', 'horizontal', 'mixed', 'paper'):
            raise ValueError(f"Invalid crossover_selection: {self.crossover_selection}. Choose 'vertical', 'horizontal', 'mixed' or 'paper'.")
        self.tournament_size    = tournament_size
        self.crossover_rate     = crossover_rate
        self.mutation_rate      = mutation_rate
        self.annealing_window   = annealing_window
        self.sigma_es           = sigma_es
        self.exact_expectation  = exact_expectation

        self.verbose                    = verbose
        self.continuous_bar             = continuous_bar
        self.log_best_each_generation   = log_best_each_generation
        self.progress_label             = progress_label
        self.target_fitness             = target_fitness
        self.optimum                    = optimum
        self.optimum_label              = optimum_label
        
        # exp_retrieval: [ativo, intervalo]
        # [True, N] = calcula a cada N gerações
        # [False, N] = calcula apenas no final
        if exp_retrieval is None:
            exp_retrieval = [False, 1]
        
        # Validar exp_retrieval
        exp_active, exp_interval = exp_retrieval
        if not isinstance(exp_active, bool):
            raise ValueError(f"exp_retrieval[0] must be bool, got {type(exp_active)}")
        if not isinstance(exp_interval, int) or exp_interval < 1:
            raise ValueError(f"exp_retrieval[1] must be positive int, got {exp_interval}")
        if exp_active and exp_interval > generations:
            print(f"  {stylePrint('Warning', '#f3af41')}: exp_retrieval interval ({exp_interval}) > generations ({generations}). Will only collect at end.")
            exp_active = False
        
        self.exp_retrieval = [exp_active, exp_interval]
        self.advanced_graphs = advanced_graphs

        # csv_flush: [ativo, frequência]
        # [True, N] = flush a cada N simulações
        # [False, 0] = sem flush (buffer Python)
        if csv_flush is None:
            csv_flush = [True, 10]
        
        # Validar csv_flush
        flush_active, flush_freq = csv_flush
        if not isinstance(flush_active, bool):
            raise ValueError(f"csv_flush[0] must be bool, got {type(flush_active)}")
        if not isinstance(flush_freq, int) or flush_freq < 0:
            raise ValueError(f"csv_flush[1] must be non-negative int, got {flush_freq}")
        
        self.csv_flush = [flush_active, flush_freq]

        self.n_jobs = n_jobs
        self.dataCollection = dataCollection

        # ── Entanglement inter-individual ───────────────────────────────────
        if entanglement_mode not in ("none", "pairwise"):
            raise ValueError(f"Invalid entanglement_mode: {entanglement_mode}. Choose 'none' or 'pairwise'.")
        if entanglement_mode == "pairwise" and exact_expectation:
            raise ValueError("entanglement_mode='pairwise' exige simulação por shots "
                             "(32 qubits/par é inviável em statevector exato). Desative exact_expectation.")
        self.entanglement_mode = entanglement_mode
        self.pairs = None
        if entanglement_mode == "pairwise":
            self.pairs = self._sorteiar_pares()

        self._validate_config()

        self._rng = random.Random(self.seed)
        
        np.random.seed(self.seed)

        self.population = self._initialize_population()
        self.best_individual = None
        self.best_fitness = float('inf') if minimize else -float('inf')
        self.history = {
            'best': [], 
            'avg': [], 
            'median': [],
            'best_individual': [],
            'best_location': [],
            'best_gates': [],
            'gate_entropy': [],
            'mutations': [],
            'centroid_distance': [],
            'population_radius': [],
            'pairwise_distance': [],
            'diversity_generations': []  # Gerações onde diversidade foi coletada
        }

        self._gen_bar = ProgressBar(total=1, style="keras", fg_color="#CC3363", bg_color="#5b5f66")
        self._fit_bar = ProgressBar(total=1, style="keras", fg_color="#f6cdb8", bg_color="#5b5f66")
        self._batch_bar = ProgressBar(total=1, style="keras", fg_color="#dcd7ba", bg_color="#5b5f66")
        self._run_dir = None
    


    # ──────────────────────────────────────────────────────────────────────────────────────────────────
    # métodos


    def _validate_config(self) -> None:
        """Valida hiperparâmetros cedo (fail-fast com mensagem clara)."""
        if not isinstance(self.population_size, int) or self.population_size < 2:
            raise ValueError(f"population_size deve ser int >= 2, got {self.population_size!r}")
        if not isinstance(self.generations, int) or self.generations < 1:
            raise ValueError(f"generations deve ser int >= 1, got {self.generations!r}")
        if not isinstance(self.num_portas, int) or self.num_portas < 1:
            raise ValueError(f"num_portas deve ser int >= 1, got {self.num_portas!r}")
        if not isinstance(self.num_variaveis, int) or self.num_variaveis < 1:
            raise ValueError(f"num_variaveis deve ser int >= 1, got {self.num_variaveis!r}")
        if not isinstance(self.elite_size, int) or not 1 <= self.elite_size <= self.population_size:
            raise ValueError(f"elite_size efetivo fora de [1, population_size], got {self.elite_size!r}")
        for rate_name in ('mutation_rate', 'crossover_rate'):
            rate = getattr(self, rate_name)
            if not isinstance(rate, (int, float)) or not 0.0 <= float(rate) <= 1.0:
                raise ValueError(f"{rate_name} deve estar em [0, 1], got {rate!r}")
        if self.selection_method not in ('tournament', 'roulette'):
            raise ValueError(f"selection_method deve ser 'tournament' ou 'roulette', got {self.selection_method!r}")
        if self.selection_method == 'tournament':
            if (not isinstance(self.tournament_size, int)
                    or not 1 <= self.tournament_size <= self.population_size):
                raise ValueError(f"tournament_size deve estar em [1, population_size], got {self.tournament_size!r}")
        if self.door_type not in DOOR_TYPES.values():
            raise ValueError(f"door_type inválido: {self.door_type!r}. Válidos: {sorted(DOOR_TYPES.values())}")
        if (len(self.domain) != self.num_variaveis
                or any(len(d) != 2 or not d[0] < d[1] for d in self.domain)):
            raise ValueError(f"domain deve ter {self.num_variaveis} pares (lo, hi) com lo < hi, got {self.domain!r}")
        if self.seed is not None and not (isinstance(self.seed, int) and 0 <= self.seed < 2 ** 32):
            raise ValueError(f"seed deve ser None ou int em [0, 2**32), got {self.seed!r}")
        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ValueError(f"n_jobs deve ser -1 ou >= 1 (joblib; 0 não tem sentido), got {self.n_jobs!r}")
        if not callable(self.fitness_function):
            raise ValueError(f"fitness_function deve ser callable, got {self.fitness_function!r}")

    def _initialize_population(self) -> List['Individuo']:

        population = Parallel(n_jobs=self.n_jobs)(
            delayed(Individuo)(
                0,
                self.num_portas,
                self.num_variaveis,
                self.door_type,
                self.domain,
                self.seed + i,
                annealing_window=self.annealing_window,
                sigma_es_ativo=self.sigma_es
            )
            for i in range(self.population_size)
        )
        return cast(List['Individuo'], population)


    def _sorteiar_pares(self) -> List[Tuple[int, int]]:
        """
        Sorteia os pares de indivíduos para o entrelaçamento inter-individual.

        Embaralha os índices da população e agrupa em pares consecutivos:
            (0, 3), (1, 5), (2, 4), ...
        Se a população for ímpar, o último índice fica sem par (avaliado sozinho).

        O pareamento é decidido UMA vez (inicialização) e mantido por todas as
        gerações — "applied consistently across generations", como no paper.
        """
        indices = list(range(self.population_size))
        random.Random(self.seed).shuffle(indices)
        pares = [(indices[i], indices[i + 1]) for i in range(0, len(indices) - 1, 2)]
        return pares


    def evaluate_population(self) -> None:
        if self.entanglement_mode == "pairwise" and self.pairs:
            evaluated = self._evaluate_population_par()
        else:
            evaluated = Parallel(n_jobs=self.n_jobs)(
                delayed(_evaluate_individual_task)(
                    ind, self.fitness_function, self.dataCollection,
                    self.exact_expectation)
                for ind in self.population
            )
            evaluated = cast(List['Individuo'], evaluated)
        self.population = evaluated
        self.population.sort(key=self._fitness_key, reverse=not self.minimize)

        current_best = self.population[0]
        current_fitness = self._fitness_key(current_best)
        if (self.minimize and current_fitness < self.best_fitness) or \
           (not self.minimize and current_fitness > self.best_fitness):
            self.best_individual = current_best
            self.best_fitness = current_fitness


    def _evaluate_population_par(self) -> List['Individuo']:
        """
        Avalia a população em pares entrelaçados (circuito conjunto 2× qubits).

        Cada par (i, j) é simulado junto via calcular_fitness_par; o fitness de
        ambos é definido pela mesma simulação correlacionada. Indivíduo sem par
        (população ímpar) é avaliado isoladamente.
        """
        pares = cast(List[Tuple[int, int]], self.pairs)
        n = self.population_size
        resultados: List['Individuo'] = [self.population[i] for i in range(n)]  # slots por índice

        em_pares = [(i, j) for i, j in pares if i < n and j < n]
        avaliados = cast(List[Tuple['Individuo', 'Individuo']],
                         Parallel(n_jobs=self.n_jobs)(
                             delayed(_avaliar_par)(
                                 self.population[i], self.population[j],
                                 self.fitness_function, self.dataCollection
                             )
                             for i, j in em_pares
                         ))
        for (i, j), (a, b) in zip(em_pares, avaliados):
            resultados[i] = a
            resultados[j] = b

        # Indivíduos sem par (população ímpar)
        com_par = {q for par in em_pares for q in par}
        for idx in range(n):
            if idx not in com_par:
                resultados[idx] = self._evaluate_individual(self.population[idx])

        return cast(List['Individuo'], resultados)


    def _evaluate_individual(self, individual: 'Individuo') -> 'Individuo':
        # Wrapper síncrono (usado no fallback de população ímpar); o dispatch
        # paralelo usa _evaluate_individual_task (payload mínimo, sem self).
        return _evaluate_individual_task(individual, self.fitness_function,
                                         self.dataCollection, self.exact_expectation)


    def _fitness_key(self, individual: 'Individuo') -> float:
        """Retorna um valor de fitness seguro para comparação."""

        if individual.fitness is not None:
            return individual.fitness

        return float('inf') if self.minimize else -float('inf')

    def _fitness_ratio(self) -> float:
        """Retorna a fração de melhoria logarítmica desde a primeira geração."""

        if len(self.history['best']) < 2:
            return 0.0

        start = self.history['best'][0]
        current = self.best_fitness
        if start <= 0.0 or current <= 0.0:
            return 0.0 if current >= start else 1.0

        max_decades = 6.0
        ratio = math.log10(start / current) / max_decades
        return max(0.0, min(1.0, ratio))


    def _extract_gates(self, individual: 'Individuo') -> Dict[str, int]:
        """Extrai a contagem de portas utilizadas por um indivíduo."""
        gate_count = {}
        counted = set()  # Para não contar portas multi-qubit múltiplas vezes
        
        for q in range(individual.num_qubits):
            for i in range(individual.num_portas):
                gene = individual.genoma[q, i]
                
                # Converter numpy array para tuple se necessário
                if isinstance(gene, np.ndarray) and gene.shape == (4,):
                    op, c1, c2, t = gene.tolist()
                elif isinstance(gene, (tuple, list)) and len(gene) == 4:
                    op, c1, c2, t = gene
                else:
                    continue
                
                if op == 'K' or op is None:
                    continue
                
                # Converter para tipos nativos
                op = str(op)
                if c1 is not None:
                    c1 = int(c1)
                if c2 is not None:
                    c2 = int(c2)
                if t is not None:
                    t = int(t)
                
                # Chave única para esta porta
                if op in individual.PORTAS1Q:
                    key = (i, op, q)
                elif op in individual.PORTAS2Q and c1 is not None and t is not None:
                    key = (i, op, min(c1, t), max(c1, t))
                elif op in individual.PORTAS3Q and c1 is not None and c2 is not None and t is not None:
                    key = (i, op, min(c1, c2, t), max(c1, c2, t))
                else:
                    continue
                
                if key not in counted:
                    counted.add(key)
                    gate_count[op] = gate_count.get(op, 0) + 1
        
        return gate_count


    def _extract_location(self, individual: 'Individuo') -> List[float]:
        """Extrai a localização média do indivíduo no domínio (via simulação ou cache)."""
        # Usar cache se disponível (dataCollection habilitado)
        if self.dataCollection and individual.location is not None:
            return individual.location
        
        try:
            qc = individual.construir_circuito_qiskit()
            resultados = individual.simular_circuito(qc, shots=256)
            
            total_shots = sum(resultados.values())
            if total_shots == 0:
                return [0.0] * individual.num_variaveis
            
            soma_valores = [0.0 for _ in range(individual.num_variaveis)]
            for estado, count in resultados.items():
                valores = individual._converter_estado_para_reais(estado[::-1])
                for i in range(individual.num_variaveis):
                    soma_valores[i] += valores[i] * count
            
            return [s / total_shots for s in soma_valores]
        except Exception:
            return [0.0] * individual.num_variaveis


    def _extract_gate_entropy(self, individual: 'Individuo') -> float:
        """Extrai a entropia de Shannon da distribuição de portas do indivíduo."""
        try:
            gate_counts = self._extract_gates(individual)
            return gate_entropy(gate_counts)
        except Exception:
            return 0.0


    def _extract_population_diversity(self) -> Dict[str, float]:
        """Extrai métricas de diversidade da população."""
        try:
            # Coletar posições decodificadas de todos os indivíduos
            positions = []
            for ind in self.population:
                location = self._extract_location(ind)
                positions.append(location)
            
            positions = np.array(positions)
            return population_diversity(positions, self.domain)
        except Exception:
            return {
                'centroid_distance': 0.0,
                'population_radius': 0.0,
                'pairwise_distance': 0.0
            }


    def select_parent(self) -> 'Individuo':

        if self.selection_method == 'tournament':
            competitors = self._rng.sample(self.population, self.tournament_size)
            return min(competitors, key=self._fitness_key) if self.minimize else max(competitors, key=self._fitness_key)

        fitnesses = np.array([self._fitness_key(ind) for ind in self.population])

        if self.minimize:
            adjusted = np.max(fitnesses) - fitnesses + 1e-10

        else:
            adjusted = fitnesses + 1e-10

        prob = adjusted / np.sum(adjusted)

        return self.population[np.random.choice(len(self.population), p=prob)]



    def _create_offspring(self, parent1: 'Individuo', parent2: 'Individuo') -> Tuple['Individuo', 'Individuo', int]:
        # Sorteio no processo principal (mantém o stream de self._rng idêntico);
        # o trabalho pesado vai p/ _create_offspring_task (payload mínimo, sem self).
        do_crossover = self._rng.random() < self.crossover_rate
        return _create_offspring_task(
            parent1, parent2,
            self.crossover_selection, do_crossover,
            self.annealing_window, self.sigma_es,
            self.mutation_rate, self.generations,
        )


    # ──────────────────────────────────────────────────────────────────────────────────────────────────
    # Output directory and CSV export


    def _make_run_dir(self, label_type: str = "qga") -> Path:
        """Cria o diretório de saída com timestamp: outputs/{label_type}_DD-MM_HHhMM/"""
        now = datetime.now()
        ts = f"{now.day:02d}-{now.month:02d}_{now.hour:02d}h{now.minute:02d}"
        run_dir = Path("outputs") / f"{label_type}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._increment_usage(run_dir)
        return run_dir


    def _usage_path(self, run_dir: Path) -> Path:
        return run_dir / ".usage"


    def _read_usage(self, run_dir: Path) -> int:
        usage_file = self._usage_path(run_dir)
        if not usage_file.exists():
            return 0
        try:
            return max(0, int(usage_file.read_text().strip()))
        except ValueError:
            return 0


    def _write_usage(self, run_dir: Path, count: int) -> None:
        self._usage_path(run_dir).write_text(str(count))


    def _increment_usage(self, run_dir: Path) -> None:
        """Incrementa o contador de acessos do diretório de saída."""
        self._write_usage(run_dir, self._read_usage(run_dir) + 1)


    def _export_csv(self, run_dir: Path) -> Tuple[Path, Path]:
        """Exporta dados da evolução para CSV.
        
        Returns:
            Tuple de (evolution_csv_path, metrics_csv_path)
        """
        evolution_path = run_dir / "evolution_data.csv"
        metrics_path = run_dir / "metrics.csv"
        
        # Helper para formatar floats
        def fmt(val, decimals=4):
            if isinstance(val, float):
                return f"{val:.{decimals}f}"
            return str(val) if val != "" else ""
        
        # ── evolution_data.csv (dados básicos) ──
        with open(evolution_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'best_fitness', 'mean_fitness', 'median_fitness',
                'best_location', 'mutations', 'gates'
            ])
            
            for i in range(len(self.history['best'])):
                gates_str = str(self.history['best_gates'][i]) if i < len(self.history['best_gates']) else ""
                mutations_str = str(self.history['mutations'][i]) if i < len(self.history['mutations']) else ""
                
                # Formatar location com 4 casas decimais
                location_str = ""
                if i < len(self.history['best_location']):
                    loc = self.history['best_location'][i]
                    location_str = "[" + ", ".join(f"{x:.4f}" for x in loc) + "]"
                
                writer.writerow([
                    i,
                    fmt(self.history['best'][i]),
                    fmt(self.history['avg'][i]),
                    fmt(self.history['median'][i]) if i < len(self.history['median']) else "",
                    location_str,
                    mutations_str,
                    gates_str
                ])
        
        # ── metrics.csv (métricas de diversidade e entropia) ─
        exp_active, exp_interval = self.exp_retrieval
        include_diversity = len(self.history['centroid_distance']) > 0
        
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Cabeçalho
            header = ['generation', 'gate_entropy']
            if include_diversity:
                header.extend(['centroid_distance', 'population_radius', 'pairwise_distance'])
            writer.writerow(header)
            
            # Dados - apenas gerações onde foram coletados
            entropy_idx = 0
            diversity_idx = 0
            
            for i in range(len(self.history['best'])):
                # Entropia é coletada toda geração
                entropy_str = ""
                if i < len(self.history['gate_entropy']):
                    entropy_str = fmt(self.history['gate_entropy'][i])
                
                # Diversidade é coletada baseado em exp_retrieval
                row = [i, entropy_str]
                
                if include_diversity:
                    should_have_diversity = (
                        (exp_active and i % exp_interval == 0) or
                        (not exp_active and i == len(self.history['best']) - 1)
                    )
                    
                    if should_have_diversity and diversity_idx < len(self.history['centroid_distance']):
                        row.extend([
                            fmt(self.history['centroid_distance'][diversity_idx]),
                            fmt(self.history['population_radius'][diversity_idx]),
                            fmt(self.history['pairwise_distance'][diversity_idx])
                        ])
                        diversity_idx += 1
                    else:
                        row.extend(['', '', ''])
                
                writer.writerow(row)
        
        return evolution_path, metrics_path


    def run_cleanup(self):
        """Lista e permite deletar diretórios de saída, mostrando quantas vezes foram usados."""
        print("\n" + "─" * 60)
        print(stylePrint("Delete outputs", C, bold=True))
        print("─" * 60 + "\n")

        outputs_dir = Path("outputs")
        if not outputs_dir.exists():
            print("  No outputs directory found.")
            return

        runs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()])
        if not runs:
            print("  No output directories found.")
            return

        print(f"  • What directories do you want to {stylePrint('delete', '#ff6b6b', bold=True)}?\n")
        for i, r in enumerate(runs):
            usage = self._read_usage(r)
            times_word = "time" if usage == 1 else "times"
            print(f"  {stylePrint('[' + str(i) + ']', C, bold=True)} {r.name} ({usage} {times_word} used)")

        raw = input("\n  > ").strip()
        if not raw:
            print("  No selection made.")
            return

        try:
            indices = [int(x.strip()) for x in raw.split(",")]
        except ValueError:
            print("  Invalid input.")
            return

        if any(idx < 0 or idx >= len(runs) for idx in indices):
            print("  Invalid selection.")
            return

        selected = [runs[i] for i in indices]

        print("\n  • Are you sure? [y/n]\n")
        confirm = input("  > ").strip().lower()
        print("\n")

        if confirm != "y":
            print("  Cancelled.")
            return

        for r in selected:
            shutil.rmtree(r)
            print("  ◦ Deleted:")
            print(f"    → {r.name}")

        print(f"\n  {stylePrint('Done', C, bold=True)}")



    # ──────────────────────────────────────────────────────────────────────────────────────────────────
    # prints e UI


    def _format_log_line(self, generation: int, elapsed: float) -> str:

        current = generation + 1
        total = self.generations
        total_width = len(str(total))

        gen_ratio = current / total if total else 0
        gen_bar = self._gen_bar.bar(gen_ratio)

        fit_ratio = self._fitness_ratio()

        fit_bar = self._fit_bar.bar(fit_ratio)

        gen_s = current / elapsed if elapsed > 0 else 0.0
        avg = self.history['avg'][-1]

        gen_label = stylePrint(f"Generation {current:>{total_width}}/{total}", C, bold=True)

        # Formatar best fitness: se > 1, usar float normal; senão, usar notação científica
        if abs(self.best_fitness) > 1:
            fit_str = f"{self.best_fitness:>12.4f}"
        else:
            fit_str = formatMantissa(self.best_fitness)

        return (
            f"{gen_label}   {gen_bar}   {fit_bar}   "
            f"{gen_s:>6.2f}g/s      {fit_str}      {avg:.4f}"
        )
    

    def _format_batch_bar(self, batch_context: Dict) -> str:
        """
        Formata a barra de progresso do batch usando o mesmo ProgressBar das outras barras.
        
        Args:
            batch_context: Dict com 'current', 'total'
        """
        current = batch_context['current']
        total = batch_context['total']
        
        total_width = len(str(total))
        ratio = current / total if total else 0
        
        batch_bar = self._batch_bar.bar(ratio)
        label = stylePrint(f"Batch {current:>{total_width}}/{total}", "#dcd7ba", bold=True)
        
        return f"{label}   {batch_bar}"
    



    def run(self, batch_context: Optional[Dict] = None) -> Dict[str, Union[Optional['Individuo'],
                                     float, List['Individuo'],
                                     List[float],
                                     Dict[str, Union[List['Individuo'],
                                     List[float]]],
                                     Path]]:
        """
        Executa o QGA.
        
        Args:
            batch_context: Se fornecido, exibe barra de batch acima da barra de iteração.
                          Deve conter: {'current': int, 'total': int, 'fg_color': str, 'bg_color': str}
        """

        start_time = time.time()
        progress = None

        colorDot = stylePrint("◦", C)
            
        door_type_name = DOOR_TYPE_NAMES.get(self.door_type)
        if door_type_name is None:
            raise ValueError(f"Invalid door_type: {self.door_type}")


        legend = (
            "                   generation bar         fitness bar           "
            "   gen/s      current best     avg\n"
        )

        header = (
            f"\n{sepComment(40, stylePrint("Iniciando availiação QGA", C , bold=True))}\n"
            
            f"  {colorDot} Fitness Function {stylePrint(self.fitness_function_name, C, bold=True)}, minimize = {stylePrint(str(self.minimize), C, bold=True)}\n" 
            f"  {colorDot} Selection: {stylePrint(self.selection_method, C, bold=True)}\n"
            f"  {colorDot} Crossover: {stylePrint(self.crossover_selection, C, bold=True)}\n"
            f"  {colorDot} Population: {stylePrint(str(self.population_size), C, bold=True)}, Generation Lifespan: {stylePrint(str(self.generations), C, bold=True)}\n"
            f"  {colorDot} Best possible fitness: {stylePrint(str(self.target_fitness), C, bold=True)}\n"
            f"  {colorDot} Door Type: {stylePrint(door_type_name, C, bold=True)}\n"
            f"  {colorDot} exp_retrieval: {stylePrint(str(self.exp_retrieval[0]), C, bold=True)} every {stylePrint(str(self.exp_retrieval[1]), C, bold=True)} generations\n"
        )

        # Em modo batch, header é impresso apenas na primeira iteração
        if batch_context is None or batch_context.get('current', 0) == 0:
            print(header, flush=True)
            print(legend, flush=True)
        



        # ──────────────────────────────────────────────────────────────────────────────────────────────────
        # loop pra valer

        
        for generation in range(self.generations):

            self.evaluate_population()


            current_fitness = np.array([ind.fitness for ind in self.population])
            self.history['best'].append(self.best_fitness)
            self.history['avg'].append(np.mean(current_fitness))
            self.history['median'].append(np.median(current_fitness))
            self.history['best_individual'].append(copy.deepcopy(self.population[0]))
            
            # Coleta dados expandidos
            best_ind = self.population[0]
            self.history['best_location'].append(self._extract_location(best_ind))
            self.history['best_gates'].append(self._extract_gates(best_ind))
            self.history['gate_entropy'].append(self._extract_gate_entropy(best_ind))
            
            # Coleta métricas de diversidade baseado em exp_retrieval
            exp_active, exp_interval = self.exp_retrieval
            should_collect_diversity = (
                exp_active and (generation % exp_interval == 0)
            )
            
            if should_collect_diversity:
                diversity = self._extract_population_diversity()
                self.history['centroid_distance'].append(diversity['centroid_distance'])
                self.history['population_radius'].append(diversity['population_radius'])
                self.history['pairwise_distance'].append(diversity['pairwise_distance'])
                self.history['diversity_generations'].append(generation)


            elapsed = time.time() - start_time
            line = self._format_log_line(generation, elapsed)


            if self.continuous_bar or not self.verbose:
                if batch_context is not None:
                    # Batch bar + iteration bar na mesma linha (sobrescreve ambos)
                    batch_bar = self._format_batch_bar(batch_context)
                    sys.stdout.write(f"\r{batch_bar}   {line}    ")
                else:
                    sys.stdout.write(f"\r{line}    ")
                sys.stdout.flush()
            else:
                # Cada geração em uma linha nova
                print(line, flush=True)


            new_population = [copy.deepcopy(ind) for ind in self.population[:self.elite_size]]

            num_pairs = (self.population_size - self.elite_size) // 2


            offspring = cast(

                List[Tuple['Individuo', 'Individuo', int]],

                Parallel(n_jobs=self.n_jobs)(
                    delayed(_create_offspring_task)(
                        self.select_parent(),
                        self.select_parent(),
                        self.crossover_selection,
                        self._rng.random() < self.crossover_rate,
                        self.annealing_window,
                        self.sigma_es,
                        self.mutation_rate,
                        self.generations,
                    )
                    for _ in range(num_pairs)
                )

            )

            # Coletar mutações e extrair filhos
            total_mutations = 0
            flat_offspring = []
            for child1, child2, mutations in offspring:
                flat_offspring.append(child1)
                flat_offspring.append(child2)
                total_mutations += mutations
            
            new_population.extend(flat_offspring)


            while len(new_population) < self.population_size:
                p1, p2 = self.select_parent(), self.select_parent()
                c1, c2, extra_mut = self._create_offspring(p1, p2)
                new_population.append(c1)
                new_population.append(c2)
                total_mutations += extra_mut
            
            self.history['mutations'].append(total_mutations)

            self.population = new_population


        # Nova linha após o loop (quando sobrescrevia a mesma linha)
        if not self.verbose:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if progress is not None:
            progress.close()

        self.evaluate_population()


        # ──────────────────────────────────────────────────────────────────────────────────────────────────
        # UX: perguntar se quer gráficos e exportar


        if self.verbose:
            # Calcular tempo total de simulação
            total_time = time.time() - start_time
            minutes = int(total_time // 60)
            seconds = total_time % 60
            
            # Formatar tempo sem "0min" ou "0sec"
            if minutes > 0 and seconds > 0:
                time_str = f"{minutes}min {seconds:.1f}sec"
            elif minutes > 0:
                time_str = f"{minutes}min"
            else:
                time_str = f"{seconds:.1f}sec"
            
            print(f"\n{sepComment(40, stylePrint('Run Complete', C, bold=True))}")
            print(f"  {colorDot} Simulation time: {stylePrint(time_str, C, bold=True)}")
            print(f"  {colorDot} Best Fitness: {stylePrint(formatMantissa(self.best_fitness), C, bold=True)}")
            if self.best_individual:
                print(f"  {colorDot} Best Location: {stylePrint(str(self._extract_location(self.best_individual)), C)}")
            
            # Perguntar se quer exportar
            print(f"\n  • Export data to CSV? [y/n]")
            export_csv = input("  > ").strip().lower()
            
            if export_csv == "y":
                self._run_dir = self._make_run_dir("qga")
                evolution_path, metrics_path = self._export_csv(self._run_dir)
                print(f"  {colorDot} Evolution CSV: {stylePrint(str(evolution_path), C)}")
                print(f"  {colorDot} Metrics CSV: {stylePrint(str(metrics_path), C)}")
            
            # Salvar circuito (sempre que houver run_dir)
            if self.best_individual:
                output_dir = self._run_dir if self._run_dir else self._make_run_dir("qga")

                try:
                    qc = self.best_individual.construir_circuito_qiskit()
                    circuit_path = output_dir / "circuit.png"
                    qc.draw(output="mpl", filename=str(circuit_path), fold=30)
                    print(f"  {colorDot} Circuit saved to: {stylePrint(str(circuit_path), C)}")

                except Exception as e:
                    print(f"  {stylePrint('Warning', '#f3af41')}: Could not save circuit: {e}")
            
            # Perguntar se quer gráficos
            print(f"\n  • Generate graphs? [y/n]")
            show_graphs = input("  > ").strip().lower()
            
            if show_graphs == "y":
                # Perguntar colorscheme
                print(f"\n  • What is the desired colorscheme? (p for paper / v for visualization)")
                colorscheme = input("  > ").strip().lower()
                if colorscheme not in ('p', 'v'):
                    colorscheme = 'v'
                
                try:
                    from visualization.graphing import plot_all
                    
                    output_dir = self._run_dir if self._run_dir else self._make_run_dir("qga")
                    plot_all(
                        best_history=self.history['best'],
                        avg_history=self.history['avg'],
                        median_history=self.history['median'],
                        gate_history=self.history['best_gates'],
                        mutation_history=self.history['mutations'],
                        target_fitness=self.target_fitness,
                        output_dir=output_dir,
                        theme=colorscheme
                    )
                    print(f"  {colorDot} Graphs saved to: {stylePrint(str(output_dir), C)}")
                except ImportError:
                    print(f"  {stylePrint('Warning', '#f3af41')}: matplotlib not installed. Skipping graphs.")
            
            # Perguntar se quer advanced EDA (apenas se exp_retrieval ativo)
            exp_active, _ = self.exp_retrieval
            if exp_active and len(self.history['centroid_distance']) > 0:
                print(f"\n  • Generate advanced diversity EDA? [y/n]")
                advanced = input("  > ").strip().lower()
                
                if advanced == "y":
                    try:
                        from visualization.advanced_eda import plot_eda_all
                        
                        output_dir = self._run_dir if self._run_dir else self._make_run_dir("qga")
                        plot_eda_all(
                            history=self.history,
                            output_dir=output_dir,
                            theme=locals().get('colorscheme', 'v')
                        )
                        print(f"  {colorDot} Advanced EDA saved to: {stylePrint(str(output_dir), C)}")
                    except ImportError:
                        print(f"  {stylePrint('Warning', '#f3af41')}: matplotlib not installed. Skipping advanced EDA.")
                    except Exception as e:
                        print(f"  {stylePrint('Error', '#ff6b6b')}: {e}")
            
            # Perguntar se quer animações (apenas para 2 variáveis)
            if self.num_variaveis == 2:
                print(f"\n  • Generate animations? [y/n]")
                animate = input("  > ").strip().lower()
                
                if animate == "y":
                    try:
                        from visualization.animation import EvolutionAnimator
                        
                        output_dir = self._run_dir if self._run_dir else self._make_run_dir("qga")
                        animator = EvolutionAnimator(
                            history=self.history,
                            domain=self.domain,
                            fitness_fn=self.fitness_function,
                            fitness_name=self.fitness_function_name,
                            optimum=self.optimum,
                            optimum_label=self.optimum_label
                        )
                        
                        print(f"  {colorDot} Generating 2D animation...")
                        animator.animate_2d(
                            output_path=output_dir / "evolution_2d.gif",
                            fps=5,
                            resolution=100
                        )
                        
                        print(f"  {colorDot} Generating 3D animation...")
                        animator.animate_3d(
                            output_path=output_dir / "evolution_3d.gif",
                            fps=5,
                            resolution=50
                        )
                        
                        print(f"  {colorDot} Animations saved to: {stylePrint(str(output_dir), C)}")
                    except ImportError:
                        print(f"  {stylePrint('Warning', '#f3af41')}: matplotlib not installed. Skipping animations.")
                    except Exception as e:
                        print(f"  {stylePrint('Error', '#ff6b6b')}: {e}")


        # ──────────────────────────────────────────────────────────────────────────────────────────────────
        # Hessian condition number (futuro)
        
        # THRESHOLD = 10.0
        # if self.best_fitness < THRESHOLD:
        #     try:
        #         from metrics import hessian_condition_number, interpret_condition_number
        #         
        #         best_location = self._extract_location(self.best_individual)
        #         kappa, eigenvalues = hessian_condition_number(
        #             self.fitness_function,
        #             np.array(best_location),
        #             self.domain
        #         )
        #         
        #         print(f"\n  {colorDot} Hessian Condition Number: {stylePrint(f'{kappa:.2e}', C, bold=True)}")
        #         print(f"  {colorDot} {interpret_condition_number(kappa)}")
        #         print(f"  {colorDot} Eigenvalues: {stylePrint(str(np.sort(eigenvalues)[:5]), C)}...")
        #         
        #     except Exception as e:
        #         print(f"  {stylePrint('Warning', '#f3af41')}: Could not compute Hessian: {e}")


        return {

            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'final_population': self.population,
            'run_dir': self._run_dir

        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
