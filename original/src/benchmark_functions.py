import math
import numpy as np
from typing import Callable, List, Tuple, Dict


'''
    Local equivalents for the DEAP benchmark functions used by the project.

'''


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _as_vector(individual):
    return np.asarray(individual, dtype=float).ravel()


def sphere(individual):
    x = _as_vector(individual)
    return (float(np.sum(x**2)),)


def rastrigin(individual):
    x = _as_vector(individual)
    value = 10 * x.size + np.sum(x**2 - 10 * np.cos(2 * math.pi * x))
    return (float(value),)


def ackley(individual):

    x = _as_vector(individual)
    if x.size == 0:
        return (0.0,)

    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * math.pi * x))
    value = (
        -20 * math.exp(-0.2 * math.sqrt(sum_sq / x.size))
        - math.exp(sum_cos / x.size)
        + 20
        + math.e
    )

    return (float(value),)


def griewank(individual):

    x = _as_vector(individual)
    indices = np.arange(1, x.size + 1, dtype=float)
    value = 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(indices)))

    return (float(value),)


def rosenbrock(individual):

    x = _as_vector(individual)
    value = np.sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (1 - x[:-1]) ** 2)

    return (float(value),)


def schwefel(individual):

    x = _as_vector(individual)
    value = 418.9828872724338 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    return (float(value),)


def zakharov(individual):

    x = _as_vector(individual)
    indices = np.arange(1, x.size + 1, dtype=float)
    linear_sum = np.sum(0.5 * indices * x)
    value = np.sum(x**2) + linear_sum**2 + linear_sum**4

    return (float(value),)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mapeamento: nome → (função, domínio padrão, ótimo global 2D)


BENCHMARKS: Dict[str, Dict] = {
    "sphere": {
        "fn": sphere,
        "domain": (-10.0, 10.0),
        "optimum": (0.0, 0.0),
        "optimum_label": "f(0,0) = 0"
    },
    "rastrigin": {
        "fn": rastrigin,
        "domain": (-5.12, 5.12),
        "optimum": (0.0, 0.0),
        "optimum_label": "f(0,0) = 0"
    },
    "ackley": {
        "fn": ackley,
        "domain": (-32.0, 32.0),
        "optimum": (0.0, 0.0),
        "optimum_label": "f(0,0) = 0"
    },
    "griewank": {
        "fn": griewank,
        "domain": (-600.0, 600.0),
        "optimum": (0.0, 0.0),
        "optimum_label": "f(0,0) = 0"
    },
    "rosenbrock": {
        "fn": rosenbrock,
        "domain": (-5.0, 10.0),
        "optimum": (1.0, 1.0),
        "optimum_label": "f(1,1) = 0"
    },
    "schwefel": {
        "fn": schwefel,
        "domain": (-500.0, 500.0),
        "optimum": (420.9687, 420.9687),
        "optimum_label": "f(420.97,420.97) ≈ 0"
    },
    "zakharov": {
        "fn": zakharov,
        "domain": (-5.0, 10.0),
        "optimum": (0.0, 0.0),
        "optimum_label": "f(0,0) = 0"
    },
}


def get_benchmark(name: str, num_variaveis: int = 2) -> Tuple[Callable, List[Tuple[float, float]], Tuple[float, ...], str]:
    """
    Retorna a função de fitness, domínio, ótimo global e label do ótimo.
    A função retornada já extrai o float da tupla interna.
    
    Args:
        name: Nome da função benchmark (sphere, ackley, rastrigin, etc.)
        num_variaveis: Número de variáveis (repete o domínio e ótimo para cada uma)
        
    Returns:
        Tuple de (fitness_function, domain_list, optimum_tuple, optimum_label)
        
    Example:
        fitness_fn, domain, optimum, label = get_benchmark("sphere", num_variaveis=2)
        # fitness_fn = sphere (retorna float, não tupla)
        # domain = [(-10.0, 10.0), (-10.0, 10.0)]
        # optimum = (0.0, 0.0)
        # label = "f(0,0) = 0"
    """
    if name not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    
    info = BENCHMARKS[name]
    raw_fn = info["fn"]
    single_domain = info["domain"]
    single_optimum = info["optimum"]
    optimum_label = info["optimum_label"]
    
    domain = [single_domain] * num_variaveis
    optimum = tuple([single_optimum[0]] * num_variaveis)  # Repete para N variáveis
    
    # Wrapper que extrai o float da tupla
    def fitness_fn(x):
        return float(raw_fn(x)[0])
    
    fitness_fn.__name__ = name
    return fitness_fn, domain, optimum, optimum_label


def list_benchmarks() -> None:
    """Imprime as funções benchmark disponíveis e seus domínios."""
    print("\nAvailable benchmark functions:\n")
    for name, info in BENCHMARKS.items():
        domain = info["domain"]
        optimum = info["optimum"]
        label = info["optimum_label"]
        print(f"  {name:<12}  domain={domain}  optimum={optimum}  {label}")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
