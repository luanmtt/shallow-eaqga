"""
Métricas de entropia e diversidade para o QGA.

Fornece funções para medir a incerteza/distribuição em diferentes
níveis do algoritmo genético quântico.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entropia de medição (por indivíduo)


def measurement_entropy(counts: Dict[str, int]) -> float:
    """
    Calcula a entropia de Shannon da distribuição de medições de um circuito quântico.
    
    H = -Σ p_i × log2(p_i)
    
    Caso de uso:
        - Medir o quão "espalhado" está o estado quântico após medição.
        - Alta entropia: circuito produz superposição uniforme (muitos estados prováveis).
        - Baixa entropia: circuito concentra probabilidade em poucos estados.
        - Útil para detectar se o circuito está convergindo para uma solução.
    
    Args:
        counts: Dicionário {estado_binário: contagem} retornado por simular_circuito().
        
    Returns:
        Entropia em bits. Máximo = log2(n_estados), Mínimo = 0.
    
    Exemplo:
        >>> counts = {'000': 512, '101': 256, '011': 256}
        >>> measurement_entropy(counts)
        1.5  # 3 estados equiprováveis → log2(3) ≈ 1.58
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def measurement_entropy_normalized(counts: Dict[str, int]) -> float:
    """
    Entropia normalizada entre 0 e 1.
    
    H_norm = H / H_max = H / log2(n_estados)
    
    Caso de uso:
        - Comparar entropia entre circuitos com diferentes números de qubits.
        - 0 = determinístico (um estado com 100% de probabilidade).
        - 1 = uniforme (todos os estados equiprováveis).
    
    Args:
        counts: Dicionário {estado_binário: contagem}.
        
    Returns:
        Entropia normalizada [0, 1].
    """
    n_states = len(counts)
    if n_states <= 1:
        return 0.0
    
    h = measurement_entropy(counts)
    h_max = math.log2(n_states)
    
    return h / h_max if h_max > 0 else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entropia da população (por geração)


def population_entropy(fitnesses: List[float], n_bins: int = 10) -> float:
    """
    Calcula a entropia de Shannon do histograma de fitness da população.
    
    Caso de uso:
        - Medir a diversidade do fitness na população.
        - Alta entropia: população com fitness variado (diversa).
        - Baixa entropia: população convergindo (fitness similar).
        - Útil como critério de parada: entropia baixa por muitas gerações = convergiu.
    
    Args:
        fitnesses: Lista de valores de fitness da população.
        n_bins: Número de bins para o histograma.
        
    Returns:
        Entropia em bits.
    """
    if not fitnesses:
        return 0.0
    
    hist, _ = np.histogram(fitnesses, bins=n_bins)
    total = hist.sum()
    if total == 0:
        return 0.0
    
    probs = hist / total
    probs = probs[probs > 0]  # Evitar log(0)
    
    return float(-np.sum(probs * np.log2(probs)))


def population_entropy_normalized(fitnesses: List[float], n_bins: int = 10) -> float:
    """
    Entropia normalizada da população entre 0 e 1.
    
    Caso de uso:
        - Comparar diversidade entre gerações.
        - 0 = todos os indivíduos têm fitness idêntico.
        - 1 = fitness uniformemente distribuído nos bins.
    """
    if not fitnesses:
        return 0.0
    
    h = population_entropy(fitnesses, n_bins)
    h_max = math.log2(n_bins)
    
    return h / h_max if h_max > 0 else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entropia dos gates (diversidade genética)


def gate_entropy(gate_counts: Dict[str, int]) -> float:
    """
    Calcula a entropia de Shannon da distribuição de portas quânticas.
    
    Caso de uso:
        - Medir a diversidade genética de um indivíduo.
        - Alta entropia: muitos tipos de gates diferentes sendo usados.
        - Baixa entropia: poucos tipos de gates dominam.
        - Útil para detectar se o genoma está "estagnado" em um padrão.
    
    Args:
        gate_counts: Dicionário {nome_porta: contagem}.
        
    Returns:
        Entropia em bits.
    """
    total = sum(gate_counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in gate_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def gate_diversity(gate_counts: Dict[str, int]) -> int:
    """
    Conta o número de tipos de gates diferentes utilizados.
    
    Caso de uso:
        - Medir a variedade de portas em um indivíduo.
        - Simples contagem de tipos únicos.
    
    Args:
        gate_counts: Dicionário {nome_porta: contagem}.
        
    Returns:
        Número de tipos de gates diferentes.
    """
    return len(gate_counts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Diversidade da população (por geração)


def centroid_distance(positions: np.ndarray, domain: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calcula a distância média ao centroide da população (normalizada pelo domínio).
    
    D_c(t) = (1/N) × Σ ||x_i(t) - x̄(t)||
    
    Caso de uso:
        - Medir a dispersão da população em torno do centro.
        - Alta distância = exploração (população espalhada).
        - Distância caindo = explotação/convergência.
        - Mais rápido que pairwise (O(N) vs O(N²)).
        - Sensível a outliers (um indivíduo distante infla a média).
    
    Args:
        positions: Array (N, dim) com posições decodificadas dos indivíduos.
        domain: Lista de tuplas (min, max) para cada variável.
        
    Returns:
        Tuple (distância_média, raio_população):
            - distância_média: distância média ao centroide (normalizada).
            - raio_população: distância máxima ao centroide (normalizada).
    """
    if len(positions) == 0:
        return 0.0, 0.0
    
    # Calcular ranges para normalização
    ranges = np.array([abs(d[1] - d[0]) for d in domain])
    ranges[ranges == 0] = 1.0  # Evitar divisão por zero
    
    # Normalizar posições pelo domínio
    normalized = positions / ranges
    
    # Centroide
    centroid = np.mean(normalized, axis=0)
    
    # Distâncias ao centroide
    distances = np.linalg.norm(normalized - centroid, axis=1)
    
    return float(np.mean(distances)), float(np.max(distances))


def pairwise_distance(positions: np.ndarray, domain: List[Tuple[float, float]]) -> float:
    """
    Calcula a distância média par-a-par da população (normalizada pelo domínio).
    
    D_p(t) = (2/(N(N-1))) × Σ_{i<j} ||x_i(t) - x_j(t)||
    
    Caso de uso:
        - Medir a diversidade considerando todos os pares.
        - Mais robusto a outliers que a distância ao centroide.
        - O(N²) — mais custoso computacionalmente.
        - Útil para detectar subgrupos na população.
    
    Args:
        positions: Array (N, dim) com posições decodificadas dos indivíduos.
        domain: Lista de tuplas (min, max) para cada variável.
        
    Returns:
        Distância média par-a-par (normalizada).
    """
    N = len(positions)
    if N <= 1:
        return 0.0
    
    # Calcular ranges para normalização
    ranges = np.array([abs(d[1] - d[0]) for d in domain])
    ranges[ranges == 0] = 1.0  # Evitar divisão por zero
    
    # Normalizar posições pelo domínio
    normalized = positions / ranges
    
    # Acumulação por chunks (economia de memória para populações grandes)
    chunk_size = 2000
    total = 0.0
    
    for i in range(0, N, chunk_size):
        chunk = normalized[i:i + chunk_size]
        diff = chunk[:, None] - normalized[None, :]
        D = np.sqrt(np.sum(diff ** 2, axis=2))
        total += np.sum(D)
    
    # Corrigir: a diagonal foi contada mas vale 0, então não afeta
    # Dividir por N*(N-1) porque contamos cada par duas vezes
    return float(total / (N * (N - 1)))


def population_diversity(positions: np.ndarray, 
                         domain: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Calcula todas as métricas de diversidade da população de uma vez.
    
    Caso de uso:
        - Coletar todas as métricas de diversidade em uma chamada.
        - Retornar dicionário pronto para inserção no CSV.
    
    Args:
        positions: Array (N, dim) com posições decodificadas dos indivíduos.
        domain: Lista de tuplas (min, max) para cada variável.
        
    Returns:
        Dicionário com:
            - centroid_distance: distância média ao centroide (normalizada).
            - population_radius: distância máxima ao centroide (normalizada).
            - pairwise_distance: distância média par-a-par (normalizada).
    """
    centroid_dist, radius = centroid_distance(positions, domain)
    pairwise_dist = pairwise_distance(positions, domain)
    
    return {
        'centroid_distance': centroid_dist,
        'population_radius': radius,
        'pairwise_distance': pairwise_dist
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Número de condição da Hessiana (para futura implementação de GD)


def hessian_condition_number(fitness_fn, point: np.ndarray, 
                             domain: List[Tuple[float, float]],
                             epsilon: float = 1e-5) -> Tuple[float, np.ndarray]:
    """
    Calcula o número de condição da Hessiana numérica no ponto dado.
    
    κ(H) = λ_max / λ_min
    
    Caso de uso:
        - Determinar se GD é viável para refinar a solução.
        - κ < 10: GD com momentum já é suficiente.
        - κ entre 10 e 10³: momentum/Nesterov ajuda, L-BFGS converge mais rápido.
        - κ > 10³: obrigatório usar informação de segunda ordem (L-BFGS, Newton).
        
    Args:
        fitness_fn: Função de fitness (recebe array, retorna float).
        point: Ponto onde calcular a Hessiana (dim,).
        domain: Lista de tuplas (min, max) para cada variável.
        epsilon: Perturbação para diferenças finitas.
        
    Returns:
        Tuple (kappa, eigenvalues):
            - kappa: número de condição da Hessiana.
            - eigenvalues: autovalores da Hessiana.
    """
    n = len(point)
    hessian = np.zeros((n, n))
    
    # Valor da função no ponto
    f_point = fitness_fn(point)
    
    # Diagonal: ∂²f/∂x_i²
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = epsilon
        
        # Diferenças finitas centrais de segunda ordem
        f_plus = fitness_fn(point + e_i)
        f_minus = fitness_fn(point - e_i)
        hessian[i, i] = (f_plus - 2 * f_point + f_minus) / (epsilon ** 2)
    
    # Fora da diagonal: ∂²f/∂x_i∂x_j
    for i in range(n):
        for j in range(i + 1, n):
            e_i = np.zeros(n)
            e_i[i] = epsilon
            e_j = np.zeros(n)
            e_j[j] = epsilon
            
            # Diferenças finitas mistas
            f_pp = fitness_fn(point + e_i + e_j)
            f_pm = fitness_fn(point + e_i - e_j)
            f_mp = fitness_fn(point - e_i + e_j)
            f_mm = fitness_fn(point - e_i - e_j)
            
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon ** 2)
            hessian[j, i] = hessian[i, j]  # Simétrica
    
    # Autovalores
    eigenvalues = np.linalg.eigvalsh(hessian)
    
    # Número de condição
    lambda_max = np.max(np.abs(eigenvalues))
    lambda_min = np.min(np.abs(eigenvalues))
    
    if lambda_min < 1e-10:
        kappa = float('inf')
    else:
        kappa = lambda_max / lambda_min
    
    return kappa, eigenvalues


def interpret_condition_number(kappa: float) -> str:
    """
    Interpreta o número de condição e sugere algoritmo.
    
    Args:
        kappa: número de condição da Hessiana.
        
    Returns:
        String com a interpretação.
    """
    if kappa < 10:
        return "κ < 10: GD com momentum é suficiente"
    elif kappa < 1000:
        return "10 ≤ κ < 10³: Momentum/Nesterov ajuda, L-BFGS converge mais rápido"
    else:
        return "κ ≥ 10³: Obrigatório usar L-BFGS, Newton ou pré-condicionamento"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Funções de conveniência


def compute_all_metrics(counts: Optional[Dict[str, int]] = None,
                        fitnesses: Optional[List[float]] = None,
                        gate_counts: Optional[Dict[str, int]] = None) -> Dict[str, float]:
    """
    Calcula todas as métricas disponíveis de uma vez.
    
    Caso de uso:
        - Coletar todas as métricas de uma geração em uma chamada.
        - Retornar dicionário pronto para inserção no CSV.
    
    Args:
        counts: Dicionário de medições (para measurement_entropy).
        fitnesses: Lista de fitness da população (para population_entropy).
        gate_counts: Dicionário de gates (para gate_entropy).
        
    Returns:
        Dicionário com todas as métricas calculadas.
    """
    metrics = {}
    
    if counts:
        metrics['measurement_entropy'] = measurement_entropy(counts)
        metrics['measurement_entropy_norm'] = measurement_entropy_normalized(counts)
    
    if fitnesses:
        metrics['population_entropy'] = population_entropy(fitnesses)
        metrics['population_entropy_norm'] = population_entropy_normalized(fitnesses)
    
    if gate_counts:
        metrics['gate_entropy'] = gate_entropy(gate_counts)
        metrics['gate_diversity'] = gate_diversity(gate_counts)
    
    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
