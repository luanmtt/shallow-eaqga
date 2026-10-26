# 02 — Plano de adaptação: nosso indivíduo dentro do EAQGA

Objetivo: usar a **heurística de entrelaçamento do EAQGA** (crossover consciente
de correlação) sobre o **nosso indivíduo** (genoma de portas, 8 qubits por
variável contínua), preservando a possibilidade de comparar com o baseline
original.

> **Atualização (arquitetura fechada):** as decisões mais recentes estão em
> [`leftovers.md`](leftovers.md), §10. Em resumo: `S = n`, portfólio QUBO,
> **Leitura A** (floresta reconstruída, genoma 1Q herdado), 1Q Hilbert +
> floresta de CNOTs, top-2 por geração, `k = 50`, melhor shot por elite. As
> opções B/C e o mapeamento `S = 8n` deste documento ficam como alternativas
> históricas.

Este documento não implementa nada; fixa o desenho, as mudanças necessárias e o
protocolo experimental.

---

## 1. O que significa "nosso indivíduo no EAQGA"

Há duas leituras possíveis, e elas levam a arquiteturas diferentes:

1. **EAQGA como gerador de circuito inicial.** O crossover do EAQGA produz um
   circuito a partir dos dois melhores; esse circuito vira o genoma inicial de
   cada indivíduo; nossa mutação/crossover passam a operar sobre ele.
2. **EAQGA como operador de crossover.** A população é nossa; a cada geração o
   crossover do EAQGA olha os dois melhores *bitstrings* e gera filhos no nosso
   genoma (em vez de gerar o circuito inteiro).

A leitura 1 é a mais fiel ao paper e a mais simples de validar: o EAQGA continua
sendo o algoritmo, e o nosso indivíduo é o "cromossomo" que ele passa a evoluir.
A leitura 2 é mais interessante cientificamente (compara operadores no mesmo
pipeline), mas exige definir como um grafo de entrelaçamento vira colunas de
portas discretas.

**Recomendação:** implementar as duas em fases, começando pela leitura 1.

## 2. Três arquiteturas

### A. Baseline fiel (sem nosso indivíduo)

Reimplementar o EAQGA como no paper: `S` qubits, 1 por bit, `RY` + `CNOT`,
elitismo top-2, `p_a`, `p_s`, `df(t)`, cadeias.

- **Prós:** valida a implementação contra os números do paper; isola bugs de
  tradução.
- **Contras:** não usa nosso indivíduo; é trabalho preliminar.
- **Uso:** obrigatório como controle experimental.

### B. Híbrido estrutural (nosso genoma, sem `RY`)

Manter o genoma discreto atual. O crossover do EAQGA gera apenas o **padrão de
entrelaçamento** e o **viés de bits**; a tradução para o nosso genoma é:

| Elemento do EAQGA | Tradução para o genoma |
|---|---|
| par entrelaçado `(i,j)` | uma coluna com `CNOT(i,j)` (ou `SWAP`, se `door_type` não tiver CNOT) |
| correlação negativa | `X(j)` na coluna anterior ao CNOT |
| bit do melhor pai = `0` | `X` inicial com probabilidade `1-p_a` (hoje é 0,5 fixo) |
| bit do melhor pai = `1` | `X` inicial com probabilidade `p_a` |
| qubit não entrelaçado | nenhuma porta (ou `I`) |

- **Prós:** não muda a representação; reutiliza 100% de `Individuo.py`; protótipo
  rápido; mantém a profundidade fixa.
- **Contras:** não reproduz os ângulos `RY`; a probabilidade `p_a` vira um viés
  na preparação (fora do genoma), então não é evoluída; o EAQGA fica reduzido a
  um gerador de CNOTs.
- **Uso:** protótipo e ablation ("só o grafo importa?").

### C. Híbrido completo (genoma com `RY` parametrizada)

Estender o genoma para aceitar `RY(θ)` com ângulo contínuo. Duas opções de
implementação:

**C1 — quinta posição na tupla do gene.** `(op, c1, c2, t, θ)`.
- **Prós:** ângulo viaja com o gene em qualquer cópia/corte.
- **Contras:** quebra todas as checagens `len(gene) == 4` (`_gene_para_tupla`,
  `_extract_gates`, `_reparar_coluna_*`, `__str__`, serialização); é a mudança
  mais invasiva.

**C2 — matriz paralela `theta`.** Gene continua `(op, c1, c2, t)`; ângulos em
`self.theta[q, i]` (float, `None` para portas sem parâmetro).
- **Prós:** mudanças localizadas em `construir_circuito_qiskit`, `_apply_gate`,
  mutação e crossover (fatiar `theta` junto com `genoma`); reparos existentes
  continuam válidos.
- **Contras:** é preciso garantir que crossover/mutação mantenham `theta`
  sincronizado; um gene `RY` órfão (sem theta) deve ser tratado.

**Recomendação:** C2 + novo `door_type = 4` (`'param'`), com `PORTAS1Q = ['RY']`
ou `['RY', 'X', 'H', ...]`. O EAQGA escreve `RY` + `CNOT`; nossa mutação pode
trocar `RY` por outra porta 1Q (e vice-versa), preservando ou re-sorteando `θ`.

## 3. Mapeamento de variáveis

EAQGA tem 1 qubit por bit. Nosso indivíduo tem 8 qubits por variável contínua.

### 3.1 Nível de bit (recomendado)

```
S = num_variaveis * 8
```

- O vetor de decisão do EAQGA é o **bitstring medido** do nosso indivíduo
  (concatenação dos 8 bits de cada variável).
- `P`/`N` são calculados sobre esses `S` bits, inclusive entre bits de variáveis
  diferentes (o que é legítimo: o genoma já permite CNOT entre qualquer par).
- O fitness continua sendo `f` aplicado à decodificação real de 8 bits, ou seja,
  a comparação com o EAQGA original fica no nível da representação, não do
  problema.
- **Cuidado:** 8 bits por variável criam `S = 16` para 2 variáveis; o número de
  pares candidatos cresce `O(S²)`. Para `n = 10`, `S = 80` e ~3.160 pares — ainda
  tratável, mas o sorteio precisa ser eficiente.

### 3.2 Nível de variável (não recomendado)

`S = num_variaveis`, cada bit = "variável incluída/excluída" ou sinal. Perde a
resolução de 8 bits e não corresponde ao nosso decoding. Só faz sentido para
problemas binários/QUBO.

### 3.3 Problema binário como caso particular

Para rodar em QUBO/portfólio, usar `N_QUBITS = 1` (ou um "modo binário") e
`f(x) = -objetivo(x)` para manter a convenção de minimização do repositório.
Isso permite comparar diretamente com os números do paper.

## 4. Extração do bitstring dos pais

O EAQGA precisa de dois bitstrings, não de dois genomas.

- **Indivíduo por shots:** o bitstring mais provável do cache
  (`simulation_results`) ou uma amostra do circuito. Para reprodutibilidade,
  usar `rng` derivado de `seed + geração + índice`.
- **Indivíduo exato:** amostrar da distribuição de probabilidades marginais ou
  usar o estado de maior probabilidade. Documentar a escolha — ela afeta o
  elitismo.
- **Alternativa determinística:** `x_i = 1` se `p1(qubit_i) ≥ 0.5`, senão `0`.
  Não é o que o paper faz (ele mede), mas elimina ruído de amostragem e é útil
  como ablation.

O pool de elitismo deve guardar `(bitstring, fitness, genoma?)`. Para a leitura 2
(crossover como operador), guardar o genoma campeão é útil para inicializar
filhos; para a leitura 1, basta o bitstring.

## 5. Laço híbrido proposto (leitura 1)

```
população ← N indivíduos (genoma aleatório, H na 1ª coluna)
avaliar população
Pool ← top-2 (bitstring, fitness)

para geração t = 1..T:
    para cada indivíduo p:
        x^p ← medir/extrair bitstring
    atualizar Pool (top-2 acumulado)
    para cada indivíduo p:
        pares ← amostrar_pares(Pool, p_s, Q, t)      # §6 do doc 01
        cadeias ← unir(pares)                         # floresta
        genoma_seed ← traduzir(cadeias, Pool)        # B ou C
        genoma_p ← crossover/mutação(genoma_seed, genoma_p)   # nossos operadores
        θ_p ← sortear/ajustar ângulos
    avaliar nova população
```

Pontos de atenção:

- **Elitismo duplo.** O EAQGA já tem elitismo top-2; o nosso pipeline tem
  `elite_size` (fração). Escolher um: ou o pool substitui o elitismo, ou o
  elitismo copia o topo e o pool só alimenta o crossover.
- **Crossover rate.** Manter `crossover_rate` do nosso pipeline decide se o
  filho recebe o genoma do EAQGA ou é cópia do pai; isso permite ablation.
- **Mutação.** Aplicar a nossa mutação **depois** da tradução, para não destruir
  o padrão de entrelaçamento; registrar quantas mutações atingiram arestas
  (`CNOT`) vs rotações (`RY`).
- **`annealing_window`.** Manter desligado: é variante nossa e confundiria a
  comparação.
- **Custo.** O EAQGA original avalia `N` circuitos de profundidade ~1 por
  geração; o híbrido avalia `N` circuitos de profundidade `num_portas`. Para
  comparação justa, reportar profundidade e contagem de portas 2Q, e também
  rodar com `num_portas` pequeno (ex.: 2–4).

## 6. Métricas

### 6.1 Do EAQGA (novas)

| Métrica | Definição |
|---|---|
| `pairs_candidates` | `|P| + |N|` antes da seleção |
| `pairs_selected` | arestas sorteadas por circuito |
| `edges` | arestas efetivas após união em cadeias |
| `chains` | número de componentes conexas |
| `max_chain_length` | maior cadeia |
| `entangled_fraction` | qubits com grau ≥ 1 / S |
| `two_qubit_gates` | CNOTs no circuito final |
| `circuit_depth` | profundidade após transpile para o basis alvo |
| `df(t)` | fator de decaimento |
| `pa`, `ps` | hiperparâmetros usados |

### 6.2 Do nosso QGA (existentes)

`gate_entropy`, `centroid_distance`, `population_radius`,
`pairwise_distance`, `measurement_entropy` (hoje não exportada), `mutations`,
`best_location`.

### 6.3 Protocolo e agregações

- Mesmos seeds, `population_size`, `generations`, shots e orçamento de
  avaliações.
- `R = 30–100` execuções por configuração. Reportar:
  - média `\bar f`, desvio `σ_f`, mediana e IQR;
  - melhor fitness por execução e curva média por geração;
  - `Δ%` apenas quando o baseline roda no **mesmo** problema/configuração.
- **AQGA não será implementado.** Para QUBO, usar as Tabelas I/II do paper como
  referência publicada; para benchmarks contínuos, comparar apenas EAQGA fiel,
  híbridos B/C e nosso QGA puro. O GA clássico de `classical/` é um controle
  opcional.
- Benchmarks: (i) QUBO/portfólio para falar com o paper; (ii) `schwefel`,
  `rastrigin`, `ackley` com 2 e 5 variáveis para falar com o repositório.
- Custo: tempo de parede, profundidade, 2Q gates, shots totais.
- Critério de sucesso: não basta melhor fitness; mostrar **fitness por
  profundidade** e **fitness por 2Q gate**, que é a tese do EAQGA (entrelaçar
  onde importa).
- Para o ótimo: brute force só até `S ≈ 20–25`; acima disso, solver
  (MILP/CP-SAT) ou comparação com o valor publicado.

Fórmulas:

$$
\bar f = \frac{1}{R}\sum_{r=1}^{R} f_r,
\qquad
\sigma_f = \sqrt{\frac{1}{R-1}\sum_{r=1}^{R}(f_r-\bar f)^2}
$$

$$
\Delta\% = \frac{\bar f_{\text{método}}-\bar f_{\text{baseline}}}
{\bar f_{\text{baseline}}}\times 100
$$

Agregação em Python (mesma convenção de `numpy.percentile`):

```python
import math

def aggregate(fitness_runs):
    n = len(fitness_runs)
    mean = sum(fitness_runs) / n
    var = (sum((x - mean) ** 2 for x in fitness_runs) / (n - 1)
           if n > 1 else 0.0)
    ordered = sorted(fitness_runs)

    def quantile(p):
        if n == 1:
            return ordered[0]
        pos = (n - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

    q1, q3 = quantile(0.25), quantile(0.75)
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "median": quantile(0.5),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }

def relative_improvement(best, baseline):
    return (best - baseline) / baseline * 100.0 if baseline else float("inf")
```

## 7. Fases

| Fase | Entrega | Critério de saída |
|---|---|---|
| **P0** | `eaqga/baseline.py`: EAQGA fiel, QUBO, sem nosso indivíduo | reproduz a ordem de grandeza da Tabela I/II para o EAQGA; AQGA/GA não são reexecutados |
| **P1** | instrumentação (métricas 6.1) + export CSV | curvas de fitness e entrelaçamento por geração |
| **P2** | `eaqga/individuo.py`: modo binário (`N_QUBITS=1` ou flag) e extração de bitstring | nosso indivíduo roda no problema QUBO |
| **P3** | tradução B (grafo → CNOT) + laço híbrido | primeira curva híbrida vs EAQGA |
| **P4** | tradução C (`RY` + `theta`) e crossover como operador | ablation B vs C vs EAQGA |
| **P5** | benchmarks contínuos + estatística | EAQGA vs B/C vs nosso QGA; AQGA fora |

## 8. Decisões pendentes

1. **Leitura 1 ou 2?** (EAQGA gera o circuito inicial × EAQGA é o crossover.)
2. **`RY` parametrizada (C) ou só estrutura (B)?** C é mais fiel; B é mais
   rápido e não muda `Individuo.py`.
3. **Bitstring dos pais:** amostra, argmax ou threshold 0,5?
4. **Elitismo:** pool top-2 do EAQGA substitui ou convive com `elite_size`?
5. **Problema:** começamos por QUBO/portfólio (fala com o paper) ou por
   `schwefel` (fala com o grupo)?
6. **`N_QUBITS`:** manter 8 fixo ou tornar configurável por indivíduo? O modo
   binário do EAQGA exige 1.
7. **Acoplamento `Q`:** usar `Σ` do portfólio, uma matriz QUBO arbitrária, ou
   derivar da função benchmark (ex.: Hessiano médio)? O paper só define para
   portfólio.
8. **Medição:** 1 shot por circuito (paper) ou múltiplos (menos variância)?
   Registrar a escolha e rodar as duas como ablation.

## Apêndice A — esqueleto corrigido do crossover

Este código é **documentação executável**, não a implementação final. Ele corrige
os problemas do snippet do Gemini: (i) o tipo `P`/`N` não é confundido com o bit
do controle; (ii) os pares são unidos em uma floresta e viram estrelas com um
controle; (iii) o RNG é explícito; (iv) o circuito é construído de fato. As
escolhas de estrela, de controle por grau e de tipo por paridade são decisões
nossas — o paper não as fixa.

```python
"""Crossover do EAQGA — esqueleto corrigido.

Decisões adotadas (o paper não fixa):
  - sorteio Bernoulli independente por par candidato;
  - união em componentes conexas; arestas que fecham ciclo são descartadas;
  - limite de S-1 arestas;
  - cada componente vira uma estrela: um controle, os demais alvos;
  - controle = nó de maior grau (desempate pelo menor índice);
  - tipo da aresta controle->nó = paridade dos tipos no caminho da árvore;
  - RNG explícito.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

from qiskit import QuantumCircuit

Kind = Literal["P", "N"]


@dataclass(frozen=True)
class Edge:
    control: int
    target: int
    kind: Kind


def correlation_pairs(x1, x2):
    """P/N conforme o paper: o padrão tem de valer nos DOIS pais."""
    P, N = [], []
    for i in range(len(x1)):
        for j in range(i + 1, len(x1)):
            if x1[i] == x1[j] and x2[i] == x2[j]:
                P.append((i, j))
            elif x1[i] != x1[j] and x2[i] != x2[j]:
                N.append((i, j))
    return P, N


def pair_probability(kind, sigma_ij, t, t_max, p_s):
    df = 0.5 + t / (2.0 * t_max)
    a = abs(float(sigma_ij))
    if kind == "P":
        return p_s * (df if sigma_ij > 0 else 1.0) * a
    return p_s * (1.0 if sigma_ij < 0 else df) * a


def sample_pairs(P, N, sigma_n, t, t_max, p_s, rng):
    chosen = []
    for kind, pairs in (("P", P), ("N", N)):
        for i, j in pairs:
            p = pair_probability(kind, sigma_n[i][j], t, t_max, p_s)
            if rng.random() < p:
                chosen.append((i, j, kind))
    return chosen


def forest_edges(pairs, S):
    """Une os pares e descarta arestas que fechariam ciclo."""
    parent = list(range(S))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out = []
    for i, j, kind in pairs:
        if len(out) >= S - 1:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[rj] = ri
        out.append((i, j, kind))
    return out


def to_star(edges, S):
    """Cada componente vira uma estrela com um único controle."""
    adj = defaultdict(list)
    for i, j, kind in edges:
        adj[i].append((j, kind))
        adj[j].append((i, kind))

    seen, star = set(), []
    for start in range(S):
        if start in seen or not adj[start]:
            continue

        component = []
        q = deque([start])
        seen.add(start)
        while q:
            u = q.popleft()
            component.append(u)
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)

        center = max(component, key=lambda u: (len(adj[u]), -u))

        # BFS a partir do centro: parent[u] = (pai, tipo da aresta)
        parent = {center: (None, "P")}
        q = deque([center])
        while q:
            u = q.popleft()
            for v, kind in adj[u]:
                if v not in parent:
                    parent[v] = (u, kind)
                    q.append(v)

        for u in component:
            if u == center:
                continue
            node, kind_acc = u, "P"
            while node != center:
                p, kind = parent[node]
                if kind == "N":
                    kind_acc = "N" if kind_acc == "P" else "P"
                node = p
            star.append(Edge(center, u, kind_acc))
    return star


def angle(bit, p_a):
    """P(0) = p_a se bit == 0; P(1) = p_a se bit == 1."""
    return 2.0 * math.acos(math.sqrt(p_a if bit == 0 else 1.0 - p_a))


def build_circuit(x_best, star, S, p_a):
    qc = QuantumCircuit(S, S)
    by_control = defaultdict(list)
    entangled = set()
    for e in star:
        by_control[e.control].append(e)
        entangled.update((e.control, e.target))

    for q in range(S):
        if q in by_control:
            qc.ry(angle(x_best[q], p_a), q)
            for e in by_control[q]:
                if e.kind == "N":
                    qc.x(e.target)
                qc.cx(e.control, e.target)
        elif q not in entangled:
            qc.ry(angle(x_best[q], p_a), q)

    qc.measure(range(S), range(S))
    return qc


def eaqga_crossover(x_best, x_second, sigma_n, t, t_max, S,
                    p_a=0.95, p_s=0.6, rng=None):
    rng = rng or random.Random(0)
    P, N = correlation_pairs(x_best, x_second)
    chosen = sample_pairs(P, N, sigma_n, t, t_max, p_s, rng)
    forest = forest_edges(chosen, S)
    star = to_star(forest, S)
    return build_circuit(x_best, star, S, p_a), star
```

`sigma_n` deve ser uma matriz `S×S` normalizada. Para problemas sem QUBO, use um
acoplamento uniforme (todos `1.0`) ou derive do Hessiano médio da função; nesse
caso `p_s` controla sozinho a densidade de arestas.

**Pontos que continuam em aberto** (e que o esqueleto apenas documenta):

- estrela vs cadeia de CNOTs: o paper diz "um controle, os demais alvos", mas
  também fala em "chained operations"; as duas leituras geram distribuições
  conjuntas diferentes quando há mais de duas arestas;
- escolha do controle: grau máximo é uma heurística; o paper não define;
- tipo da aresta sintetizada: paridade é a composição natural de correlações
  perfeitas, mas o paper não define para caminhos;
- ordem dos sorteios: a união depende da ordem; o paper não define se há
  embaralhamento antes do union-find.

