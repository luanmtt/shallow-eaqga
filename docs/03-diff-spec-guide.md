# 03 — Diff: guia do Gemini × nossos docs

Este documento registra o que o guia do Gemini (rundown de implementação, não
versionado neste repo) traz de novo, o que ele erra e o que mantivemos da nossa
leitura. O guia original **não** foi alterado; ele fica como registro histórico.

---

## 1. Tabela de cobertura

| Tema | Guia do Gemini | Nossos docs | Decisão |
|---|---|---|---|
| Universo e período (S&P 500, 01/10/2023–30/09/2024) | §1.1, §6 | `01`, §1.1 | **adotado** |
| `R_it`, `μ_i`, `Σ_ij` | §1.1 | `01`, §1.1 | **adotado** |
| QUBO e `q = 0.5` | §1.2 | `01`, §1.2 | **adotado + `Q` explícito** |
| Conjuntos `P` / `N` | §2.1 | `01`, §5 | ok |
| `Σ_n`, `df(t)`, `p_s` | §2.2 | `01`, §6 | ok |
| Bernoulli / árvore | §2.3 (afirma como se fosse do paper) | `01`, §10.6 (marca como interpretação) | **mantida a ressalva** |
| Ângulos `p_a` | §3.1 | `01`, §8.1 | ok |
| Circuito P/N | §3.2 (condiciona `X` ao bit do controle) | `01`, §8.2–8.3 e §10.13 | **corrigido** |
| Código Python | §4 (só parâmetros, sem cadeias/RNG) | `02`, Apêndice A | **substituído** |
| Média, desvio, `Δ%` | §5 | `01`, §9.3; `02`, §6.3 | **adotado** |
| Brute force `S=30/40` | §5, item 4 (repere o paper) | `01`, §9.4 | **corrigido com ressalva** |
| Ferramentas (MPS, `Sampler`, `ibm_quebec`) | §6 | `01`, §12 | **adotado** |
| Ambiguidades do paper | — | `01`, §10 | mantido |
| Métricas de entrelaçamento | — | `02`, §6.1 | mantido |
| Integração com nosso indivíduo | — | `02`, §2–5 | mantido |

## 2. Correções aplicadas

### 2.1 `P`/`N` não é o bit do controle

O guia (§3.2) aplica `X(n2)` quando `x_b1[n1] == 1`. Isso confunde o **tipo do
par** com o **valor do bit do controle**. A regra correta:

- `(n1,n2) ∈ P` → `RY(θ, n1)`, `CX(n1,n2)`, **sem** `X`;
- `(n1,n2) ∈ N` → `RY(θ, n1)`, `X(n2)`, `CX(n1,n2)`.

Contraexemplo: `x_b1 = [0,0,1,1,0]`, `x_b2 = [0,1,0,1,1]`. O par `(1,2)` é `N`,
mas `x_b1[1] = 0`; a regra do guia não aplicaria `X`, gerando correlação positiva
por engano. O pseudocódigo do paper tem o mesmo defeito; nós o tratamos como
ilustrativo, não como especificação.

### 2.2 Cadeias viram floresta e estrela

O guia sorteia pares independentes e devolve `selected_entangled_pairs` +
`entangled_qubits`, sem:

- unir componentes que compartilham qubits;
- escolher um controle por componente;
- descartar arestas que fecham ciclo;
- garantir `≤ S-1` arestas.

O Apêndice A de `02` faz união com union-find, descarta ciclos, escolhe o
controle de maior grau e converte cada componente em estrela. O tipo de cada
aresta sintetizada é a paridade dos tipos no caminho — uma decisão nossa, porque
o paper não a define.

### 2.3 RNG explícito

O guia usa `np.random.rand()` sem seed. O esqueleto usa `random.Random`, que é o
padrão do repositório (`Individuo`, `ParallelQuantumGA`).

### 2.4 O código do guia não constrói circuito

O guia só calcula parâmetros; não há `QuantumCircuit`, medição, fitness,
elitismo nem loop. O Apêndice A inclui `build_circuit` e `eaqga_crossover`, mas
ainda não é o pipeline completo (isso é P0/P1 em `02`, §7).

### 2.5 Brute force em `S=40`

`2^40 ≈ 1,1×10^12`. O guia repete "varrer todas as combinações" sem ressalva.
Em Python puro é inviável; usar solver (MILP/CP-SAT) ou C/GPU otimizado. Para
`S ≤ 20–25`, brute force direto é aceitável.

### 2.6 "Bernoulli" e "árvore" são interpretações

O paper não escreve "amostragem de Bernoulli" nem "estrutura de árvore". São
leituras razoáveis e nós as adotamos, mas ficam marcadas como decisão de
implementação em `01`, §10.6, não como fato do paper.

## 3. O que adotamos do guia

- Pipeline de dados: `YahooDataProvider` 0.4.1, S&P 500, período e subconjuntos.
  Funciona com Qiskit 2.5.2; para `yfinance` novo, usar `auto_adjust=False`.
- Fórmulas de `R_it`, `μ_i`, `Σ_ij`.
- Protocolo estatístico: `R = 100` (simulador), `R = 10` (hardware), média,
  desvio, `Δ%`, tabelas ×100.
- Ferramentas: Aer MPS, `Sampler`, 1 shot, `ibm_quebec` Eagle 127.
- A ideia de um `build_eaqga_circuit_parameters` como ponto de partida.

## 4. O que mantivemos dos nossos docs

- Lista de 14 ambiguidades/inconsistências do paper (`01`, §10).
- Métricas de entrelaçamento (`pairs_candidates`, `edges`, `chains`,
  `max_chain_length`, `entangled_fraction`, `2Q gates`, `depth`) em `02`, §6.1.
- Plano de integração do nosso indivíduo: leitura 1 vs 2, híbrido B vs C,
  matriz `theta`, mapeamento bit-level (`S = 8·num_variaveis`), elitismo duplo.
- Distinção explícita entre paper, interpretação e decisão de engenharia.

## 5. Escopo

- **AQGA não será implementado.** Os números de AQGA/GA do paper são referência
  publicada; não entram como baselines reexecutados.
- **EAQGA fiel** é o baseline a implementar; comparamos com as Tabelas I/II.
- **Problema inicial:** portfólio QUBO, `S = n`, 1Q Hilbert + floresta de CNOTs.
- **Benchmarks contínuos** ficam para depois; quando entrarem, comparar EAQGA
  fiel, híbrido e nosso QGA. O GA clássico de `classical/` é um controle
  opcional; AQGA fica de fora.
