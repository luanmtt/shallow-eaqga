# 01 — EAQGA: especificação do baseline

Paper: **EAQGA: A Quantum-Enhanced Genetic Algorithm with Novel Entanglement-Aware
Crossovers** — Kashfi Haghighi, Fortin-Deschênes, Pere, Camus (PINQ2), 2025.
[arXiv:2504.17923](https://arxiv.org/abs/2504.17923) · IEEE QCE25 (submetido).

Este documento isola o algoritmo em si, sem interpretação, e marca onde o texto
é ambíguo ou inconsistente. É a referência para a reimplementação fiel.

---

## 1. Problema, dados e fitness

### 1.1 Dados e estimadores

O paper usa ações reais do **S&P 500**:

- **Fonte:** `YahooDataProvider` do Qiskit Finance 0.4.1 (Yahoo Finance).
- **Compatibilidade verificada (2026-10-26):** `qiskit-finance==0.4.1`
  importa e roda com `qiskit==2.5.2` (testado com `qiskit-aer==0.17.2`,
  `qiskit-algorithms==0.4.0`, `qiskit-optimization==0.7.0`).
- **Ressalva do `YahooDataProvider`:** com `yfinance 1.7.0`, o default
  `auto_adjust=True` remove a coluna `Adj Close`, que o provider 0.4.1 espera;
  o erro é `QiskitFinanceError: Accessing Yahoo Data failed.` Correção:
  chamar `yf.download(..., auto_adjust=False)` ou aplicar um monkeypatch antes
  de `provider.run()`. `RandomDataProvider` funciona sem ajuste.
- **Recomendação para o repo novo:** usar `yfinance` direto com
  `auto_adjust=False` e calcular `μ`/`Σ` pelas fórmulas abaixo; `qiskit-finance`
  fica como opção de validação/fallback.
- **Workaround do provider** (se formos usá-lo):

```python
import yfinance as yf

_orig = yf.download
def _download(*args, **kwargs):
    kwargs.setdefault("auto_adjust", False)
    return _orig(*args, **kwargs)
yf.download = _download  # antes de YahooDataProvider.run()
```

- **Período:** 01/10/2023 a 30/09/2024, intervalo diário.
- `P_it` = preço do ativo `i` no dia `t`; `T` = número de dias.
- **Retorno diário simples:**

$$
R_{it} = \frac{P_{it}}{P_{i,t-1}} - 1
$$

- **Retorno médio:**

$$
\mu_i = \frac{1}{T}\sum_{t=1}^{T} R_{it}
$$

- **Covariância:**

$$
\Sigma_{ij} = \frac{1}{T-1}\sum_{t=1}^{T}(R_{it}-\mu_i)(R_{jt}-\mu_j)
$$

- **Subconjuntos do simulador:** 10 de 30 ativos e 10 de 40, sem sobreposição.
- **Hardware:** 100 ativos sorteados.

### 1.2 QUBO e fitness

$$
\max_x\ \mu^T x - q\, x^T \Sigma x
\qquad \text{s.t.}\quad x_i \in \{0,1\}\ \forall i
$$

- `x ∈ {0,1}^S`, `S` = número de ativos (bits da solução).
- `q` = aversão a risco; `q = 0.5` nos experimentos.
- Quanto maior, melhor. Sem penalidade de cardinalidade nas execuções relatadas.
- `S ∈ {30, 40, 100}`, `T_max = 20`, `N ∈ {10, 20}`.

**Matriz QUBO.** O paper não escreve `Q` explicitamente; para usar um solver,
fixe a convenção. Com `f(x) = x^T Q x` e `Q` simétrica:

$$
Q_{ii} = \mu_i - q\,\Sigma_{ii},
\qquad
Q_{ij} = -q\,\Sigma_{ij}\quad (i \ne j)
$$

Se a convenção for triangular superior (soma em `i < j`), use
`Q_ij = -2q Σ_ij`. O `Σ_n` da §6 serve **só** para a seleção de pares, não para
o fitness.

## 2. Representação

- `S` qubits, **1 qubit por variável binária** (ativo).
- Cada indivíduo/circuito é medido **uma única vez** na base computacional para
  produzir uma bitstring `x^p`.
- O fitness é avaliado sobre essa bitstring — não sobre a esperança.
- O circuito **é** o genótipo: não há genoma separado de parâmetros.

## 3. Inicialização

```
para cada circuito p = 1..N:
    |ψ_0> = |0>^{⊗S}
    aplicar H em todos os S qubits
```

Equivalente a amostrar uniformemente `{0,1}^S` na primeira medição.

## 4. Laço principal

```
Pool_elitismo ← {x^{b1}, x^{b2}}          # dois melhores já vistos
para iteração t = 1..T_max:
    avaliar/mostrar os N circuitos
    atualizar e ordenar Pool (top-2)
    para cada circuito p = 1..N:
        SelectedEntangledQubits ← sorteio(Pool, p_s, Q, t)
        QC^p ← QC_0
        codificar entrelaçamentos e rotações de QC^p a partir de Pool
```

- O pool é **global/acumulado**: guarda os dois melhores resultados vistos até
  então, não apenas os dois melhores da geração corrente.
- `N` circuitos novos por geração; todos derivam dos mesmos dois pais, mas cada
  um sorteia seu próprio subconjunto de pares entrelaçados.

## 5. Detecção de correlação

Dados os dois pais `x^{b1}`, `x^{b2}`:

- **Correlação positiva** `P`: pares `(i,j)` tais que
  `x_i^{b1} = x_j^{b1}` **e** `x_i^{b2} = x_j^{b2}`.
  Estados-alvo: `|ψ^p⟩ = α|00⟩ + β|11⟩`.
- **Correlação negativa** `N`: pares `(i,j)` tais que
  `x_i^{b1} ≠ x_j^{b1}` **e** `x_i^{b2} ≠ x_j^{b2}`.
  Estados-alvo: `|ψ^n⟩ = α|01⟩ + β|10⟩`.

Só entram pares que são Bell-like **nos dois pais simultaneamente**. O exemplo do
paper (`x^{b1} = [0,0,1,1,0]`, `x^{b2} = [0,1,0,1,1]`):

| par | pai 1 | pai 2 | conjunto |
|---|---|---|---|
| (1,2) | 0≠1 | 1≠0 | `N` |
| (1,4) | 0=0 | 1=1 | `P` |

## 6. Seleção probabilística dos pares

O acoplamento vem da matriz QUBO `Q`; no caso de portfólio, `Σ`. Normalização:

$$
\Sigma_n = \frac{\Sigma}{\max_{i,j} |\Sigma_{i,j}|}
$$

Fator de decaimento na iteração `t`:

$$
df(t) = 0.5 + \frac{t}{2\,T_{\max}}
$$

Probabilidade de selecionar o par `(i,j)`:

$$
p_s^{(i,j)} =
\begin{cases}
p_s \cdot df(t) \cdot |\Sigma_n^{(i,j)}|, & (i,j)\in\mathcal P \ \text{e}\ \Sigma_n^{(i,j)} > 0 \\
p_s \cdot |\Sigma_n^{(i,j)}|, & (i,j)\in\mathcal P \ \text{e}\ \Sigma_n^{(i,j)} \le 0 \\
p_s \cdot |\Sigma_n^{(i,j)}|, & (i,j)\in\mathcal N \ \text{e}\ \Sigma_n^{(i,j)} < 0 \\
p_s \cdot df(t) \cdot |\Sigma_n^{(i,j)}|, & (i,j)\in\mathcal N \ \text{e}\ \Sigma_n^{(i,j)} \ge 0
\end{cases}
$$

Leitura: a penalidade (`df < 1`) cai sobre pares cujo sinal empírico **contradiz**
o sinal do acoplamento. No início `df = 0.5`; no fim `df → 1.0`, e a seleção passa
a depender só da evidência dos pais. `p_s = 0.6` nos experimentos.

## 7. Cadeias de entrelaçamento

- Pares selecionados que compartilham qubits são unidos em **cadeias**
  (ex.: `(q1,q2)` e `(q2,q4)` → cadeia `{q1,q2,q4}`).
- Em cada cadeia, **um qubit é o controle** e os demais são alvos; as operações
  são encadeadas a partir do controle.
- Consequência: com `S` qubits, o total de arestas é no máximo `S - 1`
  (uma floresta, não um grafo arbitrário).
- O paper **não especifica** como o controle é escolhido, nem como evitar
  ciclos ao unir pares. Ver §10.

## 8. Codificação no circuito

### 8.1 Qubit não entrelaçado

Só `RY(θ)`. Seja `p_a` a probabilidade de reproduzir o bit do melhor pai
(`p_a = 0.95` nos experimentos). Como `P(0) = cos²(θ/2)`:

| bit do melhor pai | ângulo | resultado |
|---|---|---|
| `0` | `θ = 2·arccos(√p_a)` | `P(0) = p_a` |
| `1` | `θ = 2·arccos(√(1-p_a))` | `P(1) = p_a` |

### 8.2 Par entrelaçado positivo (`P`)

Circuito da Fig. 3(a): rotação no controle `n1`, depois `CNOT(n1, n2)`.

```
RY(θ, n1)
CX(n1, n2)
```

Estado: `cos(θ/2)|00⟩ + sin(θ/2)|11⟩`. Com `θ = 2·arccos(√p_a)`, a probabilidade
de `00` é `p_a`; com `θ = 2·arccos(√(1-p_a))`, a de `11` é `p_a`.

### 8.3 Par entrelaçado negativo (`N`)

Circuito da Fig. 3(b): `X` no alvo **antes** do CNOT.

```
RY(θ, n1)
X(n2)
CX(n1, n2)
```

Estado: `cos(θ/2)|01⟩ + sin(θ/2)|10⟩`. Os mesmos ângulos de 8.1 valem para o
bit do controle.

### 8.4 Pseudocódigo do paper (Algorithm 1)

```
1  Inicialização:
2      QC_0 ← |0>^{⊗S}
3      for p = 1..N:
4          QC^p ← QC_0
5          for n = 0..S-1:
6              QC^p.H(n)
9  for i = 1..T_max:
10     Avaliação:
11         for p = 1..N:
12             medir QC^p → x^p
14     Crossover:
15         atualizar Pool_elitismo ← {x^{b1}, x^{b2}}
16         QC_0 ← |0>^{⊗S}
17         for p = 1..N:
18             SelectedEntangledQubits ← sorteio(Pool, P_s, Q)
19             QC^p ← QC_0
20             for (n1,n2) in SelectedEntangledQubits:
21                 if x^{b1}_{n1} == 0:
22                     QC^p.RY(2*arccos(sqrt(P_a)), n1)
23                     QC^p.CX(n1, n2)
25                 else:
26                     QC^p.RY(2*arccos(sqrt(1-P_a)), n1)
27                     QC^p.X(n2)
28                     QC^p.CX(n1, n2)
31             for n = 1..S:
32                 if n1 ∉ SelectedEntangledQubits:
33                     if x^{b1}_{n1} == 0:
34                         θ ← 2*arccos(sqrt(P_a))
36                     else:
37                         θ ← 1 - 2*arccos(sqrt(P_a))     # ← inconsistente
39                     QC^p.RY(θ, n)
44 Retorna x^{b1}
```

## 9. Resultados relatados e protocolo de reporting

### 9.1 Simulador

- Aer MPS, 30 e 40 ativos, 20 iterações.
- 10 subconjuntos de 30 e 10 de 40, sem sobreposição; **100 execuções** por
  subconjunto.
- Tabelas I/II (valores **multiplicados por 100** para legibilidade):

| `S` | `N` | EAQGA | AQGA | GA |
|---|---|---|---|---|
| 30 | 10 | 2.1295 | 1.9999 | 1.9155 |
| 30 | 20 | 2.1531 | 2.0819 | 2.0080 |
| 40 | 10 | 2.3858 | 2.1678 | 2.0673 |
| 40 | 20 | 2.4435 | 2.2837 | 2.1879 |

### 9.2 Hardware

- `ibm_quebec`, Eagle de 127 qubits, 100 ativos, `N = 10`, `T_max = 20`,
  **10 execuções**.
- `+33.6%` sobre GA e `+37.2%` sobre AQGA em fitness médio.
- Os autores não reivindicam vantagem quântica formal: o número de
  entrelaçamentos é limitado e o circuito é simulável classicamente.

### 9.3 Métricas de agregação

Para `R` execuções independentes:

$$
\bar f = \frac{1}{R}\sum_{r=1}^{R} f(x_{\text{best}}^{(r)})
$$

$$
\sigma_f = \sqrt{\frac{1}{R-1}\sum_{r=1}^{R}
\left(f(x_{\text{best}}^{(r)})-\bar f\right)^2}
$$

$$
\Delta\% = \frac{\bar f_{\text{método}}-\bar f_{\text{baseline}}}
{\bar f_{\text{baseline}}}\times 100
$$

- `R = 100` no simulador; `R = 10` no hardware.
- Com `R = 10`, o desvio padrão é frágil; reportar também mediana e IQR.
- Não confundir os valores das Tabelas I/II (×100) com o fitness bruto.

### 9.4 Ótimo de referência (brute force)

O paper diz que comparou com o ótimo por **brute force** para `S = 30` e
`S = 40`. Ressalva: `2^40 ≈ 1,1×10^12`; uma varredura ingênua em Python é
inviável. Para reproduzir, usar um solver (MILP/CP-SAT) ou C/GPU otimizado;
para `S ≤ 20–25`, brute force direto é viável.

### 9.5 Escopo de comparação deste trabalho

- **AQGA não será implementado.** Os números de AQGA e GA do paper entram como
  **referência publicada**, não como baseline reexecutado.
- **EAQGA fiel** é o baseline que implementaremos; comparamos nossa implementação
  com os números reportados (mesma ordem de grandeza).
- **Benchmarks contínuos** (schwefel etc.): comparar EAQGA fiel, híbrido B,
  híbrido C e nosso QGA puro. O GA clássico de `classical/` é um controle
  opcional e barato; AQGA fica de fora.
- Em QUBO, comparar com as Tabelas I/II; em benchmarks contínuos, não há número
  de AQGA/GA do paper para comparar.
- Os números do paper só são comparáveis se usarmos os **mesmos subconjuntos**,
  o mesmo `q` e o mesmo orçamento; caso contrário, servem como *sanity check*,
  não como baseline estatístico.

## 10. Ambiguidades e inconsistências a resolver na implementação

1. **Ângulo da linha 37** do Algorithm 1 (`θ = 1 - 2·arccos(√p_a)`) contradiz a
   fórmula da §III-B (`θ = 2·arccos(√(1-p_a))`). A segunda é a correta
   matematicamente; a primeira parece erro de digitação.
2. **Índices**: o pseudocódigo mistura `n1` (índice de par) com `n` (qubit) nas
   linhas 31–39. O laço deve ser "para cada qubit `n` não participante de
   nenhum par selecionado".
3. **Escolha do controle** na cadeia não é definida. Opções naturais: primeiro
   qubit da cadeia, grau máximo, ou sorteio. Isso altera a distribuição conjunta
   quando há arestas positivas e negativas na mesma cadeia.
4. **Construção das cadeias**: o paper não diz se a união é gulosa na ordem dos
   pares, se rejeita arestas que criam ciclos, nem se um qubit pode aparecer em
   mais de uma cadeia. A restrição "no máximo `S-1` arestas" sugere floresta.
5. **Arestas negativas em cadeia**: o `X(n2)` é aplicado por aresta. Se o
   controle tem dois alvos, um positivo e um negativo, é preciso garantir que o
   `X` não seja sobrescrito e que a rotação do controle seja única.
6. **`p_a` e `p_s` por circuito**: `p_s^{(i,j)}` é uma probabilidade de seleção;
   não está claro se os sorteios são independentes por par (podendo gerar
   muitas arestas) ou se há normalização/limite. A restrição de cadeia sugere
   independência + união posterior.
7. **Atualização do pool**: "top two at each iteration" + "updated and sorted
   after each generation" pode significar top-2 da geração ou top-2 acumulado.
   A leitura mais consistente com "best results found up to each point" é
   **acumulado**.
8. **Duplicatas no pool**: se `x^{b1} = x^{b2}`, todos os pares são positivos
   triviais; o paper não trata o caso.
9. **`Q` vs `Σ`**: a fórmula usa `Σ_n`, mas o texto chama de "QUBO matrix `Q`".
   Para outros problemas, é preciso definir o acoplamento (`Q_ij`) e normalizá-lo
   pelo máximo absoluto.
10. **Medição única**: com um shot por circuito, o fitness tem variância alta.
    O paper usa 100 execuções para estimar a média, mas cada execução usa uma
    medição por circuito por geração. Reimplementações devem registrar essa
    escolha (shots = 1 vs múltiplos) para comparação justa.
11. **Sem mutação explícita**: o paper descreve o crossover como análogo a
    crossover + mutação (variação vem do sorteio de pares e da natureza
    probabilística das rotações). Não há operador de mutação separado.
12. **Profundidade**: uma camada de `RY` + CNOTs encadeados. A profundidade
    efetiva depende do transpilador e do acoplamento do hardware; para
    comparação com nosso `num_portas`, medir após transpile.
13. **O `if` do pseudocódigo testa o bit do controle, não o tipo do par.**
    As linhas 21–28 escolhem `X(n2)` quando `x^{b1}_{n1} == 1`; isso faria o
    mesmo par positivo virar negativo só porque o controle é 1, contrariando a
    §III-B, que define o tipo pelo padrão dos **dois** pais. A regra correta é:
    `(n1,n2) ∈ P` → sem `X`; `(n1,n2) ∈ N` → `X(n2)`. O pseudocódigo deve ser
    lido como ilustrativo, não como especificação.
14. **Aresta negativa e bit do controle**: como o `X(n2)` é aplicado antes do
    CNOT, a distribuição conjunta para `(n1,n2) ∈ N` fica
    `cos(θ/2)|01⟩ + sin(θ/2)|10⟩`; o bit do controle é o que carrega `p_a`.
    Se a cadeia tiver arestas positivas e negativas, o controle precisa ser
    escolhido de forma que cada aresta seja codificada corretamente.

## 11. Baselines do paper (referência, não reimplementação)

### 11.1 GA clássico

- População inicial: bitstrings binárias aleatórias.
- Seleção: roleta proporcional ao fitness, com fitness deslocado para não
  negativo.
- Crossover: ponto único, `p = 0.85`.
- Mutação: flip de um bit aleatório, taxa `0.03`.
- Elitismo não descrito no paper.

### 11.2 AQGA (quantum-inspired)

- Cromossomo quântico `[α_1 … α_n; β_1 … β_n]`.
- Direção da rotação pelo determinante:

$$
D = \det\begin{vmatrix}
\alpha_j^b & \alpha_j^i \\
\beta_j^b & \beta_j^i
\end{vmatrix},
\qquad
\text{direction} =
\begin{cases}
-\operatorname{sgn}(D), & D \ne 0 \\
\pm 1, & D = 0
\end{cases}
$$

- Magnitude adaptativa:

$$
\theta_i = \theta_{\max} -
\frac{\theta_{\max}-\theta_{\min}}{\text{iter}_{\max}}\,\text{iter}
$$

- `θmax = 0.25`, `θmin = 0.15`; mutação flip α/β com taxa `0.05`.
- *Disaster operation*: se não melhora por 6 iterações, reinicializa 20% dos
  piores em superposição uniforme.
- **Não implementar**: usar apenas os valores reportados nas Tabelas I/II e no
  hardware.

### 11.3 O que aproveitamos

- As Tabelas I/II, o desvio e o `Δ%` do hardware como referência publicada.
- A configuração de `q`, `p_a`, `p_s`, `N` e `T_max` para reproduzir o EAQGA.
- A comparação textual com GA/AQGA, sem reexecutá-los.

## 12. Ferramentas de execução

- **Dados:** `qiskit-finance==0.4.1` + `yfinance` (com `auto_adjust=False`), ou
  `yfinance` direto + as fórmulas da §1.1. `RandomDataProvider` é o fallback
  sintético. Testado com `qiskit==2.5.2`.
- **Simulador:** Qiskit Aer MPS (`matrix_product_state`), adequado a
  entrelaçamento limitado. Para os nossos 16–40 qubits, `AerSimulator` com
  statevector também é viável; para 100 qubits, MPS é obrigatório.
- **Amostragem:** `Sampler` primitive, **1 shot por circuito** por geração.
- **Hardware:** `ibm_quebec` (IBM Quantum System One, Eagle, 127 qubits), 100
  qubits usados, 10 execuções.
- **Nosso repo:** `AerSimulator` com shots ou statevector exato. Para o baseline
  EAQGA, usar 1 shot para ser fiel e registrar a variância; para o híbrido,
  permitir `shots > 1` como ablation.
