# 00 — Formulação atual do QGA do repositório

Resumo do que existe hoje, com referências de arquivo/linha. Serve de contrato
para a adaptação ao EAQGA: tudo que for reutilizado deve casar com esta
formulação.

---

## 1. Indivíduo

Arquivo: `quantum/src/Individuo.py`

### 1.1 Genoma

O genoma é uma **matriz de portas** com forma `(num_qubits, num_portas)`:

```
num_qubits  = num_variaveis * N_QUBITS        # N_QUBITS = 8  (linha 20)
num_portas  = profundidade do circuito         # colunas
```

Cada célula é uma tupla `(op, c1, c2, t)` (`_gerar_genes_aleatorios`, linha 120):

| Campo | Significado |
|-------|-------------|
| `op` | nome da porta (`'K'` = célula vazia, `'I'`, `'X'`, `'H'`, `'CNOT'`, `'TOFF'`, …) |
| `c1` | primeiro controle (ou `None`) |
| `c2` | segundo controle (ou `None`) |
| `t` | alvo (ou o próprio qubit em portas 1Q) |

Portas multi-qubit são gravadas **redundantemente** em todas as linhas dos qubits
participantes (linhas 185–186 para 2Q; 200–204 para 3Q). Isso mantém a coluna
como uma camada válida e permite reparo de crossover.

### 1.2 Conjuntos de portas (`door_type`)

Definidos em `Individuo.__init__` (linhas 53–86) e espelhados em
`quantum/src/config.py`:

| `door_type` | Nome | 1Q | 2Q | 3Q |
|---|---|---|---|---|
| `-1` | classical_pure | X, I | — | — |
| `0` | classical | X, I | CNOT, SWAP | TOFF, FREAD |
| `1` | clifford | X, I, Y, Z, S, S†, H | CNOT, SWAP | — |
| `2` | quantum (Hilbert) | + T, T† | CNOT, SWAP | TOFF, FREAD |
| `3` | no_entanglement | 1Q apenas | — | — |

### 1.3 Inicialização

- Geração 0 e `door_type > 0`: sorteia `num_h ∈ [1, num_qubits]` e aplica `H` na
  primeira coluna dos qubits sorteados (linhas 132–145).
- Demais células: portas sorteadas respeitando qubits livres na coluna.
- **Estado inicial do circuito** (`construir_circuito_qiskit`, linhas 232–238):
  para cada qubit, aplica `X` com probabilidade 0,5 usando `random.Random(seed)`.
  Isso é separado do genoma e **não é evoluído**.
- `seed` do indivíduo é `seed_global + i` na população inicial
  (`ParallelQuantumGA._initialize_population`, linhas 293–308).

### 1.4 Circuito

`construir_circuito_qiskit(measure=True)` (linha 222):

```
|0> --[X aleatório por qubit]-- barrier --[coluna 0]-- ... --[coluna d-1]-- barrier --[measure]
```

- Cada coluna é aplicada por `_apply_gate` (linha 483), que ignora `'K'`.
- `TOFF` → `ccx`, `FREAD` → `cswap`; controles iguais são descartados.
- A medição é feita na base computacional, todos os qubits.

### 1.5 Decodificação real

`_converter_estado_para_reais` (linha 373):

```
para cada variável v:
    bits = estado[v*8 : v*8+8]          # string já invertida (Qiskit little-endian)
    inteiro = int(bits, 2)
    x_v = min_v + (inteiro / 255) * (max_v - min_v)
```

Ou seja: **8 qubits por variável contínua**, mapeamento linear do inteiro de 8
bits para o domínio. O estado é invertido (`estado[::-1]`) antes da conversão
porque o Qiskit ordena o bit menos significativo primeiro.

### 1.6 Fitness

`calcular_fitness` (linha 264), em duas modalidades:

1. **Por shots** (default): mede o circuito, converte cada bitstring em reais e
   calcula a média ponderada pelas contagens; aplica `f` a essa média:
   `fitness = f( 𝔼_{z∼P_C}[x(z)] )`.
2. **Exato** (`exact=True`, linha 309): extrai o statevector, calcula as
   probabilidades marginais `p1` de cada qubit e obtém `𝔼[x]` analiticamente,
   sem ruído de amostragem.

Em ambos os casos o fitness é o da **esperança da posição**, não o melhor shot.

### 1.7 Entrelaçamento entre indivíduos (opcional)

`construir_circuito_par` (linha 397) e `calcular_fitness_par` (linha 435):

- 2×`num_qubits` qubits.
- Para cada qubit `i`, cria um par de Bell `|Φ+⟩` entre `i` e `i + num_qubits`
  (`H(i)`, `CX(i, i+num_qubits)`).
- Aplica o circuito de cada indivíduo no seu registrador.
- O par é avaliado numa única simulação; o fitness de cada um vem das marginais
  do seu registrador.
- Pareamento sorteado uma vez (`_sorteiar_pares`, linhas 311–325) e mantido.

## 2. Operadores genéticos

### 2.1 Mutação (`mutacao`, linha 972)

- Taxa por **gene** (`rng.random() < taxa_mutacao`), varrendo qubits × colunas.
- `_aplicar_mutacao_segura` (linha 769) troca o tipo da porta:
  - 1Q → 1Q: substituição direta;
  - 1Q → 2Q/3Q: ocupa slots 1Q livres na mesma coluna;
  - 2Q → 1Q/2Q/3Q; 3Q → 2Q/3Q (nunca 3Q → 1Q).
- Operações já tentadas e posições afetadas são bloqueadas na mesma chamada.
- `annealing_window=True` (variante própria, linha 999) congela colunas/qubits
  iniciais conforme o progresso — **não é do paper** e fica desligado por default.
- `sigma_es` (linhas 950–969): auto-adaptação estilo ES,
  `σ' = σ · exp(τ · N(0,1))`, `τ = 1/√(2√n)`, com `σ` limitado a `[0.01, 10]`.
  A taxa efetiva é `taxa_mutacao * sigma_es`.

### 2.2 Crossover

Quatro variantes (`crossover`, linha 1488):

| Método | Corte | Reparo |
|---|---|---|
| `vertical` | coluna `corte ∈ [1, d-1]` | nenhum (colunas inteiras) |
| `horizontal` | qubit `q_corte` | `_reparar_coluna_horizontal` |
| `mixed` | coluna por filho + corte por variável | `_reparar_coluna_mixed` |
| `paper` | camada δ na 2ª metade + `k` qubits menos significativos | portas que cruzam a fronteira não trocam |

Regras de reparo (`_apply_2q_fix`, linha 1176; `_apply_3q_fix`, linha 1188):
CNOT fragmentado → `I` no controle + `X` no alvo; SWAP → `I`/`I`;
TOFF/FREAD degradam conforme o padrão de origem.

Filhos são criados via `Individuo(...)` (genoma aleatório descartado) e recebem
`genoma` do crossover. `sigma_es` é interpolado entre os pais
(`pass_sigma_es`, linha 1254).

## 3. Pipeline (`quantum/src/ParallelQuantumGA.py`)

```
init população → evaluate → elitismo → seleção → crossover → mutação → repete
```

- Avaliação paralela com joblib (`n_jobs`), payload mínimo (funções de módulo).
- `evaluate_population` (linha 328) ordena a população pelo fitness.
- Seleção: torneio (`tournament_size`) ou roleta.
- `elite_size` é uma **fração** convertida em `max(1, int(fração * N))` (linha 192).
- `crossover_rate` decide se o par sofre crossover ou é copiado (linhas 66–71).
- Ao final há uma avaliação extra (linha 941) e menus interativos de exportação.

## 4. Métricas

### 4.1 Coletadas por geração

`history` (linhas 267–280) e exportação (`_export_csv`, linha 583):

**`evolution_data.csv`**

| Coluna | Origem |
|---|---|
| `generation` | índice |
| `best_fitness` | melhor fitness acumulado (`self.best_fitness`) |
| `mean_fitness` | média da população na geração |
| `median_fitness` | mediana |
| `best_location` | posição decodificada do melhor (`_extract_location`) |
| `mutations` | total de mutações aplicadas na geração |
| `gates` | contagem de portas do melhor (`_extract_gates`) |

**`metrics.csv`**

| Coluna | Fórmula / origem |
|---|---|
| `gate_entropy` | `H = -Σ p_g log2 p_g` sobre a contagem de portas do melhor (`features/metrics.py:137`) |
| `centroid_distance` | `(1/N) Σ ‖x_i - x̄‖`, posições normalizadas pelo range do domínio (`metrics.py:187`) |
| `population_radius` | `max_i ‖x_i - x̄‖` (mesma normalização) |
| `pairwise_distance` | `(2/(N(N-1))) Σ_{i<j} ‖x_i - x_j‖` (`metrics.py:228`) |

Diversidade é coletada conforme `exp_retrieval = [ativo, intervalo]`; por default
só no fim. `metrics.py` também oferece `measurement_entropy` (entropia de
Shannon dos shots), `population_entropy` (histograma de fitness) e
`hessian_condition_number`, mas **não são gravadas** no CSV hoje.

### 4.2 Normalização

`centroid_distance`, `population_radius` e `pairwise_distance` dividem as
posições pelo range de cada variável antes de calcular distâncias — portanto são
adimensionais e comparáveis entre funções com domínios diferentes (ver
`quantum/docs/notions.md` para as faixas de interpretação).

## 5. Funções benchmark

`quantum/src/benchmark_functions.py`:

`sphere`, `rastrigin`, `ackley`, `griewank`, `rosenbrock`, `schwefel`,
`zakharov` — todas de minimização, vetoriais. `get_benchmark(name, n)` devolve
`(fn, domain, optimum, optimum_label)`.

Foco atual do grupo: **schwefel multimodal**.

## 6. O que o EAQGA não tem e o que ele exige

| Nosso QGA | EAQGA |
|---|---|
| genoma de portas evoluído | circuito gerado por heurística a cada geração |
| 8 qubits por variável contínua | 1 qubit por variável binária |
| fitness = `f(𝔼[x])` | fitness = objetivo QUBO da bitstring medida |
| portas discretas (Clifford/T/Toffoli) | `RY(θ)` + `CNOT` (ângulos contínuos) |
| mutação/crossover de genes | seleção probabilística de pares + cadeias de entrelaçamento |
| minimização | maximização |
| profundidade fixa `num_portas` | profundidade = 1 camada de rotações + CNOTs |

Essas diferenças são o objeto de `02-plano-de-adaptacao.md`.
