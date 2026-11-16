# PLANO — comparação original × crossover EA × bias

> **Prazo:** hoje.
> **Escopo mínimo para “acabar”:** E1, E2 e E3 rodados em `S=30`, `N=10`,
> `T=20`, 10 subsets × 10 runs, com comparação pareada. E4 (transpilação) só
> se sobrar tempo; senão fica documentado como próximo passo.
>
> **Nomenclatura:** este plano usa E1/E2/E3/E4. O `Improvements.md` usa
> E1–E6 para o híbrido; na consolidação, renomear os antigos para H1–H6.

---

## 0. Definição de pronto

Para cada experimento:

- [ ] código rodando e testado;
- [ ] 10 subsets × 10 runs (100 runs por configuração);
- [ ] CSV de runs + histórico + resumo;
- [ ] comparação pareada por `(subset, run)`;
- [ ] tabela com `best`, `avg`, `generation_best`, diversidade, portas e tempo;
- [ ] conclusão curta: melhorou, empatou ou piorou.

O critério não é “bateu o baseline”, é **entender o efeito de cada troca**:
original → crossover EA → bias.

---

## 1. Protocolo comum

| item | valor |
|---|---|
| `S` | 30 (screening); 40 só se necessário |
| `N` | 10 principal; 20 secundário |
| `T` | 20 |
| subsets | 10 disjuntos |
| runs | 10 por subset |
| dados | S&P 500 de 2024-09-30, 2023-10-01..2024-09-30 |
| backend | MPS |
| seed | 0, com seed por `(subset, run)` |
| comparação | pareada por `(subset, run)` |

Métricas obrigatórias:

- `fitness_internal` (o que o algoritmo otimiza);
- `fitness_bits = QUBO(bits)` do melhor bitstring;
- `best`, `generation_best`, `avg`, `std`;
- `unique_bits`, `bit_entropy`, `mean_hamming`;
- 1Q, 2Q, 3Q, profundidade;
- tempo/run.

---

## 2. E1 — indivíduo original com parâmetros originais

### 2.1 Objetivo

Rodar o indivíduo e os operadores do `~/projects/qga/quantum` no QUBO, sem
trocar crossover nem mutação. Ver como ele se comporta e onde perde.

### 2.2 Parâmetros originais

| parâmetro | valor |
|---|---|
| `num_portas` | 6 |
| `door_type` | 2 (Hilbert) |
| portas | H, X, Y, Z, S, S†, T, T†, CNOT, SWAP, TOFF, FREAD |
| `crossover_selection` | `vertical` |
| `crossover_rate` | 0.7 |
| `mutation_rate` | 0.3 |
| `elite_size` | 0.2 |
| `selection_method` | `tournament` |
| `tournament_size` | 3 |
| `sigma_es` | `False` |
| `annealing_window` | `False` |
| `shots` | 1024 (default original) |
| `N`, `T` | 50, 50 (default do `qga.py`) |

### 2.3 Adaptações inevitáveis

O original é contínuo; para QUBO binário:

- `N_QUBITS=1` (1 qubit por ativo);
- `domain=[(0,1)]*S`, `minimize=False`;
- MPS;
- seeds determinísticas no simulador e na mutação;
- fitness do original = `f(E[x])`; bitstring = threshold em 0.5.

**Reportar os dois fitness:** o interno do original e o `QUBO(bits)`. Sem isso a
comparação com baseline/ours fica confundida.

### 2.4 Braços

- **E1a — literal:** `N=50`, `T=50`, `shots=1024`. Caro; só se sobrar tempo.
- **E1b — comparável:** `N=10` (e opcionalmente 20), `T=20`, `shots=50`,
  mesmos subsets/runs do baseline. **Principal.**

Se o dia apertar: rodar só E1b.

### 2.5 Implementação

Arquivos:

- `eaqga/original_adapter.py`:
  - expor `crossover_selection`, `sigma_es`, `annealing_window`,
    `selection_method`, `tournament_size`;
  - defaults = originais;
  - adicionar `best_fitness_bits` / `fitness_bits`;
  - registrar os parâmetros no resultado.
- `eaqga.py`:
  - flags/CSV para os parâmetros do original;
  - colunas de `fitness_bits` no runs CSV.
- `eaqga/history.py`:
  - registrar `fitness_bits` por geração, se disponível.
- testes:
  - `tests/check_integrity.py` para defaults e serialização.

### 2.6 Rodada

```bash
python eaqga.py --algorithms original \
  --sizes 30 --population 10 --generations 20 \
  --subsets 10 --runs 10 --preset paper \
  --original-runs 10 --jobs 8 --save-table
```

### 2.7 Saída

Tabela E1 × baseline × ours; curvas de fitness; portas; tempo; caso de uso do
original. Conclusão esperada: onde o genoma original ganha/perde e se o
`fitness_bits` muda a leitura.

---

## 3. E2 — trocar o crossover para o crossover EA

### 3.1 Objetivo

Manter o indivíduo original, a mutação original e o orçamento de portas;
trocar **somente** o crossover `vertical` pelo crossover EAQGA.

### 3.2 Desenho

1. Obter os bitstrings dos dois pais (threshold dos genomas originais).
2. Construir a floresta EA com `build_crossover_forest`.
3. Mapear a floresta para o genoma original:
   - CNOTs em colunas com qubits disjuntos;
   - preencher slots restantes com genes 1Q dos pais;
   - reparar slots 2Q/3Q conforme as regras do original.
4. Aplicar a mutação original (`0.3`, sem ES).
5. Seleção/elitismo originais.

Variantes:

- **E2-STAR** (paper EAQGA) — principal;
- **E2-PATH** (`max_degree=2`) — sensibilidade de baixa profundidade.

### 3.3 “Mesmo número de portas”

Dois cortes:

- **E2-depth:** mesmo `num_portas=6` colunas.
- **E2-count:** mesma média de portas 1Q/2Q/3Q do E1 (trim/pad).

Principal: `E2-depth`; `E2-count` se o mapeamento mudar muito a contagem.

### 3.4 Implementação

Novo módulo, por exemplo `eaqga/original_crossover.py`:

```python
def crossover_ea(parent1, parent2, *, shape=STAR, max_ports=6,
                 budget=None, rng=None) -> tuple[Individuo, Individuo]:
    ...
```

- precisa devolver dois `Individuo` válidos do original;
- não alterar a mutação;
- registrar `forest_edges`, `forest_layers`, `gate_budget`, `1q/2q/3q`.
- hook no `original_adapter` via flag `--original-crossover ea`.

### 3.5 Rodada

```bash
python eaqga.py --algorithms original \
  --sizes 30 --population 10 --generations 20 \
  --subsets 10 --runs 10 --preset paper \
  --original-runs 10 --original-crossover ea \
  --jobs 8 --save-table
```

### 3.6 Saída

E1b × E2-STAR × E2-PATH. Conclusão: o crossover EA melhora o mesmo genoma e o
mesmo orçamento de portas?

---

## 4. E3 — ablação do bias

### 4.1 Objetivo

Medir quanto o bias em direção à bitstring anterior ajuda a explotação.

### 4.2 Variantes

- **E3a — `p_a`:** `{0.5, 0.75, 0.9, 0.95, 0.99}`.
  - 0.5 = sem bias; 0.95 = paper; 0.99 = confiança alta.
- **E3b — bias layer discreta:** `none`, `pre`, `post`, `both` com `H T^k H`.
- **E3c — determinístico:** `I`/`X` para forçar o bit anterior.

Rodar no melhor E2.

### 4.3 Métricas de “confiança”

- fitness;
- distância de Hamming entre o melhor filho e o pai anterior;
- taxa de reprodução exata do bitstring anterior;
- match das marginais por qubit;
- diversidade e `best - avg`;
- portas.

### 4.4 Rodada

Começar por `p_a` (5 valores × 100 runs ≈ 10–15 min). Se `p_a=0.95` já for o
melhor, testar bias layer no melhor ponto.

---

## 5. E4 — transpilação de RY

> Só executar se E1–E3 terminarem. Se não, deixar como próximo passo.

### 5.1 Objetivo

Comparar custo nativo, não “1 RY = 1 porta”.

### 5.2 Circuitos

- baseline EAQGA com RY;
- E1 original;
- E2/E3 vencedores;
- ours atual.

### 5.3 Alvos

1. **NISQ:** `sx`, `rz`, `x` + 2Q nativa (`cx`/`ecr`), coupling map realista.
2. **Clifford+T:** `h`, `s`, `t`, `tdg`, `cx`; reportar `T-count`.

### 5.4 Procedimento

- mesmo `Target`, `optimization_level`, `seed_transpiler`;
- contar 1Q, 2Q, `T`/`Tdg`, profundidade 1Q/total, erro estimado;
- avaliar lógico × transpilado com mesmos shots;
- Pareto fitness × custo, separado por regime.

### 5.5 Implementação

Script em `empirical/transpilation/transpilation.py` ou
`visualization/plot_transpiled_costs.py`, lendo os melhores circuitos/circuitos
representativos e gerando CSV + gráfico.

---

## 6. Cronograma de hoje

| bloco | tarefa | saída |
|---|---|---|
| 0:00–0:40 | E1: adapter + `fitness_bits` + testes | código/testes |
| 0:40–1:10 | E1b rodada 10×10 | CSV/tabela |
| 1:10–2:30 | E2: `crossover_ea` + testes + rodada | CSV/tabela |
| 2:30–3:00 | E3a `p_a` no melhor E2 | CSV/tabela |
| 3:00–3:20 | análise/plots e conclusão | resumo |
| 3:20–4:00 | E4 se der; senão documentar | script/plano |

Se E2 estourar o tempo: entregar E1 + E3 no crossover original e deixar E2
como próximo passo. O importante é não misturar as perguntas.

---

## 7. Riscos e cortes

- **Original lento/caro:** E1a 50/50/1024 pode inviabilizar o dia; rodar só E1b.
- **Mapeamento floresta → genoma original:** é o maior risco de E2; validar com
  um teste pequeno (`S=4`, 1 geração) antes da rodada completa.
- **Fitness interno × fitness_bits:** sem separar, a comparação com baseline é
  inválida.
- **Orçamento de portas:** se o EA estourar 6 colunas, truncar/reparar e
  registrar quantas portas foram cortadas.
- **Bias:** não misturar `p_a` com bias layer na mesma rodada; uma variável por
  vez.
- **E4:** não usar contagem lógica de RY como custo; só depois da transpilação.

---

## 8. Entregáveis

- `outputs/plan/E1/`, `E2/`, `E3/`, `E4/` com runs/history/summary;
- tabela final E1 × E2 × E3 × baseline/ours;
- gráfico de fitness e tabela de portas;
- nota curta em `docs/PLANO.md` ou `docs/leftovers.md` com a conclusão;
- nenhum experimento sem seed/subsets registrados.

---

## 9. Decisões assumidas

1. E1 principal = **E1b comparável** (`N=10`, `T=20`, `shots=50`); E1a literal
   só se sobrar tempo.
2. E2 principal = **STAR**; PATH como sensibilidade.
3. Orçamento principal = **6 colunas**; contagem total como secundário.
4. E3 começa por **`p_a`**; bias layer depois.
5. E4 só se E1–E3 fecharem.
6. O `fitness_bits` é obrigatório em E1/E2/E3.
