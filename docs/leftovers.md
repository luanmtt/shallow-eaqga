# Leftovers — decisões adiadas e alternativas

Este documento reúne o que **não** foi fechado na discussão de arquitetura:
alternativas, dúvidas em espera e pontos que podem virar ablation ou artigo
separado. Não é implementação; é memória de decisão.

Ordem de leitura sugerida: `01` (baseline), `02` (plano), `03` (diff), este.

---

## 1. Agendamento de CNOTs

### 1.1 O problema

Um CNOT usa dois qubits. Numa mesma camada, só podem entrar CNOTs que usam
**qubits disjuntos**. Se várias arestas compartilham o mesmo qubit, elas são
sequenciais. Isso vale especialmente para a **estrela** (um controle, vários
alvos): todas as arestas compartilham o controle, então a profundidade é o
fan-out.

A floresta é um **grafo de correlações**; o desenho “de cima para baixo” é
arbitrário. A direção do CNOT (controle → alvo) é semântica, não espacial.

### 1.2 Formas e profundidade

| Forma | Arestas | Camadas | Profundidade |
|---|---|---|---|
| Matching | `(0,1), (2,3)` | `{(0,1), (2,3)}` | 1 |
| Path | `(0,1), (1,2), (2,3)` | `{(0,1), (2,3)}`, `{(1,2)}` | 2 |
| Star | `(0,1), (0,2), (0,3)` | `{(0,1)}`, `{(0,2)}`, `{(0,3)}` | 3 |

Para uma floresta, a profundidade mínima é o **grau máximo** `Δ` (edge
coloring). Matching: `Δ=1`; caminho: `Δ=2`; estrela com `m` folhas: `Δ=m`.

### 1.3 Soluções

1. **Matching por camada** — coloca na mesma camada apenas CNOTs disjuntos.
   Profundidade = número de matchings. Simples, mas pode precisar de muitas
   camadas.
2. **Greedy layer packing** — percorre as arestas e coloca cada CNOT na primeira
   camada em que controle e alvo estão livres. Para floresta, profundidade ≤ `Δ`.
3. **Edge coloring** — colora as arestas para que arestas adjacentes tenham
   cores diferentes; cada cor é uma camada. Garante profundidade = `Δ` para
   florestas.
4. **Degree cap** — limita o grau de cada nó a `d_max`. Com `d_max = 2`, a
   floresta vira coleção de caminhos; profundidade ≤ 2. Com `d_max = 3`,
   profundidade ≤ 3.
5. **Path decomposition** — transforma cada componente em caminho (DFS). Um
   caminho pode ser agendado em 2 camadas alternando a direção. O estado vira
   uma cadeia GHZ-like; a correlação de ponta a ponta é a paridade dos tipos.
6. **Aceitar a estrela** — se o grau máximo for 3–4, a profundidade é
   aceitável; se for `S-1`, o circuito deixa de ser raso.

### 1.4 Recomendação

`d_max = 2` (coleção de caminhos) + greedy packing. Profundidade ≤ 2 camadas de
CNOT. A estrela do paper vira caminho; nem todos os pares ficam diretamente
conectados, mas a correlação de ponta a ponta é preservada pela paridade.

Se a estrela exata for necessária, limitar o fan-out a 3–4 e usar múltiplos
centros. A profundidade fica ≤ 4.

### 1.5 Em aberto

- Aceitar caminho no lugar da estrela?
- Qual `d_max`: 2 (mais raso) ou 3–4 (mais fiel à estrela)?
- O `num_portas` do genoma precisa reservar quantas camadas para CNOTs?

---

## 2. Bias (`p_a`)

### 2.1 Por que importa

O baseline aplica `RY(θ)` para que a probabilidade de medir o bit do melhor pai
seja `p_a = 0.95`. O pareamento decide **quais** bits estão correlacionados; o
`p_a` decide **quais valores** eles tendem a assumir. Sem o bias, o filho é um
estado entrelaçado aleatório; o algoritmo perde o puxão para o melhor pai.

Exemplo de 1 qubit:

- `RY(θ)` com `θ = 2·arccos(√0.95) ≈ 0.451` → `P(0) = 0.95`;
- `I` → `P(0) = 1.0`;
- `H` → `P(0) = 0.5`;
- `X` → `P(1) = 1.0`.

### 2.2 Limitação discreta

Com o grupo Hilbert 1Q, não existe uma porta única que dê 0.95:

| Preparação | `P(0)` | Observação |
|---|---|---|
| `I` | 1.0 | determinístico |
| `H T H` | `cos²(π/8) ≈ 0.8536` | coerente, 14.6% de exploração |
| `H` | 0.5 | máxima exploração |
| `H T† H` | `≈ 0.1464` | coerente, 85.4% de exploração |
| `H T T H` | 0.5 | volta ao meio |

Para chegar perto de 0.95 com `H T^k H`, é preciso mais `T` gates; a
profundidade cresce.

### 2.3 Opções

| Opção | Marginal | Coerente? | Prós | Contras |
|---|---|---|---|---|
| Sem bias (`H`) | 0.5 | sim | máxima exploração | sem explotação |
| Determinístico `X/I` | 1.0 | sim | raso, simples | exploração só via mutação |
| `H parcial` | 0 ou 0.5 | sim | enviesa sem perder H nos outros | inicialização não uniforme |
| `H T H` | 0.8536 | sim | bias quântico discreto | 14.6% de exploração, não 5% |
| `RY` | 0.95 | sim | fiel ao baseline | volta a usar rotação |
| Clássico 0.95 | 0.95 | não | marginal exata, raso | mistura clássica, não superposição |

### 2.4 Taxa efetiva de flip

O 5% não é “alto” ou “baixo” em abstrato; é o orçamento de exploração. Em `n`
bits, o número esperado de flips é `0.05n`:

- `n = 30` → ~1,5 bits;
- `n = 100` → ~5 bits.

Para sustentar “meu algoritmo explora mais”, medir a taxa efetiva:

```python
flip_rate = (1 / n) * (1 / k) * sum(
    hamming(bitstring_t, best_bitstring) for t in range(k)
)
```

Se o seu gate set produz flip rate maior, isso é evidência — mas se você também
mudou o bias, o efeito é confundido. A ablação com o mesmo flip rate separa os
dois.

### 2.5 Recomendação e status

- Primeiro protótipo: **determinístico `X/I`** ou **clássico 0.95**.
- Ablations: `H T H`, `RY`, sem bias.
- **On-hold:** escolher o bias definitivo só depois de ver o efeito no fitness e
  na entropia.

---

## 3. Loop do EAQGA × elitismo

### 3.1 O loop do baseline

```
Init: N circuitos, cada um H^n; mede cada um uma vez -> N bitstrings.

Para t = 1..T:
    avalia os N bitstrings (fitness)
    pool = top-2 bitstrings
    para cada um dos N slots:
        gera um circuito novo a partir dos 2 bitstrings do pool
        mede uma vez -> novo bitstring
    os N novos bitstrings são a próxima geração
```

Não existem “filhos não escolhidos”: **todos os N slots são preenchidos** por
novos circuitos gerados a partir dos mesmos dois bitstrings. Os top-2 são
pais/templates, não sobreviventes. A população anterior é descartada.

O paper é ambíguo entre pool **acumulado** (top-2 global) e **por geração**.
A escolha do grupo foi **por geração**.

### 3.2 Nosso híbrido

```
Elite pool = top-2 pais (circuitos) por fitness médio.
Próxima população = N filhos.
Filho = cópia do genoma 1Q de um pai (A ou B)
      + nova floresta (P/N, p_s, cadeias)
      + mutação 1Q
```

Os pais **não** são copiados intactos: o filho herda o genoma 1Q de um pai, mas
recebe uma floresta nova e sofre mutação. O “elitismo” aqui é pressão seletiva
(só dois reproduzem), não sobrevivência dos pais. Todos os N slots são filhos; o
baseline também não copia os pais, ele regenera todos os N a partir dos dois
bitstrings.

### 3.3 Bitstrings fixos × reamostrados

No baseline, cada pai tem **um** bitstring por geração (1 shot); todos os N
filhos usam os mesmos dois.

Com `k = 50`, cada pai tem 50 bitstrings. Duas opções:

- **Fixo por geração:** escolhe um representante por pai (melhor shot), e todos
  os filhos usam esse par. É o análogo fiel ao baseline.
- **Reamostrado por filho:** cada filho tira um shot novo do circuito pai; o
  par de bitstrings varia. Mais diversidade, mas não é o baseline.

**Recomendação:** fixo por geração; reamostragem como ablation.

### 3.4 Em aberto

- Confirmar se o pool é por geração (sim) e se os pais são copiados (não: só o
  genoma 1Q é herdado).
- Definir o tie-break quando os dois pais são idênticos.

---

## 4. `k = 50` e o estimador

O ponto dos shots é estimar o **fitness médio** do indivíduo, que é volátil.
Com `k = 50`, o erro padrão cai. Em simulação, o custo dominante é a evolução do
circuito, não os shots; `k = 50` é barato.

Cuidados:

- O paper usa **1 shot**. Com `k = 50`, o estimador é diferente; a comparação
  não é apples-to-apples. Ou rode o baseline com `k = 50`, ou rotule como
  “estimador diferente”.
- O fitness médio é o critério de **seleção**; o melhor shot é o **fenótipo** do
  crossover.
- Usar as mesmas seeds comuns entre algoritmos reduz a variância da comparação.
- Two-stage (opcional): `k1 = 10` para ranquear, `k2 = 100` para refinar os
  top-2. Melhora a seleção, mas é outro algoritmo.

---

## 5. Métricas de exploração (on-hold)

O baseline reporta **média e desvio do melhor fitness final entre `R`
execuções**. Isso mede performance e estabilidade, não exploração.

Para a tese “mais/diferentes portas → mais exploração”, medir dentro da execução:

- **entropia de Shannon** dos bitstrings medidos por geração;
- **número de bitstrings únicos** por geração;
- **distância de Hamming média** par-a-par;
- **variância do fitness** da população;
- **cobertura** do espaço/ótimo (opcional);
- **taxa efetiva de flip** em relação ao melhor bitstring.

As três primeiras são baratas e diretamente ligadas à exploração. A taxa de flip
liga a métrica ao bias.

**Status:** escolher a métrica primária depois do primeiro protótipo.

---

## 6. Comparação × ablação

### 6.1 Comparação de algoritmos

Responde “qual é melhor?”:

| Algoritmo | Papel |
|---|---|
| EAQGA (RY) | referência |
| Nosso (Hilbert 1Q + CNOT) | proposto |
| GA | baseline clássico |
| AQGA | baseline quântico-inspirado (números reportados) |

### 6.2 Ablação

Responde “por quê?”. Mínima:

| Variante | Isola |
|---|---|
| Hilbert 1Q + CNOT | nosso |
| Hilbert 1Q, sem CNOT | estado produto (controle) |
| Clifford 1Q + CNOT | gate set (T/T† vs não-T) |
| Hilbert 1Q + bias RY | efeito do bias |
| Hilbert 1Q com mais/menos 1Q | efeito do número de portas (se variável) |

Sem a ablação, um fitness melhor pode ser devido ao gate set, ao número de
portas, ao entrelaçamento ou ao bias — e não dá para atribuir.

### 6.3 AQGA/GA

- “Vindos do baseline” = **números reportados**, não reexecutados. Só são
  comparáveis nos mesmos subconjuntos, mesmo `q` e mesmo orçamento.
- Para os benchmarks contínuos, os números do paper não se aplicam.
- O GA clássico de `classical/` é um controle opcional e barato.

---

## 7. Número de 1Q fixo

Se o número de portas 1Q é fixo (ou limitado), **“mais portas” não é a
variável**. As variáveis passam a ser:

- **gate set:** RY vs Hilbert (com `T/T†`) vs Clifford;
- **estrutura de CNOTs:** quantidade e topologia.

O Hilbert inclui `T/T†`; o Clifford não. O `T` cria superposições não-Clifford
(magic), o que pode aumentar a exploração **sem aumentar o número de portas**.
Hipótese limpa: mesmo número de 1Q, mesma profundidade, mesmo grafo de CNOTs,
Hilbert vs Clifford. Se Hilbert explora mais, é o gate set, não a contagem.

“Só a estrutura do indivíduo muda” — exato. Mas a alegação deve ser
“**estrutura/gate set** → exploração”, não “mais portas → exploração”. A segunda
exige variar a contagem.

---

## 8. MPS × statevector

- **Statevector** guarda `2^n` amplitudes. `n = 30` → ~16 GB; `n = 40` →
  inviável; `n = 100` → nem se fala.
- **MPS** guarda uma rede de tensores com bond dimension `χ`; custo cresce com
  `χ³`, não com `2^n`.

O baseline usa MPS porque 100 qubits. Para ser similar, usar **MPS** nos
experimentos de escala (30/40/100). Statevector fica como debug para `n ≤ 20`.

Por que a floresta importa: uma floresta de CNOTs + 1Q locais produz um **graph
state de árvore** (a menos de rotações locais). Esses estados têm entrelaçamento
limitado; o MPS os representa com `χ` pequeno (muitas vezes 2–4). Se o grafo tem
ciclos ou alta conectividade, `χ` cresce e o MPS fica caro. Portas 1Q **não
aumentam** `χ`; só as 2Q aumentam.

A floresta é um grafo, não uma direção. O desenho “de cima para baixo” é
arbitrário; a profundidade é determinada pelo compartilhamento de qubits, não
pela posição no desenho.

---

## 9. Escalabilidade e orçamento (on-hold)

- **Covariância `O(S²)`:** para `S = 30/40/100`, é trivial. Para `S ≥ 1000`, a
  matriz e o sorteio de pares pesam; top-k/threshold por ativo é uma saída.
  Isso pode virar um artigo separado de HPC aplicada ao problema.
- **Orçamento de circuitos:** `N × T` evoluções MPS. O gargalo é a evolução, não
  os shots.
- **Genoma:** `O(S × D)`; não é o gargalo.
- **“Escalabilidade dos indivíduos”** pode significar: número de qubits,
  profundidade, número de portas, memória do genoma ou tempo de simulação. Cada
  um tem um gargalo diferente. Definir qual antes de otimizar.

---

## 10. Já decidido (não reabrir sem motivo)

- Leitura **A**: a floresta é reconstruída; o genoma 1Q é herdado.
- `S = n`, portfólio QUBO.
- Top-2 **por geração**, por fitness médio.
- `k = 50`.
- Melhor shot **por elite** (não os 2 melhores shots globais).
- Mutação por **substituição** de 1Q; CNOTs intocados.
- Camada `H` inicial fixa.
- CNOT é a única porta multi-qubit.
- AQGA **não** será implementado; números do paper como referência.
- Dados: `qiskit-finance==0.4.1` funciona com `qiskit==2.5.2`; `YahooDataProvider`
  precisa de `yfinance` com `auto_adjust=False`. `RandomDataProvider` funciona
  sem ajuste. Caminho preferido: `yfinance` direto + fórmulas de `μ`/`Σ`.
- Avaliação: `EvaluationMode.SCALAR` é o default; `VECTORIZED` é opt-in via
  `HybridConfig`/`BaselineConfig` e usa `QUBOInstance.fitness_many`.

---

## 11. Decisões abertas

1. **Forma da floresta:** caminho (`d_max = 2`) ou estrela com fan-out limitado?
2. **Bias:** determinístico `X/I`, clássico 0.95, `H T H`, `RY` ou sem bias?
3. **Bitstrings fixos ou reamostrados por filho?** (recomendação: fixos)
4. **Métrica primária de exploração:** entropia, unicidade ou Hamming?
5. **`k` e two-stage:** manter 50 ou refinar o top-2 com mais shots?
6. **Prioridade da ablação:** sem CNOT + Clifford vs Hilbert + bias RY é o
   mínimo; “mais/menos 1Q” só se a contagem variar.

---

## 12. Próximos passos

1. Fechar forma da floresta e bias (itens 1–2).
2. Escrever o spec de genótipo/fenótipo (o que é herdado, gerado e mutado).
3. Implementar o EAQGA fiel (P0) e validar contra as Tabelas I/II.
4. Implementar o híbrido (P2–P4) com a ablação mínima.
5. Escolher a métrica de exploração e rodar a estatística.
