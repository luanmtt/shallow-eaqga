# 04 — Aer: memória, paralelismo e GPU

Este documento registra o que aprendemos com a codebase antiga
(`~/projects/qga/quantum/docs/aerGPU.md` e `develop.md`) e o que decidimos aqui.
Ele existe para não repetirmos dois erros: estourar memória com `statevector` e
paralelizar no lugar errado.

---

## 1. O gargalo é o `run()`, não o circuito

Medido na codebase antiga (16 qubits, 6 camadas):

| etapa (por circuito) | custo |
|---|---:|
| construir o circuito | 0.9 ms |
| construir `AerSimulator()` | 0.03 ms |
| `run()` + `result()`, 1024 shots | **169 ms** |
| `run()` + `result()`, statevector exato | **114 ms** |
| 50 circuitos em **um** `run()` | ~17% menos |

Ou seja: 99% do tempo é overhead fixo do Aer por chamada. Duas consequências:

1. **batching** (todos os circuitos de uma geração num único `run()`) rende, mas
   pouco (~17%) e muda a estrutura de avaliação;
2. **paralelizar por circuito** é desperdício: os workers passam o tempo no
   mesmo overhead e disputam CPU.

Decisão neste repo: paralelizamos **por execução** (subset × seed × algoritmo),
com `joblib`. Cada worker roda um experimento inteiro; o CSV é escrito conforme
as execuções terminam (`return_as="generator_unordered"`).

Batching fica como otimização futura, não implementada.

## 2. Memória: `statevector` não escala

O statevector de `S` qubits em `complex128` ocupa `16 B × 2^S`:

| `S` | memória |
|---:|---:|
| 20 | 16 MiB |
| 24 | 256 MiB |
| 28 | 4 GiB |
| **30** | **16 GiB** |
| 34 | 256 GiB |
| **40** | **16 TiB** |
| 100 | impraticável |

Em `float32`, metade. Mesmo assim: `S=30` não cabe confortavelmente nos 23 GB
desta máquina junto com o processo; `S=40` é impossível.

Era exatamente esse o problema da codebase antiga: 32 qubits = 64 GiB por
circuito. E com `joblib`, a memória total é `n_jobs × memória_por_processo` —
dez workers com statevector de 30 qubits pediriam 160 GiB.

Decisões:

- **MPS é o backend dos experimentos** (`BackendKind.MPS`), como no paper, que
  usou o simulador MPS justamente por isso;
- `statevector` fica restrito a `S` pequeno (testes, dummy, `S ≤ 20`);
- `experiments/paper_tables.py` imprime a estimativa de statevector no cabeçalho
  e usa MPS sempre;
- `--jobs` default é `min(8, cpu_count)`; a memória cresce linearmente com ele.

O MPS tem custo `O(S·χ²)`; para circuitos rasos `χ` fica pequeno. Medimos
`S=30`, 50 shots: ~62 ms por circuito, 24 avaliações em 1.5 s.

## 3. GPU: fora por enquanto

Benchmark da codebase antiga (RTX 2050, 4 GB, `qiskit-aer-gpu` 0.15.1):

| qubits | CPU | GPU | speedup |
|---:|---:|---:|---:|
| 16 | 7.3 ms | 4.1 ms | 1.79× |
| 24 | 6.5 ms | 6.6 ms | 0.98× |

Conclusão de lá: em 16-24 qubits o ganho é nulo ou modesto porque o custo de
transferência CPU→GPU→CPU domina. GPU compensaria com 28+ qubits, população
1000+ e placa de alto end — nenhum dos três é o nosso caso (população 10-20,
`S = 30-100` em MPS).

Decisão: **não implementar agora**. Quando entrar, o desenho já está definido:

```
statevector_gpu → statevector → automatic   (fallback, nunca quebrar)
```

com `use_gpu: bool` no config, um simulador por worker (nunca por circuito) e
paridade de fitness CPU × GPU no mesmo seed. Para `S ≥ 28` a GPU statevector
continua inviável em 4 GB; o caminho realista é MPS (CPU) ou batching.

## 3.1 Onde o tempo vai antes do primeiro resultado

O cabeçalho do `paper_tables` sai quase instantâneo; o primeiro resultado não.
Medido com `--timings` (16 cores, `S=30`):

| fase | tempo |
|---|---:|
| carregar preços do cache + `μ`/`Σ` | 0.13 s |
| busca local de referência | 0.02 s |
| **primeiro resultado** (após `tarefas`) | **2.55 s** |
| execução das 8 tarefas | 4.90 s |

O intervalo é o `joblib`/loky: criar os workers e **importar `eaqga.runner`
(→ Qiskit/Aer) em cada processo** custa ~0.95 s por worker, em paralelo. Com 4
workers, o dispatch puro leva ~1.1 s; o resto é a primeira tarefa.

Alternativas testadas:

| backend | 1º resultado | 8 tarefas |
|---|---:|---:|
| `loky` (processos) | 2.55 s | 4.90 s |
| `threading` | **9.96 s** | **21.6 s** |

`threading` é **4× pior**: a construção do circuito e o wrapper do Aer seguram o
GIL, e cada simulador abre o próprio pool OpenMP (oversubscription). Fica
`loky`.

Mitigações aplicadas: linha `• workers` antes do dispatch (explica a espera),
`--timings` para medir, e `--jobs 1` para ver a barra por geração sem spawn.

## 4. O que medimos aqui

`experiments/paper_tables.py`, 16 cores, 23 GB:

| cenário | tarefas | tempo | observação |
|---|---:|---:|---|
| `--quick` (`S=10`, pop 5, 3 ger, 2 runs, 2 subsets) | 8 | ~2 s | 4.3 execuções/s |
| `S=30`, pop 5, 3 ger, 2 runs, 1 subset | 4 | ~2.5 s | MPS |
| mesmo, `pop=4`, 1 ger, universo S&P 500 inteiro | 2 | ~4 s | cache de 495 tickers |

Primeira carga dos 495 tickers do S&P 500: **5m30s** (rede). Depois disso o
cache em `data/prices/` responde em segundos.

## 5. Regras práticas

1. `S ≥ 25` → MPS, sempre.
2. `--jobs` ≤ núcleos físicos; a memória é `jobs × por_processo`.
3. Não paralelizar por circuito; paralelizar por execução.
4. CSV incremental, nunca acumular resultados em memória até o fim.
5. Antes de medir tempo, medir memória: `16 B × 2^S × jobs`.
