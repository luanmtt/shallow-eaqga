# original/ — motor do QGA antigo (vendorizado, sem edição)

Cópia **byte-idêntica** dos arquivos do repositório `~/projects/qga` (commit
`d8dbc36`), feita em 2026-11-01, para servir de segundo baseline.

| arquivo | origem |
|---|---|
| `src/Individuo.py` | `quantum/src/Individuo.py` |
| `src/ParallelQuantumGA.py` | `quantum/src/ParallelQuantumGA.py` |
| `src/config.py` | `quantum/src/config.py` |
| `src/metrics.py` | `quantum/src/metrics.py` |
| `utils/*` | `utils/*` |

Nada aqui é editado. Se precisar mudar comportamento, mude o adapter
(`eaqga/original_adapter.py`) ou copie de novo.

## O que ficou de fora

- `benchmark_functions.py`: ninguém no motor importa — só o `qga.py` antigo
  usava. Como o nosso problema é o QUBO (domínio `(0,1)`, 1 qubit por
  variável), a função de fitness vem de `QUBOInstance`, não de benchmark.
- `visualization/`, `notebooks/`, `scripts/`, `qga.py`: são a casca do CLI
  antigo, não o motor.

## Por que `utils/` veio junto

`ParallelQuantumGA.py` importa `utils.progressbar`, `utils.formats` e
`utils.embellish` no topo. Sem essas três cópias o motor não importa. Elas são
utilitários de terminal, não visualização: `visualization/` ficou de fora.

## Por que `src/`

`ParallelQuantumGA.py` faz `from src.Individuo import *`, e `Individuo.py` faz
`from config import GAConfig`. A estrutura de diretórios é parte do contrato de
import: com `original/` e `original/src` no `sys.path`, tudo resolve sem tocar
em uma linha.

## Como é usado

`eaqga/original_adapter.py`:

1. coloca `original/` e `original/src` no `sys.path`;
2. importa `src.Individuo` **e** `ParallelQuantumGA` (mesmo objeto de módulo);
3. define `N_QUBITS = 1` (1 qubit por variável = binário), sem editar o arquivo;
4. troca o símbolo `AerSimulator` do módulo por MPS quando o backend é MPS
   (em 30 qubits, `AerSimulator()` automático tentaria statevector de 16 GiB);
5. roda o motor com `verbose=False` e captura o stdout.
