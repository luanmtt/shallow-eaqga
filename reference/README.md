# reference — código antigo só para consulta

Este diretório **não** é importado pelo pacote `eaqga`. Ele guarda o código do QGA
antigo que serve de referência para a implementação nova.

- `Individuo.py`: indivíduo do QGA contínuo (8 qubits por variável, fitness por
  média de valores reais, mutação/crossover multi-qubit). Útil para consultar
  `_apply_gate` (mapeamento 1Q/CNOT → Qiskit), `construir_circuito_qiskit` e o uso
  de `AerSimulator`. **Não** usar como classe base do EAQGA.
- `config.py`: `GAConfig` antigo. Serve de referência de estilo; a configuração
  nova fica em `eaqga/config.py`.

O novo indivíduo será binário (`S = n`), com `H^S` fixo, genoma de portas 1Q
Hilbert, floresta de CNOTs e mutação apenas nas 1Q.
