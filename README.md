# shallow-eaqga

Repositório para formular um Algoritmo Genético Quântico com baixa profundidade e
atenção aos emaranhamentos.

O alvo é o **EAQGA** ([arXiv:2504.17923](https://arxiv.org/abs/2504.17923)) como
baseline, com um indivíduo binário próprio: `H^S` fixo, portas 1Q Hilbert,
floresta de CNOTs e mutação apenas nas 1Q.

## Escopo

- **EAQGA fiel** (RY) é o baseline a bater.
- **Nosso indivíduo:** binário, `S = n`, `H^S` fixo + portas 1Q Hilbert +
  floresta de CNOTs.
- **Problema inicial:** portfólio QUBO (mesmos subconjuntos e período do paper,
  quando possível).
- **AQGA não será implementado**: os números de AQGA/GA do paper entram como
  referência publicada.
- Benchmarks contínuos ficam para depois.

## Estrutura

```text
eaqga/
├── eaqga/
│   ├── __init__.py
│   └── utils/           # utilitários copiados do repo QGA
│       ├── __init__.py
│       ├── embellish.py
│       ├── formats.py
│       └── progressbar.py
├── docs/                # especificação, plano, diff e leftovers
├── reference/           # código antigo, só para consulta
│   ├── Individuo.py
│   └── config.py
├── tests/
├── requirements.txt
└── README.md
```

## Status

- [x] Documentação de arquitetura e baseline
- [x] Cross-check com o guia do Gemini
- [x] Arquitetura: `S = n`, Leitura A, 1Q Hilbert + floresta de CNOTs
- [x] Pipeline de dados verificado: `qiskit-finance 0.4.1` + `yfinance` com
  `auto_adjust=False` (o provider falha com o default novo do `yfinance`)
- [x] Primeiro lote de utilitários (`utils/`) e referência
- [ ] Fechar forma da floresta e bias
- [ ] Spec de genótipo/fenótipo
- [ ] EAQGA fiel (baseline)
- [ ] Indivíduo híbrido + população + runner
- [ ] Portfólio QUBO e comparação estatística
